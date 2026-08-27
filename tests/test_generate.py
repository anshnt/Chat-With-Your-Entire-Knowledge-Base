"""Generation tests.

The properties that matter for a *grounded* generator are not fluency but:
citation markers attached to the right sentence, hallucinated markers stripped,
off-topic questions refused, and context packed whole.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from kb.config import GenerationProvider, Settings
from kb.generate import build_generator
from kb.generate.base import Generator
from kb.generate.extractive import ExtractiveGenerator
from kb.generate.prompt import (
    build_prompt,
    format_sources,
    looks_like_refusal,
    pack_context,
    parse_markers,
    split_into_sentences,
    strip_invalid_markers,
)
from kb.knowledge_base import KnowledgeBase
from kb.models import Chunk, PdfLocator, ScoredChunk, SourceType, TextLocator


def chunk(
    chunk_id: str,
    text: str,
    *,
    title: str = "Doc",
    heading: str = "",
    page: int | None = None,
) -> Chunk:
    locator = (
        PdfLocator(page=page, page_count=10, file_url="/files/d.pdf")
        if page
        else TextLocator(line_start=1, line_end=2, file_path="d.md")
    )
    return Chunk(
        id=chunk_id,
        document_id="doc_1",
        ordinal=0,
        text=text,
        locator=locator,
        document_title=title,
        source_type=SourceType.PDF if page else SourceType.MARKDOWN,
        heading_context=heading,
    )


def scored(chunk_id: str, text: str, score: float = 0.5, **kwargs) -> ScoredChunk:
    return ScoredChunk(chunk=chunk(chunk_id, text, **kwargs), score=score)


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #


class TestPackContext:
    def test_takes_chunks_in_rank_order(self) -> None:
        candidates = [scored(str(i), f"text {i}", score=1.0 - i / 10) for i in range(4)]
        assert [c.id for c in pack_context(candidates, token_budget=10_000)] == [
            "0",
            "1",
            "2",
            "3",
        ]

    def test_respects_the_token_budget(self) -> None:
        candidates = [scored(str(i), "word " * 400) for i in range(10)]
        packed = pack_context(candidates, token_budget=600)
        assert 0 < len(packed) < 10

    def test_never_truncates_a_chunk(self) -> None:
        """A partial chunk is a citation pointing at text the model never saw."""
        body = "word " * 400
        packed = pack_context([scored("a", body)], token_budget=10)
        assert packed[0].text == body

    def test_first_chunk_is_always_included(self) -> None:
        """Returning nothing because the top hit is large would be worse."""
        assert len(pack_context([scored("a", "word " * 5000)], token_budget=50)) == 1

    def test_max_chunks_caps_the_count(self) -> None:
        candidates = [scored(str(i), "short text") for i in range(10)]
        assert len(pack_context(candidates, token_budget=10_000, max_chunks=3)) == 3

    def test_empty_candidates(self) -> None:
        assert pack_context([], token_budget=1000) == []


class TestFormatSources:
    def test_numbers_sources_from_one(self) -> None:
        rendered = format_sources([chunk("a", "alpha"), chunk("b", "beta")])
        assert rendered.startswith("[1] Doc")
        assert "[2] Doc" in rendered

    def test_includes_the_position_label(self) -> None:
        rendered = format_sources([chunk("a", "alpha", page=7)])
        assert "p. 7 / 10" in rendered

    def test_prompt_contains_query_and_sources(self) -> None:
        prompt = build_prompt("what is RRF?", [chunk("a", "RRF fuses ranks.")])
        assert "what is RRF?" in prompt
        assert "RRF fuses ranks." in prompt
        assert "[1]" in prompt


# --------------------------------------------------------------------------- #
# marker parsing
# --------------------------------------------------------------------------- #


class TestParseMarkers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A claim [1].", [1]),
            ("A claim [1, 2].", [1, 2]),
            ("A claim [1,2].", [1, 2]),
            ("A claim [1][3].", [1, 3]),
            ("A claim [2] and another [2].", [2]),
            ("No markers here.", []),
            ("", []),
            ("Brackets [but not numbers].", []),
        ],
    )
    def test_parses_every_form(self, text: str, expected: list[int]) -> None:
        assert parse_markers(text) == expected

    def test_preserves_first_appearance_order(self) -> None:
        assert parse_markers("[3] then [1] then [3] then [2]") == [3, 1, 2]


class TestStripInvalidMarkers:
    def test_valid_markers_survive(self) -> None:
        text, invalid = strip_invalid_markers("A claim [1].", {1, 2})
        assert text == "A claim [1]."
        assert invalid == []

    def test_hallucinated_markers_are_removed(self) -> None:
        """A citation chip that leads nowhere looks like evidence. It must go."""
        text, invalid = strip_invalid_markers("A claim [9].", {1, 2})
        assert "[9]" not in text
        assert invalid == [9]

    def test_partial_group_keeps_the_valid_half(self) -> None:
        text, invalid = strip_invalid_markers("A claim [1, 9].", {1, 2})
        assert text == "A claim [1]."
        assert invalid == [9]

    def test_punctuation_is_not_orphaned(self) -> None:
        text, _ = strip_invalid_markers("A claim [9] .", {1})
        assert "  " not in text
        assert text.endswith(".")

    def test_zero_is_invalid(self) -> None:
        _, invalid = strip_invalid_markers("A claim [0].", {1})
        assert invalid == [0]


class TestSplitIntoSentences:
    def test_trailing_marker_attaches_to_its_own_sentence(self) -> None:
        """The bug this guards: citations follow the claim, so a naive split
        attributes them to the *next* sentence and every verdict is meaningless."""
        text = "The constant defaults to 60. [1] Reranking is separate. [2]"
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "defaults to 60" in sentences[0].text
        assert sentences[0].citation_markers == [1]
        assert "Reranking is separate" in sentences[1].text
        assert sentences[1].citation_markers == [2]

    def test_marker_only_tail_does_not_become_a_sentence(self) -> None:
        sentences = split_into_sentences("One claim. [1]")
        assert len(sentences) == 1
        assert sentences[0].citation_markers == [1]

    def test_inline_markers_are_attributed_correctly(self) -> None:
        text = "First claim [1]. Second claim [2]."
        sentences = split_into_sentences(text)
        assert [s.citation_markers for s in sentences] == [[1], [2]]

    def test_char_offsets_locate_the_sentence(self) -> None:
        text = "First claim [1]. Second claim [2]."
        for sentence in split_into_sentences(text):
            assert text[sentence.char_start : sentence.char_end] == sentence.text

    def test_multiple_markers_on_one_sentence(self) -> None:
        sentences = split_into_sentences("A contested claim. [1, 3]")
        assert sentences[0].citation_markers == [1, 3]

    def test_uncited_sentences_have_no_markers(self) -> None:
        sentences = split_into_sentences("An unsupported claim. Another one.")
        assert all(not s.is_cited for s in sentences)

    def test_paragraphs_are_handled(self) -> None:
        sentences = split_into_sentences("First para. [1]\n\nSecond para. [2]")
        assert [s.citation_markers for s in sentences] == [[1], [2]]

    def test_empty_text(self) -> None:
        assert split_into_sentences("") == []


class TestLooksLikeRefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "The sources do not contain an answer to this question.",
            "I cannot answer this from the provided sources.",
            "There is no information about that here.",
        ],
    )
    def test_detects_refusals(self, text: str) -> None:
        assert looks_like_refusal(text)

    def test_a_real_answer_is_not_a_refusal(self) -> None:
        assert not looks_like_refusal("The damping constant defaults to 60. [1]")


# --------------------------------------------------------------------------- #
# extractive generator
# --------------------------------------------------------------------------- #


class TestExtractiveGenerator:
    def test_answers_with_the_relevant_sentence(self) -> None:
        generator = ExtractiveGenerator()
        candidates = [
            scored("a", "The RRF damping constant defaults to 60. Other prose follows here."),
            scored("b", "Cross-encoders are slower than bi-encoders by a wide margin."),
        ]
        answer = generator.generate("what does the RRF damping constant default to", candidates)
        assert "defaults to 60" in answer.text
        assert answer.citations
        assert answer.citations[0].marker == 1

    def test_every_sentence_is_verbatim_from_a_source(self) -> None:
        """Extractive generation cannot hallucinate. That is the whole point."""
        generator = ExtractiveGenerator()
        source = (
            "Reciprocal Rank Fusion combines ranked lists using ranks rather than scores. "
            "The damping constant k defaults to 60 in the standard formulation."
        )
        candidates = [scored("a", source)]
        answer = generator.generate("reciprocal rank fusion damping constant", candidates)
        for sentence in answer.sentences:
            stripped = sentence.text.split(" [")[0].strip()
            assert stripped in source

    def test_refuses_an_off_topic_question(self) -> None:
        """The worst failure mode: a confident, fully-cited, irrelevant answer."""
        generator = ExtractiveGenerator()
        candidates = [
            scored("a", "Reciprocal Rank Fusion combines ranked lists using ranks not scores."),
            scored("b", "A cross-encoder scores each query-document pair jointly and slowly."),
        ]
        answer = generator.generate("what is the capital of France", candidates)
        assert answer.refused
        assert not answer.citations

    def test_no_context_refuses_instead_of_answering_from_memory(self) -> None:
        answer = ExtractiveGenerator().generate("anything at all", [])
        assert answer.refused
        assert answer.context_chunks == 0
        assert not answer.citations

    def test_selection_is_diversified(self) -> None:
        """Four near-copies must not become four sentences saying one thing."""
        generator = ExtractiveGenerator(max_sentences=3)
        duplicate = "The damping constant for fusion defaults to sixty in this system."
        candidates = [scored(str(i), duplicate) for i in range(4)]
        answer = generator.generate("damping constant fusion default", candidates)
        bodies = [s.text.split(" [")[0] for s in answer.sentences]
        assert len(bodies) == len(set(bodies))

    def test_heading_prefix_is_not_quoted_back(self) -> None:
        generator = ExtractiveGenerator()
        heading = "Retrieval › Fusion"
        body = "The damping constant defaults to 60 in the standard formulation."
        candidates = [scored("a", f"{heading}\n\n{body}", heading=heading)]
        answer = generator.generate("damping constant default", candidates)
        assert "›" not in answer.text

    def test_table_rows_and_fences_are_skipped(self) -> None:
        generator = ExtractiveGenerator()
        candidates = [
            scored(
                "a",
                "| metric | value | note |\n"
                "The damping constant defaults to 60 in the standard formulation.",
            )
        ]
        answer = generator.generate("damping constant default", candidates)
        assert "|" not in answer.text

    def test_context_accounting_is_reported(self) -> None:
        candidates = [scored("a", "The damping constant defaults to 60 for fusion purposes.")]
        answer = ExtractiveGenerator().generate("damping constant", candidates)
        assert answer.context_chunks == 1
        assert answer.context_tokens > 0
        assert "generation_ms" in answer.timings_ms
        assert "context_ms" in answer.timings_ms

    def test_deterministic(self) -> None:
        candidates = [
            scored("a", "The damping constant defaults to 60 for fusion purposes."),
            scored("b", "Cross-encoders score pairs jointly and are therefore slow."),
        ]
        first = ExtractiveGenerator().generate("damping constant default", candidates)
        second = ExtractiveGenerator().generate("damping constant default", candidates)
        assert first.text == second.text


# --------------------------------------------------------------------------- #
# base-class behaviour, via a stub provider
# --------------------------------------------------------------------------- #


class _StubGenerator(Generator):
    """A generator that returns whatever text the test hands it."""

    name = "stub"
    model = "stub-v1"
    supports_streaming = True

    def __init__(self, reply: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reply = reply
        self.prompts: list[str] = []

    def _generate_text(self, query: str, chunks: Sequence[Chunk]) -> str:
        self.prompts.append(build_prompt(query, chunks))
        return self.reply

    def _stream_text(self, query: str, chunks: Sequence[Chunk]) -> Iterator[str]:
        self.prompts.append(build_prompt(query, chunks))
        for i in range(0, len(self.reply), 7):
            yield self.reply[i : i + 7]


class TestGeneratorPipeline:
    def test_citations_resolve_to_chunk_positions(self) -> None:
        generator = _StubGenerator("The answer is 60. [1] Something else. [2]")
        candidates = [scored("a", "alpha text", page=3), scored("b", "beta text", page=9)]
        answer = generator.generate("q", candidates)

        assert [c.marker for c in answer.citations] == [1, 2]
        assert answer.citations[0].chunk_id == "a"
        assert answer.citations[0].deep_link == "/files/d.pdf#page=3"
        assert answer.citations[1].deep_link == "/files/d.pdf#page=9"

    def test_hallucinated_markers_never_reach_the_answer(self) -> None:
        generator = _StubGenerator("Claim one. [1] Claim two. [7]")
        answer = generator.generate("q", [scored("a", "alpha")])
        assert "[7]" not in answer.text
        assert [c.marker for c in answer.citations] == [1]

    def test_only_cited_sources_are_listed(self) -> None:
        """Three chunks in context, one cited: the answer lists one."""
        generator = _StubGenerator("Only the second matters. [2]")
        candidates = [scored(str(i), f"text {i}") for i in range(3)]
        answer = generator.generate("q", candidates)
        assert [c.marker for c in answer.citations] == [2]
        assert answer.context_chunks == 3

    def test_uncited_answer_has_sentences_but_no_citations(self) -> None:
        generator = _StubGenerator("A bare claim with no citation at all.")
        answer = generator.generate("q", [scored("a", "alpha")])
        assert answer.sentences
        assert not answer.citations
        assert not answer.sentences[0].is_cited

    def test_streaming_yields_deltas_then_the_answer(self) -> None:
        generator = _StubGenerator("The answer is 60. [1] More detail here. [1]")
        events = list(generator.stream("q", [scored("a", "alpha")]))
        deltas = [d for d, a in events if a is None]
        finals = [a for _, a in events if a is not None]

        assert len(finals) == 1
        assert "".join(deltas) == generator.reply
        assert finals[0].text == generator.reply
        assert finals[0].citations

    def test_streaming_with_no_context_still_terminates(self) -> None:
        events = list(_StubGenerator("unused").stream("q", []))
        finals = [a for _, a in events if a is not None]
        assert len(finals) == 1
        assert finals[0].refused

    def test_prompt_contains_only_packed_chunks(self) -> None:
        generator = _StubGenerator("ok", max_chunks=2)
        candidates = [scored(str(i), f"unique-body-{i}") for i in range(5)]
        generator.generate("q", candidates)
        prompt = generator.prompts[0]
        assert "unique-body-0" in prompt
        assert "unique-body-4" not in prompt


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


class TestBuildGenerator:
    def test_extractive_is_the_default(self) -> None:
        assert isinstance(build_generator(Settings()), ExtractiveGenerator)

    def test_missing_key_falls_back_to_extractive(self, monkeypatch) -> None:
        """Degrading to extractive is safe: its output is verbatim from sources."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        generator = build_generator(Settings(generation_provider=GenerationProvider.ANTHROPIC))
        assert isinstance(generator, ExtractiveGenerator)

    def test_token_budget_is_taken_from_settings(self) -> None:
        generator = build_generator(Settings(context_token_budget=1234))
        assert generator.token_budget == 1234


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


@pytest.fixture
def ask_kb(tmp_path: Path) -> KnowledgeBase:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "retrieval.md").write_text(
        "# Retrieval\n\n"
        "## Hybrid search\n\n"
        "Hybrid search combines BM25 lexical matching with dense vector retrieval "
        "over the same corpus.\n\n"
        "## Reciprocal rank fusion\n\n"
        "Reciprocal Rank Fusion combines ranked lists using ranks rather than raw "
        "scores. The damping constant k defaults to 60.\n\n"
        "## Reranking\n\n"
        "A cross-encoder reranker scores each query-document pair jointly, which is "
        "accurate but too slow to run over a whole corpus.\n"
    )
    settings = Settings(
        data_dir=tmp_path / "data", embedding_dim=256, chunk_size=700, min_chunk_size=40
    )
    instance = KnowledgeBase(settings)
    instance.ingest(str(docs))
    return instance


class TestAskEndToEnd:
    def test_answers_from_the_corpus_with_a_working_link(self, ask_kb: KnowledgeBase) -> None:
        answer = ask_kb.ask("what does the RRF damping constant default to?", top_k=4)
        assert "60" in answer.text
        assert answer.citations
        assert answer.citations[0].deep_link
        assert answer.citations[0].label

    def test_retrieval_diagnostics_are_attached(self, ask_kb: KnowledgeBase) -> None:
        answer = ask_kb.ask("damping constant", top_k=3)
        assert answer.retrieval is not None
        assert answer.retrieval.results
        assert any(k.startswith("retrieval_") for k in answer.timings_ms)

    def test_off_corpus_question_is_refused(self, ask_kb: KnowledgeBase) -> None:
        answer = ask_kb.ask("who won the 1998 world cup?", top_k=4)
        assert answer.refused

    def test_empty_corpus_refuses(self, tmp_path: Path) -> None:
        instance = KnowledgeBase(Settings(data_dir=tmp_path / "empty", embedding_dim=128))
        answer = instance.ask("anything")
        assert answer.refused
        assert answer.context_chunks == 0
        instance.close()

    def test_streaming_end_to_end(self, ask_kb: KnowledgeBase) -> None:
        events = list(ask_kb.ask_stream("damping constant default", top_k=3))
        finals = [a for _, a in events if a is not None]
        assert len(finals) == 1
        assert finals[0].text

    def test_every_marker_in_the_text_has_a_citation(self, ask_kb: KnowledgeBase) -> None:
        answer = ask_kb.ask("how does reranking differ from fusion?", top_k=4)
        for marker in parse_markers(answer.text):
            assert answer.citation_for(marker) is not None
