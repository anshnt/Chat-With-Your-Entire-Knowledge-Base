"""Citation verification tests.

The cases that matter are the failures a reader cannot spot for themselves: a
fluent paraphrase with a wrong figure, a flat contradiction that shares every
word with its source, and a factual claim that quietly cites nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from kb.config import Settings, VerifyProvider
from kb.generate.prompt import split_into_sentences
from kb.knowledge_base import KnowledgeBase
from kb.models import (
    Answer,
    AnswerCitation,
    AnswerSentence,
    Chunk,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
    SupportVerdict,
    TextLocator,
)
from kb.verify import LexicalVerifier, build_verifier, is_claim, strip_markers
from kb.verify.base import Verifier
from kb.verify.lexical import _is_negated
from kb.verify.llm import parse_verdict

SOURCE = (
    "Reciprocal Rank Fusion combines ranked lists using ranks rather than raw scores. "
    "The damping constant k defaults to 60. "
    "A cross-encoder reranker supports joint scoring of the query and the document. "
    "Recall at 50 was measured at 0.82 on the internal benchmark."
)


def source_chunk(text: str = SOURCE, chunk_id: str = "chk_1") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc_1",
        ordinal=0,
        text=text,
        locator=TextLocator(line_start=1, line_end=4),
        document_title="Retrieval",
    )


def answer_with(text: str, *chunks: Chunk) -> Answer:
    """An answer citing ``chunks`` as [1], [2], … — the shape generation produces."""
    used = chunks or (source_chunk(),)
    return Answer(
        query="q",
        text=text,
        sentences=split_into_sentences(text),
        citations=[AnswerCitation.from_chunk(i, c) for i, c in enumerate(used, start=1)],
        retrieval=RetrievalResult(
            query="q",
            results=[ScoredChunk(chunk=c, score=1.0) for c in used],
            strategy=RetrievalStrategy.HYBRID,
        ),
    )


def verdict_of(text: str, *chunks: Chunk, threshold: float = 0.5) -> AnswerSentence:
    answer = LexicalVerifier(threshold=threshold).verify(answer_with(text, *chunks))
    return answer.sentences[0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class TestStripMarkers:
    def test_removes_markers(self) -> None:
        assert strip_markers("A claim. [1]") == "A claim."

    def test_removes_grouped_markers(self) -> None:
        assert strip_markers("A claim [1, 2] mid-sentence.") == "A claim mid-sentence."

    def test_leaves_plain_text(self) -> None:
        assert strip_markers("No markers.") == "No markers."


class TestIsClaim:
    def _sentence(self, text: str) -> AnswerSentence:
        return AnswerSentence(text=text)

    def test_a_factual_statement_is_a_claim(self) -> None:
        assert is_claim(self._sentence("The damping constant k defaults to 60. [1]"))

    def test_questions_are_not_claims(self) -> None:
        assert not is_claim(self._sentence("What does the damping constant default to?"))

    def test_fragments_are_not_claims(self) -> None:
        assert not is_claim(self._sentence("Yes."))

    @pytest.mark.parametrize(
        "text",
        [
            "Here is what the sources say:",
            "In summary, the retrieval pipeline has several stages.",
            "Let me know if you want more detail on any of these.",
        ],
    )
    def test_framing_is_not_a_claim(self, text: str) -> None:
        assert not is_claim(self._sentence(text))

    def test_classification_is_conservative(self) -> None:
        """Over-classifying prose as a claim only makes the score stricter."""
        assert is_claim(self._sentence("This behaviour is configurable in the settings file."))


# --------------------------------------------------------------------------- #
# the failures that matter
# --------------------------------------------------------------------------- #


class TestSupportedClaims:
    def test_a_verbatim_claim_is_supported(self) -> None:
        sentence = verdict_of("The damping constant k defaults to 60. [1]")
        assert sentence.verdict is SupportVerdict.SUPPORTED
        assert sentence.support_score is not None
        assert sentence.support_score > 0.5

    def test_a_supporting_quote_is_returned(self) -> None:
        """Verification without a quote is an opinion."""
        sentence = verdict_of("The damping constant k defaults to 60. [1]")
        assert sentence.supporting_quote
        assert "defaults to 60" in sentence.supporting_quote

    def test_a_close_paraphrase_is_supported(self) -> None:
        sentence = verdict_of(
            "Reciprocal Rank Fusion combines ranked lists using ranks, not raw scores. [1]"
        )
        assert sentence.verdict in (SupportVerdict.SUPPORTED, SupportVerdict.PARTIAL)


class TestWrongNumbers:
    def test_a_wrong_figure_is_caught(self) -> None:
        """The characteristic RAG failure: fluent paraphrase, wrong number.

        It scores near-perfectly on word overlap and a reader cannot spot it.
        Note that 50 *does* appear in the source ("Recall at 50"), so a
        chunk-level number check would miss this — it is caught because the
        figure is absent from the sentence the claim aligns to.
        """
        sentence = verdict_of("The damping constant k defaults to 50. [1]")
        assert sentence.verdict is SupportVerdict.UNSUPPORTED
        assert sentence.verification_note
        assert "50" in sentence.verification_note
        assert "60" in sentence.verification_note

    def test_a_figure_absent_from_the_whole_source_is_also_caught(self) -> None:
        sentence = verdict_of("The damping constant k defaults to 77. [1]")
        assert sentence.verdict is SupportVerdict.UNSUPPORTED
        assert sentence.verification_note
        assert "77" in sentence.verification_note

    def test_a_numeric_contradiction_is_dispositive(self) -> None:
        """Some signals are gates, not scores: a wrong figure stays unsupported
        however lenient the caller's threshold is."""
        sentence = verdict_of("The damping constant k defaults to 50. [1]", threshold=0.15)
        assert sentence.verdict is not SupportVerdict.SUPPORTED

    def test_a_claim_spanning_two_source_sentences_is_not_punished_hard(self) -> None:
        sentence = verdict_of(
            "Reciprocal Rank Fusion combines ranked lists using ranks, and k defaults to 60. [1]"
        )
        assert sentence.verdict is SupportVerdict.SUPPORTED

    def test_a_wrong_metric_value_is_caught(self) -> None:
        sentence = verdict_of("Recall at 50 was measured at 0.91 on the benchmark. [1]")
        assert sentence.verdict is not SupportVerdict.SUPPORTED

    def test_equivalent_number_formats_still_match(self) -> None:
        chunk = source_chunk("The threshold was set to 1,000 requests per second.")
        sentence = verdict_of("The threshold was set to 1000 requests per second. [1]", chunk)
        assert sentence.verdict is SupportVerdict.SUPPORTED

    def test_decimal_equivalence(self) -> None:
        chunk = source_chunk("Measured recall was 0.820 across the evaluation set.")
        sentence = verdict_of("Measured recall was 0.82 across the evaluation set. [1]", chunk)
        assert sentence.verdict is SupportVerdict.SUPPORTED


class TestNegationDetection:
    """Contrast is not negation.

    "combines ranks, not raw scores" and "combines ranks rather than raw scores"
    mean the same thing; a keyword check calls them contradictory and produces a
    false "unsupported" on a perfectly good citation.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Reciprocal Rank Fusion combines ranked lists using ranks, not raw scores.",
            "RRF combines ranked lists using ranks rather than raw scores.",
            "It uses ranks instead of scores.",
            "Not the score but the rank is used.",
            "A cross-encoder supports joint scoring.",
            "The unsupported claim was flagged.",
        ],
    )
    def test_positive_assertions(self, text: str) -> None:
        assert not _is_negated(text)

    @pytest.mark.parametrize(
        "text",
        [
            "A cross-encoder does not support joint scoring.",
            "A cross-encoder doesn't support joint scoring.",
            "Dense retrieval never finds exact identifiers.",
            "There is no fallback path.",
            "The reranker cannot run over the corpus.",
            "It runs without a network connection.",
            "The system fails to index scanned PDFs.",
        ],
    )
    def test_true_negations(self, text: str) -> None:
        assert _is_negated(text)


class TestNegation:
    def test_a_contradiction_is_caught(self) -> None:
        """A negated claim shares almost every word with its source."""
        sentence = verdict_of(
            "A cross-encoder reranker does not support joint scoring of the query "
            "and the document. [1]"
        )
        assert sentence.verdict is SupportVerdict.UNSUPPORTED
        assert sentence.verification_note
        assert "negation" in sentence.verification_note

    def test_a_correctly_negated_claim_is_supported(self) -> None:
        chunk = source_chunk("Dense retrieval does not find exact identifiers reliably.")
        sentence = verdict_of(
            "Dense retrieval does not find exact identifiers reliably. [1]", chunk
        )
        assert sentence.verdict is SupportVerdict.SUPPORTED


class TestUncitedAndUnsupported:
    def test_an_uncited_claim_is_flagged(self) -> None:
        sentence = verdict_of("The damping constant defaults to 60 in every implementation.")
        assert sentence.verdict is SupportVerdict.UNCITED
        assert sentence.support_score == 0.0

    def test_an_off_source_claim_is_unsupported(self) -> None:
        sentence = verdict_of(
            "Voyage embeddings outperform OpenAI on the MTEB benchmark suite. [1]"
        )
        assert sentence.verdict is SupportVerdict.UNSUPPORTED

    def test_an_unresolvable_citation_is_unsupported(self) -> None:
        answer = answer_with("A claim citing nothing resolvable. [1]")
        answer.retrieval = None  # the cited chunk can no longer be found
        verified = LexicalVerifier().verify(answer)
        assert verified.sentences[0].verdict is SupportVerdict.UNSUPPORTED
        assert verified.sentences[0].verification_note
        assert "could not be resolved" in verified.sentences[0].verification_note

    def test_missing_entity_lowers_the_score(self) -> None:
        supported = verdict_of("The damping constant k defaults to 60. [1]")
        with_entity = verdict_of("In Elasticsearch the damping constant k defaults to 60. [1]")
        assert (with_entity.support_score or 0) < (supported.support_score or 0)


class TestMultipleCitations:
    def test_any_cited_source_can_support_the_claim(self) -> None:
        """Multiple markers mean "any of these", so the max is the right aggregation."""
        relevant = source_chunk("The damping constant k defaults to 60.", "chk_a")
        irrelevant = source_chunk("Unrelated notes on deployment topology.", "chk_b")
        sentence = verdict_of("The damping constant k defaults to 60. [1, 2]", relevant, irrelevant)
        assert sentence.verdict is SupportVerdict.SUPPORTED

    def test_order_does_not_matter(self) -> None:
        irrelevant = source_chunk("Unrelated notes on deployment topology.", "chk_b")
        relevant = source_chunk("The damping constant k defaults to 60.", "chk_a")
        sentence = verdict_of("The damping constant k defaults to 60. [1, 2]", irrelevant, relevant)
        assert sentence.verdict is SupportVerdict.SUPPORTED


# --------------------------------------------------------------------------- #
# answer-level behaviour
# --------------------------------------------------------------------------- #


class TestFaithfulness:
    def test_all_supported_scores_one(self) -> None:
        text = (
            "The damping constant k defaults to 60. [1] "
            "Reciprocal Rank Fusion combines ranked lists using ranks. [1]"
        )
        answer = LexicalVerifier().verify(answer_with(text))
        assert answer.faithfulness == 1.0
        assert answer.verified

    def test_a_mixed_answer_scores_between(self) -> None:
        text = (
            "The damping constant k defaults to 60. [1] The damping constant k defaults to 50. [1]"
        )
        answer = LexicalVerifier().verify(answer_with(text))
        assert answer.faithfulness == pytest.approx(0.5)
        assert len(answer.unsupported_sentences()) == 1
        assert len(answer.flagged_sentences()) == 1

    def test_framing_is_excluded_from_the_denominator(self) -> None:
        """A faithfulness metric that can be gamed by adding filler is worthless."""
        with_filler = LexicalVerifier().verify(
            answer_with(
                "In summary, the retrieval pipeline has several stages. "
                "The damping constant k defaults to 60. [1]"
            )
        )
        without = LexicalVerifier().verify(
            answer_with("The damping constant k defaults to 60. [1]")
        )
        assert with_filler.faithfulness == without.faithfulness == 1.0

    def test_a_refusal_is_not_scored_as_unfaithful(self) -> None:
        """Refusing correctly must not be punished."""
        answer = answer_with("The sources do not contain an answer to this question.")
        answer.refused = True
        verified = LexicalVerifier().verify(answer)
        assert verified.verified
        assert verified.faithfulness is None
        assert all(s.verdict is SupportVerdict.NOT_A_CLAIM for s in verified.sentences)

    def test_verification_is_timed(self) -> None:
        answer = LexicalVerifier().verify(answer_with("The constant k defaults to 60. [1]"))
        assert "verification_ms" in answer.timings_ms

    def test_no_claims_leaves_faithfulness_unset(self) -> None:
        answer = LexicalVerifier().verify(answer_with("Here is a summary:"))
        assert answer.faithfulness is None


class TestThresholds:
    def test_a_strict_threshold_demotes_borderline_claims(self) -> None:
        lenient = verdict_of("Ranked lists are combined using ranks. [1]", threshold=0.3)
        strict = verdict_of("Ranked lists are combined using ranks. [1]", threshold=0.95)
        assert lenient.verdict is SupportVerdict.SUPPORTED
        assert strict.verdict is not SupportVerdict.SUPPORTED

    def test_partial_sits_between_supported_and_unsupported(self) -> None:
        """A hard cliff would report "the citation is wrong" for a near-match."""
        verifier = LexicalVerifier(threshold=0.9, partial_margin=0.5)
        answer = verifier.verify(answer_with("Ranked lists are combined using ranks. [1]"))
        assert answer.sentences[0].verdict in (
            SupportVerdict.PARTIAL,
            SupportVerdict.SUPPORTED,
        )


class _BrokenVerifier(Verifier):
    name = "broken"

    def support(self, claim: str, sources: Sequence[Chunk]):
        raise RuntimeError("judge exploded")


class TestFailureModes:
    def test_a_failing_verifier_does_not_lose_the_answer(self) -> None:
        answer = _BrokenVerifier().verify(answer_with("The constant k defaults to 60. [1]"))
        assert answer.text
        assert answer.verified
        assert answer.sentences[0].verification_note
        assert "unavailable" in answer.sentences[0].verification_note

    def test_a_failing_verifier_does_not_claim_support(self) -> None:
        answer = _BrokenVerifier().verify(answer_with("The constant k defaults to 60. [1]"))
        assert answer.sentences[0].verdict is None

    def test_empty_chunk_text_is_unsupported(self) -> None:
        sentence = verdict_of("Some claim about the constant. [1]", source_chunk("   "))
        assert sentence.verdict is not SupportVerdict.SUPPORTED


# --------------------------------------------------------------------------- #
# LLM judge reply parsing
# --------------------------------------------------------------------------- #


class TestParseVerdict:
    def test_clean_json(self) -> None:
        parsed = parse_verdict(
            '{"supported": true, "confidence": 0.9, "quote": "q", "reason": "r"}'
        )
        assert parsed == (True, 0.9, "q", "r")

    def test_prose_around_the_json(self) -> None:
        parsed = parse_verdict('Sure.\n```json\n{"supported": false, "confidence": 0.8}\n```')
        assert parsed is not None
        assert parsed[0] is False

    def test_string_booleans(self) -> None:
        parsed = parse_verdict('{"supported": "yes", "confidence": 0.7}')
        assert parsed is not None
        assert parsed[0] is True

    def test_confidence_is_clamped(self) -> None:
        parsed = parse_verdict('{"supported": true, "confidence": 5}')
        assert parsed is not None
        assert parsed[1] == 1.0

    def test_bad_confidence_falls_back(self) -> None:
        parsed = parse_verdict('{"supported": true, "confidence": "high"}')
        assert parsed is not None
        assert parsed[1] == 0.5

    @pytest.mark.parametrize("reply", ["", "I cannot help.", "{}", "{'not': 'json'}", "null"])
    def test_unusable_replies_return_none(self, reply: str) -> None:
        """An unparseable judgement is never evidence of support."""
        assert parse_verdict(reply) is None


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


class TestBuildVerifier:
    def test_lexical_is_the_default(self) -> None:
        assert isinstance(build_verifier(Settings()), LexicalVerifier)

    def test_disabled_returns_none(self) -> None:
        assert build_verifier(Settings(verify_citations=False)) is None

    def test_provider_none_returns_none(self) -> None:
        assert build_verifier(Settings(verify_provider=VerifyProvider.NONE)) is None

    def test_threshold_comes_from_settings(self) -> None:
        verifier = build_verifier(Settings(verification_threshold=0.8))
        assert verifier is not None
        assert verifier.threshold == 0.8

    def test_missing_credentials_fall_back_to_lexical(self, monkeypatch) -> None:
        """Falling back makes the check stricter, not laxer — the safe direction."""
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        assert isinstance(
            build_verifier(Settings(verify_provider=VerifyProvider.LLM)), LexicalVerifier
        )


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


@pytest.fixture
def verify_kb(tmp_path: Path) -> KnowledgeBase:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "retrieval.md").write_text(
        "# Retrieval\n\n"
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


class TestAskWithVerification:
    def test_verification_runs_by_default(self, verify_kb: KnowledgeBase) -> None:
        answer = verify_kb.ask("what does the damping constant k default to?", top_k=3)
        assert answer.verified
        assert "verification_ms" in answer.timings_ms

    def test_an_extractive_answer_verifies_as_faithful(self, verify_kb: KnowledgeBase) -> None:
        """Extractive output is verbatim from sources, so it must verify clean.

        This is the end-to-end check that generation and verification agree.
        """
        answer = verify_kb.ask("what does the damping constant k default to?", top_k=3)
        assert not answer.refused
        assert answer.faithfulness == 1.0
        assert answer.unsupported_sentences() == []

    def test_verification_can_be_skipped(self, verify_kb: KnowledgeBase) -> None:
        answer = verify_kb.ask("damping constant", top_k=3, verify=False)
        assert not answer.verified
        assert answer.faithfulness is None

    def test_a_refusal_verifies_without_penalty(self, verify_kb: KnowledgeBase) -> None:
        answer = verify_kb.ask("who won the 1998 world cup?", top_k=3)
        assert answer.refused
        assert answer.faithfulness is None

    def test_streaming_answers_are_verified(self, verify_kb: KnowledgeBase) -> None:
        events = list(verify_kb.ask_stream("damping constant default", top_k=3))
        final = next(a for _, a in events if a is not None)
        assert final.verified

    def test_an_answer_can_be_reverified_later(self, verify_kb: KnowledgeBase) -> None:
        """Re-verification is what lets a stored answer be rechecked more strictly."""
        answer = verify_kb.ask("damping constant default", top_k=3, verify=False)
        assert not answer.verified
        reverified = verify_kb.verify_answer(answer)
        assert reverified.verified
        assert reverified.faithfulness is not None
