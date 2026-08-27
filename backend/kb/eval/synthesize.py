"""Golden-set generation from a corpus.

Hand-writing 200 evaluation questions is the reason most projects have no
evaluation. Both generators here work the same way — pick a chunk, produce a
question it answers, record the chunk as the expected source — which makes the
label correct *by construction*: the question was derived from that chunk, so
that chunk answers it.

Two important caveats, stated rather than hidden:

* **Synthetic questions are easier than real ones.** They are phrased in the
  corpus's own vocabulary, so they over-reward lexical retrieval. They are a
  regression baseline and a way to compare configurations, not an absolute
  quality measure. Real user queries (mined from ``retrieval_events`` via
  ``kb eval mine``) are the ground truth.
* **The expected source is recorded as a text snippet, not a chunk id**, so the
  set survives re-chunking and re-ingestion — which is exactly when you want to
  measure.

The offline generator needs no API key: it turns a declarative sentence into a
question using its own structure. It is blunt but genuinely usable, and it means
`kb eval generate` works on a fresh clone.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence

from kb.chunking.base import split_sentences
from kb.eval.dataset import GoldenQuery, GoldenSet
from kb.models import Chunk, ChunkKind

log = logging.getLogger(__name__)

MIN_SENTENCE_CHARS = 45
MAX_SENTENCE_CHARS = 300
SNIPPET_WORDS = 8

# Patterns that turn a declarative sentence into a question. Ordered most to
# least specific: the first match wins, so "X defaults to Y" becomes a "what
# does X default to" question rather than a generic one.
_TEMPLATES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^(?P<subject>.{5,80}?)\s+defaults?\s+to\s+(?P<value>.+?)[.?!]?$", re.I),
        "what does {subject} default to",
    ),
    (
        re.compile(r"^(?P<subject>.{5,80}?)\s+is\s+(?:the\s+)?(?P<value>.{5,}?)[.?!]?$", re.I),
        "what is {subject}",
    ),
    (
        re.compile(r"^(?P<subject>.{5,80}?)\s+are\s+(?P<value>.{5,}?)[.?!]?$", re.I),
        "what are {subject}",
    ),
    (
        re.compile(r"^(?P<subject>.{5,80}?)\s+(?:means|refers\s+to)\s+(?P<value>.+?)[.?!]?$", re.I),
        "what does {subject} mean",
    ),
    (
        re.compile(
            r"^(?P<subject>.{5,80}?)\s+(?:combines|uses|requires|supports|provides|returns|"
            r"produces|prevents|allows|handles)\s+(?P<value>.+?)[.?!]?$",
            re.I,
        ),
        "what does {subject} do",
    ),
    (
        # Affirmative modals only, and not "must not" / "should not": turning
        # "citations cannot be resolved until X" into "what can citations do?"
        # inverts the sentence's meaning while looking perfectly well-formed.
        re.compile(
            r"^(?P<subject>.{5,80}?)\s+(?:can|must|should)\s+(?!not\b)(?P<value>.+?)[.?!]?$",
            re.I,
        ),
        "what can {subject} do",
    ),
)

_LEADING_CONNECTIVE_RE = re.compile(
    r"^(?:so|and|but|then|thus|therefore|however|also|that|this|it|which|these|those)\b", re.I
)

#: A subject containing one of these spans a clause boundary, which means the
#: pattern matched a verb that is not the sentence's main verb — "X scores pairs
#: jointly and is more accurate" would yield "what is X scores pairs jointly and".
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:,|;|:"
    # conjunctions and subordinators
    r"|\b(?:and|or|but|so|because|which|that|while|when|where|than|if|since"
    r"|until|before|after|unless|though|although|whereas|whether|as)\b"
    # auxiliaries and modals: their presence means the pattern matched a verb
    # that is not the sentence's main verb
    r"|\b(?:is|are|was|were|be|been|being|has|have|had|can|cannot|could|will"
    r"|would|shall|should|must|may|might|do|does|did)\b)",
    re.I,
)

#: Longest usable subject. A question whose subject is a whole clause is not a
#: question a person would ask, and it is unanswerable as phrased.
MAX_SUBJECT_WORDS = 8

#: A subject that refers to something outside the sentence produces a question
#: nobody can answer: "what is there?", "what can it do?".
_NON_REFERENTIAL = frozenset(
    {
        "there",
        "it",
        "this",
        "that",
        "they",
        "them",
        "these",
        "those",
        "what",
        "which",
        "here",
        "one",
        "some",
        "none",
        "each",
        "both",
        "everything",
        "anything",
        "something",
        "nothing",
        "the following",
        "the above",
        "the result",
        "the point",
    }
)

#: Inline Markdown that leaks into a question if not removed: `code`, **bold**,
#: [text](link). Left in, it produces questions like "what is (RRF)`?".
_MARKDOWN_INLINE_RE = re.compile(r"`+|\*+|_{2,}|~~")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def strip_inline_markdown(text: str) -> str:
    """Remove inline Markdown formatting, keeping link text."""
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_INLINE_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _usable_subject(subject: str) -> bool:
    """True when ``subject`` is a noun phrase a person could ask about."""
    if not 5 <= len(subject) <= 70:
        return False
    words = subject.split()
    if len(words) > MAX_SUBJECT_WORDS:
        return False
    if _LEADING_CONNECTIVE_RE.match(subject):
        return False
    if _CLAUSE_BOUNDARY_RE.search(subject):
        return False
    # Strip a leading article before checking referentiality, so "the result"
    # and "result" are judged the same way.
    bare = re.sub(r"^(?:the|a|an)\s+", "", subject.lower()).strip()
    if bare in _NON_REFERENTIAL or subject.lower() in _NON_REFERENTIAL:
        return False
    # A subject *starting* with a non-referential word points outside the
    # sentence too: "what it does", "this behaviour", "those results".
    first_word = bare.split()[0] if bare.split() else ""
    if first_word in _NON_REFERENTIAL:
        return False
    # A subject with no alphabetic content is punctuation debris.
    return any(ch.isalpha() for ch in subject)


def snippet_for(sentence: str, words: int = SNIPPET_WORDS) -> str:
    """A distinctive substring of ``sentence``, used as the expectation.

    Taken from the middle rather than the start: openings are formulaic ("The
    system supports…") and repeat across a corpus, which would make one snippet
    match many chunks and silently inflate recall.
    """
    tokens = sentence.split()
    if len(tokens) <= words:
        return " ".join(tokens).rstrip(".,;:")
    start = max(0, (len(tokens) - words) // 2)
    return " ".join(tokens[start : start + words]).rstrip(".,;:")


def candidate_sentences(chunk: Chunk) -> Iterator[str]:
    """Sentences in ``chunk`` worth turning into a question."""
    if chunk.kind in (ChunkKind.CODE, ChunkKind.TABLE):
        return
    body = chunk.text
    if chunk.heading_context and body.startswith(chunk.heading_context):
        body = body[len(chunk.heading_context) :].lstrip("\n ")
    for sentence in split_sentences(body):
        raw = " ".join(sentence.split())
        if raw.startswith(("|", "```", "-", "*", "#", ">")):
            continue
        cleaned = strip_inline_markdown(raw)
        if not MIN_SENTENCE_CHARS <= len(cleaned) <= MAX_SENTENCE_CHARS:
            continue
        # A sentence opening with a connective refers to something outside
        # itself, so a question built from it is unanswerable in isolation.
        if _LEADING_CONNECTIVE_RE.match(cleaned):
            continue
        yield cleaned


def question_from_sentence(sentence: str) -> str | None:
    """Turn a declarative sentence into a question, or ``None`` if unusable."""
    for pattern, template in _TEMPLATES:
        match = pattern.match(sentence)
        if not match:
            continue
        subject = match.group("subject").strip().rstrip(",")
        if not _usable_subject(subject):
            continue
        subject = subject[0].lower() + subject[1:] if subject[:1].isupper() else subject
        return f"{template.format(subject=subject)}?"
    return None


def generate_golden_set(
    chunks: Sequence[Chunk],
    *,
    name: str = "synthetic",
    collection: str = "default",
    per_document: int = 2,
    limit: int = 100,
) -> GoldenSet:
    """Build a golden set from a corpus, offline and deterministically.

    ``per_document`` caps questions per document so a long document cannot
    dominate the set — otherwise the metrics describe how well retrieval handles
    one file.
    """
    queries: list[GoldenQuery] = []
    per_document_counts: dict[str, int] = {}
    seen_questions: set[str] = set()

    for chunk in chunks:
        if len(queries) >= limit:
            break
        if per_document_counts.get(chunk.document_id, 0) >= per_document:
            continue
        for sentence in candidate_sentences(chunk):
            question = question_from_sentence(sentence)
            if question is None:
                continue
            key = question.lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            per_document_counts[chunk.document_id] = (
                per_document_counts.get(chunk.document_id, 0) + 1
            )
            queries.append(
                GoldenQuery(
                    query=question,
                    must_contain=[snippet_for(sentence)],
                    tags=["synthetic", f"source:{chunk.source_type.value}"]
                    if chunk.source_type
                    else ["synthetic"],
                    notes=f"generated from: {sentence[:160]}",
                )
            )
            break

    return GoldenSet(
        name=name,
        collection=collection,
        description=(
            "Generated from the corpus. Questions are phrased in the corpus's own "
            "vocabulary, so they over-reward lexical retrieval: use this to compare "
            "configurations and catch regressions, not as an absolute quality measure."
        ),
        queries=queries,
    )


# --------------------------------------------------------------------------- #
# LLM generation
# --------------------------------------------------------------------------- #

LLM_SYSTEM_PROMPT = (
    "You write evaluation questions for a retrieval system. You write questions a "
    "real person would ask, and only ones the given passage actually answers."
)

LLM_PROMPT_TEMPLATE = """Passage:
\"\"\"
{passage}
\"\"\"

Write {n} question(s) that this passage answers.

Rules:
- Each question must be fully answerable from this passage alone.
- Phrase them as a person unfamiliar with the passage would: do not reuse its
  distinctive wording, or you are testing string matching rather than retrieval.
- Ask about the substance, not the passage ("what is the default damping
  constant?", not "what does this section say?").
- One question per line. No numbering, no other text."""


def generate_golden_set_with_llm(
    chunks: Sequence[Chunk],
    client: object,
    *,
    name: str = "synthetic-llm",
    collection: str = "default",
    per_chunk: int = 1,
    limit: int = 50,
) -> GoldenSet:
    """Generate questions with a language model.

    Better than the offline generator in the way that matters: the prompt asks for
    paraphrase rather than the passage's own wording, so the resulting set does
    not systematically favour lexical retrieval.
    """
    queries: list[GoldenQuery] = []
    for chunk in chunks:
        if len(queries) >= limit:
            break
        passage = " ".join(chunk.text.split())[:2000]
        if len(passage) < MIN_SENTENCE_CHARS:
            continue
        try:
            reply = client.complete(  # type: ignore[attr-defined]
                LLM_PROMPT_TEMPLATE.format(passage=passage, n=per_chunk),
                system=LLM_SYSTEM_PROMPT,
                max_tokens=300,
                temperature=0.3,
            )
        except Exception as exc:
            log.warning("question generation failed for chunk %s: %s", chunk.id, exc)
            continue

        sentences = list(candidate_sentences(chunk))
        anchor = (
            snippet_for(sentences[0]) if sentences else " ".join(passage.split()[:SNIPPET_WORDS])
        )
        for line in reply.splitlines():
            question = line.strip().lstrip("-*0123456789. ").strip()
            if len(question) < 12 or not question.endswith("?"):
                continue
            queries.append(
                GoldenQuery(
                    query=question,
                    must_contain=[anchor],
                    chunk_ids=[chunk.id],
                    grades={anchor: 2, chunk.id: 2},
                    tags=["synthetic", "llm"],
                    notes=f"generated from chunk {chunk.id}",
                )
            )
            if len(queries) >= limit:
                break

    return GoldenSet(
        name=name,
        collection=collection,
        description="Generated by a language model, prompted to paraphrase rather than "
        "reuse the passage's wording.",
        queries=queries,
    )


def mine_golden_set(
    queries: Sequence[str], *, name: str = "mined", collection: str = "default"
) -> GoldenSet:
    """Seed a golden set from real logged queries, for a human to label.

    Real traffic is the only ground truth for what people actually ask, so the
    labelling step is left to a person: the queries arrive with an empty
    ``must_contain`` for them to fill in.
    """
    return GoldenSet(
        name=name,
        collection=collection,
        description=(
            "Mined from logged retrievals. Fill in must_contain for each query with a "
            "snippet from the chunk that should answer it, then delete the ones that "
            "are not worth evaluating."
        ),
        queries=[
            GoldenQuery(
                query=q, must_contain=["TODO: snippet from the answering chunk"], tags=["mined"]
            )
            for q in queries
            if q.strip()
        ],
    )
