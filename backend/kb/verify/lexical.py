"""Offline entailment-style verifier — the default.

Textual entailment is the right frame for this problem, but a full NLI model is
a large dependency for a check that runs on every sentence of every answer. This
verifier approximates the useful part of it with lexical evidence, and is honest
about which failures it can and cannot catch.

What it measures, per claim:

1. **Content-word coverage.** The share of the claim's *content* words (stopwords
   and citation markers removed) that appear in the cited chunk, IDF-weighted
   against the chunk itself so a claim's distinctive terms count for more than
   its filler.
2. **Best-sentence alignment.** The single chunk sentence with the highest
   overlap with the claim, which becomes the ``supporting_quote``. Verification
   without a quote is an opinion; with one, a reader can check it in a glance.
3. **Number and entity agreement.** Every number and capitalised token in the
   claim must appear in the chunk. This is the highest-value check in the whole
   file, because the characteristic RAG failure is a *fluent paraphrase with a
   wrong figure* — "defaults to 50" cited to a chunk saying 60 — which scores
   near-perfectly on word overlap and is exactly the error a user cannot spot.
4. **Negation agreement.** A claim and a chunk that disagree on negation
   ("supports" vs "does not support") share almost all their words. Without this
   check, a flat contradiction reads as strongly supported.

What it cannot catch: a claim that is a genuine semantic inference from the
chunk, and a paraphrase with no lexical overlap. Both push the score *down*, so
the failure mode is a false "unsupported" rather than a false "supported" — the
safe direction. Set ``KB_VERIFY_PROVIDER=llm`` for judgements that need real
entailment.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from kb.chunking.base import split_sentences
from kb.models import Chunk
from kb.retrieval.lexical import STOPWORDS
from kb.verify.base import Verifier

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-.]*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")

# Negation detection has to distinguish *negating the assertion* from
# *contrasting two alternatives*. "combines ranks, not raw scores" and "combines
# ranks rather than raw scores" mean the same thing, and a naive keyword check
# calls them contradictory — a false "unsupported" on a perfectly good citation.
#
# So: strip contrastive constructions first, then look for negation that attaches
# to a verb ("does not support") or is inherently negative ("never", "cannot").

_CONTRASTIVE_RE = re.compile(
    r"""(
        \brather\s+than\b
      | \binstead\s+of\b
      | \bas\s+opposed\s+to\b
      | ,\s*not\b
      | \bnot\b(?=[^.;]{0,40}\bbut\b)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_TRUE_NEGATION_RE = re.compile(
    r"""(
        \b(?:do|does|did|is|are|was|were|be|been|being|has|have|had
            |can|could|will|would|shall|should|may|might|must)\s+not\b
      | \w+n't\b
      | \b(?:never|cannot|neither|nor)\b
      | \bno\s+\w+
      | \bnone\s+of\b
      | \bwithout\b
      | \bfails?\s+to\b
      | \bfailed\s+to\b
      | \bunable\s+to\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Weights. Coverage carries the signal; the agreement checks are gates that
# subtract when violated, because a single wrong number invalidates a claim that
# is otherwise a perfect lexical match.
W_COVERAGE = 0.62
W_ALIGNMENT = 0.38
NUMBER_MISMATCH_PENALTY = 0.55
#: The aligned sentence states numbers, and the claim's figure is not among them.
#: That is a contradiction, not a coincidence: the source made a numeric claim and
#: the answer reported a different one.
NUMBER_CONTRADICTION_PENALTY = 0.50
#: The aligned sentence states no numbers at all, so the claim's figure came from
#: elsewhere in the chunk. Suspicious, but a claim legitimately spanning two
#: sentences of one chunk is normal, so the penalty is mild.
NUMBER_DISPLACED_PENALTY = 0.22
ENTITY_MISMATCH_PENALTY = 0.20
NEGATION_MISMATCH_PENALTY = 0.45

#: Some signals are gates, not scores. A figure that contradicts the source, or a
#: negation that flips its meaning, is dispositive: the claim is wrong however
#: well the rest of its words line up. Capping the score here — rather than
#: subtracting and hoping the total lands low enough — keeps that guarantee
#: independent of the threshold the caller configures.
DISPOSITIVE_CEILING = 0.12


class LexicalVerifier(Verifier):
    """Approximate entailment from lexical evidence. Always available."""

    name = "lexical-verify"

    def support(self, claim: str, sources: Sequence[Chunk]) -> tuple[float, str | None, str | None]:
        """Score the claim against its cited chunks, keeping the best result.

        Multiple citations on one sentence mean "any of these supports it", so
        the maximum is the right aggregation — not the mean, which would punish
        a correct citation for being listed beside a weaker one.
        """
        best_score = 0.0
        best_quote: str | None = None
        best_note: str | None = None

        claim_words = _content_words(claim)
        if not claim_words:
            return 1.0, None, "no content words to verify"

        claim_numbers = set(_NUMBER_RE.findall(claim))
        claim_entities = {e.lower() for e in _PROPER_RE.findall(claim)}

        for chunk in sources:
            score, quote, note = self._score_one(
                claim, claim_words, claim_numbers, claim_entities, chunk
            )
            if score > best_score:
                best_score, best_quote, best_note = score, quote, note

        return best_score, best_quote, best_note

    # ------------------------------------------------------------------ #

    def _score_one(
        self,
        claim: str,
        claim_words: set[str],
        claim_numbers: set[str],
        claim_entities: set[str],
        chunk: Chunk,
    ) -> tuple[float, str | None, str | None]:
        chunk_text = chunk.text
        chunk_words = _content_words(chunk_text)
        if not chunk_words:
            return 0.0, None, "cited chunk has no content"

        idf = _idf(claim_words, chunk_text)
        total = sum(idf.values()) or 1.0
        matched = sum(weight for word, weight in idf.items() if word in chunk_words)
        coverage = matched / total

        quote, alignment = _best_sentence(idf, chunk_text)

        score = W_COVERAGE * coverage + W_ALIGNMENT * alignment
        notes: list[str] = []
        dispositive = False

        # --- gates ---------------------------------------------------- #
        # The failure these catch: a fluent paraphrase with a wrong figure. It is
        # near-perfect on word overlap and a reader cannot spot it.
        chunk_numbers = set(_NUMBER_RE.findall(chunk_text))
        missing_numbers = {n for n in claim_numbers if not _number_present(n, chunk_numbers)}
        if missing_numbers:
            score -= NUMBER_MISMATCH_PENALTY
            dispositive = True
            notes.append(
                f"figure(s) not found in the cited source: {', '.join(sorted(missing_numbers))}"
            )
        elif claim_numbers and quote:
            # The figure exists in the chunk but not in the sentence the claim
            # aligns to — a claim of "defaults to 50" against a chunk that says
            # "defaults to 60" and, separately, "recall at 50". A chunk-level
            # check passes that; it should not.
            quote_numbers = set(_NUMBER_RE.findall(quote))
            displaced = {n for n in claim_numbers if not _number_present(n, quote_numbers)}
            if displaced and quote_numbers:
                score -= NUMBER_CONTRADICTION_PENALTY
                dispositive = True
                notes.append(
                    "the supporting sentence states a different figure: claim says "
                    f"{', '.join(sorted(displaced))}, source says "
                    f"{', '.join(sorted(quote_numbers))}"
                )
            elif displaced:
                score -= NUMBER_DISPLACED_PENALTY
                notes.append(
                    "figure(s) appear elsewhere in the source but not in the supporting "
                    f"sentence: {', '.join(sorted(displaced))}"
                )

        chunk_entities = {e.lower() for e in _PROPER_RE.findall(chunk_text)}
        missing_entities = claim_entities - chunk_entities - chunk_words
        if missing_entities:
            score -= ENTITY_MISMATCH_PENALTY * min(1.0, len(missing_entities) / 2)
            notes.append(
                f"name(s) not found in the cited source: {', '.join(sorted(missing_entities))}"
            )

        if _negation_disagrees(claim, quote or chunk_text):
            score -= NEGATION_MISMATCH_PENALTY
            dispositive = True
            notes.append("the claim and the cited text disagree on negation")

        score = max(0.0, min(1.0, score))
        if dispositive:
            score = min(score, DISPOSITIVE_CEILING)
        return score, quote, "; ".join(notes) or None


def _content_words(text: str) -> set[str]:
    """Lowercase content words: stopwords, markers and single chars removed."""
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    return {
        word.lower().strip(".-'")
        for word in _WORD_RE.findall(text)
        if len(word) > 1 and word.lower() not in STOPWORDS
    }


def _idf(claim_words: set[str], chunk_text: str) -> dict[str, float]:
    """Weight each claim word by how distinctive it is within the chunk.

    A word appearing in every sentence of the chunk carries little evidence that
    *this* sentence is supported; a word appearing once carries a lot.
    """
    sentences = split_sentences(chunk_text) or [chunk_text]
    n = len(sentences)
    per_sentence = [_content_words(s) for s in sentences]
    weights: dict[str, float] = {}
    for word in claim_words:
        frequency = sum(1 for words in per_sentence if word in words)
        weights[word] = math.log(1.0 + (n - frequency + 0.5) / (frequency + 0.5)) + 0.25
    return weights


def _best_sentence(idf: dict[str, float], chunk_text: str) -> tuple[str | None, float]:
    """The chunk sentence best matching the claim, and its normalised overlap."""
    total = sum(idf.values()) or 1.0
    best_quote: str | None = None
    best = 0.0
    for sentence in split_sentences(chunk_text) or [chunk_text]:
        words = _content_words(sentence)
        overlap = sum(weight for word, weight in idf.items() if word in words) / total
        if overlap > best:
            best = overlap
            best_quote = " ".join(sentence.split())
    return best_quote, best


def _number_present(number: str, chunk_numbers: set[str]) -> bool:
    """Compare numerically where possible, so 60 matches 60.0 and 1,000."""
    if number in chunk_numbers:
        return True
    target = _as_float(number)
    if target is None:
        return False
    return any(
        candidate is not None and math.isclose(candidate, target, rel_tol=1e-9)
        for candidate in (_as_float(n) for n in chunk_numbers)
    )


def _as_float(value: str) -> float | None:
    try:
        return float(value.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _negation_disagrees(claim: str, evidence: str) -> bool:
    """True when exactly one of the two texts is negated.

    A claim and a chunk that disagree on negation share almost all their words,
    so without this check a flat contradiction scores as strongly supported.
    """
    return _is_negated(claim) != _is_negated(evidence)


def _is_negated(text: str) -> bool:
    """True when ``text`` negates its own assertion.

    Contrastive constructions are removed before the check, so "combines ranks,
    not raw scores" reads as positive — which it is.
    """
    stripped = _CONTRASTIVE_RE.sub(" ", text)
    return bool(_TRUE_NEGATION_RE.search(stripped))
