"""LLM entailment judge.

The prompt is a strict entailment question, not "is this a good citation?". The
distinction matters: a model asked to judge quality rates plausible claims highly,
whereas a model asked *"does this text state this claim, or can it be directly
deduced from it?"* will say no to a claim that is merely consistent with the
source. Consistency is exactly the failure being hunted.

Three prompt decisions carry most of the reliability:

* **A supporting quote is required.** Asking the judge to quote the sentence that
  supports the claim forces it to point at something, which suppresses "yes,
  because it seems right". If it cannot quote, it cannot claim support.
* **The default is NO.** Uncertainty resolves to unsupported. A verifier that
  guesses "supported" is worse than no verifier, because it launders the exact
  failure it was added to catch.
* **Numbers are called out explicitly.** The most damaging error is a fluent
  paraphrase with a wrong figure, and it does not look wrong.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from kb.errors import ProviderError
from kb.llm import LLMClient
from kb.models import Chunk
from kb.verify.base import Verifier

log = logging.getLogger(__name__)

MAX_SOURCE_CHARS = 2500

SYSTEM_PROMPT = (
    "You are a strict fact-checker. You decide only whether a source text states "
    "a claim or directly implies it. You never use outside knowledge, and you "
    "default to 'not supported' whenever you are unsure."
)

PROMPT_TEMPLATE = """Source text:
\"\"\"
{source}
\"\"\"

Claim: {claim}

Does the source text state this claim, or can the claim be deduced directly from it?

Rules:
- Use only the source text. Ignore anything you know from elsewhere, even if the claim is true in general.
- Check every number, date, name and quantity in the claim against the source. If any differs or is absent, the claim is NOT supported.
- If the claim is negated ("does not", "never") and the source is not, or the reverse, the claim is NOT supported.
- A claim that is merely consistent with the source, or plausible given it, is NOT supported.
- If you cannot quote a sentence from the source that supports the claim, it is NOT supported.
- When in doubt, answer no.

Reply with only JSON:
{{"supported": true|false, "confidence": 0.0-1.0, "quote": "the sentence from the source that supports the claim, or empty", "reason": "one short sentence"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMVerifier(Verifier):
    """Verifies claims with a language model acting as an entailment judge."""

    name = "llm-verify"

    def __init__(
        self,
        client: LLMClient,
        *,
        threshold: float = 0.5,
        partial_margin: float = 0.2,
        max_source_chars: int = MAX_SOURCE_CHARS,
    ) -> None:
        super().__init__(threshold=threshold, partial_margin=partial_margin)
        self.client = client
        self.name = f"llm-verify:{client.model}"
        self.max_source_chars = max_source_chars

    # ------------------------------------------------------------------ #

    def support(self, claim: str, sources: Sequence[Chunk]) -> tuple[float, str | None, str | None]:
        """Judge the claim against each cited chunk, keeping the strongest result.

        Multiple markers on one sentence mean "any of these supports it", so the
        maximum is correct — and the loop short-circuits on a confident yes, which
        keeps the common case to one call.
        """
        best: tuple[float, str | None, str | None] = (0.0, None, None)
        for chunk in sources:
            score, quote, reason = self._judge(claim, chunk)
            if score > best[0]:
                best = (score, quote, reason)
            if score >= 0.9:
                break
        return best

    def _judge(self, claim: str, chunk: Chunk) -> tuple[float, str | None, str | None]:
        prompt = PROMPT_TEMPLATE.format(
            source=_truncate(chunk.text, self.max_source_chars), claim=claim
        )
        try:
            reply = self.client.complete(
                prompt, system=SYSTEM_PROMPT, max_tokens=400, temperature=0.0
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"verification call failed: {exc}") from exc

        parsed = parse_verdict(reply)
        if parsed is None:
            # An unparseable judgement is not evidence of support.
            return 0.0, None, "verifier returned an unparseable judgement"

        supported, confidence, quote, reason = parsed
        # Map to a support score in [0, 1]: a confident "no" must land near zero,
        # not at the midpoint, or the verdict thresholds become meaningless.
        score = confidence if supported else max(0.0, 0.5 - confidence * 0.5)
        return score, quote or None, reason or None


def parse_verdict(reply: str) -> tuple[bool, float, str, str] | None:
    """Parse the judge's JSON reply, tolerating prose and fences around it.

    Returns ``None`` when nothing usable is present, which the caller treats as
    unsupported — never as support.
    """
    if not reply:
        return None
    match = _JSON_RE.search(reply)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "supported" not in payload:
        return None

    supported = payload.get("supported")
    if isinstance(supported, str):
        supported = supported.strip().lower() in ("true", "yes", "1")
    supported = bool(supported)

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    quote = str(payload.get("quote") or "").strip()
    reason = str(payload.get("reason") or "").strip()
    return supported, confidence, quote, reason


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit].rstrip()}…"
