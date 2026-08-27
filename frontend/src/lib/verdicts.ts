/**
 * Verdict presentation.
 *
 * Every verdict has a colour, a label and a one-line explanation, defined once.
 * Two rules the design follows:
 *
 * - **Never colour alone.** An underline style and a word accompany every
 *   colour, so the distinction survives colour blindness and greyscale.
 * - **`supported` is not decorated.** Marking the normal case draws the eye away
 *   from the exceptions, which are the only reason this feature exists.
 */

import type { SupportVerdict } from "./types";

export interface VerdictStyle {
  label: string;
  explanation: string;
  /** Tailwind classes for the sentence underline. */
  underline: string;
  /** Tailwind classes for a badge. */
  badge: string;
  swatch: string;
}

const STYLES: Record<SupportVerdict, VerdictStyle> = {
  supported: {
    label: "supported",
    explanation: "The cited source states this.",
    // No decoration: highlighting the normal case hides the exceptions.
    underline: "",
    badge: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
    swatch: "bg-emerald-500",
  },
  partial: {
    label: "partly supported",
    explanation: "The cited source is related but does not fully state this.",
    underline: "underline decoration-amber-500 decoration-wavy decoration-2 underline-offset-4",
    badge: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
    swatch: "bg-amber-500",
  },
  unsupported: {
    label: "not supported",
    explanation: "The cited source does not support this claim.",
    underline: "underline decoration-red-500 decoration-wavy decoration-2 underline-offset-4",
    badge: "bg-red-500/10 text-red-700 dark:text-red-400",
    swatch: "bg-red-500",
  },
  uncited: {
    label: "uncited",
    explanation: "This makes a factual claim but cites no source.",
    underline: "underline decoration-purple-500 decoration-dotted decoration-2 underline-offset-4",
    badge: "bg-purple-500/10 text-purple-700 dark:text-purple-400",
    swatch: "bg-purple-500",
  },
  not_a_claim: {
    label: "not a claim",
    explanation: "Framing or transition — nothing to verify.",
    underline: "",
    badge: "bg-slate-500/10 text-slate-600 dark:text-slate-400",
    swatch: "bg-slate-400",
  },
};

export function verdictStyle(verdict: SupportVerdict | null): VerdictStyle | null {
  return verdict ? STYLES[verdict] : null;
}

/** True when a reader should be shown this verdict rather than left to assume. */
export function isFlagged(verdict: SupportVerdict | null): boolean {
  return verdict === "unsupported" || verdict === "uncited" || verdict === "partial";
}

export function faithfulnessTone(value: number | null): string {
  if (value === null) return "text-slate-500";
  if (value >= 0.999) return "text-emerald-600 dark:text-emerald-400";
  if (value >= 0.8) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

export const ALL_VERDICTS: SupportVerdict[] = [
  "supported",
  "partial",
  "unsupported",
  "uncited",
];
