/**
 * The chat hook: streams an answer, then swaps in the verified result.
 *
 * The sequencing is what matters. Citations and verdicts cannot exist until the
 * text is complete — a marker can be half-emitted mid-stream, and verification
 * needs whole sentences — so the UI shows raw streamed text first and replaces it
 * with the structured answer on the terminal event. Rendering a citation chip for
 * a partial `[` would be worse than waiting a beat.
 *
 * In-flight requests are aborted when a new question is asked, so a slow answer
 * cannot arrive after a newer one and overwrite it.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, streamAsk, type AskOptions } from "../lib/api";
import type { ChatTurn } from "../lib/types";

let turnCounter = 0;

export function useAsk(options: AskOptions) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Options change on every render of the parent; a ref keeps `ask` stable so it
  // is not recreated (and effects depending on it not re-fired) constantly.
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => () => abortRef.current?.abort(), []);

  const update = useCallback((id: string, patch: Partial<ChatTurn>) => {
    setTurns((current) =>
      current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)),
    );
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const id = `turn-${++turnCounter}`;
      setTurns((current) => [
        ...current,
        { id, question: trimmed, streaming: "", answer: null, error: null, done: false },
      ]);
      setBusy(true);

      try {
        for await (const event of streamAsk(trimmed, optionsRef.current, controller.signal)) {
          if (event.type === "delta") {
            setTurns((current) =>
              current.map((turn) =>
                turn.id === id ? { ...turn, streaming: turn.streaming + event.text } : turn,
              ),
            );
          } else if (event.type === "done") {
            update(id, { answer: event.answer, done: true });
          } else {
            update(id, { error: event.message, done: true });
          }
        }
      } catch (cause) {
        if (controller.signal.aborted) return;
        update(id, {
          error: cause instanceof ApiError ? cause.message : "Something went wrong",
          done: true,
        });
      } finally {
        if (!controller.signal.aborted) setBusy(false);
      }
    },
    [update],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setTurns([]);
    setBusy(false);
  }, []);

  return { turns, busy, ask, stop, clear };
}
