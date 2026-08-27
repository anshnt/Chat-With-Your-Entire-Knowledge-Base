/**
 * The corpus map, rendered as SVG.
 *
 * Hand-drawn rather than pulled from a charting library: the whole component is
 * ~200 lines of SVG, and a library would add a bundle for one scatter plot while
 * making the accessibility and theming harder rather than easier.
 *
 * The rendering follows the same rules as the server-side SVG, deliberately, so
 * the two never disagree:
 *
 * - **never-retrieved points are hollow**, not a different colour, so the
 *   distinction survives colour blindness and greyscale;
 * - **size encodes retrieval count on a log scale**, because one chunk retrieved
 *   40 times would otherwise dwarf everything;
 * - **larger points are drawn last**, so a heavily-used chunk is not buried under
 *   a cloud of unused ones.
 */

import { useMemo, useState } from "react";

import type { CorpusMapResponse, MapPoint } from "../lib/types";

/** Same palette as the server renderer, so a committed SVG matches the app. */
const CLUSTER_COLOURS = [
  "#3b82f6",
  "#f59e0b",
  "#10b981",
  "#a855f7",
  "#ef4444",
  "#14b8a6",
  "#f472b6",
  "#84cc16",
  "#6366f1",
  "#fb923c",
  "#06b6d4",
  "#c084fc",
];

const SIZE = 560;
const PAD = 16;

interface Props {
  map: CorpusMapResponse;
  onSelectChunk?: (chunkId: string) => void;
}

export function CorpusMap({ map, onSelectChunk }: Props) {
  const [hovered, setHovered] = useState<MapPoint | null>(null);
  const [visibleCluster, setVisibleCluster] = useState<number | null>(null);

  const peak = useMemo(
    () => map.points.reduce((max, point) => Math.max(max, point.retrievals), 0),
    [map.points],
  );

  // Ascending by retrievals: the biggest points paint last and stay visible.
  const ordered = useMemo(
    () => [...map.points].sort((a, b) => a.retrievals - b.retrievals),
    [map.points],
  );

  if (map.points.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 p-6 text-center text-sm text-slate-500 dark:border-slate-800">
        <p>Nothing to plot yet.</p>
        {map.notes.map((note) => (
          <p key={note} className="mt-1 text-xs">
            {note}
          </p>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <MapHeader map={map} />

      <div className="relative">
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="w-full rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
          role="img"
          aria-label={
            `Corpus map: ${map.n_plotted} chunks in ${map.clusters.length} clusters, ` +
            `${(map.retrieval_coverage * 100).toFixed(0)}% retrieved at least once`
          }
        >
          {ordered.map((point) => {
            const dimmed = visibleCluster !== null && point.cluster !== visibleCluster;
            const colour = CLUSTER_COLOURS[point.cluster % CLUSTER_COLOURS.length];
            const radius = pointRadius(point.retrievals, peak);
            const cx = PAD + point.x * (SIZE - PAD * 2);
            const cy = PAD + (1 - point.y) * (SIZE - PAD * 2);
            const used = point.retrievals > 0;

            return (
              <circle
                key={point.chunk_id}
                cx={cx}
                cy={cy}
                r={radius}
                fill={used ? colour : "none"}
                fillOpacity={dimmed ? 0.15 : 0.82}
                stroke={used ? "none" : colour}
                strokeWidth={1.1}
                strokeOpacity={dimmed ? 0.12 : 0.55}
                className="cursor-pointer transition-opacity"
                onMouseEnter={() => setHovered(point)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => onSelectChunk?.(point.chunk_id)}
              >
                <title>
                  {`${point.document_title}${point.heading ? ` — ${point.heading}` : ""}\n` +
                    `${point.retrievals} retrieval(s)\n${point.snippet}`}
                </title>
              </circle>
            );
          })}
        </svg>

        {hovered && (
          <div className="pointer-events-none absolute bottom-2 left-2 right-2 animate-fade-in rounded-md border border-slate-200 bg-white/95 p-2 text-xs shadow-lg backdrop-blur dark:border-slate-700 dark:bg-slate-800/95">
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate font-medium">{hovered.document_title}</span>
              <span className="shrink-0 text-slate-500">
                {hovered.retrievals === 0
                  ? "never retrieved"
                  : `${hovered.retrievals} retrieval${hovered.retrievals === 1 ? "" : "s"}`}
              </span>
            </div>
            {hovered.heading && <div className="text-slate-500">{hovered.heading}</div>}
            <p className="mt-1 line-clamp-2 text-slate-600 dark:text-slate-400">
              {hovered.snippet}
            </p>
          </div>
        )}
      </div>

      <Legend
        map={map}
        visibleCluster={visibleCluster}
        onToggle={(id) => setVisibleCluster((current) => (current === id ? null : id))}
      />
    </div>
  );
}

function MapHeader({ map }: { map: CorpusMapResponse }) {
  const lowVariance = map.explained_variance !== null && map.explained_variance < 0.25;
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
        <span>
          {map.n_plotted.toLocaleString()}
          {map.sampled ? ` of ${map.n_chunks.toLocaleString()}` : ""} chunks
        </span>
        <span aria-hidden>·</span>
        <span className="uppercase">{map.method}</span>
        <span aria-hidden>·</span>
        <span>{map.clusters.length} clusters</span>
        <span aria-hidden>·</span>
        <span
          className={
            map.retrieval_coverage < 0.5
              ? "text-amber-600 dark:text-amber-400"
              : "text-emerald-600 dark:text-emerald-400"
          }
        >
          {(map.retrieval_coverage * 100).toFixed(0)}% ever retrieved
        </span>
        <span aria-hidden>·</span>
        <span>{map.elapsed_ms.toFixed(0)} ms</span>
      </div>
      {lowVariance && (
        /* Saying this is the difference between a useful plot and a misleading
           one: at 10% variance the layout carries very little information. */
        <p className="text-xs text-amber-600 dark:text-amber-400">
          The two axes capture {((map.explained_variance ?? 0) * 100).toFixed(0)}% of the
          variance — read the layout loosely. Install <code>umap-learn</code> for a better
          projection.
        </p>
      )}
    </div>
  );
}

function Legend({
  map,
  visibleCluster,
  onToggle,
}: {
  map: CorpusMapResponse;
  visibleCluster: number | null;
  onToggle: (id: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3 text-[0.7rem] text-slate-500">
        <span className="inline-flex items-center gap-1.5">
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="4" fill="currentColor" />
          </svg>
          retrieved at least once
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="4" fill="none" stroke="currentColor" strokeWidth="1.2" />
          </svg>
          never retrieved
        </span>
        <span>size = retrieval count</span>
      </div>

      <ul className="grid gap-1 sm:grid-cols-2">
        {map.clusters.map((cluster) => {
          const active = visibleCluster === cluster.id;
          const colour = CLUSTER_COLOURS[cluster.id % CLUSTER_COLOURS.length];
          return (
            <li key={cluster.id}>
              <button
                type="button"
                onClick={() => onToggle(cluster.id)}
                aria-pressed={active}
                className={[
                  "flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs transition-colors",
                  active
                    ? "bg-slate-200 dark:bg-slate-800"
                    : "hover:bg-slate-100 dark:hover:bg-slate-800/60",
                ].join(" ")}
              >
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: colour }}
                  aria-hidden
                />
                <span className="min-w-0 flex-1 truncate">{cluster.label}</span>
                <span className="shrink-0 text-slate-500">
                  {cluster.size} ·{" "}
                  <span
                    className={
                      cluster.retrieved_share === 0
                        ? "text-red-600 dark:text-red-400"
                        : undefined
                    }
                  >
                    {(cluster.retrieved_share * 100).toFixed(0)}%
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/** Log scale: the interesting distinction is zero vs a few vs many. */
function pointRadius(retrievals: number, peak: number): number {
  const base = 3;
  if (retrievals <= 0 || peak <= 0) return base;
  return base + 4.5 * (Math.log1p(retrievals) / Math.log1p(peak));
}
