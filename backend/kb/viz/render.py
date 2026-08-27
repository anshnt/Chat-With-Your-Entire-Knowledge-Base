"""Render a corpus map as a standalone SVG.

Hand-authored rather than plotted, for the same reasons as the evaluation charts:
no plotting dependency, deterministic output, and colours that hold contrast on
both light and dark backgrounds.

Two rendering decisions worth naming:

* **Point radius encodes retrieval count, not a constant.** The whole value of
  the map is spotting the regions nothing ever reaches, and size is the channel
  a reader picks up without consulting a legend.
* **Never-retrieved points are drawn hollow.** Colour alone would fail for
  colour-blind readers and in greyscale printing; an outline reads either way,
  and it is exactly the distinction the map exists to show.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from kb.viz.corpus_map import CorpusMap, MapPoint

#: Cluster palette. Mid-tone hues, distinguishable and legible on either
#: background, ordered so adjacent clusters differ in hue *and* lightness.
CLUSTER_COLOURS = (
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
)
_AXIS = "#8b93a1"
_TEXT = "#6b7280"


def render_corpus_map(
    corpus_map: CorpusMap,
    *,
    width: int = 900,
    height: int = 620,
    title: str | None = None,
    max_legend: int = 10,
) -> str:
    """Render the map to SVG."""
    heading = title or f"Corpus map — {corpus_map.collection}"
    if not corpus_map.points:
        return _empty(width, height, "no embedded chunks to plot")

    pad = 28
    legend_height = 22 * min(len(corpus_map.clusters), max_legend) + 34
    plot_w = width - pad * 2
    plot_h = height - pad - 46 - legend_height

    peak = max((p.retrievals for p in corpus_map.points), default=0)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_xml(heading)}" '
        f'font-family="system-ui, -apple-system, sans-serif">',
        f"<title>{_xml(heading)}</title>",
        f'<text x="{pad}" y="22" font-size="15" font-weight="600" fill="{_TEXT}">'
        f"{_xml(heading)}</text>",
        f'<text x="{pad}" y="40" font-size="11" fill="{_TEXT}" fill-opacity="0.85">'
        f"{_xml(_subtitle(corpus_map))}</text>",
        f'<rect x="{pad}" y="{pad + 26}" width="{plot_w}" height="{plot_h}" fill="none" '
        f'stroke="{_AXIS}" stroke-opacity="0.2" rx="4"/>',
    ]

    # Larger points last, so a heavily-retrieved chunk is not hidden under a
    # cloud of never-retrieved ones.
    for point in sorted(corpus_map.points, key=lambda p: p.retrievals):
        cx = pad + point.x * plot_w
        cy = pad + 26 + (1.0 - point.y) * plot_h
        colour = CLUSTER_COLOURS[point.cluster % len(CLUSTER_COLOURS)]
        radius = _radius(point.retrievals, peak)
        tooltip = _xml(
            f"{point.document_title}"
            + (f" — {point.heading}" if point.heading else "")
            + f" · {point.retrievals} retrieval(s)\n{point.snippet}"
        )
        if point.retrievals == 0:
            # Hollow: readable in greyscale and for colour-blind readers, and it
            # is the distinction the map exists to show.
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="none" '
                f'stroke="{colour}" stroke-width="1.1" stroke-opacity="0.55">'
                f"<title>{tooltip}</title></circle>"
            )
        else:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{colour}" '
                f'fill-opacity="0.82"><title>{tooltip}</title></circle>'
            )

    parts.append(_legend(corpus_map, pad, pad + 26 + plot_h + 26, max_legend))
    parts.append("</svg>")
    return "\n".join(parts)


def _radius(retrievals: int, peak: int) -> float:
    """Point radius from retrieval count, on a log scale.

    Log rather than linear: one chunk retrieved 40 times would otherwise dwarf
    everything, and the interesting distinction is between zero, a few, and many.
    """
    base = 3.0
    if retrievals <= 0 or peak <= 0:
        return base
    return base + 4.5 * (math.log1p(retrievals) / math.log1p(peak))


def _subtitle(corpus_map: CorpusMap) -> str:
    bits = [
        f"{corpus_map.n_plotted} of {corpus_map.n_chunks} chunks",
        f"{corpus_map.method.upper()}",
        f"{len(corpus_map.clusters)} clusters",
        f"{corpus_map.coverage():.0%} ever retrieved",
    ]
    if corpus_map.explained_variance is not None:
        bits.append(f"{corpus_map.explained_variance:.0%} variance in 2D")
    return " · ".join(bits)


def _legend(corpus_map: CorpusMap, x: float, y: float, max_legend: int) -> str:
    parts = [
        f'<text x="{x}" y="{y}" font-size="11" font-weight="600" fill="{_TEXT}">clusters</text>',
        f'<text x="{x + 300}" y="{y}" font-size="10" fill="{_TEXT}" fill-opacity="0.8">'
        f"filled = retrieved at least once · hollow = never retrieved · "
        f"size = retrieval count</text>",
    ]
    for index, cluster in enumerate(corpus_map.clusters[:max_legend]):
        row = y + 18 + index * 20
        colour = CLUSTER_COLOURS[int(cluster["id"]) % len(CLUSTER_COLOURS)]
        parts.append(
            f'<circle cx="{x + 6}" cy="{row - 3}" r="5" fill="{colour}" fill-opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{x + 18}" y="{row}" font-size="11" fill="{_TEXT}">'
            f"{_xml(str(cluster['label']))}</text>"
        )
        parts.append(
            f'<text x="{x + 400}" y="{row}" font-size="10" fill="{_TEXT}" '
            f'fill-opacity="0.75">{cluster["size"]} chunks · '
            f"{float(cluster['retrieved_share']):.0%} retrieved</text>"
        )
    remaining = len(corpus_map.clusters) - max_legend
    if remaining > 0:
        row = y + 18 + max_legend * 20
        parts.append(
            f'<text x="{x + 18}" y="{row}" font-size="10" fill="{_TEXT}" '
            f'fill-opacity="0.7">… and {remaining} more</text>'
        )
    return "".join(parts)


def _empty(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_xml(message)}">'
        f'<text x="{width / 2}" y="{height / 2}" font-size="12" text-anchor="middle" '
        f'fill="{_TEXT}" font-family="system-ui, sans-serif">{_xml(message)}</text></svg>'
    )


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def cluster_colour(cluster_id: int) -> str:
    """Exposed so a frontend uses the same palette as the SVG."""
    return CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)]


def sorted_by_retrievals(points: Sequence[MapPoint]) -> list[MapPoint]:
    return sorted(points, key=lambda p: -p.retrievals)
