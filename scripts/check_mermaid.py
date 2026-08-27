#!/usr/bin/env python3
"""Validate the Mermaid diagrams embedded in Markdown files.

A broken Mermaid block does not fail quietly — GitHub renders a red error box in
place of the diagram, on the README, indefinitely. So the blocks are checked in
CI like any other source.

Full validation needs a browser (Mermaid is a JS library), which is too heavy for
CI. This checks the failures that actually happen in practice:

* unbalanced brackets — the most common cause of a parse error;
* a bare ``&``, which Mermaid reads as its node-chaining operator even inside a
  quoted label, and which must be written ``#38;``;
* a missing or unknown diagram type on the first line;
* an edge referring to a node id that is never declared, which renders as a
  stray empty box rather than an error.

Exit code 1 on any finding, so it can gate a build.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)

KNOWN_TYPES = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "journey",
    "gantt",
    "pie",
    "gitGraph",
    "mindmap",
    "timeline",
    "quadrantChart",
    "xychart-beta",
)

BRACKET_PAIRS = (("[", "]"), ("(", ")"), ("{", "}"))

#: A node id immediately followed by a shape delimiter, anywhere in the source.
#: It has to match anywhere, not only at line start, because `A["x"] --> B["y"]`
#: declares both nodes in one statement.
_NODE_DECL_RE = re.compile(r"([A-Za-z_][\w-]*)\s*[\[\({]")
#: Quoted label contents, stripped before looking for declarations — a label
#: like "trailing [n] reattached" would otherwise read as a declaration of `n`.
_QUOTED_RE = re.compile(r'"[^"]*"')
#: Node ids appearing as edge endpoints, once shapes have been stripped.
_EDGE_RE = re.compile(
    r"([A-Za-z_][\w-]*)\s*(?:-{2,}>|-\.-*>|={2,}>|-{3}|-\.-)\s*(?:\|[^|]*\|\s*)?"
    r"([A-Za-z_][\w-]*)"
)
#: One shape and its contents, e.g. `[""]`, `(("" ))`, `{}`.
_SHAPE_RE = re.compile(r"[\[\({][^\[\]\(\){}]*[\]\)}]")


def strip_shapes(text: str, max_passes: int = 6) -> str:
    """Remove shape delimiters so edge endpoints become bare identifiers.

    Needed because `A["x"] --> B` puts a `]` immediately before the arrow, which
    means an endpoint regex never sees the `A`. Runs to a fixed point because
    shapes nest: `{{""}}` and `[("")]` each need more than one pass.
    """
    for _ in range(max_passes):
        reduced = _SHAPE_RE.sub(" ", text)
        if reduced == text:
            return reduced
        text = reduced
    return text


_SUBGRAPH_RE = re.compile(r"^\s*subgraph\s+([A-Za-z_][\w-]*)")


def check_block(source: str, label: str) -> list[str]:
    """Return a list of problems found in one Mermaid block."""
    problems: list[str] = []
    lines = [line for line in source.splitlines() if line.strip()]
    if not lines:
        return [f"{label}: empty block"]

    first = lines[0].strip()
    if not any(first.startswith(t) for t in KNOWN_TYPES):
        problems.append(f"{label}: first line is not a known diagram type: {first!r}")

    for open_char, close_char in BRACKET_PAIRS:
        opened = source.count(open_char)
        closed = source.count(close_char)
        if opened != closed:
            problems.append(f"{label}: unbalanced {open_char}{close_char} — {opened} vs {closed}")

    for number, line in enumerate(source.splitlines(), start=1):
        # A bare & is Mermaid's node-chaining operator; inside a label it must be
        # the entity form, or the diagram silently splits into extra nodes.
        without_entities = line.replace("#38;", "").replace("&amp;", "")
        if "&" in without_entities:
            problems.append(f"{label} line {number}: bare '&' — write it as '#38;'")

    unlabelled = _QUOTED_RE.sub('""', source)
    declared: set[str] = set(_NODE_DECL_RE.findall(unlabelled))
    for line in unlabelled.splitlines():
        subgraph = _SUBGRAPH_RE.match(line)
        if subgraph:
            declared.add(subgraph.group(1))

    referenced: set[str] = set()
    for match in _EDGE_RE.finditer(strip_shapes(unlabelled)):
        referenced.update(match.groups())

    # Only flag ids used *only* as bare edge endpoints and never declared with a
    # shape. Those render as an unlabelled box, which looks like a bug.
    undeclared = {
        node for node in referenced - declared if not any(t.startswith(node) for t in KNOWN_TYPES)
    }
    for node in sorted(undeclared):
        problems.append(f"{label}: node {node!r} is used in an edge but never given a label")

    return problems


def main(paths: list[str]) -> int:
    targets = [Path(p) for p in paths] or sorted([Path("README.md"), *Path("docs").glob("*.md")])
    total_blocks = 0
    problems: list[str] = []

    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for index, block in enumerate(BLOCK_RE.findall(text), start=1):
            total_blocks += 1
            problems.extend(check_block(block, f"{path}: block {index}"))

    if problems:
        print(f"{len(problems)} problem(s) in {total_blocks} mermaid block(s):\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"{total_blocks} mermaid block(s) look well-formed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
