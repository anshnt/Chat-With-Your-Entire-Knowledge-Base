#!/usr/bin/env python3
"""Validate the internal links in Markdown files.

Same reasoning as the Mermaid checker: a broken link on a README is visible,
embarrassing, and silent. Two classes are checked, both of which rot on their
own as files move and headings get reworded:

* **Relative file links** -- ``[LICENSE](LICENSE)``, ``![map](docs/assets/x.svg)``.
  Checked against the filesystem.
* **Internal anchors** -- ``[Config](#configuration)``. Checked against the
  headings in the target file, using GitHub's slug rules.

External ``http(s)`` links are deliberately *not* checked. Doing so would make
the build depend on other people's uptime, which turns a green build red for
reasons nobody in this repo can fix.

Exit code 1 on any finding.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

#: ``[text](target)`` and ``![alt](target)``, excluding external URLs and mailto.
LINK_RE = re.compile(r"!?\[[^\]]*\]\((?!https?://|mailto:)([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)
#: Fenced blocks, stripped first: a `# comment` in a shell block is not a heading,
#: and a link inside an example is not a link.
FENCE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
#: Inline code spans, stripped for the same reason.
CODE_SPAN_RE = re.compile(r"`[^`\n]*`")


def strip_code(text: str) -> str:
    """Remove fenced blocks and inline spans, so examples are not read as markup."""
    return CODE_SPAN_RE.sub("", FENCE_RE.sub("", text))


def slugify(heading: str) -> str:
    """GitHub's heading-anchor rules: lowercase, drop punctuation, spaces to dashes."""
    text = heading.strip().lower()
    # Inline formatting is not part of the anchor, but its content is.
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def heading_anchors(text: str) -> set[str]:
    """Every anchor the headings of ``text`` make reachable.

    GitHub disambiguates repeated headings by appending ``-1``, ``-2``, so a slug
    seen *n* times is reachable under *n* distinct anchors -- and only those. It
    also turns each space into its own dash, where a reader hand-writing a link
    tends to collapse a run into one; both spellings are accepted.
    """
    slugs = [slugify(m.group(2)) for m in HEADING_RE.finditer(strip_code(text))]
    anchors: set[str] = set()
    for slug, count in Counter(slugs).items():
        for suffix in ("", *(f"-{n}" for n in range(1, count))):
            anchors.add(slug + suffix)
            anchors.add(re.sub(r"-+", "-", slug + suffix))
    return anchors


def check_file(path: Path) -> list[str]:
    """Return the problems found in one Markdown file."""
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    anchors = heading_anchors(text)

    for match in LINK_RE.finditer(strip_code(text)):
        target = match.group(1)
        file_part, _, anchor = target.partition("#")

        if not file_part:
            if anchor and anchor not in anchors:
                problems.append(f"{path}: anchor #{anchor} matches no heading")
            continue

        resolved = path.parent / file_part
        if not resolved.exists():
            problems.append(f"{path}: {file_part} does not exist")
            continue
        # An anchor into another Markdown file is checked against that file.
        if not anchor or resolved.suffix != ".md":
            continue
        if anchor not in heading_anchors(resolved.read_text(encoding="utf-8")):
            problems.append(f"{path}: {file_part}#{anchor} matches no heading there")

    return problems


def main(paths: list[str]) -> int:
    targets = [Path(p) for p in paths] or [
        Path("README.md"),
        *sorted(Path("docs").rglob("*.md")),
        *sorted(Path("frontend").glob("*.md")),
        *sorted(Path(".github").rglob("*.md")),
    ]
    problems: list[str] = []
    checked = 0
    for path in targets:
        if not path.is_file():
            continue
        checked += 1
        problems.extend(check_file(path))

    if problems:
        print(f"{len(problems)} broken link(s) across {checked} file(s):\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"links in {checked} markdown file(s) resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
