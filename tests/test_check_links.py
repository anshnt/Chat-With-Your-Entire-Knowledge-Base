"""Tests for the Markdown link checker.

A dead link is the cheapest possible defect to ship and one of the most visible:
nothing fails, the README simply lies. These tests pin each failure class the
checker is meant to catch, and -- just as important -- the cases it must *not*
flag, since a doc checker that cries wolf gets disabled within a week.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_links import check_file, heading_anchors, main, slugify

REPO = Path(__file__).resolve().parents[1]


class TestSlugify:
    def test_lowercases_and_dashes(self) -> None:
        assert slugify("How It Works") == "how-it-works"

    def test_drops_punctuation(self) -> None:
        assert slugify("What's next? (really)") == "whats-next-really"

    def test_keeps_inline_code_content(self) -> None:
        assert slugify("The `kb eval` command") == "the-kb-eval-command"

    def test_keeps_bold_content(self) -> None:
        assert slugify("**Contents**") == "contents"

    def test_unwraps_a_link_in_a_heading(self) -> None:
        assert slugify("[Quickstart](#quickstart) notes") == "quickstart-notes"

    def test_keeps_existing_dashes_and_underscores(self) -> None:
        assert slugify("state-of-the_art") == "state-of-the_art"


class TestHeadingAnchors:
    def test_collects_every_level(self) -> None:
        anchors = heading_anchors("# One\n\n### Two\n\n###### Six\n")
        assert {"one", "two", "six"} <= anchors

    def test_a_unique_heading_gets_no_numeric_suffix(self) -> None:
        """`#config-1` with one Config heading is rot, not disambiguation."""
        anchors = heading_anchors("# Config\n")
        assert "config" in anchors
        assert "config-1" not in anchors

    def test_repeated_headings_get_one_suffix_each(self) -> None:
        anchors = heading_anchors("# Config\n\n# Config\n\n# Config\n")
        assert {"config", "config-1", "config-2"} <= anchors
        assert "config-3" not in anchors

    def test_a_comment_in_a_shell_block_is_not_a_heading(self) -> None:
        source = "# Real\n\n```sh\n# Not A Heading\nkb ingest .\n```\n"
        anchors = heading_anchors(source)
        assert "real" in anchors
        assert "not-a-heading" not in anchors

    def test_both_spellings_of_a_stripped_dash_run(self) -> None:
        """GitHub keeps a dash per space; a hand-written link collapses the run."""
        anchors = heading_anchors("# Scores — and ranks\n")
        assert "scores--and-ranks" in anchors
        assert "scores-and-ranks" in anchors


class TestCheckFile:
    def test_a_resolving_anchor_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.md"
        path.write_text("# Setup\n\nSee [setup](#setup).\n")
        assert check_file(path) == []

    def test_a_broken_anchor_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text("# Setup\n\nSee [install](#installation).\n")
        problems = check_file(path)
        assert len(problems) == 1
        assert "#installation" in problems[0]

    def test_a_missing_relative_file_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text("See the [licence](LICENSE).\n")
        problems = check_file(path)
        assert len(problems) == 1
        assert "LICENSE does not exist" in problems[0]

    def test_an_existing_relative_file_passes(self, tmp_path: Path) -> None:
        (tmp_path / "LICENSE").write_text("MIT\n")
        path = tmp_path / "ok.md"
        path.write_text("See the [licence](LICENSE).\n")
        assert check_file(path) == []

    def test_an_image_target_is_checked(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text("![map](assets/corpus-map.svg)\n")
        assert any("corpus-map.svg" in p for p in check_file(path))

    def test_a_target_with_a_title_is_parsed(self, tmp_path: Path) -> None:
        (tmp_path / "LICENSE").write_text("MIT\n")
        path = tmp_path / "ok.md"
        path.write_text('See the [licence](LICENSE "The licence").\n')
        assert check_file(path) == []

    def test_paths_resolve_from_the_file_not_the_cwd(self, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "top.txt").write_text("x\n")
        path = tmp_path / "docs" / "page.md"
        path.write_text("[up](../top.txt) and [sideways](top.txt)\n")
        problems = check_file(path)
        assert len(problems) == 1
        assert "top.txt does not exist" in problems[0]

    def test_external_links_are_left_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.md"
        path.write_text(
            "[site](https://example.invalid/nope)\n"
            "[plain](http://example.invalid)\n"
            "[mail](mailto:someone@example.invalid)\n"
        )
        assert check_file(path) == []

    def test_a_link_inside_a_code_fence_is_not_a_link(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.md"
        path.write_text("# Doc\n\n```md\n[example](missing.txt)\n```\n")
        assert check_file(path) == []

    def test_a_link_inside_an_inline_span_is_not_a_link(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.md"
        path.write_text("Write it as `[text](target.md)`.\n")
        assert check_file(path) == []

    def test_an_anchor_into_another_markdown_file_is_followed(self, tmp_path: Path) -> None:
        (tmp_path / "other.md").write_text("# Present\n")
        path = tmp_path / "doc.md"
        path.write_text("[there](other.md#present) [gone](other.md#absent)\n")
        problems = check_file(path)
        assert len(problems) == 1
        assert "other.md#absent" in problems[0]

    def test_an_anchor_into_a_non_markdown_file_is_not_followed(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("x = 1\n")
        path = tmp_path / "doc.md"
        path.write_text("[code](app.py#L1)\n")
        assert check_file(path) == []


class TestMain:
    def test_passes_on_a_clean_file(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.md"
        path.write_text("# Setup\n\n[setup](#setup)\n")
        assert main([str(path)]) == 0

    def test_fails_on_a_broken_anchor(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text("# Setup\n\n[nope](#nope)\n")
        assert main([str(path)]) == 1

    def test_fails_on_a_broken_relative_link(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text("[nope](docs/missing.md)\n")
        assert main([str(path)]) == 1

    def test_missing_files_are_skipped(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "nope.md")]) == 0

    def test_this_repositorys_own_docs_resolve(self) -> None:
        """The links shipped in this repo must actually point somewhere."""
        readme = REPO / "README.md"
        if not readme.is_file():  # pragma: no cover
            pytest.skip("README not present")
        docs = sorted(str(p) for p in (REPO / "docs").rglob("*.md"))
        assert main([str(readme), *docs]) == 0
