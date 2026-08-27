"""Tests for the Mermaid diagram checker.

The checker exists because a broken Mermaid block does not fail quietly — GitHub
renders a red error box on the README indefinitely. These tests prove it actually
catches the four failures that happen in practice, rather than only passing on
diagrams that already work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_mermaid import check_block, main, strip_shapes

GOOD = """flowchart LR
    A["query"] --> B["BM25"]
    A --> C["dense"]
    B --> D{{"fuse"}}
    C --> D
"""


class TestStripShapes:
    def test_removes_a_simple_shape(self) -> None:
        assert "A" in strip_shapes('A[""] --> B')
        assert "[" not in strip_shapes('A[""] --> B')

    def test_removes_nested_shapes(self) -> None:
        """`{{}}` and `[()]` each need more than one pass."""
        for source in ('A{{""}} --> B', 'A[(""")] --> B', 'A(("")) --> B'):
            reduced = strip_shapes(source.replace('"', ""))
            assert "{" not in reduced
            assert "[" not in reduced

    def test_leaves_bare_identifiers(self) -> None:
        assert strip_shapes("A --> B").strip() == "A --> B"


class TestCheckBlock:
    def test_a_valid_block_has_no_problems(self) -> None:
        assert check_block(GOOD, "test") == []

    def test_unbalanced_brackets(self) -> None:
        problems = check_block('flowchart TB\n    A["x --> B["y"]\n', "test")
        assert any("unbalanced" in p for p in problems)

    def test_bare_ampersand(self) -> None:
        """Mermaid reads `&` as node chaining even inside a quoted label."""
        problems = check_block('flowchart TB\n    A["a&b"] --> B["c"]\n', "test")
        assert any("bare '&'" in p for p in problems)

    def test_entity_ampersand_is_accepted(self) -> None:
        assert check_block('flowchart TB\n    A["a#38;b"] --> B["c"]\n', "test") == []

    def test_unknown_diagram_type(self) -> None:
        problems = check_block('notARealType\n    A["x"] --> B["y"]\n', "test")
        assert any("not a known diagram type" in p for p in problems)

    def test_orphan_node_in_an_edge(self) -> None:
        """An unlabelled node renders as a stray empty box, not an error."""
        problems = check_block('flowchart TB\n    A["labelled"] --> ORPHAN\n', "test")
        assert any("ORPHAN" in p for p in problems)

    def test_inline_declarations_are_recognised(self) -> None:
        """`A["x"] --> B["y"]` declares both nodes in one statement."""
        assert check_block('flowchart TB\n    A["x"] --> B["y"]\n', "test") == []

    def test_brackets_inside_a_label_are_not_declarations(self) -> None:
        source = 'flowchart TB\n    A["trailing [n] reattached"] --> B["ok"]\n'
        assert check_block(source, "test") == []

    def test_subgraph_ids_count_as_declared(self) -> None:
        source = (
            "flowchart LR\n"
            '    subgraph inner["Inner"]\n'
            '        A["x"]\n'
            "    end\n"
            '    inner --> B["y"]\n'
        )
        assert check_block(source, "test") == []

    def test_empty_block(self) -> None:
        assert any("empty" in p for p in check_block("   \n", "test"))

    @pytest.mark.parametrize(
        "first_line",
        ["flowchart TB", "graph LR", "sequenceDiagram", "erDiagram", "stateDiagram-v2"],
    )
    def test_known_types_are_accepted(self, first_line: str) -> None:
        problems = check_block(f'{first_line}\n    A["x"] --> B["y"]\n', "test")
        assert not any("diagram type" in p for p in problems)


class TestMain:
    def test_passes_on_a_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.md"
        path.write_text(f"# Doc\n\n```mermaid\n{GOOD}```\n")
        assert main([str(path)]) == 0

    def test_fails_on_an_invalid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.md"
        path.write_text('# Doc\n\n```mermaid\nflowchart TB\n    A["x --> B\n```\n')
        assert main([str(path)]) == 1

    def test_a_file_with_no_diagrams_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.md"
        path.write_text("# Doc\n\nJust prose, and a python block.\n\n```python\nx = 1\n```\n")
        assert main([str(path)]) == 0

    def test_missing_files_are_skipped(self, tmp_path: Path) -> None:
        assert main([str(tmp_path / "nope.md")]) == 0

    def test_this_repository_readme_is_valid(self) -> None:
        """The diagrams shipped in this repo must actually parse."""
        readme = Path(__file__).resolve().parents[1] / "README.md"
        if not readme.is_file():  # pragma: no cover
            pytest.skip("README not present")
        assert main([str(readme)]) == 0
