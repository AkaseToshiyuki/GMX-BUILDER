"""The page template must nest correctly.

A single unbalanced tag is invisible to every source-text assertion in this
suite and does not raise anything in the browser either: HTML has no parse
errors, only recovery. One stray `</div>` closed the panel container early,
which put the task-type grid outside the panel that is supposed to hide it, so
picking a workflow left every task card on screen with step 1 underneath.

This parses the template and checks that each container closes the element it
opened -- the one check that would have caught that before it shipped.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "gmxbuilder" / "web" / "templates" / "index.html"

# Elements that lay out the page and always require an explicit close tag.
# Anything with an optional close (p, li) is deliberately excluded: HTML lets
# those be left open, so a mismatch there is not a defect.
CONTAINERS = frozenset(
    {"div", "section", "nav", "header", "footer", "main", "form", "table", "tbody", "thead"}
)


class _NestingCheck(HTMLParser):
    """Track container open/close pairs and record every mismatch."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int, str | None]] = []
        self.problems: list[str] = []
        # id -> the ids of the containers enclosing it, outermost first.
        self.ancestors: dict[str, list[str]] = {}

    def handle_starttag(self, tag, attrs):
        element_id = dict(attrs).get("id")
        if element_id:
            self.ancestors[element_id] = [open_id for _, _, open_id in self.stack if open_id]
        if tag in CONTAINERS:
            self.stack.append((tag, self.getpos()[0], element_id))

    def handle_startendtag(self, tag, attrs):
        pass  # self-closing; opens nothing

    def handle_endtag(self, tag):
        if tag not in CONTAINERS:
            return
        line = self.getpos()[0]
        if not self.stack:
            self.problems.append(f"line {line}: </{tag}> closes nothing")
            return
        opened, opened_at, _ = self.stack.pop()
        if opened != tag:
            self.problems.append(
                f"line {line}: </{tag}> closes <{opened}> opened on line {opened_at}"
            )


def test_page_containers_and_step_panels_are_nested():
    """The task grid and each step must remain inside their intended containers."""
    checker = _NestingCheck()
    checker.feed(TEMPLATE.read_text(encoding="utf-8"))
    assert checker.problems == []
    assert checker.stack == [], f"still open: {checker.stack}"
    assert "panel-task-type" in checker.ancestors["task-grid"]
    for panel_id in (
        "panel-task-type",
        "panel-input",
        "panel-forcefield",
        "panel-membrane",
        "panel-ions",
    ):
        assert checker.ancestors[panel_id][-1] == "panels", panel_id
