"""Contracts for optional deployment extensions.

Extensions live outside the published tree, so the properties that matter are
about their *absence* and their *misbehaviour*: a public build has none, and a
private one must not be able to break the page or inject markup into it.

Each test here is written so it would fail if the guarantee were removed --
several of these could otherwise pass against an implementation that does
nothing at all.
"""

from __future__ import annotations

import sys
import textwrap
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from gmxbuilder import extensions
from gmxbuilder.extensions import (
    Announcement,
    collect_announcements,
    render_announcements_html,
)
from gmxbuilder.web.server import app


@pytest.fixture(autouse=True)
def no_ambient_extensions(monkeypatch):
    monkeypatch.delenv(extensions.ENVIRONMENT_VARIABLE, raising=False)


def _install_module(monkeypatch, name: str, source: str) -> None:
    """Register a throwaway module and point the loader at it."""
    module = ModuleType(name)
    exec(textwrap.dedent(source), module.__dict__)  # noqa: S102 - test-local source
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(extensions.ENVIRONMENT_VARIABLE, name)


# --------------------------------------------------------------------------
# Absence: the public build's behaviour


def test_no_extension_renders_nothing_at_all():
    """Not an empty panel -- nothing, so the page looks unchanged."""
    assert collect_announcements() == []
    assert render_announcements_html() == ""


def test_the_homepage_consumes_its_placeholder_even_with_no_extension():
    with TestClient(app) as client:
        body = client.get("/").text
    assert "{{ announcements }}" not in body
    assert 'class="announcements"' not in body


# --------------------------------------------------------------------------
# Misbehaviour: an extension must not be able to take the page down


def test_an_extension_that_cannot_be_imported_is_skipped(monkeypatch):
    monkeypatch.setenv(extensions.ENVIRONMENT_VARIABLE, "gmxbuilder_no_such_extension")
    assert collect_announcements() == []
    with TestClient(app) as client:
        assert client.get("/").status_code == 200


def test_an_extension_that_raises_is_contained(monkeypatch):
    _install_module(
        monkeypatch,
        "gmxbuilder_raising_extension",
        """
        def announcements():
            raise RuntimeError("extension is broken")
        """,
    )
    assert collect_announcements() == []
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert 'class="announcements"' not in response.text


def test_one_broken_extension_does_not_suppress_a_working_one(monkeypatch):
    for name, source in (
        (
            "gmxbuilder_broken_ext",
            """
            def announcements():
                raise RuntimeError("broken")
            """,
        ),
        (
            "gmxbuilder_working_ext",
            """
            def announcements():
                return [{"title": "Maintenance", "body": "Back at 09:00 UTC."}]
            """,
        ),
    ):
        module = ModuleType(name)
        exec(textwrap.dedent(source), module.__dict__)  # noqa: S102
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(
        extensions.ENVIRONMENT_VARIABLE,
        "gmxbuilder_broken_ext,gmxbuilder_working_ext",
    )

    collected = collect_announcements()
    assert [item.title for item in collected] == ["Maintenance"]


@pytest.mark.parametrize(
    "returned",
    ["a string", 42, None, {"title": "not a sequence"}],
)
def test_a_wrong_return_shape_contributes_nothing(monkeypatch, returned):
    _install_module(
        monkeypatch,
        "gmxbuilder_wrong_shape_ext",
        f"def announcements():\n    return {returned!r}\n",
    )
    assert collect_announcements() == []


def test_unusable_items_are_dropped_without_losing_the_rest(monkeypatch):
    """Only an entry with neither a title nor a body is unusable.

    A heading on its own is a perfectly good bulletin line, so requiring both
    fields would reject something an author reasonably wrote.
    """
    _install_module(
        monkeypatch,
        "gmxbuilder_mixed_ext",
        """
        def announcements():
            return [
                None,
                "not a mapping",
                {"title": "", "body": "   "},
                {"title": "Title only"},
                {"body": "Body only"},
                {"title": "Kept", "body": "This one is usable."},
            ]
        """,
    )
    assert [item.title for item in collect_announcements()] == [
        "Title only",
        "",
        "Kept",
    ]


# --------------------------------------------------------------------------
# Injection: extensions supply text, never markup


def test_markup_from_an_extension_is_escaped_not_rendered(monkeypatch):
    _install_module(
        monkeypatch,
        "gmxbuilder_injecting_ext",
        """
        def announcements():
            return [{
                "title": "<script>alert(1)</script>",
                "body": "<img src=x onerror=alert(2)>",
            }]
        """,
    )
    with TestClient(app) as client:
        body = client.get("/").text

    # The escaped text necessarily contains the attribute as characters, so
    # assert that no real tag was produced rather than that the substring is
    # absent.
    assert "<script>alert(1)</script>" not in body
    assert "<img src=x" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "&lt;img src=x onerror=alert(2)&gt;" in body


def test_an_unknown_level_cannot_inject_a_class_name():
    """The level becomes a CSS class, so it must come from a fixed set."""
    rendered = render_announcements_html([Announcement("t", "b", level='info" onload="alert(1)')])
    assert 'announcement-info"' in rendered
    assert "onload" not in rendered


@pytest.mark.parametrize("level", extensions.ANNOUNCEMENT_LEVELS)
def test_each_supported_level_reaches_the_markup(level):
    rendered = render_announcements_html([Announcement("t", "b", level=level)])
    assert f"announcement-{level}" in rendered


# --------------------------------------------------------------------------
# Presence: a working extension actually appears


def test_a_working_extension_reaches_the_homepage(monkeypatch):
    """The counterpart to the absence tests: this must fail if nothing renders."""
    _install_module(
        monkeypatch,
        "gmxbuilder_panel_ext",
        """
        def announcements():
            return [{
                "title": "Scheduled maintenance",
                "body": "The queue pauses at 02:00 UTC.",
                "level": "warning",
            }]
        """,
    )
    with TestClient(app) as client:
        body = client.get("/").text

    assert 'class="announcements"' in body
    assert "Scheduled maintenance" in body
    assert "The queue pauses at 02:00 UTC." in body
    assert "announcement-warning" in body


# --------------------------------------------------------------------------
# Presentation controls: pinning, level, links


def test_pinned_entries_are_rendered_before_unpinned_ones():
    rendered = render_announcements_html(
        [
            Announcement("second", "b"),
            Announcement("first", "b", pinned=True),
        ]
    )
    assert rendered.index("first") < rendered.index("second")
    assert "announcement-pinned" in rendered


def test_pinning_preserves_file_order_within_each_group():
    """Ordering inside a group is the author's, expressed by file position."""
    rendered = render_announcements_html(
        [
            Announcement("plain-a", "b"),
            Announcement("pin-a", "b", pinned=True),
            Announcement("plain-b", "b"),
            Announcement("pin-b", "b", pinned=True),
        ]
    )
    order = [rendered.index(name) for name in ("pin-a", "pin-b", "plain-a", "plain-b")]
    assert order == sorted(order)


@pytest.mark.parametrize(
    "target",
    [
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "java\tscript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "/relative/path",
        "example.org",
        "",
    ],
)
def test_a_disallowed_link_is_dropped(target):
    """The link is the one place an attribute value comes from content."""
    rendered = render_announcements_html([Announcement("t", "b", link=target)])
    assert "<a href=" not in rendered


@pytest.mark.parametrize(
    "target",
    [
        "https://example.org/status",
        "http://example.org/status",
        "mailto:someone@example.org",
    ],
)
def test_a_permitted_link_is_rendered(target):
    rendered = render_announcements_html([Announcement("t", "b", link=target)])
    assert f'<a href="{target}"' in rendered


def test_link_text_is_escaped_and_falls_back_to_the_target():
    rendered = render_announcements_html(
        [Announcement("t", "b", link="https://example.org", link_text="<b>x</b>")]
    )
    assert "<b>x</b>" not in rendered
    assert "&lt;b&gt;x&lt;/b&gt;" in rendered

    plain = render_announcements_html([Announcement("t", "b", link="https://example.org")])
    assert ">https://example.org<" in plain


def test_a_quote_in_a_link_cannot_break_out_of_the_attribute():
    rendered = render_announcements_html(
        [Announcement("t", "b", link='https://example.org/"onmouseover="alert(1)')]
    )
    assert 'onmouseover="alert(1)"' not in rendered
    assert "&quot;" in rendered or "<a href=" not in rendered


def test_coerce_accepts_the_bulletin_file_shape():
    """The mapping shape is what a YAML file yields, so it is pinned here."""
    announcement = Announcement.coerce(
        {
            "title": "Maintenance",
            "body": "Back soon.",
            "level": "critical",
            "pinned": True,
            "link": "mailto:someone@example.org",
            "link_text": "Contact",
        }
    )
    assert announcement == Announcement(
        "Maintenance", "Back soon.", "critical", True, "mailto:someone@example.org", "Contact"
    )
