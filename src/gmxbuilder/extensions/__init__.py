"""Optional feature modules that are not part of the public distribution.

A deployment may add features that this repository does not ship -- a homepage
announcement panel, for instance. Those live outside the published tree, so
everything here is written on the assumption that **no extension is present**:
the public build must behave exactly as it does today, and a broken or missing
extension must never keep the service from starting.

Two discovery routes, both optional:

``GMXBUILDER_EXTENSIONS``
    Comma-separated importable module names. Suits a private module that is
    checked out beside the application rather than installed as a package.

``gmxbuilder.extensions`` entry points
    For an extension distributed as a proper package.

A module is an extension if it defines any of the hook functions below. It need
not import anything from here, which keeps a private module free of a build-time
dependency on this package's version.

Hooks
-----
``announcements() -> Sequence[Announcement | Mapping]``
    Items for the homepage panel. Returning an empty sequence renders nothing
    at all, not an empty box.

Extensions return **data, never markup**. The homepage is assembled by string
substitution with no template autoescaping, so an extension that could return
HTML would be an injection route straight into every page. Escaping happens
here, once, where it cannot be forgotten.
"""

from __future__ import annotations

import html
import importlib
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType

logger = logging.getLogger("gmxbuilder.extensions")

ENTRY_POINT_GROUP = "gmxbuilder.extensions"
ENVIRONMENT_VARIABLE = "GMXBUILDER_EXTENSIONS"

#: Levels an announcement may claim. Anything else is treated as "info", so a
#: typo cannot smuggle an arbitrary CSS class into the page.
ANNOUNCEMENT_LEVELS = ("info", "notice", "warning", "critical")

#: Link schemes an announcement may use. A bulletin often needs to point at a
#: contact address or a page, and offering one escaped field is far safer than
#: accepting markup. "javascript:" and "data:" are absent on purpose.
ANNOUNCEMENT_LINK_SCHEMES = ("https://", "http://", "mailto:")


@dataclass(frozen=True)
class Announcement:
    """One homepage message. Plain text only; markup is not accepted."""

    title: str
    body: str
    level: str = "info"
    #: Pinned items are shown first, keeping their relative order otherwise.
    pinned: bool = False
    #: Optional link target, restricted to ANNOUNCEMENT_LINK_SCHEMES.
    link: str = ""
    #: Text for the link; falls back to the target itself.
    link_text: str = ""

    @classmethod
    def coerce(cls, value: object) -> Announcement | None:
        """Build an announcement from a dataclass or a plain mapping.

        Returns None for anything unusable rather than raising, so one
        malformed item cannot suppress the rest of the panel.
        """
        if isinstance(value, cls):
            candidate = value
        elif isinstance(value, dict):
            title = value.get("title", "")
            body = value.get("body", "")
            if not isinstance(title, str) or not isinstance(body, str):
                return None
            candidate = cls(
                title,
                body,
                str(value.get("level", "info")),
                bool(value.get("pinned", False)),
                str(value.get("link", "")),
                str(value.get("link_text", "")),
            )
        else:
            return None

        title = candidate.title.strip()
        body = candidate.body.strip()
        if not title and not body:
            return None
        level = candidate.level if candidate.level in ANNOUNCEMENT_LEVELS else "info"
        return cls(
            title,
            body,
            level,
            bool(candidate.pinned),
            safe_link(candidate.link),
            candidate.link_text.strip(),
        )


def safe_link(target: object) -> str:
    """Return *target* if it is a permitted link, otherwise an empty string.

    Only the schemes in ANNOUNCEMENT_LINK_SCHEMES are accepted, so a bulletin
    file cannot turn a link into script execution. Whitespace is rejected
    outright rather than stripped from the middle, since "java\tscript:" is a
    known way past a naive prefix check.
    """
    if not isinstance(target, str):
        return ""
    candidate = target.strip()
    if not candidate or any(character.isspace() for character in candidate):
        return ""
    lowered = candidate.lower()
    if not lowered.startswith(ANNOUNCEMENT_LINK_SCHEMES):
        return ""
    return candidate


def _named_modules() -> Iterator[ModuleType]:
    raw = os.environ.get(ENVIRONMENT_VARIABLE, "")
    for name in (item.strip() for item in raw.split(",")):
        if not name:
            continue
        try:
            yield importlib.import_module(name)
        except Exception:
            # A deployment misconfiguration must not take the service down.
            logger.warning("Could not load extension module %r", name, exc_info=True)


def _entry_point_modules() -> Iterator[ModuleType]:
    try:
        import importlib.metadata

        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:
        logger.warning("Could not enumerate %s entry points", ENTRY_POINT_GROUP, exc_info=True)
        return
    for entry_point in entry_points:
        try:
            yield entry_point.load()
        except Exception:
            logger.warning("Could not load extension %r", entry_point, exc_info=True)


def loaded_extensions() -> list[ModuleType]:
    """Return every extension that could be loaded, ignoring those that could not."""
    return [*_named_modules(), *_entry_point_modules()]


def collect_announcements() -> list[Announcement]:
    """Gather homepage announcements from every extension.

    An extension that raises, returns the wrong shape, or produces unusable
    items contributes nothing and is logged; the others are unaffected.
    """
    collected: list[Announcement] = []
    for module in loaded_extensions():
        hook = getattr(module, "announcements", None)
        if hook is None:
            continue
        try:
            produced = hook()
        except Exception:
            logger.warning(
                "Extension %r raised while producing announcements",
                getattr(module, "__name__", module),
                exc_info=True,
            )
            continue
        if not isinstance(produced, Sequence) or isinstance(produced, (str, bytes)):
            logger.warning(
                "Extension %r returned %s from announcements(); expected a sequence",
                getattr(module, "__name__", module),
                type(produced).__name__,
            )
            continue
        for item in produced:
            announcement = Announcement.coerce(item)
            if announcement is not None:
                collected.append(announcement)
    return collected


def render_announcements_html(announcements: Sequence[Announcement] | None = None) -> str:
    """Return the homepage panel markup, or an empty string when there is none.

    Empty output is deliberate: the placeholder collapses to nothing rather
    than leaving a blank panel on a build with no extensions, which is every
    public build.
    """
    items = list(announcements) if announcements is not None else collect_announcements()
    if not items:
        return ""

    # Pinned items first, otherwise file order. sorted() is stable, so an
    # author controls the order within each group by moving entries in the file.
    items = sorted(items, key=lambda item: not item.pinned)

    parts = ['<section class="announcements" aria-label="Announcements">']
    for item in items:
        # Both of these become part of an attribute, so they are re-checked
        # here rather than trusted from the object: coerce() is not the only
        # way an Announcement can be built, and this is the last gate before
        # markup.
        level = item.level if item.level in ANNOUNCEMENT_LEVELS else "info"
        classes = f"announcement announcement-{level}"
        if item.pinned:
            classes += " announcement-pinned"
        parts.append(f'  <article class="{classes}">')
        if item.title:
            pin = (
                '<span class="announcement-pin" aria-hidden="true">*</span> ' if item.pinned else ""
            )
            label = ' <span class="visually-hidden">(pinned)</span>' if item.pinned else ""
            parts.append(f"    <h3>{pin}{html.escape(item.title)}{label}</h3>")
        if item.body:
            parts.append(f"    <p>{html.escape(item.body)}</p>")
        link = safe_link(item.link)
        if link:
            text = item.link_text.strip() or link
            parts.append(
                f'    <p class="announcement-link">'
                f'<a href="{html.escape(link, quote=True)}">{html.escape(text)}</a></p>'
            )
        parts.append("  </article>")
    parts.append("</section>")
    return "\n".join(parts)
