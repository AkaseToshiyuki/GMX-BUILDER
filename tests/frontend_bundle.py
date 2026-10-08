"""Helpers for source-level tests of the ordered browser application bundle."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SCRIPT_PATHS = (
    ROOT / "src/gmxbuilder/web/static/app.js",
    ROOT / "src/gmxbuilder/web/static/app_parts/custom_lipids.js",
    ROOT / "src/gmxbuilder/web/static/app_parts/structure_processing.js",
    ROOT / "src/gmxbuilder/web/static/app_parts/simulation.js",
    ROOT / "src/gmxbuilder/web/static/app_parts/system_verification.js",
)


def frontend_source() -> str:
    """Return classic scripts in the same order used by the page template."""
    return "\n".join(path.read_text(encoding="utf-8") for path in FRONTEND_SCRIPT_PATHS)
