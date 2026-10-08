#!/usr/bin/env python3
"""Switch the frontend's installation default without changing backend behaviour.

This edits the tracked HTML shell. Review/commit the resulting configuration
change through the normal private release workflow. Existing tabs with an
explicit appearance override retain their selection.
"""

import argparse
import os
import re
import tempfile
from pathlib import Path


def set_default(template: Path, appearance: str) -> None:
    if appearance not in {"classic", "dark"}:
        raise ValueError("Appearance must be classic or dark")
    source = template.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(<link id="dark-appearance"[^>]*data-default=")(classic|dark)(")',
        lambda match: match[1] + appearance + match[3],
        source,
    )
    if count != 1:
        raise ValueError("Expected exactly one frontend appearance default; nothing changed")
    if updated == source:
        return
    descriptor, name = tempfile.mkstemp(prefix=".appearance-", dir=template.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(updated)
        os.chmod(name, template.stat().st_mode & 0o777)
        os.replace(name, template)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("appearance", choices=("classic", "dark"))
    args = parser.parse_args()
    template = Path(__file__).resolve().parents[1] / "src/gmxbuilder/web/templates/index.html"
    set_default(template, args.appearance)
    print(f"Frontend default: {args.appearance}. Refresh the page to apply.")
    print("Review and commit the HTML change; use ?appearance=classic or dark for a tab override.")


if __name__ == "__main__":
    main()
