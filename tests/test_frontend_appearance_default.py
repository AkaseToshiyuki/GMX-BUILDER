"""The rollback helper must not rewrite controls or damage an unknown shell."""

from pathlib import Path

import pytest

from scripts.set_frontend_appearance import set_default


def test_default_switch_round_trip_preserves_the_entire_shell(tmp_path):
    original = (Path(__file__).parents[1] / "src/gmxbuilder/web/templates/index.html").read_text()
    template = tmp_path / "index.html"
    template.write_text(original)
    set_default(template, "classic")
    assert template.read_text().replace('data-default="classic"', 'data-default="dark"') == (
        original
    )
    set_default(template, "dark")
    assert template.read_text() == original


def test_unknown_shell_is_preserved_on_failed_switch(tmp_path):
    template = tmp_path / "index.html"
    template.write_text("<p>Preserve this deployment</p>")
    with pytest.raises(ValueError, match="exactly one"):
        set_default(template, "classic")
    assert template.read_text() == "<p>Preserve this deployment</p>"
