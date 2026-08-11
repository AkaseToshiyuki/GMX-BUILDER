from pathlib import Path


def test_gaff2_frontend_requires_explicit_integer_ligand_charge():
    root = Path(__file__).parents[1]
    source = (root / "src/gmxbuilder/web/static/app.js").read_text()

    assert "input.required = true; input.placeholder = 'Required';" in source
    assert "input.type = 'number'; input.step = '1'; input.value = '';" in source
    assert "GAFF2 then assigns AM1-BCC partial charges" in source
    assert "'/api/ligand-charge-suggestions/' + state.taskId" in source
    assert "ligand_pH: _systemPH" in source
    assert "User override:" in source
    assert "Recalculate charge suggestions at target pH" in source


def test_lipid_picker_uses_selected_parameter_family_label():
    root = Path(__file__).parents[1]
    source = (root / "src/gmxbuilder/web/static/app.js").read_text()

    assert "availableSources.indexOf(selectedSource) >= 0" in source
    assert "gaff2: 'Amber/GAFF2'" in source
