from tests.frontend_bundle import frontend_source


def test_gaff2_frontend_requires_explicit_integer_ligand_charge():
    source = frontend_source()

    assert "input.required = true; input.placeholder = 'Required';" in source
    assert "input.type = 'number'; input.step = '1'; input.value = '';" in source
    assert "GAFF2 then assigns AM1-BCC partial charges" in source
    assert "'/api/ligand-charge-suggestions/'" in source
    assert "ligand_pH: _systemPH" in source
    assert "User override:" in source
    assert "Recalculate charge suggestions at target pH" in source


def test_lipid_picker_uses_selected_parameter_family_label():
    source = frontend_source()

    assert "availableSources.indexOf(selectedSource) >= 0" in source
    assert "lipidParameterSourceLabel(selectedSource)" in source
