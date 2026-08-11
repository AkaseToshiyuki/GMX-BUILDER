import pytest

from gmxbuilder.modules.forcefield.lipid_policy import (
    charmm_lipid_capability,
    lipid_has_rtp,
    lipid_rtp_identity_issues,
    lipid_rtp_name,
    lipid_rtp_template,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry


@pytest.mark.parametrize(
    "lipid_name,template_name",
    [
        ("BSM", "LSM"),
        ("CER16", "CER160"),
        ("CER18", "CER180"),
        ("CER24", "CER240"),
        ("CHOL", "CHL1"),
        ("DPEPE", "DYPE"),
        ("POP2", "POPI25"),
        ("POP3", "POPI35"),
        ("PUPC", "PDOPC"),
        ("SAPI", "SAPI25"),
        ("TMCL", "TMCL2"),
        ("TOCL", "TOCL2"),
    ],
)
def test_charmm_lipid_identity_mapping(lipid_name, template_name):
    assert lipid_rtp_name(lipid_name, "charmm36m") == template_name
    assert lipid_has_rtp(lipid_name, "charmm36m")


def test_mapping_is_not_applied_to_amber():
    assert lipid_rtp_name("CHOL", "amber14sb") == "CHOL"


def test_release_specific_npt_quality_failure_names_validated_alternative():
    supported, reason = charmm_lipid_capability("SAPI", "charmm36m")
    assert not supported
    assert "use CHARMM36" in reason
    assert charmm_lipid_capability("SAPI", "charmm36")[0]


def test_current_lipid_stream_is_compatible_with_classic_charmm_protein():
    assert lipid_has_rtp("PUPC", "charmm36m")
    assert lipid_has_rtp("PUPC", "charmm36")


@pytest.mark.parametrize("lipid_name", ["POP2", "POP3", "SAPI", "TOCL"])
def test_mapped_template_matches_formula_and_charge(lipid_name):
    assert lipid_rtp_identity_issues(lipid_name, "charmm36m") == ()


@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_every_advertised_charmm_lipid_passes_identity_audit(force_field):
    for lipid_name in LipidRegistry.list():
        if lipid_has_rtp(lipid_name, force_field):
            assert lipid_rtp_identity_issues(lipid_name, force_field) == ()


@pytest.mark.parametrize(
    "lipid_name",
    [
        "DLIPA", "DLIPG", "DLIPS", "PAPC", "PAPE", "PAPG", "PAPI",
        "PIPI", "PMPC", "SMPC", "SOP2", "SOP3", "SOPI", "LPC16",
        "LPC18", "LPE16", "LYSPG", "DSM",
        "DOPGD", "DPPGD",
        "PPCPL", "PPEPL",
        "MGDG", "DGDG",
        "CAMP",
    ],
)
def test_modular_charmm36m_lipid_template_is_connected_and_exact(lipid_name):
    template_name, template = lipid_rtp_template(lipid_name, "charmm36m")
    assert template_name == lipid_name
    assert template is not None
    atom_names = {atom[0] for atom in template["atoms"]}
    assert len(atom_names) == len(template["atoms"])
    assert all(left in atom_names and right in atom_names
               for left, right in template["bonds"])
    assert lipid_rtp_identity_issues(lipid_name, "charmm36m") == ()


@pytest.mark.parametrize(
    "lipid_name",
    [
        "DLIPA", "DLIPC", "DLIPE", "DLIPG", "DLIPS", "ERG",
        "LPC16", "LPC18", "LPE16", "LYSPG", "PUPC", "SITO", "STIG",
    ],
)
def test_current_lipid_stream_is_exposed_to_classic_charmm(lipid_name):
    assert lipid_has_rtp(lipid_name, "charmm36")
    assert lipid_rtp_identity_issues(lipid_name, "charmm36") == ()


@pytest.mark.parametrize("lipid_name", ["DOPGD", "DPPGD"])
@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_published_dag_template_is_available_in_both_charmm_releases(
    lipid_name, force_field,
):
    template_name, template = lipid_rtp_template(lipid_name, force_field)
    assert template_name == lipid_name
    assert template is not None
    assert lipid_rtp_identity_issues(lipid_name, force_field) == ()
    atoms = {atom[0]: atom[1:3] for atom in template["atoms"]}
    assert atoms["C1"] == ("CTL2", 0.05)
    assert atoms["O11"] == ("OHL", -0.65)
    assert atoms["HO1"] == ("HOL", 0.42)
    assert ("O11", "HO1") in template["bonds"]


@pytest.mark.parametrize("lipid_name", ["PPCPL", "PPEPL"])
@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_published_plasmalogen_template_is_available_in_both_charmm_releases(
    lipid_name, force_field,
):
    _template_name, template = lipid_rtp_template(lipid_name, force_field)
    assert template is not None
    assert lipid_rtp_identity_issues(lipid_name, force_field) == ()
    atoms = {atom[0]: atom[1:3] for atom in template["atoms"]}
    assert "O32" not in atoms
    assert "H2Y" not in atoms
    assert atoms["O31"] == ("OG301", -0.36)
    assert atoms["C31"] == ("CEL1", 0.0)
    assert atoms["C32"] == ("CEL1", -0.2)
    assert atoms["H1X"] == ("HEL1", 0.08)


@pytest.mark.parametrize("lipid_name", ["MGDG", "DGDG"])
@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_published_galactolipid_patches_are_exact(lipid_name, force_field):
    _template_name, template = lipid_rtp_template(lipid_name, force_field)
    assert template is not None
    assert lipid_rtp_identity_issues(lipid_name, force_field) == ()
    atoms = {atom[0]: atom[1:3] for atom in template["atoms"]}
    assert atoms["C1"] == ("CTO2", 0.0)
    assert atoms["O1G"] == ("OC301", -0.36)
    assert ("O1G", "C1") in template["bonds"]
    if lipid_name == "DGDG":
        assert atoms["O6G"] == ("OC301", -0.36)
        assert ("O6G", "C1A") in template["bonds"]


@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_campesterol_uses_exact_charmm_plant_sterol_fragments(force_field):
    _template_name, template = lipid_rtp_template("CAMP", force_field)
    assert template is not None
    assert lipid_rtp_identity_issues("CAMP", force_field) == ()
    atoms = {atom[0]: atom[1:3] for atom in template["atoms"]}
    assert "C29" not in atoms
    assert atoms["C28"] == ("CTL3", -0.27)
    assert atoms["H28C"] == ("HAL3", 0.09)
    assert ("C28", "H28C") in template["bonds"]


@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_gm1_uses_native_charmm_glycolipid_patches(force_field):
    _template_name, template = lipid_rtp_template("GM1", force_field)
    assert template is not None
    assert lipid_rtp_identity_issues("GM1", force_field) == ()
    names = [atom[0] for atom in template["atoms"]]
    assert len(names) == len(set(names)) == 237
    assert max(map(len, names)) <= 5
    atoms = {atom[0]: atom[1:3] for atom in template["atoms"]}
    assert atoms["C1S"] == ("CTO2", 0.0)       # CERB
    assert atoms["O1X"] == ("OC301", -0.36)   # Glc-Cer
    assert atoms["O4X"] == ("OC301", -0.36)   # Gal(beta1-4)Glc
    assert atoms["O4Y"] == ("OC301", -0.36)   # GalNAc(beta1-4)Gal
    assert atoms["O3Z"] == ("OC301", -0.36)   # Gal(beta1-3)GalNAc
    assert atoms["C2A"] == ("CC3062", 0.28)   # Neu5Ac(alpha2-3)Gal
    assert ("O1X", "C1S") in template["bonds"]
    assert ("O4X", "C1Y") in template["bonds"]
    assert ("O4Y", "C1Z") in template["bonds"]
    assert ("O3Z", "C1Q") in template["bonds"]
    assert ("O3Y", "C2A") in template["bonds"]
