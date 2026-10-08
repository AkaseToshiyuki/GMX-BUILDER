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
        ("POP3", "POPI35"),
        ("PUPC", "PDOPC"),
        ("TMCL", "TMCL2"),
        ("TOCL", "TOCL2"),
    ],
)
def test_charmm_lipid_identity_mapping(lipid_name, template_name):
    assert lipid_rtp_name(lipid_name, "charmm36m") == template_name
    assert lipid_has_rtp(lipid_name, "charmm36m")


def test_mapping_is_not_applied_to_amber():
    assert lipid_rtp_name("CHOL", "amber14sb") == "CHOL"


@pytest.mark.parametrize("name", ["POP2", "PAPI", "SAPI", "SOP2"])
@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_corrected_pip_mapping_accepts_exact_identity_and_rejects_old_protonation(
    name, force_field, monkeypatch
):
    from gmxbuilder.modules.forcefield import lipid_policy as policy

    assert lipid_has_rtp(name, force_field)
    assert lipid_rtp_identity_issues(name, force_field) == ()
    with policy.rebuilding_library_entry(retry_failed_validation=True):
        assert charmm_lipid_capability(name, force_field) == (True, "")

    # Reintroduce the original, wrong proton location. Equal formula and net
    # charge must never allow a different molecular graph through admission.
    if name in {"POP2", "SAPI"}:
        monkeypatch.setitem(
            policy._CHARMM_RTP_IDENTITIES, name, {"POP2": "POPI25", "SAPI": "SAPI25"}[name]
        )
    else:
        monkeypatch.setitem(
            policy._CHARMM_RTP_TAIL_COMBINATIONS,
            name,
            {"PAPI": ("POPI25", None, "SAPI25"), "SOP2": ("POPI2D", "SOPC", None)}[name],
        )
    assert not lipid_has_rtp(name, force_field)
    assert any(
        "bond graph differs" in issue for issue in lipid_rtp_identity_issues(name, force_field)
    )
    supported, reason = charmm_lipid_capability(name, force_field)
    assert not supported and "no exact" in reason


def test_current_lipid_stream_is_compatible_with_classic_charmm_protein():
    assert lipid_has_rtp("PUPC", "charmm36m")
    assert lipid_has_rtp("PUPC", "charmm36")


@pytest.mark.parametrize("lipid_name", ["POP3", "TOCL"])
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
        "DLIPA",
        "DLIPG",
        "DLIPS",
        "PAPC",
        "PAPE",
        "PAPG",
        "PIPI",
        "PMPC",
        "SMPC",
        "SOP3",
        "SOPI",
        "LPC16",
        "LPC18",
        "LPE16",
        "LYSPG",
        "DSM",
        "DOPGD",
        "DPPGD",
        "PPCPL",
        "PPEPL",
        "MGDG",
        "DGDG",
        "CAMP",
    ],
)
def test_modular_charmm36m_lipid_template_is_connected_and_exact(lipid_name):
    template_name, template = lipid_rtp_template(lipid_name, "charmm36m")
    assert template_name == lipid_name
    assert template is not None
    atom_names = {atom[0] for atom in template["atoms"]}
    assert len(atom_names) == len(template["atoms"])
    assert all(left in atom_names and right in atom_names for left, right in template["bonds"])
    assert lipid_rtp_identity_issues(lipid_name, "charmm36m") == ()


@pytest.mark.parametrize(
    "lipid_name",
    [
        "DLIPA",
        "DLIPC",
        "DLIPE",
        "DLIPG",
        "DLIPS",
        "ERG",
        "LPC16",
        "LPC18",
        "LPE16",
        "LYSPG",
        "PUPC",
        "SITO",
        "STIG",
    ],
)
def test_current_lipid_stream_is_exposed_to_classic_charmm(lipid_name):
    assert lipid_has_rtp(lipid_name, "charmm36")
    assert lipid_rtp_identity_issues(lipid_name, "charmm36") == ()


@pytest.mark.parametrize("lipid_name", ["DOPGD", "DPPGD"])
@pytest.mark.parametrize("force_field", ["charmm36", "charmm36m"])
def test_published_dag_template_is_available_in_both_charmm_releases(
    lipid_name,
    force_field,
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
    lipid_name,
    force_field,
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
    assert atoms["C1S"] == ("CTO2", 0.0)  # CERB
    assert atoms["O1X"] == ("OC301", -0.36)  # Glc-Cer
    assert atoms["O4X"] == ("OC301", -0.36)  # Gal(beta1-4)Glc
    assert atoms["O4Y"] == ("OC301", -0.36)  # GalNAc(beta1-4)Gal
    assert atoms["O3Z"] == ("OC301", -0.36)  # Gal(beta1-3)GalNAc
    assert atoms["C2A"] == ("CC3062", 0.28)  # Neu5Ac(alpha2-3)Gal
    assert ("O1X", "C1S") in template["bonds"]
    assert ("O4X", "C1Y") in template["bonds"]
    assert ("O4Y", "C1Z") in template["bonds"]
    assert ("O3Z", "C1Q") in template["bonds"]
    assert ("O3Y", "C2A") in template["bonds"]


def test_the_superseded_report_does_not_block_its_own_remedy():
    """A rebuild must not be refused for the entry it is replacing.

    `library_entry_superseded` withdraws a lipid whose pre-equilibrated entry
    was built from a structure since corrected. The process that rebuilds that
    entry asks for exactly the backend the report withdraws -- rebuilding a
    GAFF2 entry needs a GAFF2 backend for the lipid whose GAFF2 entry is
    superseded -- so without a scope the report blocks the only thing that
    clears it. That cost the V4 queue a 2.4 hour build before it was caught.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import (
        library_entry_superseded,
        rebuilding_library_entry,
    )

    with rebuilding_library_entry():
        assert library_entry_superseded("DAPC", "amber14sb", "gaff2") == ""
        assert library_entry_superseded("POPC", "amber14sb", "gaff2") == ""

    # and the scope is not sticky once it exits
    from gmxbuilder.modules.forcefield.lipid_policy import _REBUILDING

    assert _REBUILDING.get() is False


def test_the_library_builder_enters_the_rebuild_scope(tmp_path):
    """The scope has to be entered by the builder, not merely to exist.

    Asserted by running the builder rather than by reading its source: the
    same guarantee used to be checked by looking for a call in the text of
    ``build``, which said nothing about whether the call ran and broke the
    moment the lock it lives in was shared with ``extend``. Both entry points
    are checked here, because an entry rebuilt by continuing its run needs the
    scope for exactly the reason a rebuilt one does -- it asks for a backend
    for the lipid whose entry it is replacing.
    """
    from gmxbuilder.modules.forcefield.lipid_policy import _REBUILDING
    from gmxbuilder.modules.membrane.equilibrated_library import EquilibratedLipidLibrary
    from gmxbuilder.modules.membrane.lipid_equilibration import LipidEquilibrationBuilder

    builder = LipidEquilibrationBuilder.__new__(LipidEquilibrationBuilder)
    builder.library = EquilibratedLipidLibrary(roots=[tmp_path, tmp_path])

    seen: list[bool] = []
    builder._build_once = lambda *a, **k: seen.append(_REBUILDING.get())
    builder._extend_once = lambda *a, **k: seen.append(_REBUILDING.get())

    assert _REBUILDING.get() is False
    builder.build("POPC", "charmm36m")
    builder.extend("POPC", "charmm36m")
    assert seen == [True, True]
    # and neither entry point leaves the scope standing behind it
    assert _REBUILDING.get() is False


def test_a_replaced_entry_clears_the_superseded_report(tmp_path, monkeypatch):
    """The report promises to clear itself; keyed on the lipid alone it never did.

    A long-running web service asked once, cached the refusal for the process's
    lifetime, and went on refusing the lipid after its rebuilt entry had landed
    -- which is exactly the moment V4 replaces an entry. Keying the cache on the
    metadata file's identity as well makes publishing a replacement invalidate
    the answer by construction.
    """
    import json

    from gmxbuilder.modules.forcefield import lipid_policy
    from gmxbuilder.modules.membrane.lipids import LipidRegistry

    entry = tmp_path / "amber-gaff2" / "POPC"
    entry.mkdir(parents=True)
    metadata = entry / "metadata.json"

    def write(smiles, atoms):
        metadata.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "canonical_smiles": smiles,
                    "atom_names": [f"A{i}" for i in range(atoms)],
                }
            )
        )

    class OneEntryLibrary:
        def _candidate_dirs(self, *_args, **_kwargs):
            return [entry]

    from gmxbuilder.modules.membrane import equilibrated_library

    monkeypatch.setattr(equilibrated_library, "get_equilibrated_lipid_library", OneEntryLibrary)
    lipid_policy._SUPERSEDED_CACHE.clear()

    # A stale entry: a different molecule with a different atom count.
    write("CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCC", 60)
    stale = lipid_policy.library_entry_superseded("POPC", "amber14sb", "gaff2")
    assert stale, "an entry holding a different molecule must be refused"
    assert lipid_policy.library_entry_superseded("POPC", "amber14sb", "gaff2") == stale

    # The rebuild lands: same molecule as the registry now records.
    write(LipidRegistry.get("POPC").smiles, lipid_policy._explicit_atoms("POPC"))
    assert lipid_policy.library_entry_superseded("POPC", "amber14sb", "gaff2") == "", (
        "the replacement must clear the report without restarting the process"
    )
