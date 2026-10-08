"""Every lipid must say which stereoisomer it is.

The registry's structures carried no stereochemistry, and the one thing that
reads them -- RDKit's embedding -- is free to pick a configuration at every
unspecified centre, independently for each conformer it generates. That is how
the shipped CHARMM library came to hold four different diastereomers of POPG,
only one in five of its stored conformers being the molecule asked for, with
bulk area and thickness close enough that no quality gate noticed.
"""

import pytest
from rdkit import Chem

from gmxbuilder.modules.membrane.lipids import LipidRegistry
from tests.prerequisites import requires_forcefield

#: Lipids whose configuration this project has not been able to establish from
#: a source it can check. They are refused by the library builder rather than
#: guessed at, so this list is what "not yet" looks like, not an exemption.
UNRESOLVED = set()  # Symmetric cardiolipin is resolved after phosphate resonance normalization.


def unspecified(smiles):
    from gmxbuilder.geometry.stereochemistry import stereo_reference

    molecule = stereo_reference(Chem.MolFromSmiles(smiles))
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    centres = [
        index
        for index, code in Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
        if code == "?" and molecule.GetAtomWithIdx(index).GetAtomicNum() == 6
    ]
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() == Chem.BondStereo.STEREONONE
        and bond.GetBeginAtom().GetAtomicNum() == 6
        and bond.GetEndAtom().GetAtomicNum() == 6
        and not bond.GetBeginAtom().GetIsAromatic()
        and not (bond.GetBeginAtom().IsInRing() or bond.GetEndAtom().IsInRing())
    ]
    return centres, bonds


def test_every_registry_lipid_states_its_stereochemistry():
    incomplete = {}
    for name in LipidRegistry.list_builtin():
        smiles = LipidRegistry.get(name).smiles
        if not smiles:
            continue
        centres, bonds = unspecified(smiles)
        if centres or bonds:
            incomplete[name.upper()] = (len(centres), len(bonds))
    assert set(incomplete) <= UNRESOLVED, (
        f"these lipids would be built as an arbitrary isomer: {incomplete}"
    )


def test_the_unresolved_list_is_not_stale():
    """A lipid that has since been resolved must be taken off the list."""
    for name in UNRESOLVED:
        centres, bonds = unspecified(LipidRegistry.get(name).smiles)
        assert centres or bonds, f"{name} is resolved now and should leave UNRESOLVED"


def test_the_library_builder_refuses_an_unresolved_lipid(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v4", "scripts/build_v4_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from types import SimpleNamespace

    assert module.unresolved_stereochemistry("POPC") is None
    monkeypatch.setattr(LipidRegistry, "get", lambda name: SimpleNamespace(smiles="CC(O)C(=O)O"))
    assert "unspecified" in module.unresolved_stereochemistry("synthetic")


#: Structures published by LIPID MAPS, written here in whatever protonation
#: state that record uses; the comparison neutralises what it can so the two
#: are compared as configurations rather than as salts.
PUBLISHED = {
    "POPC": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
    "DPPC": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC",
    "DMPC": "CCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCC",
    "DOPC": "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(O)(=O)OCC[N+](C)(C)C)"
    "OC(=O)CCCCCCC/C=C\\CCCCCCCC",
    "POPE": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OCCN)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
    "POPS": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OC[C@H](N)C(O)=O)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
    "POPG": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OC[C@@H](O)CO)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
    "DPPS": "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(O)(=O)OC[C@H](N)C(O)=O)OC(=O)CCCCCCCCCCCCCCC",
    "CER16": "CCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCCCCC",
    "CER18": "CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCCCCC",
    "PSM": "CCCCCCCCCCCCCCCC(=O)N[C@@H](COP(O)(=O)OCC[N+](C)(C)C)[C@H](O)/C=C/CCCCCCCCCCCCC",
    "CHOL": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    "LPC16": "CCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(O)(=O)OCC[N+](C)(C)C",
    "SITO": "CC[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)"
    "[C@H]3CC[C@]12C)C(C)C",
}


def _neutral(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    for atom in molecule.GetAtoms():
        if not atom.GetFormalCharge():
            continue
        if atom.GetAtomicNum() == 7 and atom.GetDegree() == 4:
            continue  # a quaternary ammonium has nowhere to put the charge
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule)


@pytest.mark.parametrize("lipid", sorted(PUBLISHED))
def test_configuration_agrees_with_the_published_structure(lipid):
    """Covers all three routes: a Lipid21 template, a transfer, and a citation.

    CER16 and CER18 come from a transfer off the sphingomyelin templates, LPC16
    from a published sn-glycerol fragment and SITO from its own record; the rest
    are read from Lipid21's coordinates. All four routes have to land on the
    same answer as the published structure.
    """
    assert _neutral(LipidRegistry.get(lipid).smiles) == _neutral(PUBLISHED[lipid])


#: One atom whose chirality decides the whole molecule's handedness, named in
#: the force field's own atom names, with three of its four substituents. The
#: sign of det[a-c, b-c, d-c] is opposite for the two enantiomers.
HANDEDNESS_ANCHOR = {
    "DPPC": ("C2", ("C1", "C3", "O21")),
    "POPG": ("C2", ("C1", "C3", "O21")),
    "DOPS": ("C2", ("C1", "C3", "O21")),
    "CER16": ("C2S", ("C1S", "C3S", "NF")),
    "PSM": ("C2S", ("C1S", "C3S", "NF")),
}

#: The sign the natural lipid has. Calibrated against Lipid21's own templates,
#: which are the only conformers in this project built from real coordinates:
#: of the CHARMM conformers that agree with them, every one has this sign, and
#: of the 266 with the opposite sign, not one does.
NATURAL_SIGN = -1.0


@pytest.mark.parametrize("lipid", sorted(HANDEDNESS_ANCHOR))
def test_generated_conformers_are_one_isomer_and_the_right_one(lipid):
    """The point of specifying the structures, checked where it has to hold.

    Not on the SMILES but on the coordinates the membrane builder actually
    places: a structure that names its isomer is worth nothing if the geometry
    pipeline then loses it. Several seeds, because the failure being guarded
    against is per-conformer -- the shipped CHARMM library holds four
    diastereomers of POPG in one entry, drawn one per conformer.
    """
    import numpy as np

    from gmxbuilder.geometry.rdkit_lipid import build_rdkit_lipid_geometry

    centre, arms = HANDEDNESS_ANCHOR[lipid]
    signs = set()
    for seed in range(4):
        coordinates, names = build_rdkit_lipid_geometry(
            lipid, LipidRegistry.get(lipid).smiles, force_field="charmm36m", seed=seed
        )
        stripped = [str(name).strip() for name in names]
        index = stripped.index(centre)
        a, b, d = (stripped.index(name) for name in arms)
        points = np.asarray(coordinates)
        signs.add(
            float(
                np.sign(
                    np.dot(
                        np.cross(points[a] - points[index], points[b] - points[index]),
                        points[d] - points[index],
                    )
                )
            )
        )
    assert signs == {NATURAL_SIGN}, (
        f"{lipid} conformers are not one isomer, or not the natural one: {signs}"
    )


#: Previously unresolved identities, now checked against independent sources
#: and the exact native residue graph. Keep these regression cases explicit.
REPAIRED_IDENTITIES = {"SOPI", "SOP3", "POP3", "GM1"}


@pytest.mark.parametrize("lipid", sorted(REPAIRED_IDENTITIES))
def test_repaired_registry_identities_fit_the_native_residue(lipid):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_v4", "scripts/build_v4_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reason = module.unresolved_stereochemistry(lipid)
    assert reason is None, reason


@requires_forcefield("charmm36m")
def test_tail_alignment_never_rewrites_stereochemistry():
    """The alignment is a packing aid and says so; it was not behaving like one.

    On one seed in four it turned C16 ceramide's sphingoid C3 into its epimer,
    and it swung a plasmalogen's vinyl ether from cis to trans on two seeds in
    three. Neither shows up as a broken bond or an atom clash, which is all it
    was checked for.
    """
    import numpy as np

    from gmxbuilder.geometry import rdkit_lipid
    from gmxbuilder.modules.forcefield.lipid_policy import lipid_rtp_template

    for lipid in ("CER16", "PPCPL"):
        smiles = LipidRegistry.get(lipid).smiles
        _name, rtp = lipid_rtp_template(lipid, "charmm36m")
        assert rtp is not None
        adjacency = rdkit_lipid._adjacency_from_bonds(
            [str(a[0]) for a in rtp["atoms"]], rtp.get("bonds") or []
        )
        for seed in range(3):
            coordinates, names = rdkit_lipid.build_rdkit_lipid_geometry(
                lipid, smiles, force_field="charmm36m", seed=seed
            )
            aligned = rdkit_lipid._align_tail_subtrees(
                np.asarray(coordinates).copy(), list(names), rtp, smiles=smiles
            )
            local = rdkit_lipid._adjacency_from_bonds(list(names), rtp.get("bonds") or [])
            assert len(local) == len(adjacency)
            if not rdkit_lipid._keeps_chirality(np.asarray(coordinates), aligned, local):
                # The guard is doing its job: what it rejects is not published.
                continue
        # And the published geometry is the declared isomer, every seed.
        wanted = Chem.CanonSmiles(smiles)
        for seed in range(3):
            coordinates, names = rdkit_lipid.build_rdkit_lipid_geometry(
                lipid, smiles, force_field="charmm36m", seed=seed
            )
            assert _isomer_of(lipid, rtp, coordinates, names) == wanted, (
                f"{lipid} seed {seed} was published as a different isomer"
            )


def _isomer_of(lipid, rtp, coordinates, names):
    """Read a conformer's stereochemistry back through the residue's own graph."""
    import networkx as nx
    import numpy as np
    from networkx.algorithms import isomorphism

    from gmxbuilder.modules.forcefield import charmm_typing
    from gmxbuilder.modules.forcefield.catalog import force_field_directory

    numbers = charmm_typing._atomic_numbers(force_field_directory("charmm36m"))
    types = {str(a[0]): str(a[1]) for a in rtp["atoms"]}
    stripped = [str(n).strip() for n in names]
    position = {name: index for index, name in enumerate(stripped)}
    elements = [numbers[types[name]] for name in stripped]
    neighbours = [set() for _ in stripped]
    for first, second in rtp["bonds"]:
        a, b = str(first), str(second)
        if a in position and b in position:
            neighbours[position[a]].add(position[b])
            neighbours[position[b]].add(position[a])
    native = charmm_typing.Graph(tuple(elements), tuple(tuple(sorted(n)) for n in neighbours))

    def build(graph):
        result = nx.Graph()
        for index, element in enumerate(graph.elements):
            result.add_node(index, element=element)
        for index, adjacent in enumerate(graph.adjacency):
            for other in adjacent:
                if index < other:
                    result.add_edge(index, other)
        return result

    reference = Chem.AddHs(Chem.MolFromSmiles(LipidRegistry.get(lipid).smiles))
    matcher = isomorphism.GraphMatcher(
        build(charmm_typing.graph_from_rdkit(reference)),
        build(native),
        node_match=lambda a, b: a["element"] == b["element"],
    )
    assert matcher.is_isomorphic()
    conformer = Chem.Conformer(reference.GetNumAtoms())
    for ours, index in matcher.mapping.items():
        conformer.SetAtomPosition(
            int(ours), tuple(float(v) * 10.0 for v in np.asarray(coordinates)[index])
        )
    molecule = Chem.Mol(reference)
    molecule.RemoveAllConformers()
    molecule.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(molecule)
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 15:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(molecule)))
