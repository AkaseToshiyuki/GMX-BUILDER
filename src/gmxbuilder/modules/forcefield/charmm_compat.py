"""Offline, complete-template CHARMM ligand assignment.

Chemical identities are reviewed annotations, not atom-typing rules. All
charges and interaction parameters come from the selected installed CHARMM
release. New chemistry is rejected instead of borrowing GAFF charges or
inventing a CGenFF penalty model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.modules.forcefield.catalog import force_field_directory, get_force_field_profile
from gmxbuilder.modules.forcefield.cgenff_import import CGenFFTemplate
from gmxbuilder.modules.forcefield.charmm_templates import EXTRA_TEMPLATES
from gmxbuilder.modules.forcefield.rtp_parser import RTPParser
from gmxbuilder.runtime.hardware import find_gromacs_executable

BACKEND_VERSION = "3"


class CharmmCompatError(ModuleConfigError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"Local CHARMM [{code}]: {message}")


@dataclass(frozen=True)
class ChemicalTemplate:
    residue: str
    smiles: str
    heavy_names: tuple[str, ...]
    label: str


# Chemical identity and atom correspondence only; no force-field values.
# Atom order follows RDKit's reading of the explicit SMILES, never residue name.
TEMPLATES = (
    ChemicalTemplate("MEOH", "CO", ("CB", "OG"), "methanol"),
    ChemicalTemplate("ETOH", "CCO", ("C2", "C1", "O1"), "ethanol"),
    ChemicalTemplate("BENZ", "c1ccccc1", ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"), "benzene"),
    ChemicalTemplate("MAM1", "CN", ("C1", "N1"), "methylamine"),
    ChemicalTemplate("MAMM", "C[NH3+]", ("CE", "NZ"), "methylammonium"),
    ChemicalTemplate("EAMM", "CC[NH3+]", ("C1", "CE", "NZ"), "ethylammonium"),
    ChemicalTemplate("DMAM", "CNC", ("C1", "N1", "C2"), "dimethylamine"),
    ChemicalTemplate("MMAM", "C[NH2+]C", ("C1", "N", "C2"), "dimethylammonium"),
    ChemicalTemplate("ACEM", "CC(N)=O", ("CC", "C", "N", "O"), "acetamide"),
    ChemicalTemplate("PRPA", "CCC", ("C1", "C2", "C3"), "propane"),
    ChemicalTemplate("DMEE", "COC", ("C1", "O2", "C3"), "dimethyl ether"),
    ChemicalTemplate("DETE", "CCOCC", ("C1", "C2", "O3", "C4", "C5"), "diethyl ether"),
    ChemicalTemplate("PROH", "CCCO", ("C3", "C2", "C1", "O1"), "1-propanol"),
    ChemicalTemplate("METE", "CCOC", ("C1", "C2", "O3", "C4"), "ethyl methyl ether"),
    ChemicalTemplate(
        "EBEN", "CCc1ccccc1", ("CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"), "ethylbenzene"
    ),
    ChemicalTemplate(
        "TOLU", "Cc1ccccc1", ("CT", "CZ", "CE1", "CD1", "CG", "CD2", "CE2"), "toluene"
    ),
    ChemicalTemplate(
        "CUME",
        "CC(C)c1ccccc1",
        ("C3", "C4", "C5", "CA4", "CA3", "CA2", "CA1", "CA6", "CA5"),
        "cumene",
    ),
    ChemicalTemplate(
        "BZAM",
        "[NH3+]Cc1ccccc1",
        ("N8", "C7", "C5", "C4", "C3", "C2", "C1", "C6"),
        "benzylammonium",
    ),
) + tuple(ChemicalTemplate(*row) for row in EXTRA_TEMPLATES)


def inspect_smiles(smiles: str, *, allow_unknown: bool = False):
    """Require an explicit connected chemical state; never infer it from PDB."""
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 4096:
        raise CharmmCompatError(
            "INPUT_PARSE_ERROR", "provide a SMILES with explicit protonation state"
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise CharmmCompatError("INPUT_PARSE_ERROR", "SMILES could not be parsed")
    if molecule.GetNumAtoms() > 128 or len(Chem.GetMolFrags(molecule)) != 1:
        raise CharmmCompatError(
            "UNSUPPORTED_CHEMISTRY", "one connected molecule, at most 128 atoms"
        )
    if any(
        a.GetAtomicNum() not in {1, 6, 7, 8, 9, 15, 16}
        or a.GetIsotope()
        or a.GetNumRadicalElectrons()
        for a in molecule.GetAtoms()
    ):
        raise CharmmCompatError(
            "UNSUPPORTED_CHEMISTRY", "this release supports reviewed H/C/N/O/F/P/S templates"
        )
    if any(
        s.specified != Chem.StereoSpecified.Specified for s in Chem.FindPotentialStereo(molecule)
    ):
        raise CharmmCompatError(
            "STEREOCHEMISTRY_UNSPECIFIED",
            "specify every stereocenter in SMILES; coordinates will be checked against it",
        )
    maps = [a.GetAtomMapNum() for a in molecule.GetAtoms()]
    mapped = any(maps)
    if mapped and (
        sorted(maps) != list(range(1, molecule.GetNumAtoms() + 1))
        or any(a.GetAtomicNum() == 1 for a in molecule.GetAtoms())
    ):
        raise CharmmCompatError(
            "ATOM_MAPPING_AMBIGUOUS",
            "map every heavy atom exactly once with numbers 1..N in retained PDB heavy-atom order",
        )
    for atom in molecule.GetAtoms():
        if mapped:
            atom.SetIntProp("_InputCoordinateIndex", atom.GetAtomMapNum() - 1)
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    if not mapped:
        molecule = Chem.MolFromSmiles(canonical)
    for template in TEMPLATES:
        reference = Chem.MolFromSmiles(template.smiles)
        if Chem.MolToSmiles(reference, canonical=True) == canonical:
            return molecule, template
    if allow_unknown:
        return molecule, None
    raise CharmmCompatError(
        "CHARGE_MODEL_MISSING",
        "no complete reviewed template for this chemical state; use CGenFF / ParamChem import. "
        "General atom typing, charge increments and parameter analogy "
        "are not validated in this release",
    )


def template_database(force_field: str) -> Path:
    root = force_field_directory(force_field)
    if root is None:
        raise CharmmCompatError("DEPENDENCY_UNAVAILABLE", "force-field directory is missing")
    return root / ("merged.rtp" if force_field == "charmm36" else "cgenff.rtp")


def availability(force_field: str) -> tuple[bool, str]:
    from importlib.util import find_spec

    root = force_field_directory(force_field)
    if force_field not in {"charmm36", "charmm36m"} or root is None:
        return False, "the selected CHARMM force field is not installed"
    prefix = "merged" if force_field == "charmm36" else "cgenff"
    if not all(
        (root / name).is_file()
        for name in (f"{prefix}.rtp", f"{prefix}.hdb", "ffbonded.itp", "ffnonbonded.itp")
    ):
        return False, "CHARMM template/parameter files are missing"
    if not find_gromacs_executable():
        return False, "local GROMACS is required for template topology generation"
    if any(find_spec(name) is None for name in ("rdkit", "openmm")):
        return False, "local RDKit and OpenMM are required for identity and numerical checks"
    return True, "automatic chemical identification; optional MOL2/SMILES clarification"


def source_manifest(force_field: str) -> dict:
    enabled, reason = availability(force_field)
    if not enabled:
        raise CharmmCompatError("DEPENDENCY_UNAVAILABLE", reason)
    root = force_field_directory(force_field)
    profile = get_force_field_profile(force_field)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.iterdir())
        if p.is_file()
    }
    return {
        "family": "charmm",
        "force_field": force_field,
        "release": profile.release,
        "cgenff_version": profile.cgenff_version,
        "files_sha256": files,
        "parameter_source": "user-installed CHARMM force field",
        "data_redistribution": "not_granted_by_this_project",
        "backend_version": BACKEND_VERSION,
    }


def _map_heavy_atoms(
    structure: Structure,
    indices: list[int],
    template: ChemicalTemplate,
    rtp: dict,
    coordinate_molecule=None,
):
    """Map known chemistry to PDB coordinates using connectivity, not bond-order guesses."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    reference = Chem.MolFromSmiles(template.smiles)
    if len(indices) != reference.GetNumAtoms():
        raise CharmmCompatError(
            "ATOM_MAPPING_AMBIGUOUS", "SMILES heavy-atom count differs from retained PDB atoms"
        )
    names = [structure.atom_names[i].strip() for i in indices]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[A-Za-z0-9_]{1,5}", n) for n in names
    ):
        raise CharmmCompatError(
            "ATOM_MAPPING_AMBIGUOUS", "unique portable PDB atom names are required"
        )
    atoms = {a[0]: a for a in rtp["atoms"]}
    bonds = {frozenset(b[:2]) for b in rtp["bonds"]}
    if set(template.heavy_names) != {name for name in atoms if not name.startswith("H")}:
        raise CharmmCompatError(
            "FF_VERSION_MISMATCH", "installed RTP heavy atoms differ from reviewed identity"
        )
    for atom, name in zip(reference.GetAtoms(), template.heavy_names, strict=True):
        bonded_h = sum(name in bond and any(n.startswith("H") for n in bond) for bond in bonds)
        if bonded_h != atom.GetTotalNumHs():
            raise CharmmCompatError(
                "FF_VERSION_MISMATCH", "installed RTP hydrogen state differs from reviewed identity"
            )
    expected_edges = {
        frozenset(
            (template.heavy_names[b.GetBeginAtomIdx()], template.heavy_names[b.GetEndAtomIdx()])
        )
        for b in reference.GetBonds()
    }
    if expected_edges != {b for b in bonds if not any(n.startswith("H") for n in b)}:
        raise CharmmCompatError(
            "FF_VERSION_MISMATCH", "installed RTP connectivity differs from reviewed identity"
        )
    total_charge = sum(float(a[2]) for a in rtp["atoms"])
    if abs(total_charge - Chem.GetFormalCharge(reference)) > 1e-6:
        raise CharmmCompatError(
            "CHARGE_CONSERVATION_FAILED", "RTP charge and explicit chemical state differ"
        )

    observed = Chem.RWMol()
    conformer = Chem.Conformer(len(indices))
    for local, index in enumerate(indices):
        element = structure.elements[index].strip().title()
        if element not in {"C", "N", "O", "F", "P", "S"}:
            raise CharmmCompatError(
                "UNSUPPORTED_CHEMISTRY",
                "the retained PDB must contain supported organic heavy atoms",
            )
        observed.AddAtom(Chem.Atom(element))
        conformer.SetAtomPosition(local, tuple(structure.coordinates[index] * 10))
    observed.AddConformer(conformer)
    observed = observed.GetMol()
    rdDetermineBonds.DetermineConnectivity(observed, useHueckel=False, covFactor=1.25, useVdw=True)
    query = Chem.RWMol()
    for atom in reference.GetAtoms():
        query.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in reference.GetBonds():
        query.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    matches = observed.GetSubstructMatches(query.GetMol(), uniquify=False, maxMatches=1025)
    if not matches or len(matches) > 1024 or observed.GetNumBonds() != reference.GetNumBonds():
        raise CharmmCompatError(
            "ATOM_MAPPING_AMBIGUOUS",
            "PDB connectivity does not uniquely support the supplied SMILES",
        )
    if coordinate_molecule is not None and coordinate_molecule.GetAtomWithIdx(0).HasProp(
        "_InputCoordinateIndex"
    ):
        explicit_matches = set()
        for candidate in coordinate_molecule.GetSubstructMatches(
            reference, uniquify=False, useChirality=True, maxMatches=1025
        ):
            if all(
                coordinate_molecule.GetAtomWithIdx(i).GetTotalNumHs() == a.GetTotalNumHs()
                and coordinate_molecule.GetAtomWithIdx(i).GetFormalCharge() == a.GetFormalCharge()
                for i, a in zip(candidate, reference.GetAtoms(), strict=True)
            ):
                explicit_matches.add(
                    tuple(
                        coordinate_molecule.GetAtomWithIdx(i).GetIntProp("_InputCoordinateIndex")
                        for i in candidate
                    )
                )
        matches = [m for m in matches if m in explicit_matches]
        if not matches:
            raise CharmmCompatError(
                "ATOM_MAPPING_AMBIGUOUS", "explicit atom-map numbers disagree with PDB connectivity"
            )
    expected_stereo = dict(Chem.FindMolChiralCenters(reference, includeUnassigned=True))
    if expected_stereo:
        valid_matches = []
        for candidate in matches:
            spatial = Chem.Mol(reference)
            Chem.RemoveStereochemistry(spatial)
            coords = Chem.Conformer(reference.GetNumAtoms())
            coords.Set3D(True)
            for ref_index, local in enumerate(candidate):
                coords.SetAtomPosition(ref_index, tuple(structure.coordinates[indices[local]] * 10))
            spatial.AddConformer(coords)
            Chem.AssignAtomChiralTagsFromStructure(spatial, replaceExistingTags=True)
            Chem.AssignStereochemistry(spatial, cleanIt=True, force=True)
            actual = dict(Chem.FindMolChiralCenters(spatial, includeUnassigned=True))
            nonplanar = True
            for center in expected_stereo:
                neighbors = [a.GetIdx() for a in spatial.GetAtomWithIdx(center).GetNeighbors()]
                vectors = np.asarray(
                    [coords.GetAtomPosition(i) for i in neighbors[:3]]
                ) - np.asarray(coords.GetAtomPosition(center))
                lengths = np.linalg.norm(vectors, axis=1)
                if np.any(lengths < 1e-8) or abs(np.linalg.det(vectors / lengths[:, None])) < 1e-3:
                    nonplanar = False
            if actual == expected_stereo and nonplanar:
                valid_matches.append(candidate)
        if not valid_matches:
            raise CharmmCompatError(
                "STEREOCHEMISTRY_MISMATCH",
                "PDB coordinates disagree with the explicit SMILES stereochemistry "
                "or are planar at a stereocenter",
            )
        matches = valid_matches
    signatures = set()
    for match in matches:
        assigned = [None] * len(indices)
        for ref_index, observed_index in enumerate(match):
            name = template.heavy_names[ref_index]
            hydrogens = sorted(
                (atoms[n][1], atoms[n][2])
                for b in bonds
                if name in b
                for n in b
                if n.startswith("H")
            )
            assigned[observed_index] = (atoms[name][1], atoms[name][2], tuple(hydrogens))
        signatures.add(tuple(assigned))
    if len(signatures) != 1:
        raise CharmmCompatError(
            "ATOM_MAPPING_AMBIGUOUS",
            "equivalent PDB graph mappings assign different parameters; supply atom-mapped SMILES "
            "with 1..N denoting PDB heavy-atom order to identify protonation/bond-order sites",
        )
    match = min(matches, key=lambda m: tuple(names[i] for i in m))
    return {name: indices[match[i]] for i, name in enumerate(template.heavy_names)}


def _molecule_itp(text: str, ligand_name: str, name_map: dict[str, str]) -> str:
    """Keep the native molecule terms, renaming labels only; keep all Fourier/UB terms."""
    output = ["; Local CHARMM-compatible complete-template replay"]
    section = ""
    for raw in text.splitlines():
        code = raw.partition(";")[0].strip()
        if code.startswith("["):
            section = code.strip("[] ").lower()
            if section == "system":
                break
            output.append(code)
        elif code.startswith("#"):
            if section and not re.match(r'#(?:ifdef POSRES|include "posre.itp"|endif)', code):
                raise CharmmCompatError(
                    "UNSUPPORTED_BACKEND_FEATURE", "unexpected directive in generated molecule"
                )
        elif code and section:
            fields = code.split()
            if section == "moleculetype":
                fields[0] = ligand_name
            elif section == "atoms":
                fields[3] = ligand_name
                fields[4] = name_map[fields[4]]
            output.append(" ".join(fields))
    return "\n".join(output) + "\n"


def _energy_equivalence(source_top: Path, target_top: Path, coordinates: np.ndarray) -> dict:
    """Compare native and adapted topologies at identical coordinates; execute no MD."""
    import openmm as mm
    from openmm import app, unit

    results = []
    for path in (source_top, target_top):
        topology = app.GromacsTopFile(str(path), includeDir=str(path.parent))
        system = topology.createSystem(
            nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False
        )
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
        context.setPositions(coordinates * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        results.append(
            (
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                ),
            )
        )
        del context, integrator
    energy_error = abs(results[0][0] - results[1][0])
    force_error = float(np.max(np.abs(results[0][1] - results[1][1])))
    if not np.isfinite(energy_error + force_error) or energy_error > 1e-6 or force_error > 1e-5:
        raise CharmmCompatError(
            "NUMERICAL_VALIDATION_FAILED", "native/adapted energy or force differs"
        )
    return {
        "status": "passed",
        "energy_abs_error_kj_mol": energy_error,
        "force_max_abs_error_kj_mol_nm": force_error,
        "platform": "OpenMM Reference",
        "comparison": "native GROMACS topology versus adapted ITP; identical coordinates",
        "physical_validation": "not_evaluated",
        "dynamics_steps": 0,
    }


def prepare_local_molecule(
    name: str,
    structure: Structure,
    indices: list[int],
    smiles: str,
    force_field: str,
    output_dir: Path,
    *,
    allow_research: bool = False,
    environment_pH: float | None = None,
    chemical_identity_report: dict | None = None,
) -> CGenFFTemplate:
    """Generate a complete, numerically checked local template replay, or fail closed."""
    if not re.fullmatch(r"[A-Z0-9_]{1,8}", name):
        raise CharmmCompatError("INPUT_PARSE_ERROR", "invalid ligand residue name")
    if environment_pH is not None and (
        isinstance(environment_pH, bool)
        or not isinstance(environment_pH, (float, int))
        or not 1 <= environment_pH <= 13
    ):
        raise CharmmCompatError("INPUT_PARSE_ERROR", "environment pH must be between 1 and 13")
    molecule, template = inspect_smiles(smiles, allow_unknown=True)
    coordinate_molecule = molecule
    manifest = source_manifest(force_field)
    root = force_field_directory(force_field)
    parser = RTPParser(template_database(force_field))
    research = template is None
    assignment = {}
    if research:
        from rdkit import Chem

        from gmxbuilder.modules.forcefield.charmm_research import assign_research, check_domain

        # Coordinate atom maps must not change the research RTP/ITP atom order.
        # Keep the mapped object for placement, and type a canonical graph.
        molecule = Chem.MolFromSmiles(Chem.MolToSmiles(molecule, canonical=True))
        check_domain(molecule, force_field)
        if not allow_research:
            raise CharmmCompatError(
                "PHYSICAL_VALIDATION_REQUIRED",
                "no complete reference template; "
                "explicitly enable experimental assignment or use CGenFF import",
            )
        molecule, template, rtp, assignment = assign_research(molecule, parser, force_field)
    else:
        rtp = parser.get_residue(template.residue)
    if not rtp:
        raise CharmmCompatError(
            "FF_VERSION_MISMATCH", "reviewed template is absent from this force-field release"
        )
    mapping = _map_heavy_atoms(structure, indices, template, rtp, coordinate_molecule)
    # These Jul2022 HDB entries reference absent or inconsistently named H
    # atoms. Use explicit chemistry only after checking parent-H equivalence.
    explicit_hydrogens = not research and (
        force_field == "charmm36" or template.residue in {"ETAC", "ETSH"}
    )
    explicit_names = []
    if explicit_hydrogens:
        from gmxbuilder.modules.forcefield.charmm_research import reference_record

        molecule, _types, _charges, explicit_names = reference_record(template, parser)
        atoms = {a[0]: a for a in rtp["atoms"]}
        for parent in template.heavy_names:
            signatures = {
                tuple(atoms[n][1:3])
                for bond in rtp["bonds"]
                if parent in bond
                for n in bond[:2]
                if n.startswith("H")
            }
            if len(signatures) > 1:
                raise CharmmCompatError(
                    "ATOM_MAPPING_AMBIGUOUS",
                    "the selected native HDB cannot safely place this template: "
                    "template has inequivalent hydrogen parameters; use CGenFF import",
                )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Each request has its own staging directory; failed runs cannot leave an
    # old successful ITP beside a new diagnostic report.
    work = Path(tempfile.mkdtemp(prefix="assign-", dir=output_dir))
    try:
        if research or explicit_hydrogens:
            # Stage explicit RTP atom labels/hydrogens without native rename or
            # hydrogen-addition rules. The installed database stays unchanged.
            (work / "compat.ff").mkdir()
            for source in root.iterdir():
                if source.suffix in {".arn", ".hdb"}:
                    # All H atoms and labels already match the RTP. Do not let
                    # native renaming/addition reinterpret explicit input atoms.
                    continue
                (work / "compat.ff" / source.name).symlink_to(source.resolve())
        else:
            (work / "compat.ff").symlink_to(root.resolve(), target_is_directory=True)
        if research:
            rtp_text = "[ bondedtypes ]\n1 5 9 2 1 3 1 0\n[ LCMP ]\n[ atoms ]\n"
            rtp_text += "".join(f"{n} {t} {q:.12f} {g}\n" for n, t, q, g in rtp["atoms"])
            rtp_text += "[ bonds ]\n" + "".join(f"{a} {b}\n" for a, b in rtp["bonds"])
            (work / "compat.ff" / "local_compat.rtp").write_text(rtp_text)
        if research or explicit_hydrogens:
            from rdkit import Chem

            molecule = Chem.RemoveHs(molecule)
            conformer = Chem.Conformer(molecule.GetNumAtoms())
            for i, atom in enumerate(template.heavy_names):
                conformer.SetAtomPosition(i, tuple(structure.coordinates[mapping[atom]] * 10))
            molecule.AddConformer(conformer, assignId=True)
            molecule = Chem.AddHs(molecule, addCoords=True)
            input_names = explicit_names if explicit_hydrogens else [a[0] for a in rtp["atoms"]]
            input_coordinates = molecule.GetConformer().GetPositions() / 10
        else:
            input_names = list(template.heavy_names)
            input_coordinates = np.asarray([structure.coordinates[mapping[n]] for n in input_names])
        heavy = Structure(
            coordinates=input_coordinates,
            box_vectors=np.eye(3) * 6,
            atom_names=input_names,
            resnames=[template.residue] * len(input_names),
            resids=[1] * len(input_names),
        )
        GROWriter.write(heavy, work / "input.gro")
        gmx = find_gromacs_executable()
        result = subprocess.run(
            [
                gmx,
                "pdb2gmx",
                "-f",
                "input.gro",
                "-o",
                "native.gro",
                "-p",
                "native.top",
                "-ff",
                "compat",
                "-water",
                "none",
                "-noignh" if research or explicit_hydrogens else "-ignh",
            ],
            cwd=work,
            input="",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "GMXLIB": str(work), "OMP_NUM_THREADS": "1"},
        )
        (work / "pdb2gmx.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise CharmmCompatError(
                "PARAMETER_MISSING", "native CHARMM template/HDB generation failed"
            )
        # Any research assignment, not only the tied-charge-class one. The
        # analogy is keyed on specific ammonium-adjacent type names, so on
        # chemistry that does not carry them it matches nothing and the missing
        # term is still refused. Gating it on which model produced the types
        # therefore widened no assumption -- it only meant that a molecule
        # which moved between research models silently lost its bonded terms.
        if research:
            from gmxbuilder.modules.forcefield.charmm_bonded_research import resolve_missing

            shutil.copyfile(work / "native.top", work / "native-unresolved.top")
            assignment["bonded_parameter_analogies"] = resolve_missing(
                work / "native.top", root / "ffbonded.itp"
            )
            assignment["parameter_policy"] = (
                "native exact/wildcard first; missing terms may use "
                "recorded single-substitution alkyl/ammonium research analogy"
            )
        native = GROReader().read(work / "native.gro")
        if set(native.atom_names) != {a[0] for a in rtp["atoms"]}:
            raise CharmmCompatError(
                "UNSUPPORTED_BACKEND_FEATURE",
                "native topology changed the complete template atom set",
            )
        name_map = {n: structure.atom_names[i] for n, i in mapping.items()}
        used = set(name_map.values())
        counter = 1
        elements = []
        for index, atom in enumerate(native.atom_names):
            if atom in mapping:
                native.coordinates[index] = structure.coordinates[mapping[atom]]
                elements.append(structure.elements[mapping[atom]])
            else:
                if not atom.startswith("H"):
                    raise CharmmCompatError(
                        "UNSUPPORTED_BACKEND_FEATURE", "unexpected non-hydrogen particle"
                    )
                while f"HL{counter}" in used:
                    counter += 1
                name_map[atom] = f"HL{counter}"
                used.add(name_map[atom])
                elements.append("H")
        itp = _molecule_itp((work / "native.top").read_text(), name, name_map)
        (work / "ligand.itp").write_text(itp)
        (work / "atomtypes.itp").write_text(
            "; Types and NBFIX inherited unchanged from selected CHARMM force field\n"
        )
        (work / "adapted.top").write_text(
            '#include "compat.ff/forcefield.itp"\n#include "ligand.itp"\n'
            f"[ system ]\nLocal template\n[ molecules ]\n{name} 1\n"
        )
        try:
            numerical = _energy_equivalence(
                work / "native.top", work / "adapted.top", native.coordinates
            )
        except (ValueError, KeyError) as error:
            raise CharmmCompatError(
                "PARAMETER_MISSING", f"native parameter lookup failed: {error}"
            ) from error
        # GROMACS itself must resolve every interaction with zero allowed warnings.
        native.atom_names = [name_map[n] for n in native.atom_names]
        native.resnames = [name] * native.num_atoms
        GROWriter.write(native, work / "validated.gro")
        (work / "check.mdp").write_text(
            "integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n"
            "coulombtype = Cut-off\nrcoulomb = 1.0\nrvdw = 1.0\nrlist = 1.0\n"
        )
        checked = subprocess.run(
            [
                gmx,
                "grompp",
                "-f",
                "check.mdp",
                "-c",
                "validated.gro",
                "-p",
                "adapted.top",
                "-o",
                "checked.tpr",
                "-pp",
                "processed.top",
                "-maxwarn",
                "0",
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "GMXLIB": str(work), "OMP_NUM_THREADS": "1"},
        )
        (work / "grompp.log").write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            raise CharmmCompatError(
                "PARAMETER_MISSING", "GROMACS preprocessing rejected the assigned model"
            )
        report = {
            "schema_version": 1,
            "backend": "charmm_compat",
            "assignment_method": "template_replay",
            "template": template.residue,
            "chemical_identity": template.smiles,
            "input_smiles": smiles,
            "chemical_identification": chemical_identity_report,
            "environment_pH": environment_pH,
            "protonation_policy": (chemical_identity_report or {}).get(
                "protonation_policy",
                "explicit SMILES state; environment pH does not rewrite hydrogens or charges",
            ),
            "source_manifest": manifest,
            "atom_mapping": name_map,
            "parameter_completeness": "complete",
            "charge_method": "complete_template",
            "numerical_validation": numerical,
            "physical_validation": "not_evaluated",
            "export_eligibility": "eligible",
            "heavy_coordinates_preserved": True,
            "stereochemistry_validation": "explicit_SMILES_centers_checked_against_coordinates",
            "hydrogen_geometry": "RDKit explicit chemistry"
            if research or explicit_hydrogens
            else "native HDB",
            "native_atom_renaming": "disabled_for_explicit_RTP_names"
            if research or explicit_hydrogens
            else "native",
            "official_cgenff_output": False,
            "gromacs_preprocessing": "passed_maxwarn_0",
            **assignment,
        }
        from importlib.metadata import version

        from gmxbuilder import __version__

        input_record = {
            "smiles": smiles,
            "names": [structure.atom_names[i] for i in indices],
            "elements": [structure.elements[i] for i in indices],
            "coordinates_nm": structure.coordinates[indices].tolist(),
        }
        report["input_sha256"] = hashlib.sha256(
            json.dumps(input_record, sort_keys=True).encode()
        ).hexdigest()
        report["software"] = {name: version(name) for name in ("rdkit", "openmm", "numpy")}
        report["software"]["gmxbuilder"] = __version__
        report["gromacs"] = next(
            (line.strip() for line in result.stderr.splitlines() if "GROMACS -" in line),
            "version not reported",
        )
        if research:
            report.update(
                assignment_method="experimental_environment_bci", export_eligibility="research_only"
            )
        report["artifact_sha256"] = {
            p: hashlib.sha256((work / p).read_bytes()).hexdigest()
            for p in ("ligand.itp", "atomtypes.itp")
        }
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        # Native files are diagnostic only; the returned paths identify this
        # immutable successful assignment, never a partially overwritten cache.
        return CGenFFTemplate(
            name=name,
            atom_names=tuple(native.atom_names),
            elements=tuple(elements),
            coordinates=native.coordinates,
            net_charge=round(sum(a[2] for a in rtp["atoms"])),
            itp_path=work / "ligand.itp",
            atomtypes_path=work / "atomtypes.itp",
            cgenff_version=manifest["cgenff_version"],
            maximum_penalty=None,
        )
    except Exception as error:
        (output_dir / f"{work.name}-failure.json").write_text(
            json.dumps(
                {
                    "backend": "charmm_compat",
                    "export_eligibility": "blocked",
                    "error_code": getattr(error, "code", "ASSIGNMENT_FAILED"),
                    "message": str(error),
                    "diagnostic_logs": {p.name: p.read_text()[-8000:] for p in work.glob("*.log")},
                },
                indent=2,
            )
            + "\n"
        )
        shutil.rmtree(work, ignore_errors=True)
        raise


def validate_artifacts(itp_path: Path, force_field: str) -> dict:
    """Bind cached ligand parameters to the exact selected force field and report."""
    report_path = itp_path.parent / "report.json"
    if not report_path.is_file():
        raise CharmmCompatError("FF_VERSION_MISMATCH", "local assignment report is missing")
    report = json.loads(report_path.read_text())
    if report.get("source_manifest", {}).get("force_field") != force_field:
        raise CharmmCompatError(
            "FF_VERSION_MISMATCH", "assignment belongs to a different force field"
        )
    if report.get("source_manifest") != source_manifest(force_field):
        raise CharmmCompatError(
            "FF_VERSION_MISMATCH", "force field changed; run the force-field Check again"
        )
    for name in ("ligand.itp", "atomtypes.itp"):
        if hashlib.sha256((itp_path.parent / name).read_bytes()).hexdigest() != report.get(
            "artifact_sha256", {}
        ).get(name):
            raise CharmmCompatError(
                "NUMERICAL_VALIDATION_FAILED", "assigned parameter files changed since validation"
            )
    return report
