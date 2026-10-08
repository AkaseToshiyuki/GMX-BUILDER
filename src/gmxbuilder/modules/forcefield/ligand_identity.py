"""Recover ligand chemistry without moving the retained heavy atoms.

Identity and parameter assignment are separate operations. CCD/embedded bond
tables are chemical evidence, not force-field templates. Coordinate perception
is a model, with explicit ambiguity gates; its partial charges are never used
as CHARMM charges. Explicit user states always take precedence over pH models.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from gmxbuilder.core.exceptions import ModuleConfigError

IDENTITY_VERSION = "1"
MAX_IDENTITY_BYTES = 2 * 1024 * 1024
MAX_HEAVY_ATOMS = 128
LOG = logging.getLogger(__name__)


class LigandIdentityError(ModuleConfigError):
    def __init__(self, reason: str):
        super().__init__(
            f"Ligand chemistry needs clarification: {reason}. "
            "Provide a MOL2 with correct bonds/protonation or an explicit SMILES."
        )


def _chem():
    from rdkit import Chem

    return Chem


def _canonical(molecule):
    Chem = _chem()
    molecule = Chem.RemoveHs(Chem.Mol(molecule))
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _validate(molecule):
    Chem = _chem()
    if molecule is None:
        raise LigandIdentityError("the chemical structure could not be parsed")
    Chem.SanitizeMol(molecule)
    if len(Chem.GetMolFrags(molecule)) != 1 or not 1 < molecule.GetNumHeavyAtoms() <= 128:
        raise LigandIdentityError("expected one connected small molecule with 2–128 heavy atoms")
    if any(a.GetIsotope() or a.GetNumRadicalElectrons() for a in molecule.GetAtoms()):
        raise LigandIdentityError("isotopes or radical states need explicit specialist preparation")
    if any(
        s.specified != Chem.StereoSpecified.Specified for s in Chem.FindPotentialStereo(molecule)
    ):
        raise LigandIdentityError("stereochemistry is not uniquely defined")
    return molecule


def coordinate_graph(structure, indices):
    Chem = _chem()
    from rdkit.Chem import rdDetermineBonds

    indices = [int(i) for i in indices if structure.elements[int(i)].upper() != "H"]
    if not 1 < len(indices) <= MAX_HEAVY_ATOMS:
        raise LigandIdentityError("expected 2–128 retained heavy atoms")
    molecule = Chem.RWMol()
    conformer = Chem.Conformer(len(indices))
    conformer.Set3D(True)
    for local, index in enumerate(indices):
        element = structure.elements[index].strip().title()
        if element not in {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"}:
            raise LigandIdentityError(f"unsupported element {element}")
        molecule.AddAtom(Chem.Atom(element))
        conformer.SetAtomPosition(local, tuple(structure.coordinates[index] * 10))
    molecule.AddConformer(conformer)
    molecule = molecule.GetMol()
    if not np.isfinite(molecule.GetConformer().GetPositions()).all():
        raise LigandIdentityError("non-finite coordinates")
    rdDetermineBonds.DetermineConnectivity(molecule, covFactor=1.25, useVdw=True)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise LigandIdentityError("retained atoms form disconnected fragments")
    distances = np.linalg.norm(
        molecule.GetConformer().GetPositions()[:, None]
        - molecule.GetConformer().GetPositions()[None, :],
        axis=2,
    )
    if np.any(distances[np.triu_indices(len(indices), 1)] < 0.65):
        raise LigandIdentityError("overlapping heavy atoms")
    return molecule, indices


def _graph(molecule):
    Chem = _chem()
    result = Chem.RWMol()
    for atom in molecule.GetAtoms():
        result.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in molecule.GetBonds():
        result.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return result.GetMol()


def _mapped_state(molecule, structure, indices, atom_names=None, *, ordered=False):
    """Return chemistry in retained atom order; reject inequivalent graph maps."""
    Chem = _chem()
    molecule = Chem.RemoveHs(molecule)
    observed, indices = coordinate_graph(structure, indices)
    if (molecule.GetNumAtoms(), molecule.GetNumBonds()) != (
        observed.GetNumAtoms(),
        observed.GetNumBonds(),
    ):
        raise LigandIdentityError("chemical definition and retained atoms/bonds differ")
    matches = observed.GetSubstructMatches(_graph(molecule), uniquify=False, maxMatches=1025)
    if len(matches) > 1024:
        raise LigandIdentityError("too many possible atom mappings")
    names = [structure.atom_names[i].strip() for i in indices]
    if ordered:
        matches = [m for m in matches if m == tuple(range(len(indices)))]
    elif atom_names and len(set(atom_names)) == len(names) and set(atom_names) == set(names):
        match = tuple(names.index(n) for n in atom_names)
        matches = [m for m in matches if m == match]
    if not matches:
        raise LigandIdentityError("chemical definition does not match retained connectivity")
    states = {}
    for match in matches:
        candidate = Chem.Mol(molecule)
        candidate.RemoveAllConformers()
        conf = Chem.Conformer(len(indices))
        conf.Set3D(True)
        for ref, local in enumerate(match):
            conf.SetAtomPosition(ref, observed.GetConformer().GetAtomPosition(local))
        candidate.AddConformer(conf)
        before = dict(Chem.FindMolChiralCenters(candidate, includeUnassigned=False))
        before.update(
            {
                a.GetIdx(): a.GetProp("_CCD_CIP")
                for a in candidate.GetAtoms()
                if a.HasProp("_CCD_CIP")
            }
        )
        spatial = Chem.Mol(candidate)
        Chem.RemoveStereochemistry(spatial)
        Chem.AssignStereochemistryFrom3D(spatial)
        after = dict(Chem.FindMolChiralCenters(spatial, includeUnassigned=False))
        if any(after.get(i) != tag for i, tag in before.items()):
            continue
        if any(
            b.GetStereo() in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}
            and spatial.GetBondWithIdx(b.GetIdx()).GetStereo() != b.GetStereo()
            for b in candidate.GetBonds()
        ):
            continue
        planar = False
        for center in after:
            neighbors = [a.GetIdx() for a in spatial.GetAtomWithIdx(center).GetNeighbors()]
            vectors = conf.GetPositions()[neighbors[:3]] - conf.GetPositions()[center]
            lengths = np.linalg.norm(vectors, axis=1)
            if (
                len(vectors) != 3
                or np.any(lengths < 1e-8)
                or abs(np.linalg.det(vectors / lengths[:, None])) < 1e-3
            ):
                planar = True
        if planar:
            continue
        # For undefined centers, the actual uploaded geometry supplies stereo.
        candidate = spatial
        try:
            _validate(candidate)
        except LigandIdentityError:
            continue
        for ref, local in enumerate(match):
            candidate.GetAtomWithIdx(ref).SetAtomMapNum(local + 1)
        # The mapped chemical graph distinguishes protonation/tautomer sites,
        # while allowing true chemical automorphisms (e.g. benzene).
        key = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
        states[key] = candidate
    if len(states) != 1:
        raise LigandIdentityError(
            "atom mapping, protonation sites or stereochemistry are ambiguous; "
            "use MOL2 with matching atom names or map SMILES atoms 1..N "
            "to retained heavy-atom order"
        )
    key, molecule = next(iter(states.items()))
    return molecule, key


def read_mol2(payload: bytes):
    Chem = _chem()
    if len(payload) > MAX_IDENTITY_BYTES:
        raise LigandIdentityError("MOL2 exceeds the 2 MiB limit")
    text = payload.decode("utf-8")
    if text.upper().count("@<TRIPOS>MOLECULE") != 1:
        raise LigandIdentityError("MOL2 must contain exactly one molecule")
    molecule = Chem.MolFromMol2Block(
        text, sanitize=True, removeHs=False, cleanupSubstructures=False
    )
    if molecule is None:
        raise LigandIdentityError("MOL2 chemical structure could not be parsed")
    names = [a.GetProp("_TriposAtomName") for a in molecule.GetAtoms() if a.GetAtomicNum() != 1]
    if len(set(names)) != len(names):
        raise LigandIdentityError("MOL2 heavy atom names must be unique")
    if not any(a.GetAtomicNum() == 1 for a in molecule.GetAtoms()) and any(
        a.GetNumRadicalElectrons() for a in molecule.GetAtoms()
    ):
        # RDKit's MOL2 reader needs explicit H for formal-charge estimation.
        # Honor the supplied Sybyl types/bonds; do not apply a pH override.
        molecule = _convert(text, "mol2", None)
    _validate(molecule)
    return molecule, names


def _cif_rows(raw, prefix):
    from gmxbuilder.io.cif import CIFParser

    fields, values = CIFParser._extract_loop(raw, prefix)
    if not fields:
        return []
    if len(values) % len(fields):
        raise LigandIdentityError("incomplete chemical-component table")
    names = [f.split(".", 1)[1] for f in fields]
    return [dict(zip(names, values[i : i + len(names)])) for i in range(0, len(values), len(names))]


def molecule_from_cif(raw: str, name: str):
    """Read atom formal charges and bond orders, never CCD ideal coordinates."""
    Chem = _chem()
    atoms = [r for r in _cif_rows(raw, "_chem_comp_atom.") if r.get("comp_id") == name]
    bonds = [r for r in _cif_rows(raw, "_chem_comp_bond.") if r.get("comp_id") == name]
    if not atoms or not bonds:
        return None
    if len(atoms) > 512 or len(bonds) > 1024:
        raise LigandIdentityError("chemical-component definition is too large")
    molecule = Chem.RWMol()
    lookup = {}
    names = []
    for record in atoms:
        name = record["atom_id"]
        if name in lookup:
            raise LigandIdentityError("duplicate chemical-component atom name")
        atom = Chem.Atom(record["type_symbol"].title())
        charge = record.get("charge", "?")
        if charge in {"?", "."}:
            raise LigandIdentityError("chemical-component formal charges are missing")
        atom.SetFormalCharge(int(charge))
        if record.get("pdbx_stereo_config") in {"R", "S"}:
            atom.SetProp("_CCD_CIP", record["pdbx_stereo_config"])
        lookup[name] = molecule.AddAtom(atom)
        if atom.GetAtomicNum() != 1:
            names.append(name)
    types = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "AROM": Chem.BondType.AROMATIC,
    }
    for record in bonds:
        kind = record.get("value_order", "").upper()
        if kind not in types:
            raise LigandIdentityError("chemical-component bond order is undefined")
        molecule.AddBond(lookup[record["atom_id_1"]], lookup[record["atom_id_2"]], types[kind])
    result = molecule.GetMol()
    Chem.SanitizeMol(result)
    return result, names


def _ccd(name: str):
    """Fetch a public dictionary entry by ID only; no coordinates are submitted."""
    if not re.fullmatch(r"[A-Z0-9]{1,5}", name) or name in {"LIG", "UNK", "UNL", "UNX"}:
        return None
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    root = Path(os.environ.get("GMXBUILDER_CCD_CACHE") or cache_home / "gmxbuilder/ccd")
    cached = root / f"{name}.cif"
    # Cache successful definitions on disk. A temporary network failure must
    # not be memoized for the lifetime of the service.
    try:
        if cached.is_file():
            with cached.open("rb") as handle:
                payload = handle.read(MAX_IDENTITY_BYTES + 1)
            if len(payload) <= MAX_IDENTITY_BYTES:
                raw = payload.decode("utf-8")
                if molecule_from_cif(raw, name) is not None:
                    return raw
    except (OSError, ValueError, KeyError, ModuleConfigError):
        LOG.debug("CCD cache could not supply a valid definition for %s", name, exc_info=True)
    url = f"https://files.rcsb.org/ligands/download/{name}.cif"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            payload = response.read(MAX_IDENTITY_BYTES + 1)
        if len(payload) > MAX_IDENTITY_BYTES:
            return None
        raw = payload.decode("utf-8")
        if molecule_from_cif(raw, name) is None:
            return None
    except (OSError, ValueError, KeyError, ModuleConfigError):
        LOG.debug("CCD lookup failed for %s", name, exc_info=True)
        return None
    # Persistence is optional: a validated definition remains usable when the
    # cache is read-only, full, or denied by the managed write sandbox.
    temporary = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(cached)
    except OSError:
        LOG.warning("CCD definition %s validated but could not be cached", name)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                LOG.warning("Could not remove incomplete CCD cache file for %s", name)
    return raw


def _obabel():
    from gmxbuilder.modules.forcefield.gaff_backend import (
        _gaff_tool_environment,
        gaff_environment_path,
    )

    installed = gaff_environment_path() / "bin" / "obabel"
    if installed.is_file():
        return str(installed), _gaff_tool_environment()
    executable = shutil.which("obabel")
    if executable:
        return executable, os.environ.copy()
    raise LigandIdentityError("Open Babel is unavailable for automatic bond/protonation perception")


def _convert(payload: str, input_format: str, pH: float | None):
    Chem = _chem()
    executable, env = _obabel()
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-identity-") as work:
        source = Path(work) / f"input.{input_format}"
        source.write_text(payload)
        try:
            result = subprocess.run(
                [
                    executable,
                    f"-i{input_format}",
                    str(source),
                    "-osdf",
                    *(["-p", f"{pH:.3f}"] if pH is not None else ["-h"]),
                ],
                capture_output=True,
                timeout=20,
                env=env,
                cwd=work,
            )
        except subprocess.TimeoutExpired as exc:
            raise LigandIdentityError("chemical perception timed out") from exc
    if result.returncode or len(result.stdout) > MAX_IDENTITY_BYTES:
        raise LigandIdentityError("automatic bond/protonation perception failed")
    if any(word in result.stderr.lower() for word in (b"warning", b"error", b"failed")):
        raise LigandIdentityError("Open Babel reported uncertain chemical perception")
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(result.stdout), removeHs=False))
    if len(molecules) != 1 or molecules[0] is None:
        raise LigandIdentityError("automatic perception did not produce one valid molecule")
    return molecules[0]


def _at_ph(molecule, pH):
    """Select a model state only away from detected protonation transitions."""
    Chem = _chem()
    payload = Chem.MolToMolBlock(molecule)
    states = [
        _convert(payload, "mol", value)
        for value in sorted({max(1.0, pH - 0.5), pH, min(13.0, pH + 0.5)})
    ]
    if len({_canonical(m) for m in states}) != 1:
        raise LigandIdentityError("multiple protonation states occur within pH ±0.5 in the model")
    result = Chem.RemoveHs(states[0])
    base = Chem.RemoveHs(molecule)
    if [a.GetAtomicNum() for a in result.GetAtoms()] != [a.GetAtomicNum() for a in base.GetAtoms()]:
        raise LigandIdentityError("protonation changed heavy-atom identity or order")
    if not np.allclose(
        result.GetConformer().GetPositions(), base.GetConformer().GetPositions(), atol=0.002, rtol=0
    ):
        raise LigandIdentityError("protonation changed heavy-atom coordinates")
    return _validate(result)


def _perceive(structure, indices):
    from gmxbuilder.core.structure import Structure
    from gmxbuilder.io.pdb import PDBWriter

    Chem = _chem()
    observed, indices = coordinate_graph(structure, indices)
    ligand = Structure(
        coordinates=structure.coordinates[indices],
        box_vectors=structure.box_vectors,
        atom_names=[structure.atom_names[i] for i in indices],
        resnames=["LIG"] * len(indices),
        resids=[1] * len(indices),
        elements=[structure.elements[i] for i in indices],
    )
    with tempfile.TemporaryDirectory(prefix="gmxbuilder-extract-") as work:
        path = Path(work) / "ligand.pdb"
        PDBWriter.write(ligand, path)
        molecule = _convert(path.read_text(), "pdb", None)
    heavy = Chem.RemoveHs(molecule)
    if heavy.GetNumAtoms() != observed.GetNumAtoms() or not np.allclose(
        heavy.GetConformer().GetPositions(),
        observed.GetConformer().GetPositions(),
        atol=0.002,
        rtol=0,
    ):
        raise LigandIdentityError("coordinate perception changed retained heavy atoms")
    _mapped_state(heavy, structure, indices, ordered=True)
    # Do not invent a tautomer from a coordinate-only bond-order guess.
    from rdkit.Chem.MolStandardize import rdMolStandardize

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(16)
    enumerator.SetMaxTransforms(64)
    tautomers = enumerator.Enumerate(heavy)
    if (
        len(tautomers) != 1
        or tautomers.status != rdMolStandardize.TautomerEnumeratorStatus.Completed
    ):
        raise LigandIdentityError(
            "coordinate-only input permits multiple tautomer/bond-order states"
        )
    return molecule


def _pdb_definition(path, name, structure, indices):
    """Keep an original, hydrogen-complete ligand's chemical evidence."""
    Chem = _chem()
    first = int(indices[0])
    target = (name, structure.chain_ids[first].strip(), structure.resids[first])
    atoms = {}
    connections = []
    first_model_ended = False
    with Path(path).open() as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ENDMDL":
                first_model_ended = True
            if record in {"ATOM", "HETATM"} and not first_model_ended:
                try:
                    key = (line[17:20].strip(), line[21:22].strip(), int(line[22:26]))
                except ValueError:
                    continue
                if key == target and line[16:17] in {" ", "A"}:
                    atoms[int(line[6:11])] = line
                    if len(atoms) > 512:
                        raise LigandIdentityError("source ligand exceeds the supported atom limit")
            elif record == "CONECT":
                try:
                    values = [
                        int(line[i : i + 5])
                        for i in range(6, len(line.rstrip()), 5)
                        if line[i : i + 5].strip()
                    ]
                except ValueError:
                    continue
                if values:
                    connections.append((values, line))
    if not atoms:
        return None
    selected = set(atoms)
    included_bonds = []
    for values, line in connections:
        if values[0] in selected:
            if any(v not in selected for v in values[1:]):
                raise LigandIdentityError(
                    "PDB records link this ligand to another residue or metal"
                )
            included_bonds.append(line)
        elif any(v in selected for v in values[1:]):
            raise LigandIdentityError("PDB records link this ligand to another residue or metal")
    block = "".join(atoms.values()) + "".join(included_bonds) + "END\n"
    original = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if original is None:
        raise LigandIdentityError("original ligand records could not be interpreted")
    hydrogen_count = sum(a.GetAtomicNum() == 1 for a in original.GetAtoms())
    if not hydrogen_count:
        return None
    molecule = _convert(block, "pdb", None)
    # Incomplete H records cannot settle the chemical state by themselves.
    if sum(a.GetAtomicNum() == 1 for a in molecule.GetAtoms()) != hydrogen_count:
        return None
    names = [
        a.GetPDBResidueInfo().GetName().strip()
        for a in original.GetAtoms()
        if a.GetAtomicNum() != 1
    ]
    molecule, _ = _mapped_state(molecule, structure, indices, names)
    return molecule, hashlib.sha256(block.encode()).hexdigest()


def resolve_identity(
    name, structure, indices, *, pH=7.0, smiles="", mol2_path=None, source_path=None, use_ccd=True
):
    """Resolve explicit overrides, embedded definitions, CCD, then coordinates."""
    Chem = _chem()
    if isinstance(pH, bool) or not isinstance(pH, (int, float)) or not 1 <= pH <= 13:
        raise LigandIdentityError("pH must be between 1 and 13")
    coordinate_graph(structure, indices)
    warnings = []
    source_hash = None
    if smiles and mol2_path:
        raise LigandIdentityError("choose either MOL2 or SMILES, not both")
    if smiles:
        if not isinstance(smiles, str) or len(smiles) > 4096:
            raise LigandIdentityError("SMILES must contain at most 4096 characters")
        molecule = _validate(Chem.MolFromSmiles(smiles))
        mapped = any(a.GetAtomMapNum() for a in molecule.GetAtoms())
        if mapped:
            # Existing numbered override convention: 1..N = retained heavy order.
            if sorted(a.GetAtomMapNum() for a in molecule.GetAtoms()) != list(
                range(1, molecule.GetNumAtoms() + 1)
            ) or any(a.GetAtomicNum() == 1 for a in molecule.GetAtoms()):
                raise LigandIdentityError("map every heavy atom exactly once as 1..N")
            order = sorted(
                range(molecule.GetNumAtoms()),
                key=lambda i: molecule.GetAtomWithIdx(i).GetAtomMapNum(),
            )
            molecule = Chem.RenumberAtoms(molecule, order)
            molecule, resolved = _mapped_state(molecule, structure, indices, ordered=True)
        else:
            molecule, resolved = _mapped_state(molecule, structure, indices)
        source = "user_smiles"
        policy = "explicit user state; pH does not rewrite it"
    elif mol2_path:
        with Path(mol2_path).open("rb") as handle:
            payload = handle.read(MAX_IDENTITY_BYTES + 1)
        molecule, names = read_mol2(payload)
        molecule, resolved = _mapped_state(molecule, structure, indices, names)
        source_hash = hashlib.sha256(payload).hexdigest()
        source, policy = "user_mol2", "explicit user state; pH does not rewrite it"
    else:
        definition = None
        if source_path and Path(source_path).is_file():
            # Original mmCIF bytes remain available even after conversion/filtering.
            from gmxbuilder.modules.input.pdb_input import PDBInputModule

            if PDBInputModule._is_cif_format(source_path):
                raw = Path(source_path).read_text()
                embedded = molecule_from_cif(raw, name)
                if embedded is not None:
                    molecule, names = embedded
                    molecule, _ = _mapped_state(molecule, structure, indices, names)
                    definition = molecule
                    source, source_hash = "input_mmcif", hashlib.sha256(raw.encode()).hexdigest()
            else:
                original = _pdb_definition(source_path, name, structure, list(indices))
                if original:
                    definition, source_hash = original
                    source = "input_pdb"
        if definition is None and use_ccd:
            raw = _ccd(name)
            if raw:
                try:
                    parsed = molecule_from_cif(raw, name)
                    if parsed:
                        molecule, names = parsed
                        molecule, _ = _mapped_state(molecule, structure, indices, names)
                        definition = molecule
                        source = "wwpdb_ccd"
                        source_hash = hashlib.sha256(raw.encode()).hexdigest()
                except (ValueError, KeyError, ModuleConfigError):
                    warnings.append("CCD ID did not match retained chemistry; used coordinates")
        if definition is None:
            molecule = _perceive(structure, indices)
            source = "coordinate_perception"
            warnings.append("Bond orders were inferred from coordinates by Open Babel")
        else:
            # Reorder a trusted definition into retained order before pH processing.
            molecule = definition
            order = sorted(
                range(molecule.GetNumAtoms()),
                key=lambda i: molecule.GetAtomWithIdx(i).GetAtomMapNum(),
            )
            molecule = Chem.RenumberAtoms(molecule, order)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecule = _at_ph(molecule, float(pH))
        molecule, resolved = _mapped_state(molecule, structure, indices, ordered=True)
        policy = (
            "Open Babel solution-pH model; stable over pH ±0.5; not a bound-state pKa prediction"
        )
        warnings.append(
            "Automatic protonation is a solution-pH model; explicit overrides are available"
        )
    return {
        "status": "ok",
        "smiles": _canonical(molecule),
        "mapped_smiles": resolved,
        "source": source,
        "source_sha256": source_hash,
        "pH": pH,
        "net_charge": Chem.GetFormalCharge(molecule),
        "protonation_policy": policy,
        "warnings": warnings,
        "identity_version": IDENTITY_VERSION,
    }
