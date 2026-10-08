"""Offline explicit-solvent equilibration for the Step 5 lipid library."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.io.gro import GROReader, GROWriter
from gmxbuilder.io.top import TopologyWriter
from gmxbuilder.modules.membrane.area_observable import (
    box_series_from_edr,
    measure_area_per_lipid,
    summarise,
)
from gmxbuilder.modules.membrane.equilibrated_library import (
    ACCEPTED_METHOD,
    MIN_CONFORMERS,
    SCHEMA_VERSION,
    EquilibratedLipidLibrary,
    lipid_parameter_family,
    topology_signature,
)
from gmxbuilder.modules.membrane.lipid_ensemble import (
    MIN_ORIENTED_FRACTION,
    orientation_population,
    orientation_summary,
)
from gmxbuilder.modules.membrane.lipid_orientation import (
    MAX_TAIL_CORE_GAP_NM,
    MIN_INWARD_COSINE,
    MIN_INWARD_PROJECTION_NM,
    infer_lipid_orientation,
    orient_lipid_to_outward_normal,
    outward_orientation,
    rotate_to_opposite_leaflet,
)

_gpu_device: ContextVar[int | None] = ContextVar("gmxbuilder_lipid_gpu_device", default=None)

#: The seed every library entry built before V4 used. Kept as the default so
#: rebuilding an entry reproduces it exactly; a replica asks for its own.
#:
#: One seed per entry is also why a V3 area cannot be checked: with a single
#: trajectory there is nothing to compare a value against, so protocol noise
#: and force-field difference are indistinguishable.
DEFAULT_REPLICA_SEED = 20260713


class PreparationError(RuntimeError):
    """A recoverable geometry/identity failure before any dynamics starts."""


class BilayerThicknessError(RuntimeError):
    """A finite frame whose periodic core geometry cannot identify membrane thickness."""

    def __init__(self, core_separation: float, box_z: float):
        self.evidence = {
            "kind": "ambiguous_periodic_leaflet_cores",
            "core_separation_nm": float(core_separation),
            "box_z_nm": float(box_z),
            "maximum_core_separation_fraction": MAX_CORE_SEPARATION_FRACTION,
        }
        super().__init__(
            "Cannot measure bilayer thickness: the two leaflet cores are "
            f"{abs(core_separation):.2f} nm apart in a {box_z:.2f} nm box, so which "
            "way round the box the membrane lies is ambiguous. This is not a "
            "bilayer the thickness gate can judge."
        )


def validate_prepared_work(
    work: Path, protocol: dict | None, *, allow_started: bool = False
) -> dict:
    """Read-only gate shared by the pair scheduler and production launcher."""
    from gmxbuilder.modules.membrane.parameter_provenance import validate_work_parameters

    record = json.loads((work / "prepared.json").read_text())
    required_fields = {
        "lipid_name",
        "force_field",
        "lipid_ff",
        "temperature",
        "npt_steps",
        "replica_seed",
        "genion_rmin",
        "test_mode",
        "build_started",
        "sampling_parameters",
    }
    if not isinstance(record, dict) or not required_fields <= record.keys():
        raise ValueError("Incomplete preparation record")
    if not isinstance(record["sampling_parameters"], dict):
        raise ValueError("Invalid prepared parameter record")
    from gmxbuilder.modules.membrane.v4_reanalysis import analysis_upgrade_allowed

    compatible = record.get("v4_protocol") == protocol or analysis_upgrade_allowed(
        record.get("v4_protocol"), protocol
    )
    if record.get("schema") != 1 or not compatible:
        raise ValueError("Prepared structure has an incompatible protocol")
    required = {"em.gro", "nvt.tpr", "npt.mdp", "topol.top"}
    if set(record.get("files", {})) != required:
        raise ValueError("Prepared structure is missing its input fingerprints")
    for filename, digest in record["files"].items():
        if hashlib.sha256((work / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared input changed: {filename}")
    validate_work_parameters(work, record["sampling_parameters"])
    from gmxbuilder.modules.membrane.initial_water import initial_water_evidence

    if initial_water_evidence(work) != record.get("initial_water_exclusion"):
        raise ValueError("Initial dry-membrane evidence is absent or changed")
    if not allow_started and (
        (work / "production-started.json").exists() or any(work.glob("nvt*.log"))
    ):
        raise RuntimeError(
            "Dynamics already started; preserve checkpoints for explicit continuation"
        )
    if protocol:
        validate_prepared_geometry(work, record, protocol)
    return record


def validate_prepared_geometry(work: Path, record: dict, protocol: dict) -> None:
    """Recheck stored coordinates, including hosts, before accepting a preparation.

    Hashes prove persistence, not chemical correctness. Walk complete molecular
    graphs so multi-residue lipid templates cannot split a molecule into tails.
    """
    from collections import Counter

    from gmxbuilder.geometry.molecular_identity import validate_stereochemistry, whole_molecule
    from gmxbuilder.modules.membrane.lipid_graph import molecular_graph, ordered_graph
    from gmxbuilder.modules.membrane.lipids import LipidRegistry
    from gmxbuilder.modules.membrane.local_conformations import validate_intrinsic_geometry
    from gmxbuilder.modules.membrane.v4_comparison import SALT_RESIDUES, WATER_RESIDUES

    structure = GROReader().read(work / "em.gro")
    force_field, lipid_ff = record["force_field"], record["lipid_ff"]
    composition = protocol["composition"]
    aliases = _simulation_lipid_resname_map(set(composition), force_field, lipid_ff)
    expected = {
        name: 2 * int(protocol["lipids_per_leaflet"] * fraction / 100)
        for name, fraction in composition.items()
    }
    counts = Counter()
    index = 0
    while index < structure.num_atoms:
        raw = str(structure.resnames[index]).strip().upper()
        name = aliases.get(raw, raw)
        if name in WATER_RESIDUES | SALT_RESIDUES:
            index += 1
            continue
        if name not in composition:
            raise ValueError(f"Unrecognized prepared lipid/solute: {raw}")
        lipid = LipidRegistry.get(name)
        names, _, _ = molecular_graph(name, force_field, lipid_ff, lipid.smiles)
        span = slice(index, index + len(names))
        elements, bonds = ordered_graph(
            name, force_field, lipid_ff, lipid.smiles, structure.atom_names[span]
        )
        xyz = whole_molecule(structure.coordinates[span], bonds, structure.box_vectors)
        validate_stereochemistry(lipid.smiles, elements, bonds, xyz)
        validate_intrinsic_geometry(xyz, elements, bonds)
        counts[name] += 1
        index += len(names)
    if dict(counts) != expected:
        raise ValueError(f"Prepared lipid composition {dict(counts)} differs from {expected}")


def _orientation_gate(
    lipid_name: str,
    projections: np.ndarray,
    cosines: np.ndarray,
    host_projections: np.ndarray,
    host_cosines: np.ndarray,
    *,
    upper_count: int,
    lower_count: int,
) -> tuple[bool, str, np.ndarray, np.ndarray]:
    """Select the scientifically appropriate production orientation gate."""
    gate_projections, gate_cosines, profile = orientation_population(
        lipid_name, projections, cosines, host_projections, host_cosines
    )
    summary = orientation_summary(gate_projections, gate_cosines)
    passed = bool(
        summary["passed"] and upper_count >= MIN_CONFORMERS and lower_count >= MIN_CONFORMERS
    )
    return passed, profile, gate_projections, gate_cosines


@contextmanager
def lipid_gpu_device(device_id: int | None) -> Iterator[None]:
    """Bind GROMACS mdrun calls in this execution to one visible GPU."""
    token = _gpu_device.set(None if device_id is None else int(device_id))
    try:
        yield
    finally:
        _gpu_device.reset(token)


def find_gromacs() -> str:
    from gmxbuilder.runtime.hardware import find_gromacs_executable

    candidate = find_gromacs_executable()
    if candidate:
        return candidate
    raise RuntimeError("GROMACS was not found; set GMX_BIN")


def _outer_headgroup_anchor(
    coordinates: np.ndarray,
    atom_names: list[str],
    box_midplane_z: float,
    *,
    upper_leaflet: bool | None = None,
) -> tuple[int, bool]:
    """Return an outward polar anchor and whether the lipid is upper-leaflet."""
    elements = [next((char for char in name.upper() if char.isalpha()), "") for name in atom_names]
    candidates = [
        index for index, element in enumerate(elements) if element in {"O", "N", "P", "S"}
    ]
    if not candidates:
        candidates = [index for index, element in enumerate(elements) if element != "H"]
    if not candidates:
        candidates = list(range(len(atom_names)))
    # Classify the leaflet by the chemical head region, not by the whole
    # molecule COM (which is tail-dominated and misclassifies inverted lipids).
    upper = (
        bool(upper_leaflet)
        if upper_leaflet is not None
        else float(coordinates[candidates, 2].mean()) >= box_midplane_z
    )
    stripped = [str(name).strip() for name in atom_names]
    if "P" in stripped:
        return stripped.index("P"), upper
    # Amber's Lipid21 names the backbone phosphorus P31, so the exact match
    # above finds nothing for all 39 of its templates and the fallback below
    # picks whichever polar atom happens to sit furthest out -- a different
    # reference plane for every headgroup, and a systematically wrong bilayer
    # thickness. A numeric suffix is accepted; a letter suffix is not, because
    # CHARMM names the *other* phosphates of cardiolipin and the
    # phosphoinositides PA, PB, PC and PD, and a thickness is referenced to
    # the backbone phosphate rather than to a headgroup one. Those residues
    # all carry a bare P as well, so they never reach here.
    numbered_phosphorus = [
        index
        for index, name in enumerate(stripped)
        if len(name) > 1 and name[0].upper() == "P" and name[1:].isdigit()
    ]
    if len(numbered_phosphorus) == 1:
        return numbered_phosphorus[0], upper
    z_values = coordinates[candidates, 2]
    local_index = int(np.argmax(z_values) if upper else np.argmin(z_values))
    return candidates[local_index], upper


#: Largest tail-centroid separation, as a fraction of the box height, for which
#: the two leaflet cores can still be identified by minimum image without
#: ambiguity. Measured across every bilayer this project has equilibrated the
#: ratio spans 0.18 to 0.26, so 0.40 leaves a wide margin and still refuses a
#: box in which the assumption has genuinely broken down.
MAX_CORE_SEPARATION_FRACTION = 0.40

#: Residue names the bundled water models use.
_WATER_RESIDUES = frozenset({"SOL", "WAT", "TIP3", "HOH", "T3P"})


#: Headgroup categories that are pre-equilibrated inside a POPC host.
#:
#: Sterols, lysolipids, diacylglycerols, ceramides, GM1 and the
#: phosphoinositides are not stable or biologically representative as a neat
#: 100% bilayer. What is extracted from such a run is the guest's conformers;
#: what is *measured* is the mixture's area, which is why these entries publish
#: no reference area of their own.
HOST_CATEGORIES = frozenset({"ST", "LPC", "DG", "CER", "GM1", "PIP"})


def membrane_molecule_offsets(system) -> list[int] | None:
    """Where each lipid molecule starts, from the membrane component.

    A residue is not a molecule. Lipid21 builds DOPC out of three residues --
    OL, PC, OL -- so grouping atoms by residue hands a half tail to anything
    that asks what a lipid looks like: its leaflet, its head, its orientation.
    The membrane component records the real molecule sizes and they are used
    wherever they exist.
    """
    for component in getattr(system, "components", []) or []:
        if component.kind is not ComponentKind.MEMBRANE:
            continue
        sizes = [int(value) for value in (component.metadata or {}).get("lipid_sizes", [])]
        if sizes:
            return list(np.cumsum([0] + sizes))
    return None


def _molecule_slices(resids, resnames, offsets=None) -> list[slice]:
    """Contiguous atom ranges, one per molecule."""
    if offsets is not None:
        return [
            slice(int(offsets[index]), int(offsets[index + 1])) for index in range(len(offsets) - 1)
        ]
    boundaries: list[slice] = []
    start = 0
    for index in range(1, len(resids) + 1):
        if (
            index < len(resids)
            and resids[index] == resids[start]
            and resnames[index] == resnames[start]
        ):
            continue
        boundaries.append(slice(start, index))
        start = index
    return boundaries


def _unwrapped_membrane_z(structure, offsets=None) -> tuple:
    """Atom z coordinates unwrapped about the bilayer, and where its centre is.

    A freshly built bilayer often straddles the periodic boundary, and then the
    median of the lipid z coordinates lands in the water rather than in the
    membrane -- and everything downstream is wrong from there: which leaflet a
    lipid belongs to, whether a water molecule is in the core. The first run of
    this gate called 890 waters "in the hydrophobic core" of a perfectly good
    sphingomyelin bilayer for exactly that reason. The circular mean is the
    centre of a distribution that wraps; the median is not.
    """
    coordinates = np.asarray(structure.coordinates, dtype=float)
    resnames = [str(name).strip().upper() for name in structure.resnames]
    lipid_atoms = [index for index, name in enumerate(resnames) if name not in _WATER_RESIDUES]
    if not lipid_atoms:
        return coordinates[:, 2], float("nan"), lipid_atoms
    box_z = float(np.asarray(structure.box_vectors, dtype=float)[2][2])
    if not np.isfinite(box_z) or box_z <= 0.0:
        return coordinates[:, 2], float(np.median(coordinates[lipid_atoms, 2])), lipid_atoms
    # Locate the hydrophobic slab, giving each molecule one vote. An all-atom
    # circular mean can select the water layer when bulky polar heads occupy
    # more than half the periodic Z interval (notably hosted phosphoinositides).
    tail_centres = []
    for span in _molecule_slices(structure.resids, resnames, offsets):
        if resnames[span.start] in _WATER_RESIDUES or span.stop - span.start < 4:
            continue
        profile = infer_lipid_orientation(coordinates[span], structure.atom_names[span])
        tail_centres.append(float(profile.tail_centroid[2]))
    if not tail_centres:
        return coordinates[:, 2], float("nan"), lipid_atoms
    centre = _circular_mean_z(np.asarray(tail_centres), box_z)
    unwrapped = (coordinates[:, 2] - centre + box_z / 2.0) % box_z
    # Lipids have already been made whole along their covalent graph. Image
    # each as a rigid object: a per-atom cut can tear a CER18 bond across Lz.
    for span in _molecule_slices(structure.resids, resnames, offsets):
        if resnames[span.start] in _WATER_RESIDUES or span.stop - span.start < 4:
            continue
        z = coordinates[span, 2]
        shifted = z - centre + box_z / 2.0
        unwrapped[span] = shifted - np.floor(shifted.mean() / box_z) * box_z
    return unwrapped, box_z / 2.0, lipid_atoms


def construction_core_gap(structure, offsets=None) -> float:
    """The gap between the leaflets' innermost tails, before any water is added.

    The same quantity the production gate measures after the simulation, asked
    at the point where it can still be acted on. The offline library builder
    deliberately lets the bootstrap lattice come out with an open core, because
    it then regrids the lipids and precompresses them -- but nothing checked
    whether that worked, and an open core at solvation is a permanent one.
    """
    coordinates = np.asarray(structure.coordinates, dtype=float)
    names = [str(name).strip() for name in structure.atom_names]
    resids = list(structure.resids)
    resnames = [str(name).strip().upper() for name in structure.resnames]
    unwrapped, centre, lipid_atoms = _unwrapped_membrane_z(structure, offsets)
    if not lipid_atoms or not np.isfinite(centre):
        return float("nan")
    local = coordinates.copy()
    local[:, 2] = unwrapped

    upper = []
    lower = []
    for span in _molecule_slices(resids, resnames, offsets):
        if resnames[span.start] in _WATER_RESIDUES:
            continue
        block = local[span]
        block_names = names[span]
        if len(block) < 4:
            continue  # an ion, not a lipid
        _anchor, is_upper = _outer_headgroup_anchor(block, block_names, centre)
        profile = infer_lipid_orientation(block, block_names)
        tails = block[list(profile.tail_indices), 2]
        if tails.size:
            (upper if is_upper else lower).append(tails)
    if not upper or not lower:
        return float("nan")
    upper_inner = float(np.percentile(np.concatenate(upper), 1.0))
    lower_inner = float(np.percentile(np.concatenate(lower), 99.0))
    return upper_inner - lower_inner


def hydrophobic_half_thickness(structure, offsets=None) -> float:
    """Half the thickness of the acyl-chain slab, measured from the midplane.

    Water does not belong anywhere in it. This is not a claim about equilibrium
    hydration -- a real bilayer has water down to the glycerols -- but about a
    bilayer that has just been packed and has never been simulated: any water
    the insertion grid dropped between two chains is an artefact, and it stays,
    because a hydrophobic slab does not dewet.
    """
    names = [str(name).strip() for name in structure.atom_names]
    resids = list(structure.resids)
    resnames = [str(name).strip().upper() for name in structure.resnames]
    unwrapped, centre, lipid_atoms = _unwrapped_membrane_z(structure, offsets)
    if not lipid_atoms or not np.isfinite(centre):
        return 0.0
    reach = 0.0
    for span in _molecule_slices(resids, resnames, offsets):
        if resnames[span.start] in _WATER_RESIDUES:
            continue
        block = np.asarray(structure.coordinates, dtype=float)[span].copy()
        block[:, 2] = unwrapped[span]
        block_names = names[span]
        if len(block) < 4:
            continue
        profile = infer_lipid_orientation(block, block_names)
        tails = block[list(profile.tail_indices), 2]
        if tails.size:
            reach = max(reach, float(np.abs(tails - centre).max()))
    return reach


def core_water_count(structure, offsets=None) -> int:
    """Water molecules inside the acyl-chain slab of a freshly built bilayer.

    The slab is measured rather than assumed: it is how far this bilayer's own
    tails reach from its own midplane, so the question asked is the physical one
    -- is there water among the chains -- and not a question about a distance
    somebody chose. A bilayer that has just been packed and solvated returns
    zero; any other answer means water was placed where nothing will take it out
    again.
    """
    names = [str(name).strip() for name in structure.atom_names]
    resnames = [str(name).strip().upper() for name in structure.resnames]
    unwrapped, centre, lipid_atoms = _unwrapped_membrane_z(structure, offsets)
    if not lipid_atoms or not np.isfinite(centre):
        return 0
    half_thickness = hydrophobic_half_thickness(structure, offsets)
    if half_thickness <= 0.0:
        return 0
    return int(
        sum(
            1
            for index, name in enumerate(resnames)
            if name in _WATER_RESIDUES
            and names[index].upper().startswith("O")
            and abs(unwrapped[index] - centre) <= half_thickness
        )
    )


def _circular_mean_z(values: np.ndarray, box_z: float) -> float:
    """Mean of periodic z coordinates, correct when the set straddles the box."""
    angles = np.asarray(values, dtype=float) * (2.0 * np.pi / box_z)
    angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    return (angle % (2.0 * np.pi)) * box_z / (2.0 * np.pi)


def head_to_head_distance(
    head_z: np.ndarray,
    tail_offsets: np.ndarray,
    upper: np.ndarray,
    box_z: float,
) -> float:
    """Head-plane separation of a bilayer, measured through its own core.

    The distance between the two leaflets' head planes is larger than half the
    box for most bilayers -- a 4.6 nm membrane in a 7.6 nm box is the ordinary
    case, not an extreme one. So it must not be reduced to its minimum image:
    doing that returns the distance the other way round the box, through the
    water, and reports that 4.6 nm bilayer as 2.9 nm. Reported thicknesses then
    come out low by however much the membrane exceeds half the box, which looks
    like a plausible number rather than an obvious failure.

    Nothing here images a distance that spans the membrane. Within one lipid
    the head-to-tail offset is exact: `trjconv -pbc whole` has already made the
    molecule whole, so no imaging is involved. Across the leaflets only the two
    tail-centroid planes are compared, and those meet at the hydrophobic core,
    so they sit far closer together than half the box.
    """
    head_z = np.asarray(head_z, dtype=float)
    tail_offsets = np.asarray(tail_offsets, dtype=float)
    upper = np.asarray(upper, dtype=bool)
    if not upper.any() or not (~upper).any():
        return float("nan")
    if not np.isfinite(head_z).all() or not np.isfinite(tail_offsets).all():
        return float("nan")

    core_upper = _circular_mean_z(head_z[upper] + tail_offsets[upper], box_z)
    core_lower = _circular_mean_z(head_z[~upper] + tail_offsets[~upper], box_z)
    core_separation = (core_upper - core_lower + box_z / 2.0) % box_z - box_z / 2.0
    if abs(core_separation) > MAX_CORE_SEPARATION_FRACTION * box_z:
        raise BilayerThicknessError(core_separation, box_z)
    return float(
        core_separation - float(tail_offsets[upper].mean()) + float(tail_offsets[~upper].mean())
    )


def _simulation_lipid_resname_map(
    lipid_names: set[str],
    force_field: str,
    lipid_ff: str,
) -> dict[str, str]:
    """Map five-character GROMACS output residue names to registry names."""
    mapping = {name[:5].upper(): name for name in lipid_names}
    if lipid_ff in {"gaff2", "amber-mixed"}:
        from gmxbuilder.modules.forcefield.gaff_backend import prepare_gaff_lipid
        from gmxbuilder.modules.forcefield.lipid_policy import lipid_backend_for
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        for name in lipid_names:
            if lipid_backend_for(name, lipid_ff) != "gaff2":
                continue
            lipid = LipidRegistry.get(name)
            template = prepare_gaff_lipid(name, lipid.smiles, lipid.charge)
            mapping[template.name[:5].upper()] = name
    return mapping


@dataclass(frozen=True)
class _EntryContext:
    """Everything about a library entry that its trajectory does not decide.

    A continued run has to arrive at the same expected area, the same host
    composition and the same residue naming as the run it continues, and it
    gets there without repeating the construction that produced the
    trajectory. Deriving them once, here, is what keeps the two paths from
    drifting apart -- the failure that made a sterol's measured area look like
    0.81 of an expectation that construction and the gate computed two
    different ways.
    """

    name: str
    lipid: object
    host_lipid: object | None
    membrane_lipid_names: set[str]
    membrane_config: dict
    simulation_resnames: dict
    target_apl: float
    target_dhh: float
    force_field: str
    lipid_ff: str
    output_dir: Path


class LipidEquilibrationBuilder:
    """Build a validated conformer entry from a solvated bilayer."""

    require_gpu_update = False
    v4_protocol = None

    def __init__(
        self,
        library: EquilibratedLipidLibrary | None = None,
        gmx: str | None = None,
        *,
        require_gpu_update: bool = False,
        v4_protocol: dict | None = None,
    ):
        from gmxbuilder.runtime.hardware import lipid_worker_threads

        self.library = library or EquilibratedLipidLibrary()
        self.gmx = gmx or find_gromacs()
        self.threads = lipid_worker_threads()
        self.require_gpu_update = require_gpu_update
        self.v4_protocol = v4_protocol

    @staticmethod
    def _mdp(
        stage: str,
        nsteps: int,
        temperature: float,
        seed: int = DEFAULT_REPLICA_SEED,
        *,
        force_field: str = "charmm36m",
        emtol: float = 1000,
    ) -> str:
        from gmxbuilder.io.mdp import _nonbond_params

        common = _nonbond_params({"force_field": force_field}) + "\npbc = xyz\n"
        if stage == "em":
            return (
                f"integrator = steep\nemtol = {emtol:g}\nemstep = 0.01\nnsteps = {nsteps}\n"
                + common
            )
        pressure = ""
        if stage == "npt":
            pressure = (
                "pcoupl = C-rescale\npcoupltype = Semiisotropic\n"
                "tau-p = 5.0\nref-p = 1.0 1.0\ncompressibility = 4.5e-5 4.5e-5\n"
            )
        # The area is measured from this file, so how often it is written is a
        # scientific parameter rather than an output-size preference. 1 ps is
        # far below the area's correlation time (measured in nanoseconds), so
        # the series resolves the autocorrelation instead of aliasing it.
        return (
            "integrator = md\ndt = 0.002\nnstenergy = 500\nnstcalcenergy = 100\n"
            "nstxout = 0\nnstvout = 0\nnstfout = 0\nnstxout-compressed = 5000\n"
            f"nsteps = {nsteps}\n"
            "constraints = h-bonds\nconstraint-algorithm = lincs\n"
            "tcoupl = V-rescale\ntc-grps = System\ntau-t = 1.0\n"
            f"ref-t = {temperature:.2f}\ngen-vel = {'yes' if stage == 'nvt' else 'no'}\n"
            f"gen-temp = {temperature:.2f}\ngen-seed = {int(seed)}\n"
            + ("continuation = yes\n" if stage == "npt" else "")
            + pressure
            + common
        )

    def _energy_end_time(self, work: Path, stage: str) -> float:
        """The last time present in a stage's energy file, in ps, or NaN."""
        try:
            times, _box_x, _box_y = box_series_from_edr(self.gmx, work / f"{stage}.edr", work)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError):
            return float("nan")
        return float(times[-1]) if len(times) else float("nan")

    def _mdrun_continue(self, stage: str, work: Path, *, until_ps: float, timeout: int) -> float:
        """Resume one MD stage from its checkpoint and return where it stopped.

        Deliberately not ``_mdrun``. That method treats an existing ``.gro`` as
        evidence that a GPU run which failed during device teardown had in fact
        integrated every step it was asked for, and on a continuation the
        previous run always left a ``.gro`` behind -- so the same reasoning
        would accept a resumed run that never started. What is checked here is
        the energy file, which only grows when steps are actually integrated.
        """
        from gmxbuilder.modules.membrane.grompp_policy import ensure_dynamics_policy

        ensure_dynamics_policy(self.gmx, work, stage)
        if self.v4_protocol:
            from gmxbuilder.modules.membrane.v4_tpr import verify_tpr

            verify_tpr(
                self.gmx, work / f"{stage}.tpr", self.v4_protocol, self.v4_protocol["family"], stage
            )
        if self.require_gpu_update and os.environ.get("GMXBUILDER_LIPID_LIBRARY_GPU", "1") != "1":
            raise RuntimeError("V4 requires GPU update, but lipid GPU execution is disabled")
        args = [
            self.gmx,
            "mdrun",
            "-deffnm",
            stage,
            "-cpi",
            f"{stage}.cpt",
            "-ntmpi",
            "1",
            "-ntomp",
            str(self.threads),
        ]
        if os.environ.get("GMXBUILDER_LIPID_LIBRARY_GPU", "1") == "1":
            args.extend(["-nb", "gpu", "-pme", "gpu"])
            if self.require_gpu_update:
                args.extend(["-update", "gpu"])
            selected_gpu = _gpu_device.get()
            if selected_gpu is not None:
                args.extend(["-gpu_id", str(selected_gpu)])
        failure: RuntimeError | None = None
        try:
            self._run(args, work, timeout=timeout)
        except RuntimeError as exc:
            failure = exc
        if failure is not None and self.require_gpu_update:
            raise failure
        reached = self._energy_end_time(work, stage)
        # ``_mdp`` writes an energy frame every 500 steps of 2 fs, so the last
        # one can sit one picosecond short of the requested end with nothing
        # wrong. Written as a lower bound rather than a comparison so an
        # unreadable energy file, which reads back as NaN, fails here instead
        # of comparing false and passing.
        if not reached >= until_ps - 1.0:
            raise RuntimeError(
                f"Continuing {stage} reached {reached:.0f} ps of the {until_ps:.0f} ps requested"
            ) from failure
        return reached

    def _run(
        self,
        args: list[str],
        cwd: Path,
        *,
        input_text: str | None = None,
        timeout: int = 3600,
        charge_stage: str = "neutralized",
    ) -> None:
        if len(args) > 1 and args[1] == "grompp":
            from gmxbuilder.modules.membrane.grompp_policy import run_grompp

            run_grompp(args, cwd, stage=charge_stage, timeout=timeout)
            return
        result = subprocess.run(
            args,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode:
            output = (result.stdout + "\n" + result.stderr)[-6000:]
            raise RuntimeError(f"Command failed ({' '.join(args)}):\n{output}")

    @staticmethod
    def _assert_finite_minimization(log_path: Path, *, force_limit: float | None = None) -> None:
        """Reject an EM output that GROMACS wrote after infinite forces."""
        text = log_path.read_text(errors="replace")
        potential_matches = re.findall(r"Potential Energy\s*=\s*([^\s]+)", text)
        force_matches = re.findall(r"Maximum force\s*=\s*([^\s]+)", text)
        if not potential_matches or not force_matches:
            raise PreparationError(
                f"Energy minimization diagnostics are missing from {log_path.name}"
            )
        try:
            potential = float(potential_matches[-1])
            maximum_force = float(force_matches[-1])
        except ValueError as exc:
            raise PreparationError(
                f"Energy minimization diagnostics are invalid in {log_path.name}"
            ) from exc
        if not math.isfinite(potential) or not math.isfinite(maximum_force):
            raise PreparationError(
                f"Energy minimization {log_path.name} left non-finite energy or force; "
                "the structure contains unresolved atomic overlaps"
            )

        if force_limit is not None and maximum_force > force_limit:
            raise PreparationError(
                f"Final minimization Fmax {maximum_force:g} exceeds emtol {force_limit:g}"
            )

    @staticmethod
    def _solvent_padding(net_charge_per_lipid: int) -> float:
        """Reserve enough water for neutralising extreme pure-anion bilayers."""
        return 2.0 if abs(int(net_charge_per_lipid)) >= 4 else 1.2

    @staticmethod
    def _ion_minimization_mdp(*, test_mode: bool, force_field: str = "charmm36m") -> str:
        """Use conservative steepest descent after close ion placement."""
        steps = 10000 if test_mode else 20000
        return LipidEquilibrationBuilder._mdp("em", steps, 310.0, force_field=force_field).replace(
            "emstep = 0.01", "emstep = 0.001"
        )

    def _mdrun(self, stage: str, work: Path, *, timeout: int) -> str:
        """Run one MD stage, preferring CUDA but retrying safely on CPU."""
        from gmxbuilder.modules.membrane.grompp_policy import ensure_dynamics_policy

        ensure_dynamics_policy(self.gmx, work, stage)
        if self.v4_protocol:
            from gmxbuilder.modules.membrane.v4_tpr import verify_tpr

            verify_tpr(
                self.gmx, work / f"{stage}.tpr", self.v4_protocol, self.v4_protocol["family"], stage
            )
        if self.require_gpu_update and os.environ.get("GMXBUILDER_LIPID_LIBRARY_GPU", "1") != "1":
            raise RuntimeError("V4 requires GPU update, but lipid GPU execution is disabled")
        base = [
            self.gmx,
            "mdrun",
            "-deffnm",
            stage,
            "-ntmpi",
            "1",
            "-ntomp",
            str(self.threads),
        ]
        if os.environ.get("GMXBUILDER_LIPID_LIBRARY_GPU", "1") == "1":
            try:
                gpu_args = ["-nb", "gpu", "-pme", "gpu"]
                if self.require_gpu_update:
                    gpu_args.extend(["-update", "gpu"])
                selected_gpu = _gpu_device.get()
                if selected_gpu is not None:
                    gpu_args.extend(["-gpu_id", str(selected_gpu)])
                self._run(base + gpu_args, work, timeout=timeout)
                return stage
            except RuntimeError:
                if self.require_gpu_update:
                    # A V4 GPU-resident run must not silently change execution
                    # policy or accept a stale final coordinate file on error.
                    raise
                # Some CUDA/GROMACS combinations complete the requested steps
                # but fail during device-buffer teardown. A final GRO is
                # written only after all requested integration steps; let the
                # next grompp validate it instead of repeating a long run.
                gpu_output = work / f"{stage}.gro"
                if gpu_output.is_file() and gpu_output.stat().st_size > 100:
                    return stage
                # Otherwise re-run to a separate basename so partial GPU files
                # are never consumed.
                cpu_stage = stage + "_cpu"
                self._run(
                    [
                        self.gmx,
                        "mdrun",
                        "-s",
                        f"{stage}.tpr",
                        "-deffnm",
                        cpu_stage,
                        "-ntmpi",
                        "1",
                        "-ntomp",
                        str(self.threads),
                        "-nb",
                        "cpu",
                        "-pme",
                        "cpu",
                    ],
                    work,
                    timeout=timeout,
                )
                return cpu_stage
        self._run(base + ["-nb", "cpu", "-pme", "cpu"], work, timeout=timeout)
        return stage

    def _genion_with_retry(self, work: Path) -> float:
        """Add neutralising/salt ions with bounded solvent-packing fallback."""
        topology = work / "topol.top"
        original_topology = topology.read_text()
        last_error: RuntimeError | None = None
        for rmin in (0.40, 0.35, 0.30):
            topology.write_text(original_topology)
            (work / "ionized.gro").unlink(missing_ok=True)
            try:
                self._run(
                    [
                        self.gmx,
                        "genion",
                        "-s",
                        "ions.tpr",
                        "-o",
                        "ionized.gro",
                        "-p",
                        "topol.top",
                        "-neutral",
                        "-conc",
                        "0.15",
                        "-pname",
                        "NA",
                        "-nname",
                        "CL",
                        "-rmin",
                        f"{rmin:.2f}",
                    ],
                    work,
                    input_text="SOL\n",
                    timeout=600,
                )
                return rmin
            except RuntimeError as exc:
                last_error = exc
                if "No more replaceable solvent" not in str(exc):
                    topology.write_text(original_topology)
                    raise
        topology.write_text(original_topology)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _repack_bootstrap_bilayer(system: System, spacing: float = 1.2) -> None:
        """Rigidly place bootstrap lipids on staggered, clash-free lattices."""
        membrane_components = [
            component for component in system.components if component.kind == ComponentKind.MEMBRANE
        ]
        if len(membrane_components) != 1:
            raise RuntimeError("Offline lipid equilibration requires one membrane component")
        metadata = membrane_components[0].metadata
        sizes = [int(value) for value in metadata.get("lipid_sizes", [])]
        upper_count = int(metadata.get("n_lipids_upper", 0))
        lower_count = int(metadata.get("n_lipids_lower", 0))
        if len(sizes) != upper_count + lower_count or min(upper_count, lower_count) <= 0:
            raise RuntimeError("Membrane lipid partition metadata is inconsistent")
        offsets = np.cumsum([0] + sizes)
        # A bounding-box span is not rotation invariant.  Two long lipids at
        # neighbouring grid points can therefore overlap when their tails
        # point toward each other even if each X/Y span is smaller than the
        # lattice spacing.  Bound every molecule by its maximum XY radius
        # about the same centre used for translation; twice the largest
        # radius plus a margin guarantees a clash-free initial lattice for
        # any azimuthal orientation.  Gradual rigid-centre precompression then
        # returns the system to the target APL before solvent is introduced.
        maximum_xy_radius = max(
            float(
                np.linalg.norm(
                    system.coordinates[offsets[index] : offsets[index + 1], :2]
                    - system.coordinates[offsets[index] : offsets[index + 1], :2].mean(axis=0),
                    axis=1,
                ).max()
            )
            for index in range(len(sizes))
        )
        spacing = max(float(spacing), 2.0 * maximum_xy_radius + 0.15)
        count = max(upper_count, lower_count)
        rows = max(1, int(np.floor(np.sqrt(count))))
        row_sizes = [count // rows + (i < count % rows) for i in range(rows)]
        box_xy = max(row_sizes) * spacing
        # Every periodic row is filled, including non-square sizes such as
        # 200/leaflet. Truncating a square grid left an unoccupied terminal band.
        grid = np.asarray(
            [
                ((j - (size - 1) / 2.0) * (box_xy / size), (i - (rows - 1) / 2.0) * (box_xy / rows))
                for i, size in enumerate(row_sizes)
                for j in range(size)
            ]
        )
        lower_grid = grid.copy()
        lower_grid[:, :2] += spacing / 2.0
        lower_grid = (lower_grid + box_xy / 2.0) % box_xy - box_xy / 2.0

        for molecule_index in range(len(sizes)):
            indices = slice(offsets[molecule_index], offsets[molecule_index + 1])
            current_xy = system.coordinates[indices, :2].mean(axis=0)
            if molecule_index < upper_count:
                target_xy = grid[molecule_index]
                # Keep the two bootstrap tail slabs apart during the vacuum
                # precompression.  At 0.15 nm, long anionic lipids can put an
                # upper-leaflet terminal carbon within 0.01 nm of a lower-
                # leaflet hydrogen before the first EM.  The explicit-solvent
                # NVT/NPT stages subsequently close this temporary core gap.
                z_shift = 0.55
            else:
                target_xy = lower_grid[molecule_index - upper_count]
                z_shift = -0.55
            system.coordinates[indices, :2] += target_xy - current_xy
            system.coordinates[indices, 2] += z_shift
        # Highly polyunsaturated chains can interdigitate far beyond the
        # nominal midplane.  Regridding changes their XY neighbours, so a
        # fixed shift alone can still create sub-0.02-nm cross-leaflet pairs.
        # Separate the complete atomic slabs before the first vacuum EM; NPT
        # later restores the physical DHH and core contact.
        upper_atoms = slice(0, int(offsets[upper_count]))
        lower_atoms = slice(int(offsets[upper_count]), int(offsets[-1]))
        upper_inner = float(system.coordinates[upper_atoms, 2].min())
        lower_inner = float(system.coordinates[lower_atoms, 2].max())
        minimum_slab_gap = 0.18
        additional_z = max(
            (lower_inner + minimum_slab_gap - upper_inner) / 2.0,
            0.0,
        )
        if additional_z > 0.0:
            system.coordinates[upper_atoms, 2] += additional_z
            system.coordinates[lower_atoms, 2] -= additional_z
        dimensions = system.structure.dimensions()
        system.structure.box_vectors = np.diag(
            [
                box_xy,
                box_xy,
                float(dimensions[2]) + 2.0 * additional_z + 0.3,
            ]
        )

    @staticmethod
    def _reimage_bilayer_z(system: System) -> None:
        """Move whole lipids to the nearest intended leaflet periodic image."""
        component = next(item for item in system.components if item.kind == ComponentKind.MEMBRANE)
        sizes = [int(value) for value in component.metadata.get("lipid_sizes", [])]
        upper_count = int(component.metadata.get("n_lipids_upper", 0))
        lower_count = int(component.metadata.get("n_lipids_lower", 0))
        if len(sizes) != upper_count + lower_count or not sizes:
            raise RuntimeError("Membrane lipid partition metadata is inconsistent")
        offsets = np.cumsum([0] + sizes)
        box_z = float(system.structure.dimensions()[2])
        if box_z <= 0.0:
            raise RuntimeError("Membrane box Z dimension is invalid")
        dhh = float(
            component.metadata.get(
                "bilayer_thickness",
                component.metadata.get("bilayer_thickness_nominal", 3.8),
            )
        )
        for molecule_index in range(len(sizes)):
            start = int(offsets[molecule_index])
            end = int(offsets[molecule_index + 1])
            upper = molecule_index < upper_count
            force_field = system.metadata.get("force_field")
            if force_field:
                from gmxbuilder.geometry.molecular_identity import whole_molecule
                from gmxbuilder.modules.membrane.lipid_graph import ordered_graph
                from gmxbuilder.modules.membrane.lipids import LipidRegistry

                name = str(system.structure.resnames[start]).strip().upper()
                lipid = LipidRegistry.get(name)
                _elements, bonds = ordered_graph(
                    name,
                    force_field,
                    system.metadata.get("lipid_ff", force_field),
                    lipid.smiles,
                    system.structure.atom_names[start:end],
                )
                system.coordinates[start:end] = whole_molecule(
                    system.coordinates[start:end],
                    bonds,
                    system.structure.box_vectors,
                )
            molecule_names = [
                str(value).strip() for value in system.structure.atom_names[start:end]
            ]
            anchor_index, _ = _outer_headgroup_anchor(
                system.coordinates[start:end],
                molecule_names,
                0.0,
                upper_leaflet=upper,
            )
            anchor_z = float(system.coordinates[start + anchor_index, 2])
            target_z = (dhh / 2.0) * (1.0 if upper else -1.0)
            image_shift = round((target_z - anchor_z) / box_z) * box_z
            system.coordinates[start:end, 2] += image_shift

    def _precompress_bilayer(
        self,
        system: System,
        work: Path,
        force_field: str,
        water_model: str,
        target_apl: float,
        *,
        test_mode: bool,
    ) -> None:
        """Shrink a safe lattice to target APL using rigid-centre EM cycles."""
        component = next(item for item in system.components if item.kind == ComponentKind.MEMBRANE)
        sizes = [int(value) for value in component.metadata["lipid_sizes"]]
        upper_count = int(component.metadata["n_lipids_upper"])
        target_xy = float(np.sqrt(upper_count * target_apl))
        current = work / "compress_00.gro"
        GROWriter.write(system.structure, current)
        topology = work / "compress.top"
        TopologyWriter(
            force_field,
            ff_config={
                "protein": force_field,
                "lipid_ff": system.metadata.get("lipid_ff", force_field),
                "water_model": water_model,
            },
        ).write_top(system.structure, topology, system_name="Lipid library precompression")
        # Dry packing needs more relaxation than the hydrated EM target:
        # stopping at 1000 reproduced a POPG double-bond inversion. This is
        # an optimizer target; identity checks remain the acceptance gate.
        (work / "compress.mdp").write_text(
            self._mdp("em", 1000 if test_mode else 3000, 310.0, force_field=force_field, emtol=500)
        )
        offsets = np.cumsum([0] + sizes)
        cycle = 0
        # Every construction attempt gets the approved full budget. A gentler
        # retry can need more than 64 cycles; a predicted count must not stop
        # it before the 128-cycle limit. The loop still exits as soon as ready.
        contraction = getattr(self, "_construction_contraction", 0.96)
        if not math.isfinite(contraction) or not 0 < contraction < 1:
            raise ValueError("Precompression requires a finite contraction between zero and one")
        max_cycles = 128
        accepted = None
        rejected = []

        # Keep the last validated pre-shrink state. A failed EM must never
        # seed the next trial, even when it wrote a syntactically valid GRO.
        def shrink(source, scale, destination):
            coordinates = source.coordinates.copy()
            centers = np.asarray(
                [
                    coordinates[offsets[i] : offsets[i + 1], :2].mean(axis=0)
                    for i in range(len(sizes))
                ]
            )
            box_center = centers.mean(axis=0)
            for i, center in enumerate(centers):
                coordinates[offsets[i] : offsets[i + 1], :2] += (
                    box_center + (center - box_center) * scale - center
                )
            trial = source.copy()
            trial.coordinates = coordinates
            dimensions = source.dimensions()
            trial.box_vectors = np.diag(
                [dimensions[0] * scale, dimensions[1] * scale, dimensions[2]]
            )
            GROWriter.write(trial, destination)

        while float(GROReader().read(current).dimensions()[0]) > target_xy * 1.01:
            cycle += 1
            if cycle > max_cycles:
                raise RuntimeError(
                    f"Precompression did not reach the target APL in {max_cycles} cycles"
                )
            tpr = work / f"compress_{cycle:02d}.tpr"
            deffnm = f"compress_em_{cycle:02d}"
            self._run(
                [
                    self.gmx,
                    "grompp",
                    "-f",
                    "compress.mdp",
                    "-c",
                    current.name,
                    "-p",
                    topology.name,
                    "-o",
                    tpr.name,
                    "-maxwarn",
                    "0",
                ],
                work,
                charge_stage="dry",
            )
            self._run(
                [
                    self.gmx,
                    "mdrun",
                    "-s",
                    tpr.name,
                    "-deffnm",
                    deffnm,
                    "-ntmpi",
                    "1",
                    "-ntomp",
                    str(self.threads),
                    "-nb",
                    "cpu",
                ],
                work,
                timeout=3600,
            )
            previous_xyz = system.coordinates.copy()
            previous_box = system.structure.box_vectors.copy()
            try:
                self._assert_finite_minimization(work / f"{deffnm}.log")
                minimized = GROReader().read(work / f"{deffnm}.gro")
                system.structure.coordinates = minimized.coordinates
                system.structure.box_vectors = minimized.box_vectors
                self._reimage_bilayer_z(system)
                if self.v4_protocol:
                    self._check_preparation_identity(system)
            except PreparationError as exc:
                system.structure.coordinates = previous_xyz
                system.structure.box_vectors = previous_box
                rejected.append({"cycle": cycle, "input": current.name, "reason": str(exc)})
                (work / "compression-rejections.json").write_text(json.dumps(rejected, indent=2))
                if accepted is None:
                    raise
                contraction = 1.0 - (1.0 - contraction) / 2.0
                if 1.0 - contraction < 0.0005:
                    raise PreparationError("Precompression exhausted its bounded backoff") from exc
                current = work / f"compress_{cycle:02d}.gro"
                shrink(accepted, contraction, current)
                continue
            minimized.coordinates = system.coordinates.copy()
            accepted = minimized.copy()
            old_xy = float(minimized.dimensions()[0])
            scale = max(target_xy / old_xy, contraction)
            current = work / f"compress_{cycle:02d}.gro"
            shrink(accepted, scale, current)

        # The loop minimises before each shrink. The target-size structure
        # therefore needs one final full EM before water is introduced.
        (work / "compress_final.mdp").write_text(
            self._mdp("em", 5000, 310.0, force_field=force_field, emtol=500)
        )
        self._run(
            [
                self.gmx,
                "grompp",
                "-f",
                "compress_final.mdp",
                "-c",
                current.name,
                "-p",
                topology.name,
                "-o",
                "compress_final.tpr",
                "-maxwarn",
                "0",
            ],
            work,
            charge_stage="dry",
        )
        self._run(
            [
                self.gmx,
                "mdrun",
                "-s",
                "compress_final.tpr",
                "-deffnm",
                "compress_final",
                "-ntmpi",
                "1",
                "-ntomp",
                str(self.threads),
                "-nb",
                "cpu",
            ],
            work,
            timeout=7200,
        )
        self._assert_finite_minimization(work / "compress_final.log")
        previous_xyz = system.coordinates.copy()
        previous_box = system.structure.box_vectors.copy()
        compressed = GROReader().read(work / "compress_final.gro")
        try:
            system.structure.coordinates = compressed.coordinates
            system.structure.box_vectors = compressed.box_vectors
            self._reimage_bilayer_z(system)
            if self.v4_protocol:
                self._check_preparation_identity(system)
        except PreparationError:
            system.structure.coordinates = previous_xyz
            system.structure.box_vectors = previous_box
            raise
        self._close_leaflet_gap(system, work, offsets, upper_count)

    @staticmethod
    def _drain_core_water(system: System) -> int:
        """Legacy defensive drain for imported offline construction coordinates.

        Normal solvation now excludes the complete lipid envelope. This narrower
        hydrophobic-core check remains useful for older offline checkpoints.
        """
        offsets = membrane_molecule_offsets(system)
        structure = system.structure
        unwrapped, centre, lipid_atoms = _unwrapped_membrane_z(structure, offsets)
        if not lipid_atoms or not np.isfinite(centre):
            return 0
        half_thickness = hydrophobic_half_thickness(structure, offsets)
        if half_thickness <= 0.0:
            return 0
        solvent = next(
            (
                component
                for component in system.components
                if component.kind is ComponentKind.SOLVENT
            ),
            None,
        )
        if solvent is None or not len(solvent.atom_indices):
            return 0
        indices = np.asarray(solvent.atom_indices, dtype=int)
        molecules = int(solvent.metadata.get("n_molecules", 0) or 0)
        if molecules <= 0 or len(indices) % molecules:
            return 0
        per_molecule = len(indices) // molecules
        first = int(indices.min())
        if not np.array_equal(indices, np.arange(first, first + len(indices))):
            return 0  # not the contiguous tail block this assumes

        # The test is made where the slab is unambiguous. Expressed as an
        # absolute interval it straddles the periodic boundary whenever the
        # bilayer sits at the box edge -- which is where re-imaging puts it --
        # and "between 7.0 and 9.0 nm" in an 8 nm box excludes every molecule
        # it should catch.
        oxygen_z = unwrapped[indices].reshape(molecules, per_molecule)[:, 0]
        inside = np.abs(oxygen_z - centre) <= half_thickness
        if not inside.any():
            return 0

        keep = list(range(first))
        for molecule, drop in enumerate(inside):
            if not drop:
                start = first + molecule * per_molecule
                keep.extend(range(start, start + per_molecule))
        system.structure.select_atoms(keep)
        solvent.atom_indices = np.arange(first, system.num_atoms, dtype=np.int64)
        solvent.metadata["n_molecules"] = int(molecules - int(inside.sum()))
        return int(inside.sum())

    def _close_leaflet_gap(
        self,
        system: System,
        work: Path,
        offsets: np.ndarray,
        upper_count: int,
    ) -> None:
        """Bring the two leaflets into contact before any water is added.

        The bootstrap repack deliberately holds the leaflets 1.1 nm apart so
        that rigid-centre precompression cannot drive an upper tail through a
        lower one, and left a comment saying the solvated NVT/NPT stages would
        close the gap again. They do not. Solvation fills whatever space it
        finds, a hydrophobic slab full of water does not dewet on any timescale
        this project simulates, and every entry built this way came out of 50 ns
        with a water layer where its core belonged -- 1052 molecules within
        0.4 nm of the midplane of what should have been a DPPC bilayer.

        So the gap is closed here, while the box is still dry and the only cost
        of being wrong is one more energy minimisation.
        """
        gap = construction_core_gap(system.structure, offsets)
        if not np.isfinite(gap) or gap <= 0.0:
            return
        best_coordinates = system.structure.coordinates.copy()
        best_box = system.structure.box_vectors.copy()
        best_gap = gap
        # Close incrementally and retain only chemically intact candidates.
        # Clash screening precedes EM; identity and force checks follow it.
        # Rejected attempts restore the last accepted coordinates and back off.
        step_limit = getattr(self, "_closure_step_limit", 0.2)
        attempt = min(best_gap, step_limit)
        audit = work / "closure-attempts.jsonl"
        for cycle in range(32):
            # GRO stores 0.001 nm coordinates. Each leaflet moves half a step;
            # stop every rejection path before retries become unrepresentable.
            if attempt < 0.002:
                break
            system.structure.coordinates = best_coordinates.copy()
            system.structure.box_vectors = best_box.copy()
            coordinates = system.structure.coordinates
            for molecule_index in range(len(offsets) - 1):
                indices = slice(int(offsets[molecule_index]), int(offsets[molecule_index + 1]))
                coordinates[indices, 2] += (
                    -attempt / 2.0 if molecule_index < upper_count else attempt / 2.0
                )
            from scipy.spatial import cKDTree

            dimensions = system.structure.dimensions()
            boundary = int(offsets[upper_count])
            upper = coordinates[:boundary] % dimensions
            lower = coordinates[boundary : int(offsets[-1])] % dimensions
            # Floating-point remainder of a tiny negative coordinate can round
            # to L exactly; cKDTree requires the half-open interval [0, L).
            upper = np.where(upper >= dimensions, 0.0, upper)
            lower = np.where(lower >= dimensions, 0.0, lower)
            nearest = float(cKDTree(lower, boxsize=dimensions).query(upper)[0].min())
            record = {"cycle": cycle, "step_nm": attempt, "minimum_contact_nm": nearest}
            if nearest < 0.12:
                record["rejected"] = "interleaflet contact below 0.12 nm"
                with audit.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                attempt /= 2
                if attempt < 0.002:
                    break
                continue
            current = work / f"closed_{cycle:02d}.gro"
            GROWriter.write(system.structure, current)
            (work / "closed.mdp").write_text(
                self._mdp("em", 5000, 310.0, force_field=system.metadata["force_field"])
            )
            try:
                self._run(
                    [
                        self.gmx,
                        "grompp",
                        "-f",
                        "closed.mdp",
                        "-c",
                        current.name,
                        "-p",
                        "compress.top",
                        "-o",
                        f"closed_{cycle:02d}.tpr",
                        "-maxwarn",
                        "0",
                    ],
                    work,
                    charge_stage="dry",
                )
                self._run(
                    [
                        self.gmx,
                        "mdrun",
                        "-s",
                        f"closed_{cycle:02d}.tpr",
                        "-deffnm",
                        f"closed_em_{cycle:02d}",
                        "-ntmpi",
                        "1",
                        "-ntomp",
                        str(self.threads),
                        "-nb",
                        "cpu",
                    ],
                    work,
                    timeout=7200,
                )
                self._assert_finite_minimization(
                    work / f"closed_em_{cycle:02d}.log", force_limit=1000.0
                )
                closed = GROReader().read(work / f"closed_em_{cycle:02d}.gro")
                system.structure.coordinates = closed.coordinates
                system.structure.box_vectors = closed.box_vectors
                self._reimage_bilayer_z(system)
                if self.v4_protocol:
                    self._check_preparation_identity(system)
            except (RuntimeError, OSError) as exc:
                record["rejected"] = str(exc)
                with audit.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                attempt /= 2.0
                continue
            measured = construction_core_gap(system.structure, offsets)
            record["gap_nm"] = float(measured)
            record["accepted"] = bool(np.isfinite(measured) and measured < best_gap)
            with audit.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if np.isfinite(measured) and measured < best_gap:
                best_coordinates = system.structure.coordinates.copy()
                best_box = system.structure.box_vectors.copy()
                best_gap = measured
                if measured <= 0.0:
                    break
                attempt = min(best_gap, step_limit)
            else:
                attempt /= 2.0

        system.structure.coordinates = best_coordinates
        system.structure.box_vectors = best_box

    @contextmanager
    def _entry_lock(self, lipid_name: str, force_field: str) -> Iterator[None]:
        """Serialize every writer of one library entry across processes."""
        safe_name = self.library._safe_component(str(lipid_name).strip().upper(), "lipid name")
        safe_force_field = self.library._safe_component(
            str(force_field).strip().lower(), "force field"
        )
        lock_root = self.library.roots[0].expanduser().resolve() / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        # Deliberately omit lipid_ff from the lock key: an automatic Amber
        # backend selection and an explicit GAFF2/Lipid21 request for the same
        # lipid must not publish concurrently to related cache paths.
        lock_path = lock_root / f"{safe_force_field}-{safe_name}.lock"
        from gmxbuilder.modules.forcefield.lipid_policy import rebuilding_library_entry

        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                # This *is* the process that replaces a superseded entry, so it
                # must not be refused for the entry being superseded. Without
                # the scope the report blocks its own remedy: rebuilding a
                # GAFF2 entry asks for a GAFF2 backend for exactly the lipid
                # whose GAFF2 entry is superseded.
                with rebuilding_library_entry(retry_failed_validation=self.v4_protocol is not None):
                    yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def build(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        **kwargs,
    ) -> Path:
        """Serialize builders for the same library entry across processes."""
        with self._entry_lock(lipid_name, force_field):
            work = kwargs.get("retain_work")
            attempts = 2 if self.v4_protocol and work is not None else 1
            for attempt in range(attempts):
                self._construction_contraction = 0.96 if attempt == 0 else 0.98
                self._closure_step_limit = 0.2 if attempt == 0 else 0.1
                try:
                    return self._build_once(lipid_name, force_field, lipid_ff, **kwargs)
                except PreparationError as exc:
                    if work is None:
                        raise
                    path = Path(work)
                    if any(path.glob("nvt*.log")) or any(path.glob("npt*.log")):
                        raise RuntimeError("Refusing construction retry after dynamics") from exc
                    (path / "preparation-failure.json").write_text(
                        json.dumps(
                            {
                                "attempt": attempt + 1,
                                "maximum_attempts": attempts,
                                "reason": str(exc),
                                "stage": "construction",
                            },
                            indent=2,
                        )
                    )
                    if attempt + 1 == attempts:
                        raise
                    archive = path.with_name(
                        path.name + ".failed-preparation-" + str(time.time_ns())
                    )
                    path.rename(archive)
                    print(
                        f"Construction retry; failed evidence preserved at {archive}: {exc}",
                        flush=True,
                    )

    def _check_preparation_identity(self, system):
        try:
            self._validate_membrane_identity(system)
        except ValueError as exc:
            raise PreparationError(f"Construction chemical identity check failed: {exc}") from exc

    def extend(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        **kwargs,
    ) -> Path:
        """Continue a published entry's NPT run and republish the entry."""
        with self._entry_lock(lipid_name, force_field):
            return self._extend_once(lipid_name, force_field, lipid_ff, **kwargs)

    def reanalyse(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        work_dir: Path,
        replica_seed: int = DEFAULT_REPLICA_SEED,
    ) -> Path:
        """Explicitly upgrade analysis of a completed run without more dynamics."""
        from gmxbuilder.modules.membrane.v4_reanalysis import reanalyse

        with self._entry_lock(lipid_name, force_field):
            return reanalyse(
                self,
                lipid_name,
                force_field,
                lipid_ff,
                work_dir=work_dir,
                replica_seed=replica_seed,
            )

    def _extend_once(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        to_ps: float,
        work_dir: Path,
        temperature: float | None = None,
        replica_seed: int = DEFAULT_REPLICA_SEED,
    ) -> Path:
        """Carry a finished NPT run further and measure the longer trajectory.

        An entry can finish its run having sampled the area too few times
        independently to average -- not because the run was wrong, but because
        the area's correlation time turned out to be long. Rebuilding that
        entry at a greater length answers the question and pays for the
        construction, the equilibration and the discarded head a second time.
        Continuing from the checkpoint is the same Markov chain carried
        further: every frame already collected stays in the average, and only
        the new ones cost anything.

        The frames already collected are also what makes this worth doing at
        all. Where the equilibration detector starts the analysis window is
        itself an estimate made from the series it is given, and on a short
        series it is bounded away from the answer -- BSM's 50 ns run put
        equilibration at 25 ns because that was as far as the detector was
        allowed to look, and the same system measured over 100 ns put it at
        2.6 ns. So a continuation does not merely add a tail; it can re-date
        the head, which is why the count is re-measured after every step
        rather than predicted before the first.
        """
        context = self._entry_context(lipid_name, force_field, lipid_ff, replica_seed=replica_seed)
        configured_temperature = self.v4_protocol["temperature_K"] if self.v4_protocol else 310.0
        if self.v4_protocol and temperature is not None and temperature != configured_temperature:
            raise ValueError("Temperature differs from the reviewed V4 protocol")
        temperature = configured_temperature if temperature is None else temperature
        published = context.output_dir / "metadata.json"
        if not published.is_file():
            raise RuntimeError(f"{context.name} has no published {force_field} entry to continue")
        metadata = json.loads(published.read_text())
        from gmxbuilder.modules.membrane.parameter_provenance import validate_work_parameters

        validate_work_parameters(Path(work_dir), metadata)
        if self.v4_protocol and metadata.get("v4_protocol") != self.v4_protocol:
            raise RuntimeError("Retained trajectory uses a different or absent V4 protocol")
        published_ps = float(metadata.get("npt_ps") or 0.0)
        if to_ps <= published_ps:
            raise ValueError(
                f"{context.name} has already run {published_ps:.0f} ps of NPT; "
                f"asked to continue it to {to_ps:.0f} ps"
            )
        # Recorded by the build that produced the trajectory. Recomputing it
        # would re-run genion against a system this process does not have.
        genion_rmin = float(metadata.get("genion_rmin_nm") or 0.0)

        work_dir = Path(work_dir)
        required = ("npt.tpr", "npt.cpt", "npt.edr", "topol.top")
        missing = [artefact for artefact in required if not (work_dir / artefact).is_file()]
        if missing:
            raise RuntimeError(
                f"{context.name} cannot be continued: {work_dir} is missing " + ", ".join(missing)
            )

        build_started = time.time()
        with self._extension_workspace(context.name, work_dir) as temp:
            work = Path(temp)
            # ``-until`` rather than ``-extend``: an absolute end time, so the
            # step is idempotent. A continuation that integrated its steps and
            # then failed while being measured leaves a trajectory longer than
            # the entry it has not yet republished, and a relative extension
            # retried against that state would add the same interval a second
            # time -- running past the target while publishing the shorter
            # length as the entry's own. There is no arithmetic to get wrong
            # when the run is told where to stop.
            self._run(
                [
                    self.gmx,
                    "convert-tpr",
                    "-s",
                    "npt.tpr",
                    "-until",
                    f"{to_ps:.3f}",
                    "-o",
                    "npt_extended.tpr",
                ],
                work,
            )
            (work / "npt_extended.tpr").replace(work / "npt.tpr")
            self._mdrun_continue("npt", work, until_ps=to_ps, timeout=172800)
            return self._measure_and_publish(
                work,
                "npt",
                name=context.name,
                lipid=context.lipid,
                force_field=force_field,
                lipid_ff=context.lipid_ff,
                temperature=temperature,
                npt_steps=int(round(to_ps * 500.0)),
                host_lipid=context.host_lipid,
                membrane_lipid_names=context.membrane_lipid_names,
                simulation_resnames=context.simulation_resnames,
                target_apl=context.target_apl,
                target_dhh=context.target_dhh,
                genion_rmin=genion_rmin,
                output_dir=context.output_dir,
                test_mode=False,
                build_started=build_started,
                retain_work=work_dir,
            )

    @contextmanager
    def _extension_workspace(self, name, work_dir):
        if self.v4_protocol:
            # Preserve new checkpoints and trajectories even if continuation
            # fails. The entry lock prevents simultaneous modification, and
            # -until makes retrying an absolute target idempotent.
            yield Path(work_dir)
        else:
            with tempfile.TemporaryDirectory(prefix=f"gmxbuilder-{name.lower()}-") as temp:
                shutil.copytree(work_dir, temp, dirs_exist_ok=True)
                yield Path(temp)

    def _entry_context(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None,
        *,
        replica_seed: int = DEFAULT_REPLICA_SEED,
    ) -> _EntryContext:
        """Resolve an entry's fixed properties from its name and force field."""
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        name = lipid_name.strip().upper()
        lipid = LipidRegistry.get(name)
        # Some amphiphiles are not stable or biologically representative as a
        # neat 100% bilayer (sterols, lysolipids, DAG, bulky glycolipids and
        # highly charged phosphoinositides).  Pre-equilibrate them at 40 mol%
        # in a POPC host so extracted conformers experience a sealed bilayer
        # while still providing >=20 molecules per leaflet for validation.
        host_lipid = LipidRegistry.get("POPC") if lipid.category in HOST_CATEGORIES else None
        if self.v4_protocol:
            from gmxbuilder.modules.membrane.v4_protocol import resolve_protocol

            expected = resolve_protocol(name, lipid_parameter_family(force_field, lipid_ff))
            if self.v4_protocol != expected:
                raise ValueError("V4 protocol identity or fingerprint mismatch")
            host_lipid = LipidRegistry.get("POPC") if expected["environment"] == "host" else None
        n_leaflet = self.v4_protocol["lipids_per_leaflet"] if self.v4_protocol else 64
        guest_percent = 10 if self.v4_protocol else 40
        host_percent = 100 - guest_percent
        membrane_lipid_names = {name}
        if host_lipid is not None:
            membrane_lipid_names.add(host_lipid.name)
            membrane_config = {
                "lipid_composition": {
                    "upper": [
                        {"name": host_lipid.name, "ratio": host_percent},
                        {"name": name, "ratio": guest_percent},
                    ],
                    "lower": [
                        {"name": host_lipid.name, "ratio": host_percent},
                        {"name": name, "ratio": guest_percent},
                    ],
                },
                "n_lipids_per_leaflet": n_leaflet,
                "seed": int(replica_seed),
            }
            # Ideal mixing of cross-sectional areas is the wrong model for
            # exactly the guests that need a host. A sterol's cross-section is
            # 0.38-0.41 nm^2; its *partial molar* area in a PC bilayer is 0.24,
            # because the condensing effect is what a sterol does. The area
            # model has held that number, with its provenance, all along --
            # construction already used it and only this gate did not, which
            # made 22RHC's measured 0.438 look like 0.81 of an expectation that
            # should have been 0.486.
            from gmxbuilder.modules.membrane.area_model import species_area

            # No force field is passed: that path would read a previously
            # published library area, and this is the process that publishes it.
            host_area, _host_source = species_area(host_lipid.name)
            guest_area, guest_area_source = species_area(name)
            target_apl = (host_percent * host_area + guest_percent * guest_area) / 100.0
            # Thickness has no equivalent model. Linear mixing predicts that a
            # sterol *thins* a PC bilayer, and sterols thicken it, so the
            # expectation is wrong in direction and no run can satisfy it. The
            # number is kept for the record and the gate on it is not applied
            # to host systems -- see production_quality below.
            target_dhh = (
                host_percent * host_lipid.bilayer_thickness
                + guest_percent * lipid.bilayer_thickness
            ) / 100.0
        else:
            membrane_config = {
                "lipid_type": name,
                "n_lipids_per_leaflet": n_leaflet,
                "seed": int(replica_seed),
            }
            target_apl = float(lipid.area_per_lipid)
            target_dhh = float(lipid.bilayer_thickness)
        if lipid_ff is None and force_field.startswith("amber"):
            from gmxbuilder.modules.forcefield.lipid_policy import amber_lipid_backend

            lipid_ff, reason = amber_lipid_backend(sorted(membrane_lipid_names))
            if lipid_ff is None:
                raise ValueError(reason)
        lipid_ff = lipid_ff or force_field
        if lipid_ff in {"gaff2", "amber-mixed"} and not force_field.startswith("amber"):
            raise ValueError("GAFF2 lipids require an Amber protein force-field family")
        if force_field == "oplsaa":
            raise ValueError("No general OPLS lipid parameterization backend is installed")
        simulation_resnames = _simulation_lipid_resname_map(
            membrane_lipid_names,
            force_field,
            lipid_ff,
        )

        output_dir = self.library.entry_dir(name, force_field, lipid_ff, writable=True)
        return _EntryContext(
            name=name,
            lipid=lipid,
            host_lipid=host_lipid,
            membrane_lipid_names=membrane_lipid_names,
            membrane_config=membrane_config,
            simulation_resnames=simulation_resnames,
            target_apl=target_apl,
            target_dhh=target_dhh,
            force_field=force_field,
            lipid_ff=lipid_ff,
            output_dir=output_dir,
        )

    @contextmanager
    def _build_workspace(self, name, retain_work):
        if self.v4_protocol and retain_work is not None:
            path = Path(retain_work)
            path.mkdir(parents=True, exist_ok=False)
            (path / "v4-protocol.json").write_text(json.dumps(self.v4_protocol, indent=2))
            yield path
        else:
            with tempfile.TemporaryDirectory(prefix=f"gmxbuilder-{name.lower()}-") as temp:
                yield Path(temp)

    def _validate_membrane_identity(self, system):
        from gmxbuilder.geometry.molecular_identity import validate_stereochemistry, whole_molecule
        from gmxbuilder.modules.membrane.lipid_graph import ordered_graph
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        structure = system.structure
        offsets = membrane_molecule_offsets(system)
        for span in _molecule_slices(structure.resids, structure.resnames, offsets):
            name = str(structure.resnames[span.start]).strip().upper()
            lipid = LipidRegistry.get(name)
            elements, bonds = ordered_graph(
                name,
                system.metadata["force_field"],
                system.metadata["lipid_ff"],
                lipid.smiles,
                structure.atom_names[span],
            )
            xyz = whole_molecule(structure.coordinates[span], bonds, structure.box_vectors)
            validate_stereochemistry(lipid.smiles, elements, bonds, xyz)
            structure.coordinates[span] = xyz

    def _repair_ceramide_bootstrap(self, system):
        """Select intact, correctly oriented two-chain seeds; never relax gates."""
        from gmxbuilder.geometry.rdkit_lipid import build_rdkit_lipid_geometry
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        component = next(c for c in system.components if c.kind == ComponentKind.MEMBRANE)
        offsets = membrane_molecule_offsets(system)
        upper_count = component.metadata["n_lipids_upper"]
        candidates = {}

        def chains_inward(xyz, names, upper):
            sign = 1 if upper else -1
            for suffix, root in (("F", "C1F"), ("S", "C3S")):
                chain = [
                    (int(n[1:-1]), i)
                    for i, n in enumerate(names)
                    if re.fullmatch(r"C\d+" + suffix, n)
                ]
                if root not in names or not chain:
                    raise ValueError("CER bootstrap lacks exact CHARMM chain atom names")
                vector = xyz[max(chain)[1]] - xyz[names.index(root)]
                inward = -sign * vector[2]
                if (
                    inward < MIN_INWARD_PROJECTION_NM
                    or inward / max(np.linalg.norm(vector), 1e-12) < MIN_INWARD_COSINE
                ):
                    return False
            return True

        for index, span in enumerate(
            _molecule_slices(system.structure.resids, system.structure.resnames, offsets)
        ):
            name = str(system.structure.resnames[span.start]).strip().upper()
            if name not in {"CER16", "CER18", "CER24"}:
                continue
            upper = index < upper_count
            names = [str(n).strip() for n in system.structure.atom_names[span]]
            original = system.coordinates[span]
            if chains_inward(original, names, upper):
                continue
            if name not in candidates:
                lipid = LipidRegistry.get(name)
                candidates[name] = []
                for seed in range(5):
                    xyz, generated_names = build_rdkit_lipid_geometry(
                        name,
                        lipid.smiles,
                        system.metadata["force_field"],
                        seed,
                        lipid_ff=system.metadata["lipid_ff"],
                    )
                    xyz = orient_lipid_to_outward_normal(xyz, generated_names, upper=True)
                    if generated_names == names and chains_inward(xyz, names, True):
                        candidates[name].append(xyz)
                if not candidates[name]:
                    raise RuntimeError(
                        f"No chemically intact, inward two-chain bootstrap for {name}"
                    )
            chosen = candidates[name][index % len(candidates[name])].copy()
            if not upper:
                chosen = rotate_to_opposite_leaflet(chosen)
            anchor, _ = _outer_headgroup_anchor(original, names, 0, upper_leaflet=upper)
            chosen += original[anchor] - chosen[anchor]
            system.coordinates[span] = chosen

    def _build_once(
        self,
        lipid_name: str,
        force_field: str,
        lipid_ff: str | None = None,
        *,
        temperature: float | None = None,
        npt_ps: float = 1000.0,
        test_mode: bool = False,
        force: bool = False,
        replica_seed: int = DEFAULT_REPLICA_SEED,
        retain_work: Path | None = None,
        prepare_only: bool = False,
        dry_membrane_checkpoint: Path | None = None,
    ) -> Path:
        from gmxbuilder.modules.membrane.builder import MembraneBuilder
        from gmxbuilder.modules.solvation.solvate import SolvationBuilder

        if prepare_only and (not self.v4_protocol or retain_work is None):
            raise ValueError("Preparation-only builds require a retained V4 workspace")

        context = self._entry_context(lipid_name, force_field, lipid_ff, replica_seed=replica_seed)
        configured_temperature = self.v4_protocol["temperature_K"] if self.v4_protocol else 310.0
        if self.v4_protocol and temperature is not None and temperature != configured_temperature:
            raise ValueError("Temperature differs from the reviewed V4 protocol")
        temperature = configured_temperature if temperature is None else temperature
        name = context.name
        lipid = context.lipid
        host_lipid = context.host_lipid
        membrane_lipid_names = context.membrane_lipid_names
        membrane_config = context.membrane_config
        target_apl = context.target_apl
        lipid_ff = context.lipid_ff
        output_dir = context.output_dir
        if not force and self.library.has(name, force_field, lipid_ff):
            return self.library.inspect(name, force_field, lipid_ff).path

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        build_started = time.time()
        with self._build_workspace(name, retain_work) as temp:
            work = Path(temp)
            initial = System(
                Structure(np.empty((0, 3)), np.eye(3) * 10.0),
                metadata={
                    "force_field": force_field,
                    "lipid_ff": lipid_ff,
                    "water_model": "tip3p",
                    "selected_lipid_names": sorted(membrane_lipid_names),
                    "seed": int(replica_seed),
                },
            )
            membrane = MembraneBuilder(
                use_equilibrated_library=False,
                allow_repairable_core_gap=True,
            ).run(
                initial,
                membrane_config,
            )
            if not membrane.success:
                raise RuntimeError("Bootstrap membrane failed: " + "; ".join(membrane.log))
            if self.v4_protocol:
                self._repair_ceramide_bootstrap(membrane.system)
                self._validate_membrane_identity(membrane.system)
            if dry_membrane_checkpoint is not None:
                if not self.v4_protocol or not prepare_only:
                    raise ValueError("Dry checkpoint recovery requires V4 preparation-only mode")
                saved = GROReader().read(dry_membrane_checkpoint)
                dry = membrane.system.structure
                count = len(dry.atom_names)
                if len(saved.atom_names) < count or any(
                    list(getattr(saved, field)[:count]) != list(getattr(dry, field))
                    for field in ("atom_names", "resnames", "resids")
                ):
                    raise PreparationError("Dry checkpoint atom order or identity differs")
                dry.coordinates = saved.coordinates[:count].copy()
                dry.box_vectors = saved.box_vectors.copy()
                self._check_preparation_identity(membrane.system)
            else:
                self._repack_bootstrap_bilayer(membrane.system)
                self._precompress_bilayer(
                    membrane.system,
                    work,
                    force_field,
                    "tip3p",
                    target_apl,
                    test_mode=test_mode,
                )
            # The bootstrap lattice is allowed to come out with an open core
            # because regridding and precompression are supposed to close it.
            # Whether they did was never checked, and solvating an open core
            # fills it with water that never leaves: 22RHC spent 50 ns with a
            # three-nanometre water slab where its hydrophobic core belonged,
            # and the outcome was fixed before the simulation started. The
            # interactive builder has held this invariant all along; the
            # library path now asks the same question at the last moment it can
            # still be answered cheaply.
            membrane_offsets = membrane_molecule_offsets(membrane.system)
            construction_gap = construction_core_gap(membrane.system.structure, membrane_offsets)
            if not np.isfinite(construction_gap) or construction_gap > MAX_TAIL_CORE_GAP_NM:
                raise PreparationError(
                    f"Bilayer for {name} is not sealed before solvation: leaflet tail gap "
                    f"{construction_gap:.3f} nm exceeds {MAX_TAIL_CORE_GAP_NM:.2f} nm. "
                    "Solvating this would put water in the hydrophobic core, where it "
                    "would stay. Precompression could not close the lattice the "
                    "conformer generator produced."
                )

            solvated = SolvationBuilder().run(
                membrane.system,
                {
                    "water_model": "tip3p",
                    "box_padding": self._solvent_padding(lipid.charge),
                    "seed": int(replica_seed),
                },
            )
            if not solvated.success:
                raise RuntimeError("Solvation failed: " + "; ".join(solvated.log))

            drained = self._drain_core_water(solvated.system)
            if drained:
                log_note = (
                    f"Removed {drained} water molecules the solvator placed among the "
                    f"acyl chains of {name}"
                )
                print(log_note, flush=True)
            core_waters = core_water_count(
                solvated.system.structure, membrane_molecule_offsets(solvated.system)
            )
            if core_waters:
                raise RuntimeError(
                    f"Solvation left {core_waters} water molecules among the acyl chains "
                    f"of {name}. The solvator was told this bilayer has no pore and asked "
                    "to keep them out, so any that remain are in a place nothing will "
                    "remove them from."
                )

            GROWriter.write(solvated.system.structure, work / "solvated.gro")
            TopologyWriter(
                force_field,
                ff_config={
                    "protein": force_field,
                    "lipid_ff": lipid_ff,
                    "water_model": "tip3p",
                },
            ).write_top(
                solvated.system.structure, work / "topol.top", system_name=f"{name} library"
            )

            # Even smoke mode must perform a real minimisation: the bootstrap
            # bilayer is densely packed and a 100-step EM can leave enormous
            # LJ contacts that make an MD smoke test meaningless.
            (work / "em.mdp").write_text(
                self._mdp("em", 5000, temperature, force_field=force_field)
            )
            self._run(
                [
                    self.gmx,
                    "grompp",
                    "-f",
                    "em.mdp",
                    "-c",
                    "solvated.gro",
                    "-p",
                    "topol.top",
                    "-o",
                    "ions.tpr",
                    "-maxwarn",
                    "0",
                ],
                work,
                charge_stage="before_ions",
            )
            # The offline library also builds deliberately extreme pure
            # anionic bilayers.  GROMACS' 0.6-nm default can exhaust the thin
            # solvent slabs before all neutralising ions fit; 0.4 nm remains
            # outside normal first-shell contact while allowing those valid
            # high-charge test systems to be neutralised.
            genion_rmin = self._genion_with_retry(work)
            (work / "ion_em.mdp").write_text(
                self._ion_minimization_mdp(test_mode=test_mode, force_field=force_field)
            )
            self._run(
                [
                    self.gmx,
                    "grompp",
                    "-f",
                    "ion_em.mdp",
                    "-c",
                    "ionized.gro",
                    "-p",
                    "topol.top",
                    "-o",
                    "em.tpr",
                    "-maxwarn",
                    "0",
                ],
                work,
            )
            self._run(
                [
                    self.gmx,
                    "mdrun",
                    "-deffnm",
                    "em",
                    "-ntmpi",
                    "1",
                    "-ntomp",
                    str(self.threads),
                ],
                work,
            )
            try:
                self._assert_finite_minimization(work / "em.log", force_limit=1000.0)
            except RuntimeError as exc:
                raise PreparationError(str(exc)) from exc
            if self.v4_protocol:
                final_em = GROReader().read(work / "em.gro")
                solvated.system.structure.coordinates = final_em.coordinates
                solvated.system.structure.box_vectors = final_em.box_vectors
                self._check_preparation_identity(solvated.system)
                gap = construction_core_gap(final_em, membrane_molecule_offsets(solvated.system))
                if not np.isfinite(gap) or gap > MAX_TAIL_CORE_GAP_NM:
                    raise PreparationError(
                        f"Post-solvation minimization left an open membrane core ({gap:.3f} nm)"
                    )

            nvt_steps = 100 if test_mode else 50000
            npt_steps = 250 if test_mode else max(50000, int(npt_ps * 500.0))
            (work / "nvt.mdp").write_text(
                self._mdp("nvt", nvt_steps, temperature, replica_seed, force_field=force_field)
            )
            (work / "npt.mdp").write_text(
                self._mdp("npt", npt_steps, temperature, replica_seed, force_field=force_field)
            )
            self._run(
                [
                    self.gmx,
                    "grompp",
                    "-f",
                    "nvt.mdp",
                    "-c",
                    "em.gro",
                    "-p",
                    "topol.top",
                    "-o",
                    "nvt.tpr",
                    "-maxwarn",
                    "0",
                ],
                work,
            )
            from gmxbuilder.modules.membrane.parameter_provenance import (
                fingerprint_from_metadata,
                record_work_parameters,
            )

            sampling_parameters = {
                "lipid_name": name,
                "force_field": force_field,
                "lipid_ff": lipid_ff,
                "equilibration_host": (
                    {
                        "lipid_name": host_lipid.name,
                        "ratio_percent": 90 if self.v4_protocol else 60,
                        "target_ratio_percent": 10 if self.v4_protocol else 40,
                    }
                    if host_lipid is not None
                    else None
                ),
            }
            record_work_parameters(work, fingerprint_from_metadata(sampling_parameters))
            from gmxbuilder.modules.membrane.initial_water import initial_water_evidence

            prepared = {
                "schema": 1,
                "lipid_name": name,
                "force_field": force_field,
                "lipid_ff": lipid_ff,
                "temperature": temperature,
                "npt_steps": npt_steps,
                "replica_seed": replica_seed,
                "genion_rmin": genion_rmin,
                "test_mode": test_mode,
                "build_started": build_started,
                "v4_protocol": self.v4_protocol,
                "initial_water_exclusion": initial_water_evidence(work),
                "sampling_parameters": {
                    **sampling_parameters,
                    "parameter_fingerprint": fingerprint_from_metadata(sampling_parameters),
                },
                "files": {
                    filename: hashlib.sha256((work / filename).read_bytes()).hexdigest()
                    for filename in ("em.gro", "nvt.tpr", "npt.mdp", "topol.top")
                },
            }
            if dry_membrane_checkpoint is not None:
                prepared["dry_membrane_checkpoint_sha256"] = hashlib.sha256(
                    Path(dry_membrane_checkpoint).read_bytes()
                ).hexdigest()
            if self.v4_protocol:
                try:
                    validate_prepared_geometry(work, prepared, self.v4_protocol)
                except ValueError as exc:
                    raise PreparationError(f"Prepared geometry check failed: {exc}") from exc
            # Publish readiness last; incomplete preparation can never acquire
            # this marker. Queue production is a separate phase/process.
            temporary = work / "prepared.json.tmp"
            temporary.write_text(json.dumps(prepared, indent=2) + "\n")
            temporary.replace(work / "prepared.json")
            if prepare_only:
                return work
            return self._run_prepared_once(work, retain_work=retain_work)

    def run_prepared(self, work_dir: Path) -> Path:
        """Start dynamics only from a complete, unchanged preparation record."""
        work = Path(work_dir)
        record = json.loads((work / "prepared.json").read_text())
        with self._entry_lock(record["lipid_name"], record["force_field"]):
            return self._run_prepared_once(work, retain_work=work)

    def resume_prepared(self, work_dir: Path) -> Path:
        """Continue an interrupted first NPT without rebuilding or rerunning NVT."""
        work = Path(work_dir)
        record = validate_prepared_work(work, self.v4_protocol, allow_started=True)
        if not (work / "production-started.json").is_file() or not all(
            (work / name).is_file() and (work / name).stat().st_size > 0
            for name in ("npt.tpr", "npt.cpt", "npt.edr", "nvt.gro", "nvt.cpt")
        ):
            raise RuntimeError("No complete first-NPT continuation checkpoint; preserve work")
        with self._entry_lock(record["lipid_name"], record["force_field"]):
            context = self._entry_context(
                record["lipid_name"],
                record["force_field"],
                record["lipid_ff"],
                replica_seed=record["replica_seed"],
            )
            reached = self._energy_end_time(work, "npt")
            if math.isfinite(reached) and reached >= record["npt_steps"] * 0.002 - 1.0:
                return self._recover_prepared_once(work, record, context)
            self._mdrun_continue("npt", work, until_ps=record["npt_steps"] * 0.002, timeout=172800)
            return self._publish_prepared(work, "npt", record, context, retain_work=work)

    def recover_prepared(self, work_dir: Path) -> Path:
        """Finish analysis of completed first production, with no MD fallback."""
        work = Path(work_dir)
        record = validate_prepared_work(work, self.v4_protocol, allow_started=True)
        with self._entry_lock(record["lipid_name"], record["force_field"]):
            context = self._entry_context(
                record["lipid_name"],
                record["force_field"],
                record["lipid_ff"],
                replica_seed=record["replica_seed"],
            )
            return self._recover_prepared_once(work, record, context)

    def _recover_prepared_once(self, work, record, context):
        from mdtraj.formats import XTCTrajectoryFile

        from gmxbuilder.modules.membrane.v4_tpr import verify_tpr

        required = (
            "production-started.json",
            "npt.tpr",
            "npt.cpt",
            "npt.edr",
            "npt.xtc",
            "npt.gro",
        )
        if not all((work / name).is_file() and (work / name).stat().st_size for name in required):
            raise RuntimeError("Completed production recovery requires all original MD artifacts")
        verify_tpr(self.gmx, work / "npt.tpr", self.v4_protocol, context.force_field, "npt")
        expected = record["npt_steps"] * 0.002
        reached = self._energy_end_time(work, "npt")
        if not math.isfinite(reached) or abs(reached - expected) > 1.0:
            raise RuntimeError("Completed production energy endpoint differs from preparation")
        with XTCTrajectoryFile(str(work / "npt.xtc")) as stream:
            if not len(stream):
                raise RuntimeError("Completed production trajectory is empty")
            stream.seek(len(stream) - 1)
            xyz, times, _, boxes = stream.read(n_frames=1)
        if abs(float(times[0]) - expected) > 0.01:
            raise RuntimeError("Completed production coordinate endpoint differs from preparation")
        final = GROReader().read(work / "npt.gro")
        if final.coordinates.shape != xyz[0].shape or not np.allclose(
            final.box_vectors, boxes[0], atol=1e-4, rtol=0
        ):
            raise RuntimeError("Completed production final coordinates differ from trajectory")
        delta = (final.coordinates - xyz[0]) @ np.linalg.inv(boxes[0])
        delta = (delta - np.rint(delta)) @ boxes[0]
        if not np.isfinite(delta).all() or np.max(np.abs(delta)) > 0.002:
            raise RuntimeError("Completed production final frame differs from trajectory endpoint")
        context.output_dir.parent.mkdir(parents=True, exist_ok=True)
        return self._publish_prepared(work, "npt", record, context, retain_work=work)

    def _run_prepared_once(self, work: Path, *, retain_work: Path | None) -> Path:
        from gmxbuilder.modules.membrane.grompp_policy import ensure_dynamics_policy

        record = validate_prepared_work(work, self.v4_protocol)
        context = self._entry_context(
            record["lipid_name"],
            record["force_field"],
            record["lipid_ff"],
            replica_seed=record["replica_seed"],
        )
        ensure_dynamics_policy(self.gmx, work, "nvt")
        (work / "production-started.json").write_text(json.dumps({"started": time.time()}))
        nvt_output = self._mdrun("nvt", work, timeout=3600)
        self._run(
            [
                self.gmx,
                "grompp",
                "-f",
                "npt.mdp",
                "-c",
                f"{nvt_output}.gro",
                "-t",
                f"{nvt_output}.cpt",
                "-p",
                "topol.top",
                "-o",
                "npt.tpr",
                "-maxwarn",
                "0",
            ],
            work,
        )
        npt_output = self._mdrun(
            "npt",
            work,
            timeout=7200 if record["test_mode"] else 172800,
        )
        return self._publish_prepared(work, npt_output, record, context, retain_work=retain_work)

    def _publish_prepared(self, work, npt_output, record, context, *, retain_work):
        return self._measure_and_publish(
            work,
            npt_output,
            name=context.name,
            lipid=context.lipid,
            force_field=context.force_field,
            lipid_ff=context.lipid_ff,
            temperature=record["temperature"],
            npt_steps=record["npt_steps"],
            host_lipid=context.host_lipid,
            membrane_lipid_names=context.membrane_lipid_names,
            simulation_resnames=context.simulation_resnames,
            target_apl=context.target_apl,
            target_dhh=context.target_dhh,
            genion_rmin=record["genion_rmin"],
            output_dir=context.output_dir,
            test_mode=record["test_mode"],
            build_started=record["build_started"],
            retain_work=retain_work,
        )

    def _measure_and_publish(
        self,
        work: Path,
        npt_output: str,
        *,
        name: str,
        lipid,
        force_field: str,
        lipid_ff: str,
        temperature: float,
        npt_steps: int,
        host_lipid,
        membrane_lipid_names: set[str],
        simulation_resnames: dict,
        target_apl: float,
        target_dhh: float,
        genion_rmin: float,
        output_dir: Path,
        test_mode: bool,
        build_started: float,
        retain_work: Path | None,
    ) -> Path:
        """Turn a finished NPT run into a published library entry.

        Split out of ``_build_once`` because a run that needs more sampling is
        continued from its checkpoint rather than rebuilt, and the continuation
        has to reach exactly this code -- the same measurement, the same gates
        and the same metadata -- from a different starting point. Everything
        here is a function of the NPT outputs in ``work`` plus parameters that
        are fixed by the lipid and the force field, so it does not care which
        of the two produced the trajectory.
        """
        from gmxbuilder.modules.membrane.initial_water import initial_water_evidence

        dry_initial = initial_water_evidence(work)
        prepared = json.loads((work / "prepared.json").read_text())
        if dry_initial != prepared.get("initial_water_exclusion"):
            raise ValueError("Initial dry-membrane evidence changed during simulation")
        analysis_history = []
        if self.v4_protocol:
            from gmxbuilder.modules.membrane.v4_reanalysis import archive_analysis

            analysis_history = archive_analysis(work, output_dir)
        n_leaflet = self.v4_protocol["lipids_per_leaflet"] if self.v4_protocol else 64
        from gmxbuilder.modules.membrane.parameter_provenance import work_parameter_files

        guest_percent = 10 if self.v4_protocol else 40
        host_percent = 100 - guest_percent
        # Measure the area from the NPT trajectory rather than from its
        # final frame. The final-frame value is one sample of a quantity
        # that fluctuates by several percent and drifts for tens of
        # nanoseconds, and publishing it as ``area_per_lipid_nm2`` is what
        # made the V3 numbers scatter by +-10% with no consistent sign.
        if retain_work is not None:
            # Copied before the quality gates run, not after: the run worth
            # keeping is usually the one that failed, and a tempdir that
            # has already been removed cannot be asked why.
            retain_work.mkdir(parents=True, exist_ok=True)
            for pattern in ("npt.*", "nvt.*", "topol.top", "*.mdp", "parameter-provenance.json"):
                for artefact in work.glob(pattern):
                    if (
                        artefact.is_file()
                        and artefact.resolve() != (retain_work / artefact.name).resolve()
                    ):
                        shutil.copy2(artefact, retain_work / artefact.name)

            for relative in work_parameter_files(work):
                source = work / relative
                target = retain_work / relative
                if source.resolve() != target.resolve():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)

        area_measurement = None
        try:
            series = box_series_from_edr(self.gmx, work / f"{npt_output}.edr", work)
            area_measurement = measure_area_per_lipid(*series, n_leaflet)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            # A missing or unreadable energy file must not lose the run:
            # the conformers are still valid, the area simply goes
            # unmeasured and is reported as such.
            area_note = f"area not measured: {exc}"
        else:
            area_note = summarise(area_measurement)

        whole = work / "whole.gro"
        self._run(
            [
                self.gmx,
                "trjconv",
                "-s",
                "npt.tpr",
                "-f",
                f"{npt_output}.gro",
                "-o",
                str(whole),
                "-pbc",
                "mol",
                "-center",
            ],
            work,
            input_text="System\nSystem\n",
            timeout=600,
        )
        structure = GROReader().read(whole)
        validation_groups: list[list[int]] = []
        current: list[int] = []
        previous = None
        for index, (resname, resid) in enumerate(
            zip(structure.resnames, structure.resids, strict=True)
        ):
            key = (resname, resid)
            canonical_resname = simulation_resnames.get(
                str(resname).strip().upper(),
                str(resname).strip().upper(),
            )
            if canonical_resname not in membrane_lipid_names:
                if current:
                    validation_groups.append(current)
                    current = []
                previous = None
                continue
            if previous is not None and key != previous:
                validation_groups.append(current)
                current = []
            current.append(index)
            previous = key
        if current:
            validation_groups.append(current)
        if len(validation_groups) % 2:
            raise RuntimeError("Equilibrated membrane has an odd number of lipid molecules")
        from gmxbuilder.modules.membrane.leaflet_identity import initial_leaflets

        initial_structure = GROReader().read(work / "ionized.gro")
        validation_records = initial_leaflets(initial_structure, structure, validation_groups)
        target_records = [
            (indices, upper_leaflet)
            for indices, upper_leaflet in validation_records
            if simulation_resnames.get(
                str(structure.resnames[indices[0]]).strip().upper(),
                str(structure.resnames[indices[0]]).strip().upper(),
            )
            == name
        ]
        if len(target_records) < MIN_CONFORMERS * 2:
            raise RuntimeError(
                f"Only {len(target_records)} intact {name} molecules were extracted; "
                f"need at least {MIN_CONFORMERS} in each leaflet"
            )

        sampler = None
        trajectory_analysis = None
        if self.v4_protocol:
            from gmxbuilder.modules.membrane.v4_trajectory import TrajectorySampler

            sampler = TrajectorySampler(
                structure,
                validation_records,
                simulation_resnames,
                self.v4_protocol,
                force_field,
                lipid_ff,
            )
            if not test_mode:
                trajectory_analysis = sampler.analyse(
                    work / f"{npt_output}.xtc", work, expected_end_ps=npt_steps * 0.002
                )
                if retain_work is not None and work.resolve() != retain_work.resolve():
                    for evidence_file in (
                        "trajectory_observables.npz",
                        "trajectory_geometry.npz",
                        "trajectory_unmeasurable_frames.json",
                        "trajectory_local_conformations.npz",
                    ):
                        shutil.copy2(work / evidence_file, retain_work)
        from gmxbuilder.modules.membrane.local_conformations import (
            canonical_coordinates,
            source_orientation,
        )

        source_groups = {
            tuple(indices): group
            for group in (sampler.groups if sampler is not None else [])
            for indices in group["indices"]
        }
        box_z = float(structure.dimensions()[2])
        prepared: list[tuple[np.ndarray, list[str]]] = []
        anchor_positions = []
        tail_offsets: list[float] = []
        leaflet_flags: list[bool] = []
        for indices, upper_leaflet in target_records:
            coords = structure.coordinates[indices].copy()
            atom_names = [structure.atom_names[index].strip() for index in indices]
            anchor_index, _ = _outer_headgroup_anchor(
                coords,
                atom_names,
                box_z / 2.0,
                upper_leaflet=upper_leaflet,
            )
            leaflet_flags.append(upper_leaflet)
            anchor = coords[anchor_index].copy()
            anchor_positions.append(anchor.copy())
            # Where this lipid's own hydrophobic centre sits relative to its
            # own head. The molecule is whole, so this is exact and needs no
            # periodic imaging -- which is what lets the bilayer thickness
            # be reconstructed below without ever imaging across the
            # membrane. Recorded before the frame change on the next line.
            profile = (
                source_orientation(coords, source_groups[tuple(indices)])
                if sampler is not None
                else infer_lipid_orientation(coords, atom_names)
            )
            tail_offsets.append(float(profile.tail_centroid[2]) - float(anchor[2]))
            coords -= anchor
            if not upper_leaflet:
                coords = rotate_to_opposite_leaflet(coords)
            # Store every accepted conformation in the canonical upper-
            # leaflet frame.  This rigid rotation preserves the NPT
            # internal geometry while making the on-disk contract explicit.
            coords = (
                canonical_coordinates(coords, sampler.local_definition)
                if sampler is not None
                else orient_lipid_to_outward_normal(coords, atom_names, upper=True)
            )
            coords -= coords[anchor_index]
            prepared.append((coords, atom_names))

        atom_names = prepared[0][1]
        if any(names != atom_names for _, names in prepared):
            raise RuntimeError("Extracted lipid atom order is inconsistent")
        # Deterministic, even sampling from both leaflets.
        selected = np.arange(len(prepared))
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.building-",
                dir=output_dir.parent,
            )
        )
        provenance = []
        if trajectory_analysis is not None:
            provenance = sampler.collect(work / f"{npt_output}.xtc", trajectory_analysis, staging)
            selected = np.arange(len(provenance))
        else:
            for number, index in enumerate(selected):
                coords, names = prepared[int(index)]
                np.savez_compressed(
                    staging / f"conf_{number:04d}.npz",
                    coords=coords,
                    atom_names=np.asarray(names),
                )
        signature = topology_signature(atom_names, force_field, lipid_ff)
        box_dimensions = structure.dimensions()
        upper_count = int(sum(leaflet_flags))
        lower_count = len(leaflet_flags) - upper_count
        area_per_lipid = float(box_dimensions[0] * box_dimensions[1] / float(n_leaflet))
        expected_apl = target_apl
        apl_ratio = area_per_lipid / expected_apl
        head_z = np.asarray([position[2] for position in anchor_positions], dtype=float)
        flags = np.asarray(leaflet_flags, dtype=bool)
        measured_dhh = head_to_head_distance(
            head_z,
            np.asarray(tail_offsets, dtype=float),
            flags,
            float(box_dimensions[2]),
        )
        dhh_ratio = measured_dhh / target_dhh
        # Validate the entire simulated bilayer, including the POPC host
        # used for sterols that cannot form a stable pure bilayer.
        validation_projections: list[float] = []
        validation_cosines: list[float] = []
        host_projections: list[float] = []
        host_cosines: list[float] = []
        upper_tail_z: list[np.ndarray] = []
        lower_tail_z: list[np.ndarray] = []
        for indices, validation_upper in validation_records:
            validation_coords = structure.coordinates[indices]
            validation_names = [structure.atom_names[index].strip() for index in indices]
            validation_profile = (
                source_orientation(validation_coords, source_groups[tuple(indices)])
                if sampler is not None
                else infer_lipid_orientation(validation_coords, validation_names)
            )
            validation_projection, validation_cosine = outward_orientation(
                validation_profile,
                upper=validation_upper,
            )
            validation_projections.append(validation_projection)
            validation_cosines.append(validation_cosine)
            validation_resname = simulation_resnames.get(
                str(structure.resnames[indices[0]]).strip().upper(),
                str(structure.resnames[indices[0]]).strip().upper(),
            )
            if validation_resname != name:
                host_projections.append(validation_projection)
                host_cosines.append(validation_cosine)
            validation_anchor, _ = _outer_headgroup_anchor(
                validation_coords,
                validation_names,
                box_z / 2.0,
                upper_leaflet=validation_upper,
            )
            head_plane = measured_dhh / 2.0 * (1.0 if validation_upper else -1.0)
            tail_relative = (
                validation_coords[validation_profile.tail_indices, 2]
                - validation_coords[validation_anchor, 2]
            )
            (upper_tail_z if validation_upper else lower_tail_z).append(head_plane + tail_relative)
        projections = np.asarray(validation_projections, dtype=float)
        cosines = np.asarray(validation_cosines, dtype=float)
        orientation_passed, orientation_profile, gate_projections, gate_cosines = _orientation_gate(
            name,
            projections,
            cosines,
            np.asarray(host_projections, dtype=float),
            np.asarray(host_cosines, dtype=float),
            upper_count=upper_count,
            lower_count=lower_count,
        )
        gate_orientation_valid = (
            np.isfinite(gate_projections)
            & np.isfinite(gate_cosines)
            & (gate_projections >= MIN_INWARD_PROJECTION_NM)
            & (gate_cosines >= MIN_INWARD_COSINE)
        )
        oriented_fraction = (
            float(gate_orientation_valid.mean()) if len(gate_orientation_valid) else 0.0
        )
        if upper_tail_z and lower_tail_z:
            upper_inner = float(np.percentile(np.concatenate(upper_tail_z), 1.0))
            lower_inner = float(np.percentile(np.concatenate(lower_tail_z), 99.0))
            tail_core_gap = upper_inner - lower_inner
        else:
            tail_core_gap = float("nan")
        frame_geometry = None
        if sampler is not None:
            # Use the same mixed-head-plane definition and atom selections as
            # XTC screening. The legacy dhh_nm below remains the TARGET metric.
            _, _, frame_geometry = sampler.metrics(
                structure.box_vectors,
                [structure.coordinates[group["indices"]] for group in sampler.groups],
            )
            tail_core_gap = frame_geometry["core_gap_nm"]
            oriented_fraction = frame_geometry["oriented_fraction"]
            orientation_passed = bool(
                frame_geometry["orientation"]["gate"]["passed"]
                and upper_count >= MIN_CONFORMERS
                and lower_count >= MIN_CONFORMERS
            )
        core_passed = bool(np.isfinite(tail_core_gap) and tail_core_gap <= MAX_TAIL_CORE_GAP_NM)
        # A guest in a POPC host is a different kind of entry
        # from a lipid that forms its own bilayer, and it is judged
        # differently -- not more leniently. What it does not get judged on
        # is a head-to-head thickness, because the number measured is the
        # mixture's and the number it would be compared against comes from
        # ideal mixing, which is wrong in direction for exactly these
        # guests. Everything structural still applies: the leaflets face
        # the right way and the hydrophobic core is sealed. The area is
        # kept as a diagnostic and is not published for these entries.
        hosted = host_lipid is not None
        production_quality = (
            not test_mode
            and npt_steps * 0.002 >= (100.0 if hosted else 500.0)
            and np.isfinite(area_per_lipid)
            and 0.75 <= apl_ratio <= 1.35
            and orientation_passed
            and core_passed
            and (hosted or (np.isfinite(measured_dhh) and 0.70 <= dhh_ratio <= 1.30))
        )
        metadata = {
            "v4_protocol": self.v4_protocol,
            "schema_version": SCHEMA_VERSION,
            "coordinate_handedness": "preserved",
            "leaflet_transform": "proper_rotation",
            "status": ("ready" if test_mode or production_quality else "failed"),
            "method": ACCEPTED_METHOD,
            "lipid_name": name,
            "canonical_smiles": lipid.smiles,
            "force_field": force_field,
            "lipid_ff": lipid_ff,
            "parameter_family": lipid_parameter_family(force_field, lipid_ff),
            "topology_sha256": signature,
            "atom_names": atom_names,
            "n_conformations": len(selected),
            "temperature_K": temperature,
            "salt_molar": 0.15,
            "genion_rmin_nm": genion_rmin,
            "npt_ps": npt_steps * 0.002,
            "test_mode": bool(test_mode),
            "equilibration_host": (
                {
                    "lipid_name": host_lipid.name,
                    "ratio_percent": host_percent,
                    "target_ratio_percent": guest_percent,
                }
                if host_lipid is not None
                else None
            ),
            # V4. Absent on V3 entries, which is how a reader tells the two
            # generations apart without a version negotiation.
            #
            # ``measures`` exists because a sterol has no pure lamellar
            # phase, so its entry is built in a 60:40 host bilayer and the
            # area belongs to the *mixture*. V3 stored that under
            # ``quality.area_per_lipid_nm2`` with nothing to say so, and
            # anyone reading cholesterol's 0.4612 would reasonably have
            # taken it for cholesterol's own area.
            "observables": {
                "measures": (
                    "pure-bilayer-area-per-lipid"
                    if host_lipid is None
                    else "host-mixture-area-per-lipid"
                ),
                "composition": (
                    {name: 100}
                    if host_lipid is None
                    else {host_lipid.name: host_percent, name: guest_percent}
                ),
                "lipids_per_leaflet": n_leaflet,
                **(
                    area_measurement.as_metadata()
                    if area_measurement is not None
                    else {"converged": False, "rejection": area_note}
                ),
            },
            "quality": {
                "initial_water_exclusion": dry_initial,
                "passed": bool(production_quality),
                "reason": (
                    "Production structural screening passed; equilibrium is assessed separately"
                    if production_quality
                    else "test mode or a production APL/DHH/orientation/core-seal gate failed"
                ),
                "area_per_lipid_nm2": area_per_lipid,
                "expected_area_per_lipid_nm2": expected_apl,
                "apl_ratio": apl_ratio,
                "dhh_nm": measured_dhh,
                "expected_dhh_nm": target_dhh,
                "dhh_ratio": dhh_ratio,
                "orientation": {
                    "passed": orientation_passed,
                    "profile": orientation_profile,
                    "n_lipids_checked": int(len(projections)),
                    "n_gate_lipids": int(len(gate_projections)),
                    "correct_fraction": oriented_fraction,
                    "minimum_correct_fraction": MIN_ORIENTED_FRACTION,
                    "outlier_count": int((~gate_orientation_valid).sum()),
                    "upper_lipids": upper_count,
                    "lower_lipids": lower_count,
                    "minimum_inward_projection_nm": (
                        float(gate_projections.min()) if len(gate_projections) else None
                    ),
                    "minimum_inward_cosine": (
                        float(gate_cosines.min()) if len(gate_cosines) else None
                    ),
                    "observed_target_minimum_projection_nm": (
                        float(projections.min()) if len(projections) else None
                    ),
                    "observed_target_minimum_cosine": (
                        float(cosines.min()) if len(cosines) else None
                    ),
                },
                "hydrophobic_core": {
                    "passed": core_passed,
                    "tail_core_gap_nm": tail_core_gap,
                    "maximum_tail_core_gap_nm": MAX_TAIL_CORE_GAP_NM,
                },
            },
            "elapsed_s": round(time.time() - build_started, 2),
        }
        if frame_geometry is not None:
            metadata["quality"]["geometry"] = frame_geometry
            gate = frame_geometry["orientation"]["gate"]
            metadata["quality"]["orientation"].update(
                profile=frame_geometry["orientation"]["profile"],
                n_gate_lipids=gate["n_lipids"],
                outlier_count=gate["outlier_count"],
                minimum_inward_projection_nm=gate["minimum_projection_nm"],
                minimum_inward_cosine=gate["minimum_cosine"],
            )
            metadata["quality"]["dhh_definition"] = "target-head-plane-separation"
            metadata["quality"]["hydrophobic_core"]["definition"] = (
                "mixed-head-plane-relative-tail-z-quantiles"
            )
        if trajectory_analysis is not None:
            block = trajectory_analysis["observables"]["area_per_lipid_nm2"]
            metadata["quality"]["orientation"]["final_frame_lipids_checked"] = len(projections)
            metadata["quality"]["orientation"]["n_lipids_checked"] = len(provenance)
            metadata["trajectory_analysis"] = trajectory_analysis
            metadata["analysis_history"] = analysis_history
            original_protocol = work / "v4-protocol.json"
            metadata["simulation_protocol"] = (
                json.loads(original_protocol.read_text())
                if original_protocol.is_file()
                else self.v4_protocol
            )
            metadata["conformer_provenance"] = provenance
            metadata["conformer_validation"] = {
                "identity_passed": True,
                "intrinsic_geometry_passed": True,
                "source": "local-relaxation-window-XTC-candidates",
                "normalization": "rigid-chemical-frame-only",
            }
            metadata["observables"].update(
                {
                    "area_per_lipid_nm2": block,
                    "analysis_window_ps": trajectory_analysis["analysis_window_ps"],
                    "equilibration_ps": trajectory_analysis["analysis_window_ps"][0],
                    "retained_fraction": trajectory_analysis["retained_fraction"],
                    "converged": False,
                    "rejection": (
                        "Quantitative area awaits independent-replica agreement and 2% CI precision"
                    ),
                    "criteria": {"min_effective_samples": 10, "max_relative_ci95_half_width": 0.02},
                }
            )
        from gmxbuilder.modules.membrane.parameter_provenance import (
            validate_work_parameters,
        )
        from gmxbuilder.modules.membrane.v4_sampling import json_safe

        metadata["parameter_fingerprint"] = json.loads(
            (work / "parameter-provenance.json").read_text()
        )["fingerprint"]
        validate_work_parameters(work, metadata)
        (staging / "metadata.json").write_text(
            json.dumps(json_safe(metadata), indent=2, allow_nan=False)
        )
        publish_dir = (
            output_dir
            if test_mode or production_quality
            else output_dir.with_name(output_dir.name + ".failed")
        )
        if publish_dir.exists():
            shutil.rmtree(publish_dir)
        staging.replace(publish_dir)
        if not test_mode and not production_quality:
            gate_projection_minimum = (
                float(gate_projections.min()) if len(gate_projections) else float("nan")
            )
            raise RuntimeError(
                f"Production quality gates failed for {name}: "
                f"APL ratio {apl_ratio:.3f}, DHH ratio {dhh_ratio:.3f}, "
                f"correctly oriented fraction {oriented_fraction:.3f}, "
                f"minimum inward projection "
                f"{gate_projection_minimum:.3f} nm, "
                f"minimum inward cosine "
                f"{float(gate_cosines.min()) if len(gate_cosines) else float('nan'):.3f}, "
                f"tail-core gap {tail_core_gap:.3f} nm. Diagnostics were retained in "
                f"{publish_dir / 'metadata.json'}"
            )
        return output_dir
