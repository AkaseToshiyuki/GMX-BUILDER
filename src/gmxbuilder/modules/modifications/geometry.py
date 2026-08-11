"""Force-field-native local geometry construction for modified residues.

The RTP supplies the bonded graph and atom types; the force-field bonded
parameter tables supply equilibrium bond lengths and angles.  Only atoms that
are absent from the uploaded parent residue are optimized.  Common heavy atoms
remain fixed so applying a modification cannot silently move the protein.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from gmxbuilder.modules.forcefield.rtp_parser import _force_field_path
from gmxbuilder.modules.modifications.patches import StereoConstraint


class ModificationGeometryError(ValueError):
    """Raised when a force-field-consistent modified residue cannot be built."""


@dataclass(frozen=True)
class GeometryQuality:
    """Quality measurements for a newly constructed heavy-atom group."""

    added_atoms: tuple[str, ...]
    max_bond_error_nm: float
    max_angle_error_deg: float
    min_nonbonded_distance_nm: float | None
    stereo_centres: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BondedGeometryParameters:
    bonds: dict[tuple[str, str], float]
    angles: dict[tuple[str, str, str], float]

    def bond(self, first: str, second: str) -> float:
        try:
            return self.bonds[(first, second)]
        except KeyError as error:
            raise ModificationGeometryError(
                f"No equilibrium bond length for atom types {first}-{second}"
            ) from error

    def angle(self, first: str, centre: str, third: str) -> float:
        try:
            return self.angles[(first, centre, third)]
        except KeyError as error:
            raise ModificationGeometryError(
                f"No equilibrium angle for atom types {first}-{centre}-{third}"
            ) from error


def _parse_parameter_file(
    path: Path,
    bonds: dict[tuple[str, str], float],
    angles: dict[tuple[str, str, str], float],
) -> None:
    section = ""
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        fields = line.split()
        try:
            if section == "bondtypes" and len(fields) >= 5:
                value = float(fields[3])
                bonds[(fields[0], fields[1])] = value
                bonds[(fields[1], fields[0])] = value
            elif section == "angletypes" and len(fields) >= 6:
                value = float(fields[4])
                angles[(fields[0], fields[1], fields[2])] = value
                angles[(fields[2], fields[1], fields[0])] = value
        except ValueError:
            # Preprocessor macros and symbolic parameters cannot provide a
            # numerical construction target and are deliberately ignored.
            continue


@lru_cache(maxsize=None)
def _load_geometry_parameters(force_field: str) -> _BondedGeometryParameters:
    bonds: dict[tuple[str, str], float] = {}
    angles: dict[tuple[str, str, str], float] = {}
    for path in sorted(_force_field_path(force_field).glob("*.itp")):
        _parse_parameter_file(path, bonds, angles)
    return _BondedGeometryParameters(bonds=bonds, angles=angles)


def _is_hydrogen(name: str) -> bool:
    value = name.strip().upper()
    return value.startswith("H") or (
        len(value) > 1 and value[0].isdigit() and value[1] == "H"
    )


def _element(name: str) -> str:
    value = "".join(character for character in name.upper() if character.isalpha())
    if value.startswith("CL"):
        return "CL"
    if value.startswith("BR"):
        return "BR"
    return value[:1] or "C"


_VDW_RADII_NM = {
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "P": 0.180,
    "S": 0.180,
    "F": 0.147,
    "CL": 0.175,
    "BR": 0.185,
}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ModificationGeometryError("Degenerate local coordinate reference")
    return vector / norm


def _angle(first: np.ndarray, centre: np.ndarray, third: np.ndarray) -> float:
    left = _unit(first - centre)
    right = _unit(third - centre)
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0))))


def _heavy_template(template: dict) -> tuple[dict[str, str], list[tuple[str, str]]]:
    atom_types = {
        str(atom[0]).strip(): str(atom[1]).strip()
        for atom in template.get("atoms", [])
        if not _is_hydrogen(str(atom[0]))
    }
    bonds = [
        (str(first).strip(), str(second).strip())
        for first, second in template.get("bonds", [])
        if first in atom_types
        and second in atom_types
        and not str(first).startswith(("+", "-"))
        and not str(second).startswith(("+", "-"))
    ]
    return atom_types, bonds


def _neighbours(
    atom_types: dict[str, str], bonds: list[tuple[str, str]]
) -> dict[str, list[str]]:
    graph = {name: [] for name in atom_types}
    for first, second in bonds:
        graph[first].append(second)
        graph[second].append(first)
    return graph


def _angle_terms(
    graph: dict[str, list[str]],
    atom_types: dict[str, str],
    parameters: _BondedGeometryParameters,
) -> list[tuple[str, str, str, float]]:
    terms = []
    for centre, bonded in graph.items():
        for first, third in combinations(bonded, 2):
            target = parameters.angle(
                atom_types[first], atom_types[centre], atom_types[third]
            )
            terms.append((first, centre, third, target))
    return terms


def _graph_distances(graph: dict[str, list[str]]) -> dict[tuple[str, str], int]:
    distances: dict[tuple[str, str], int] = {}
    for start in graph:
        queue = [(start, 0)]
        visited = {start}
        while queue:
            current, distance = queue.pop(0)
            distances[(start, current)] = distance
            for neighbour in graph[current]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))
    return distances


def _resolve_stereo_constraints(
    constraints: tuple[StereoConstraint, ...],
    atom_types: dict[str, str],
) -> list[tuple[str, str, str, str, int, str]]:
    """Resolve force-field atom-name aliases and fail on incomplete metadata."""
    resolved = []
    available = set(atom_types)
    for constraint in constraints:
        selectors = (constraint.center, *constraint.ordered_neighbors)
        names = []
        for selector in selectors:
            name = next((candidate for candidate in selector if candidate in available), None)
            if name is None:
                raise ModificationGeometryError(
                    f"Stereochemistry {constraint.label} references unavailable atom names "
                    f"{'/'.join(selector)}"
                )
            names.append(name)
        if len(set(names)) != 4:
            raise ModificationGeometryError(
                f"Stereochemistry {constraint.label} does not identify four distinct atoms"
            )
        if constraint.expected_sign not in {-1, 1}:
            raise ModificationGeometryError(
                f"Stereochemistry {constraint.label} has an invalid expected sign"
            )
        resolved.append((*names, constraint.expected_sign, constraint.label))
    return resolved


def _signed_volume(
    coordinates: dict[str, np.ndarray],
    center: str,
    first: str,
    second: str,
    third: str,
) -> float:
    origin = coordinates[center]
    return float(np.dot(
        np.cross(coordinates[first] - origin, coordinates[second] - origin),
        coordinates[third] - origin,
    ))


def _seed_stereochemistry(
    coordinates: dict[str, np.ndarray],
    missing: set[str],
    constraints: list[tuple[str, str, str, str, int, str]],
) -> None:
    """Reflect one new substituent when initialization chose the enantiomer."""
    for center, first, second, third, expected_sign, label in constraints:
        ordered = (first, second, third)
        movable = next((name for name in ordered if name in missing), None)
        if movable is None:
            oriented = expected_sign * _signed_volume(
                coordinates, center, first, second, third
            )
            if oriented < 2.0e-4:
                raise ModificationGeometryError(
                    f"Retained atoms contradict required stereochemistry {label}"
                )
            continue
        fixed = [name for name in ordered if name != movable]
        if len(fixed) != 2:
            raise ModificationGeometryError(
                f"Cannot seed stereochemistry {label} with multiple new reference atoms"
            )
        oriented = expected_sign * _signed_volume(
            coordinates, center, first, second, third
        )
        if oriented >= 2.0e-4:
            continue
        origin = coordinates[center]
        normal = np.cross(
            coordinates[fixed[0]] - origin,
            coordinates[fixed[1]] - origin,
        )
        normal = _unit(normal)
        displacement = coordinates[movable] - origin
        coordinates[movable] = (
            coordinates[movable] - 2.0 * float(np.dot(displacement, normal)) * normal
        )
        if expected_sign * _signed_volume(
            coordinates, center, first, second, third
        ) < 2.0e-4:
            raise ModificationGeometryError(
                f"Cannot initialize required stereochemistry {label}"
            )


@lru_cache(maxsize=1)
def _sphere_directions() -> np.ndarray:
    """Deterministic near-uniform unit directions used only for initialization."""
    count = 1536
    indices = np.arange(count, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = golden_angle * indices
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def _local_basis(
    anchor: str,
    coordinates: dict[str, np.ndarray],
    graph: dict[str, list[str]],
) -> np.ndarray:
    known_neighbours = [name for name in graph[anchor] if name in coordinates]
    if not known_neighbours:
        raise ModificationGeometryError(
            f"Cannot define a local frame for disconnected atom {anchor}"
        )
    first = known_neighbours[0]
    axis = _unit(coordinates[first] - coordinates[anchor])

    references = [
        coordinates[name] - coordinates[anchor]
        for name in known_neighbours[1:]
    ]
    references.extend(
        coordinates[name] - coordinates[first]
        for name in graph[first]
        if name != anchor and name in coordinates
    )
    side = None
    for reference in references:
        projected = reference - float(np.dot(reference, axis)) * axis
        if float(np.linalg.norm(projected)) > 1e-6:
            side = _unit(projected)
            break
    if side is None:
        # This fallback is reached only for a perfectly linear two-atom input.
        # Select the least parallel Cartesian direction without changing any
        # chemical target; normal protein side chains provide a local reference.
        cartesian = np.eye(3)
        reference = cartesian[int(np.argmin(np.abs(cartesian @ axis)))]
        side = _unit(reference - float(np.dot(reference, axis)) * axis)
    normal = _unit(np.cross(axis, side))
    return np.column_stack((axis, side, normal))


def _clash_threshold(first: str, second: str) -> float:
    first_radius = _VDW_RADII_NM.get(_element(first), 0.170)
    second_radius = _VDW_RADII_NM.get(_element(second), 0.170)
    return 0.55 * (first_radius + second_radius)


def _initial_score(
    name: str,
    candidate: np.ndarray,
    coordinates: dict[str, np.ndarray],
    graph_distances: dict[tuple[str, str], int],
    bond_targets: list[tuple[str, str, float]],
    angle_targets: list[tuple[str, str, str, float]],
    environment: np.ndarray,
) -> float:
    trial = dict(coordinates)
    trial[name] = candidate
    score = 0.0
    for first, second, target in bond_targets:
        if name not in {first, second} or first not in trial or second not in trial:
            continue
        score += ((float(np.linalg.norm(trial[first] - trial[second])) - target) / 0.008) ** 2
    for first, centre, third, target in angle_targets:
        if name not in {first, centre, third}:
            continue
        if first not in trial or centre not in trial or third not in trial:
            continue
        score += ((_angle(trial[first], trial[centre], trial[third]) - target) / 7.5) ** 2
    for other, position in coordinates.items():
        if graph_distances.get((name, other), 99) <= 2:
            continue
        distance = float(np.linalg.norm(candidate - position))
        threshold = _clash_threshold(name, other)
        if distance < threshold:
            score += 25.0 * ((threshold - distance) / 0.02) ** 2
    if environment.size:
        distances = np.linalg.norm(environment - candidate, axis=1)
        violations = np.maximum(0.0, 0.17 - distances)
        score += 25.0 * float(np.sum((violations / 0.02) ** 2))
    return score


def _initialize_missing_atoms(
    missing: list[str],
    coordinates: dict[str, np.ndarray],
    graph: dict[str, list[str]],
    bond_targets: list[tuple[str, str, float]],
    angle_targets: list[tuple[str, str, str, float]],
    environment: np.ndarray,
) -> dict[str, np.ndarray]:
    graph_distances = _graph_distances(graph)
    pending = list(missing)
    while pending:
        buildable = [
            name for name in pending if any(neighbour in coordinates for neighbour in graph[name])
        ]
        if not buildable:
            raise ModificationGeometryError(
                "Modified-residue template has heavy atoms disconnected from the parent residue: "
                + ", ".join(pending)
            )
        name = max(
            buildable,
            key=lambda item: sum(neighbour in coordinates for neighbour in graph[item]),
        )
        known_bonds = [neighbour for neighbour in graph[name] if neighbour in coordinates]
        anchor = known_bonds[0]
        target_length = next(
            target
            for first, second, target in bond_targets
            if {first, second} == {name, anchor}
        )
        basis = _local_basis(anchor, coordinates, graph)
        directions = _sphere_directions() @ basis.T
        candidates = coordinates[anchor][None, :] + target_length * directions
        scores = np.asarray([
            _initial_score(
                name,
                candidate,
                coordinates,
                graph_distances,
                bond_targets,
                angle_targets,
                environment,
            )
            for candidate in candidates
        ])
        coordinates[name] = candidates[int(np.argmin(scores))]
        pending.remove(name)
    return coordinates


def build_modified_heavy_atom_geometry(
    *,
    force_field: str,
    template: dict,
    retained_coordinates: dict[str, np.ndarray],
    environment_coordinates: np.ndarray | None = None,
    stereo_constraints: tuple[StereoConstraint, ...] = (),
) -> tuple[dict[str, np.ndarray], GeometryQuality]:
    """Build missing product heavy atoms while keeping retained atoms fixed."""
    parameters = _load_geometry_parameters(force_field.strip().lower())
    atom_types, bonds = _heavy_template(template)
    graph = _neighbours(atom_types, bonds)
    resolved_stereo = _resolve_stereo_constraints(stereo_constraints, atom_types)
    coordinates = {
        name: np.asarray(position, dtype=float).copy()
        for name, position in retained_coordinates.items()
        if name in atom_types
    }
    missing = [name for name in atom_types if name not in coordinates]
    if not missing:
        return coordinates, GeometryQuality((), 0.0, 0.0, None)
    if not coordinates:
        raise ModificationGeometryError("Modified residue has no retained heavy-atom anchor")

    bond_targets = [
        (first, second, parameters.bond(atom_types[first], atom_types[second]))
        for first, second in bonds
        if first in missing or second in missing
    ]
    angle_targets = [
        term for term in _angle_terms(graph, atom_types, parameters)
        if any(name in missing for name in term[:3])
    ]
    environment = np.asarray(
        environment_coordinates if environment_coordinates is not None else [],
        dtype=float,
    ).reshape((-1, 3))
    coordinates = _initialize_missing_atoms(
        missing,
        coordinates,
        graph,
        bond_targets,
        angle_targets,
        environment,
    )
    _seed_stereochemistry(coordinates, set(missing), resolved_stereo)

    initial = np.concatenate([coordinates[name] for name in missing])
    graph_distances = _graph_distances(graph)
    internal_clashes = [
        (first, second, _clash_threshold(first, second))
        for first, second in combinations(atom_types, 2)
        if (first in missing or second in missing)
        and graph_distances.get((first, second), 99) > 2
    ]

    def unpack(values: np.ndarray) -> dict[str, np.ndarray]:
        trial = dict(coordinates)
        reshaped = values.reshape((-1, 3))
        for index, name in enumerate(missing):
            trial[name] = reshaped[index]
        return trial

    def residuals(values: np.ndarray) -> np.ndarray:
        trial = unpack(values)
        result: list[float] = []
        for first, second, target in bond_targets:
            distance = float(np.linalg.norm(trial[first] - trial[second]))
            result.append((distance - target) / 0.003)
        for first, centre, third, target in angle_targets:
            result.append((_angle(trial[first], trial[centre], trial[third]) - target) / 2.0)
        for first, second, threshold in internal_clashes:
            distance = float(np.linalg.norm(trial[first] - trial[second]))
            result.append(max(0.0, threshold - distance) / 0.01)
        if environment.size:
            for name in missing:
                distances = np.linalg.norm(environment - trial[name], axis=1)
                result.extend((np.maximum(0.0, 0.17 - distances) / 0.01).tolist())
        for center, first, second, third, expected_sign, _label in resolved_stereo:
            if all(name in trial for name in (center, first, second, third)):
                oriented = expected_sign * _signed_volume(
                    trial, center, first, second, third
                )
                # Bond/angle targets determine the physical volume magnitude;
                # this one-sided residual only selects the documented enantiomer
                # and keeps the centre safely away from a planar ambiguity.
                result.append(max(0.0, 2.0e-4 - oriented) / 1.0e-4)
        # Resolve unconstrained torsional degeneracy near the local, clash-free
        # initializer without competing with bonded force-field targets.
        result.extend((1e-4 * (values - initial) / 0.1).tolist())
        return np.asarray(result, dtype=float)

    lower = np.tile(np.min(np.vstack(list(retained_coordinates.values())), axis=0) - 1.0, len(missing))
    upper = np.tile(np.max(np.vstack(list(retained_coordinates.values())), axis=0) + 1.0, len(missing))
    optimized = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        max_nfev=2000,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise ModificationGeometryError(
            "Local force-field geometry optimization did not converge"
        )
    coordinates = unpack(optimized.x)

    bond_errors = [
        abs(float(np.linalg.norm(coordinates[first] - coordinates[second])) - target)
        for first, second, target in bond_targets
    ]
    angle_errors = [
        abs(_angle(coordinates[first], coordinates[centre], coordinates[third]) - target)
        for first, centre, third, target in angle_targets
    ]
    nonbonded_distances = [
        float(np.linalg.norm(coordinates[first] - coordinates[second]))
        for first, second, _threshold in internal_clashes
    ]
    if environment.size:
        nonbonded_distances.extend(
            float(np.min(np.linalg.norm(environment - coordinates[name], axis=1)))
            for name in missing
        )
    quality = GeometryQuality(
        added_atoms=tuple(missing),
        max_bond_error_nm=max(bond_errors, default=0.0),
        max_angle_error_deg=max(angle_errors, default=0.0),
        min_nonbonded_distance_nm=min(nonbonded_distances, default=None),
        stereo_centres=tuple(label for *_atoms, label in resolved_stereo),
    )
    if quality.max_bond_error_nm > 0.015:
        raise ModificationGeometryError(
            f"Modified-residue bond geometry is outside tolerance "
            f"({quality.max_bond_error_nm:.4f} nm maximum error)"
        )
    if quality.max_angle_error_deg > 15.0:
        raise ModificationGeometryError(
            f"Modified-residue angle geometry is outside tolerance "
            f"({quality.max_angle_error_deg:.1f} degrees maximum error)"
        )
    if quality.min_nonbonded_distance_nm is not None and quality.min_nonbonded_distance_nm < 0.08:
        raise ModificationGeometryError(
            "Modified-residue geometry contains a heavy-atom overlap below 0.08 nm"
        )
    for center, first, second, third, expected_sign, label in resolved_stereo:
        oriented = expected_sign * _signed_volume(
            coordinates, center, first, second, third
        )
        if oriented < 2.0e-4:
            raise ModificationGeometryError(
                f"Modified-residue stereochemistry does not match {label}"
            )
    return coordinates, quality


def validate_modified_template_parameters(
    *,
    force_field: str,
    product_template: dict,
    parent_template: dict,
    stereo_constraints: tuple[StereoConstraint, ...] = (),
) -> tuple[str, ...]:
    """Fail if a native product lacks geometry targets for every new heavy atom."""
    parameters = _load_geometry_parameters(force_field.strip().lower())
    atom_types, bonds = _heavy_template(product_template)
    _resolve_stereo_constraints(stereo_constraints, atom_types)
    parent_types, _parent_bonds = _heavy_template(parent_template)
    missing = tuple(name for name in atom_types if name not in parent_types)
    if not missing:
        return ()
    graph = _neighbours(atom_types, bonds)
    for first, second in bonds:
        if first in missing or second in missing:
            parameters.bond(atom_types[first], atom_types[second])
    for first, centre, third, _target in _angle_terms(
        graph, atom_types, parameters
    ):
        if any(name in missing for name in (first, centre, third)):
            # _angle_terms has already performed the strict parameter lookup.
            continue
    return missing


def crosslink_bond_length(
    force_field: str,
    residue_template: dict,
    atom_name: str,
) -> float:
    """Return the native equilibrium length for a symmetric crosslink bond."""
    atoms = {
        str(atom[0]).strip(): str(atom[1]).strip()
        for atom in residue_template.get("atoms", [])
    }
    try:
        atom_type = atoms[atom_name]
    except KeyError as error:
        raise ModificationGeometryError(
            f"Crosslink residue has no {atom_name} atom"
        ) from error
    return _load_geometry_parameters(force_field.strip().lower()).bond(
        atom_type, atom_type
    )
