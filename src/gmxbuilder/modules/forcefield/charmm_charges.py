"""Partial charges read from the environments the installed release documents.

The previous model fitted bond charge increments to five molecules and required
them to reproduce the release exactly, to 1e-8 e. That gate passed only because
five molecules over eight bond features form an exactly determined system. Run
the same fit across the whole installed release and it fails outright: 570 bond
features over 23986 atom charges solves at full rank but leaves **RMSE 0.073 e,
worst case 0.78 e**, with only half of all atoms inside 0.01 e. CGenFF charges
are optimised against QM water-interaction energies, not generated from bond
increments, so no linear increment model can reproduce them and a gate demanding
that it do so can only ever be satisfied by keeping the domain tiny.

What does work is asking the release directly. Atoms sharing a neighbourhood
descriptor also share a charge, and sharply so once the description is wide
enough -- at radius 4, 95.1% of shared environments have a charge spread below
0.01 e and the median spread is exactly zero. So this reads the charge off the
widest environment the release documents for each atom and reports the spread
among the witnesses as that atom's uncertainty, rather than fitting anything.

Charges obtained this way do not sum to the molecule's formal charge, because
they come from different parent molecules. The shortfall is distributed and its
size reported: a large correction means the environments matched came from
chemistry unlike this molecule, and that is exactly when the answer should be
doubted.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from gmxbuilder.modules.forcefield import charmm_typing as ct

CHARGE_MODEL_ID = "installed-release-environment-charge-v1"

#: Charge spread, in e, above which a matched environment is reported as poorly
#: determined. Set at the measured 90th percentile of radius-3 spread, so it
#: flags the tail rather than routine variation.
SPREAD_WARNING_E = 0.02

#: Total correction, in e, above which the assignment is reported as unreliable.
#: A whole molecule's charges being off by more than this means the matched
#: environments came from chemistry that does not resemble the query.
MAX_TOTAL_CORRECTION_E = 0.5


@dataclass(frozen=True)
class ChargeAssignment:
    atom_index: int
    charge: float
    #: Standard deviation of the charge across corpus atoms sharing this
    #: environment. Zero when only one witness exists.
    spread: float
    radius: int
    support: int
    correction: float = 0.0

    @property
    def well_determined(self) -> bool:
        return self.support > 1 and self.spread <= SPREAD_WARNING_E


@dataclass
class ChargeIndex:
    """Mean charge per environment, at every radius, from the installed release."""

    force_field: str
    source_sha256: str
    #: radius -> descriptor -> (mean charge, spread, witness count)
    charges: dict[int, dict[bytes, tuple[float, float, int]]] = field(default_factory=dict)
    n_atoms: int = 0

    def report(self) -> dict:
        return {
            "model_id": CHARGE_MODEL_ID,
            "force_field": self.force_field,
            "source_sha256": self.source_sha256,
            "atoms_read": self.n_atoms,
            "environments": {r: len(m) for r, m in sorted(self.charges.items())},
            "method": "mean charge of corpus atoms sharing the widest matched environment",
            "fitted_coefficients": "none",
            "distributed_parameters": "none",
        }


_CACHE: dict[tuple[str, str], ChargeIndex] = {}


def build_index(force_field: str, *, exclude: frozenset[str] | None = None) -> ChargeIndex:
    """Index the charge every documented environment carries.

    ``exclude`` withholds residues so generalisation can be measured against
    molecules that did not build the index. Excluded builds are never cached.
    """
    from gmxbuilder.modules.forcefield.catalog import force_field_directory
    from gmxbuilder.modules.forcefield.charmm_compat import template_database
    from gmxbuilder.modules.forcefield.rtp_parser import RTPParser

    root = force_field_directory(force_field)
    if root is None:
        raise ct.TypingUnavailable(f"{force_field} is not installed")
    database = template_database(force_field)
    signature = ct._source_hash(root, database)
    if exclude is None:
        cached = _CACHE.get((force_field, signature))
        if cached is not None:
            return cached

    numbers = ct._atomic_numbers(root)
    parser = RTPParser(database)
    observed: dict[int, dict[bytes, list[float]]] = {r: {} for r in range(1, ct.MAX_RADIUS + 1)}
    n_atoms = 0
    for name in parser.residue_names:
        if exclude and name in exclude:
            continue
        record = parser.get_residue(name)
        if not record:
            continue
        built = ct._residue_graph(record, numbers)
        if built is None:
            continue
        graph, _types = built
        charges = [float(atom[2]) for atom in record["atoms"]]
        n_atoms += len(charges)
        current = ct.base_labels(graph)
        for radius in range(1, ct.MAX_RADIUS + 1):
            current = ct.descriptors(graph, 1, current)
            shell = observed[radius]
            for position, key in enumerate(current):
                shell.setdefault(key, []).append(charges[position])

    index = ChargeIndex(force_field=force_field, source_sha256=signature, n_atoms=n_atoms)
    for radius, shell in observed.items():
        index.charges[radius] = {
            key: (
                statistics.fmean(values),
                statistics.pstdev(values) if len(values) > 1 else 0.0,
                len(values),
            )
            for key, values in shell.items()
        }
    if exclude is None:
        _CACHE[(force_field, signature)] = index
    return index


def assign_charges(molecule, index: ChargeIndex, formal_charge: int):
    """Charge every atom from its widest documented environment.

    The widest rather than the narrowest match, unlike typing: a type only has
    to be unambiguous, but a charge gets sharper the more of the molecule the
    environment covers, so the most specific description available is the best
    available answer.
    """
    from gmxbuilder.modules.forcefield.charmm_compat import CharmmCompatError

    graph = ct.graph_from_rdkit(molecule)
    shells = []
    current = ct.base_labels(graph)
    for _ in range(ct.MAX_RADIUS):
        current = ct.descriptors(graph, 1, current)
        shells.append(current)

    raw: list[tuple[float, float, int, int]] = []
    for position in range(len(graph.elements)):
        best = None
        for radius in range(ct.MAX_RADIUS, 0, -1):
            found = index.charges[radius].get(shells[radius - 1][position])
            if found is not None:
                best = (found[0], found[1], radius, found[2])
                break
        if best is None:
            raise CharmmCompatError(
                "CHARGE_MODEL_MISSING",
                f"atom {position + 1} has an environment the installed release does not "
                "document, so no charge can be read for it; use CGenFF import",
            )
        raw.append(best)

    total = sum(entry[0] for entry in raw)
    shortfall = float(formal_charge) - total
    if abs(shortfall) > MAX_TOTAL_CORRECTION_E:
        raise CharmmCompatError(
            "CHARGE_CONSERVATION_FAILED",
            f"environment charges sum to {total:+.3f} e against a formal charge of "
            f"{formal_charge:+d} e; the matched environments do not describe this "
            "molecule closely enough to be corrected; use CGenFF import",
        )
    # Spread the shortfall evenly. Weighting it toward the least certain atoms
    # was considered and rejected: it hides the correction inside exactly the
    # atoms whose charges are already least trustworthy, and the size of the
    # correction is more useful reported than disguised.
    per_atom = shortfall / len(raw) if raw else 0.0
    assignments = [
        ChargeAssignment(position, charge + per_atom, spread, radius, support, per_atom)
        for position, (charge, spread, radius, support) in enumerate(raw)
    ]
    report = {
        "charge_method": "environment_lookup",
        "charge_model": index.report(),
        "total_before_correction_e": total,
        "formal_charge_e": int(formal_charge),
        "total_correction_e": shortfall,
        "correction_per_atom_e": per_atom,
        "max_environment_spread_e": max((a.spread for a in assignments), default=0.0),
        "poorly_determined_atoms": [a.atom_index + 1 for a in assignments if not a.well_determined],
        "spread_warning_threshold_e": SPREAD_WARNING_E,
        "physical_validation": "not_evaluated",
    }
    return [a.charge for a in assignments], assignments, report
