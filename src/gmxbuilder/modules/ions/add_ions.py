"""Module 7: neutralize a solvated system and add salt by replacing water."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gmxbuilder.core.component import Component
from gmxbuilder.core.enums import ComponentKind
from gmxbuilder.core.exceptions import ModuleConfigError
from gmxbuilder.core.structure import Structure
from gmxbuilder.core.system import System
from gmxbuilder.modules import register_module
from gmxbuilder.modules.ions.catalog import (
    KNOWN_ANIONS,
    KNOWN_CATIONS,
    ion_charge,
    supported_ions,
)
from gmxbuilder.modules.ions.neutralize import compute_net_charge
from gmxbuilder.modules.solvation.water_models import WaterRegistry
from gmxbuilder.pipeline.base import BaseModule, ModuleResult

_AVOGADRO_NM3 = 0.602214076
_WATER_VOLUME_NM3 = 0.0299
_METHODS = {"replace", "random", "mc"}


@dataclass(frozen=True)
class _WaterSite:
    oxygen_index: int
    atom_indices: tuple[int, ...]
    coordinate: np.ndarray


@register_module
class IonBuilder(BaseModule):
    """Add ions at solvent sites, removing each selected water molecule."""

    name = "ions"
    description = "Neutralize system charge and add salt ions"

    def validate_config(self, config: dict) -> bool:
        self.validate_config_keys(config, {
            "cations", "anions", "cation", "anion", "concentration",
            "neutralize", "neutralize_cation", "neutralize_anion",
            "ion_method", "exclusion_radius", "seed",
        })
        cations, anions, concentrations, neutralize, neut_cat, neut_ani = self._parse(config)
        if not cations or not anions:
            raise ModuleConfigError("At least one cation and one anion are required")
        if len(cations) != len(set(cations)) or len(anions) != len(set(anions)):
            raise ModuleConfigError("Each ion species may be selected only once")
        unknown_cat = sorted(set(cations) - KNOWN_CATIONS)
        unknown_ani = sorted(set(anions) - KNOWN_ANIONS)
        if unknown_cat:
            raise ModuleConfigError(f"Unknown or non-cation species: {', '.join(unknown_cat)}")
        if unknown_ani:
            raise ModuleConfigError(f"Unknown or non-anion species: {', '.join(unknown_ani)}")
        if neut_cat not in KNOWN_CATIONS:
            raise ModuleConfigError(f"Invalid neutralizing cation: {neut_cat}")
        if neut_ani not in KNOWN_ANIONS:
            raise ModuleConfigError(f"Invalid neutralizing anion: {neut_ani}")
        for name in cations + anions:
            value = concentrations.get(name)
            if value is None or not math.isfinite(value) or value < 0 or value > 2.0:
                raise ModuleConfigError(f"Concentration for {name} must be finite and between 0 and 2 M")
        continuous_charge = sum(concentrations[name] * ion_charge(name) for name in cations + anions)
        if abs(continuous_charge) > 1e-8:
            raise ModuleConfigError(
                "Salt concentrations are not charge-balanced: "
                f"Σ(c×z)={continuous_charge:+.6g} M·e. "
                "For example, 0.15 M CaCl2 requires 0.15 M CA and 0.30 M CL."
            )
        method = str(config.get("ion_method", "random")).strip().lower()
        if method not in _METHODS:
            raise ModuleConfigError(f"Unknown ion placement method: {method}")
        exclusion = self._finite_float(config.get("exclusion_radius", 0.35), "exclusion_radius")
        if not 0.1 <= exclusion <= 1.0:
            raise ModuleConfigError("exclusion_radius must be between 0.1 and 1.0 nm")
        if not isinstance(neutralize, bool):
            raise ModuleConfigError("neutralize must be true or false")
        return True

    @staticmethod
    def _finite_float(value, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ModuleConfigError(f"{label} must be a number") from exc
        if not math.isfinite(number):
            raise ModuleConfigError(f"{label} must be finite")
        return number

    def _parse(self, config: dict) -> tuple[list[str], list[str], dict[str, float], bool, str, str]:
        raw_cations = config.get("cations", [config.get("cation", "NA")])
        raw_anions = config.get("anions", [config.get("anion", "CL")])
        if not isinstance(raw_cations, list) or not isinstance(raw_anions, list):
            raise ModuleConfigError("cations and anions must be lists")
        cations = [str(item).strip().upper() for item in raw_cations]
        anions = [str(item).strip().upper() for item in raw_anions]
        raw_conc = config.get("concentration", 0.15)
        if isinstance(raw_conc, dict):
            concentrations = {
                str(key).strip().upper(): self._finite_float(value, f"concentration[{key}]")
                for key, value in raw_conc.items()
            }
        else:
            value = self._finite_float(raw_conc, "concentration")
            concentrations = {name: value for name in cations + anions}
        return (
            cations,
            anions,
            concentrations,
            config.get("neutralize", True),
            str(config.get("neutralize_cation", cations[0] if cations else "NA")).strip().upper(),
            str(config.get("neutralize_anion", anions[0] if anions else "CL")).strip().upper(),
        )

    def run(self, system: System, config: dict) -> ModuleResult:
        self.validate_config(config)
        cations, anions, concentrations, neutralize, neut_cat, neut_ani = self._parse(config)
        system, pre_topology_log = self._release_validated_crosslink_topology(system)
        force_field = str(system.metadata.get("force_field", "amber14sb"))
        water_model_name = str(system.metadata.get("water_model", "tip3p")).lower()
        solute_charge = compute_net_charge(system)
        rounded_charge = round(solute_charge)
        if abs(solute_charge - rounded_charge) > 1e-6:
            raise ModuleConfigError(
                f"Solute formal charge is non-integral ({solute_charge:+.6f} e); "
                "review protonation and ligand formal charges before adding ions"
            )
        available = supported_ions(force_field, water_model_name)
        requested = set(cations + anions)
        if neutralize and rounded_charge > 0:
            requested.add(neut_ani)
        elif neutralize and rounded_charge < 0:
            requested.add(neut_cat)
        missing = sorted(requested - available)
        if missing:
            raise ModuleConfigError(
                f"Ion(s) {', '.join(missing)} are not defined by {force_field} "
                f"with {water_model_name.upper()} water"
            )
        sites, water_model = self._water_sites(system, water_model_name)
        if not sites:
            raise ModuleConfigError("No complete solvent water molecules are available for ion replacement")
        n_water = len(sites)
        water_volume = n_water * _WATER_VOLUME_NM3
        salt_counts = {
            name: round(concentrations[name] * _AVOGADRO_NM3 * water_volume)
            for name in cations + anions
        }
        self._balance_rounded_salt(salt_counts, cations, anions)

        neutralizing_counts: dict[str, int] = {}
        if neutralize and rounded_charge:
            counterion = neut_ani if rounded_charge > 0 else neut_cat
            valence = abs(ion_charge(counterion))
            if abs(rounded_charge) % valence:
                raise ModuleConfigError(
                    f"{counterion} ({ion_charge(counterion):+d}) cannot exactly neutralize "
                    f"a {rounded_charge:+d} e solute; choose a monovalent counterion"
                )
            neutralizing_counts[counterion] = abs(rounded_charge) // valence

        counts = dict(salt_counts)
        for name, count in neutralizing_counts.items():
            counts[name] = counts.get(name, 0) + count
        total_ions = sum(counts.values())
        if total_ions == 0:
            result = system.copy()
            result.metadata["ions"] = self._ion_metrics(
                salt_counts, neutralizing_counts, counts, concentrations,
                solute_charge, 0, 0, config,
            )
            return ModuleResult(
                True, result,
                pre_topology_log + ["No ions requested; system left unchanged"],
            )
        if total_ions > n_water:
            raise ModuleConfigError(f"Requested {total_ions} ions but only {n_water} waters are available")

        exclusion = float(config.get("exclusion_radius", 0.35))
        z_regions = self._water_regions(system, sites)
        eligible = self._eligible_sites(system, sites, z_regions, exclusion)
        if len(eligible) < total_ions:
            raise ModuleConfigError(
                f"Only {len(eligible)} water sites satisfy the {exclusion:.2f} nm exclusion "
                f"and membrane-water constraints; {total_ions} ions were requested"
            )
        rng = np.random.default_rng(system.metadata.get("seed", config.get("seed", 42)))
        method = str(config.get("ion_method", "random")).lower()
        chosen = self._select_sites(system, eligible, counts, method, rng, exclusion)
        # _select_sites returns all cation sites followed by all anion sites.
        # Preserve the same order when assigning chemical identities, including
        # a neutralizing species that was not part of the salt selection.
        ion_names = [
            name for name, count in counts.items() if ion_charge(name) > 0
            for _ in range(count)
        ] + [
            name for name, count in counts.items() if ion_charge(name) < 0
            for _ in range(count)
        ]
        if len(chosen) != len(ion_names):
            raise ModuleConfigError("Ion placement did not produce every requested ion site")

        remove_indices = sorted({idx for site in chosen for idx in site.atom_indices})
        stripped = self._remove_atoms(system, remove_indices, len(chosen))
        ion_structure = Structure(
            coordinates=np.asarray([site.coordinate for site in chosen], dtype=np.float64),
            box_vectors=stripped.structure.box_vectors.copy(),
            atom_names=ion_names,
            resnames=ion_names,
            resids=list(range(1, total_ions + 1)),
            elements=[name[0] + name[1:].lower() for name in ion_names],
        )
        merged = stripped.merge(System(structure=ion_structure))
        ion_start = stripped.num_atoms
        metrics = self._ion_metrics(
            salt_counts, neutralizing_counts, counts, concentrations,
            solute_charge, len(chosen), len(remove_indices), config,
        )
        merged.add_component(Component(
            name="IONS_ADDED",
            kind=ComponentKind.IONS,
            atom_indices=np.arange(ion_start, merged.num_atoms, dtype=int),
            metadata=metrics,
        ))
        merged.metadata["ions"] = metrics
        ion_charge_total = sum(count * ion_charge(name) for name, count in counts.items())
        if neutralize and abs(solute_charge + ion_charge_total) > 1e-6:
            raise ModuleConfigError("Internal ion-count error: final system is not neutral")
        log = pre_topology_log + [
            f"Solute formal charge: {solute_charge:+.0f} e",
            f"Salt ion counts: {salt_counts}",
            f"Neutralizing ion counts: {neutralizing_counts}",
            f"Replaced {len(chosen)} complete {water_model.full_name} waters with {len(chosen)} ions",
            f"Final formal charge: {solute_charge + ion_charge_total:+.0f} e",
        ]
        return ModuleResult(True, merged, log)

    @staticmethod
    def _release_validated_crosslink_topology(system: System) -> tuple[System, list[str]]:
        """Discard only the Step-3 crosslink stub before atom replacement.

        Structure Processing records validated disulfides both as authoritative
        metadata and as a minimal bond-only topology for its own checkpoint.
        Final topology assignment reconstructs those bonds from the metadata.
        Ion placement may therefore release that narrow stub before removing
        waters, but must still reject a complete or unrecognised topology.
        """
        topology = system.topology
        if topology is None:
            return system, []
        has_non_stub_terms = any((
            topology.atom_types,
            topology.angles,
            topology.dihedrals,
            topology.impropers,
            topology.pairs,
            topology.exclusions,
            topology.molecule_blocks,
        ))
        crosslinks = system.metadata.get("crosslinks")
        if has_non_stub_terms or not topology.bonds or not isinstance(crosslinks, list):
            raise ModuleConfigError(
                "Ions must be added before final topology assignment"
            )

        expected_pairs: set[tuple[int, int]] = set()
        structure = system.structure
        for record in crosslinks:
            if (
                not isinstance(record, dict)
                or record.get("type") != "disulfide"
                or record.get("status") != "passed"
            ):
                raise ModuleConfigError(
                    "Ion placement found an unvalidated crosslink topology"
                )
            endpoints: list[int] = []
            for label in ("first", "second"):
                endpoint = record.get(label)
                if not isinstance(endpoint, dict):
                    raise ModuleConfigError(
                        "Ion placement found incomplete disulfide metadata"
                    )
                try:
                    resid = int(endpoint["resid"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ModuleConfigError(
                        "Ion placement found invalid disulfide residue metadata"
                    ) from exc
                chain = str(endpoint.get("chain", ""))
                matches = [
                    index for index, (atom_chain, atom_resid, atom_name) in enumerate(zip(
                        structure.chain_ids, structure.resids, structure.atom_names
                    ))
                    if str(atom_chain) == chain
                    and int(atom_resid) == resid
                    and str(atom_name).strip().upper() == "SG"
                ]
                if len(matches) != 1:
                    raise ModuleConfigError(
                        "Ion placement could not resolve a validated disulfide SG atom"
                    )
                endpoints.append(matches[0])
            expected_pairs.add(tuple(sorted(endpoints)))
        observed_pairs = {tuple(sorted((bond.i, bond.j))) for bond in topology.bonds}
        if observed_pairs != expected_pairs:
            raise ModuleConfigError(
                "Ion placement found topology bonds not represented by validated crosslinks"
            )
        released = system.copy()
        released.topology = None
        return released, [
            (
                "Validated disulfide topology stub retained through authoritative "
                "crosslink metadata until final topology assignment"
            )
        ]

    @staticmethod
    def _balance_rounded_salt(counts: dict[str, int], cations: list[str], anions: list[str]) -> None:
        charge = sum(count * ion_charge(name) for name, count in counts.items())
        if charge > 0:
            counts[anions[0]] += charge  # supported anions are monovalent
        elif charge < 0:
            remove = min(counts[anions[0]], -charge)
            counts[anions[0]] -= remove
            charge += remove
            if charge < 0:
                counts[cations[0]] += math.ceil(-charge / ion_charge(cations[0]))
        final = sum(count * ion_charge(name) for name, count in counts.items())
        if final != 0:
            raise ModuleConfigError("Rounded salt counts cannot be made charge-neutral")

    @staticmethod
    def _water_sites(system: System, water_model_name: str) -> tuple[list[_WaterSite], object]:
        water_model = WaterRegistry.get(water_model_name)
        sites: list[_WaterSite] = []
        for comp in system.component_by_kind(ComponentKind.SOLVENT):
            indices = [int(item) for item in comp.atom_indices]
            n_molecules = int(comp.metadata.get("n_molecules", 0))
            if len(indices) != n_molecules * water_model.n_atoms:
                raise ModuleConfigError(
                    f"Solvent component {comp.name} has {len(indices)} atoms but metadata "
                    f"declares {n_molecules} {water_model.full_name} waters"
                )
            for offset in range(0, len(indices), water_model.n_atoms):
                molecule = tuple(indices[offset:offset + water_model.n_atoms])
                oxygen = next(
                    (idx for idx in molecule if system.structure.atom_names[idx].strip().upper().startswith("O")),
                    None,
                )
                if oxygen is None:
                    raise ModuleConfigError(f"Water molecule at solvent offset {offset} has no oxygen atom")
                sites.append(_WaterSite(oxygen, molecule, system.coordinates[oxygen].copy()))
        return sites, water_model

    @staticmethod
    def _water_regions(system: System, sites: list[_WaterSite]) -> list[tuple[float, float]]:
        z = np.asarray([site.coordinate[2] for site in sites])
        regions = [(float(z.min()), float(z.max()))]
        membranes = system.component_by_kind(ComponentKind.MEMBRANE)
        if membranes:
            indices = np.concatenate([comp.atom_indices for comp in membranes]).astype(int)
            membrane_z = system.coordinates[indices, 2]
            lower, upper = float(membrane_z.min() - 0.4), float(membrane_z.max() + 0.4)
            regions = []
            if z.min() < lower:
                regions.append((float(z.min()), lower))
            if z.max() > upper:
                regions.append((upper, float(z.max())))
        return regions

    @staticmethod
    def _eligible_sites(
        system: System,
        sites: list[_WaterSite],
        regions: list[tuple[float, float]],
        exclusion: float,
    ) -> list[_WaterSite]:
        from scipy.spatial import cKDTree

        solute_indices: list[int] = []
        for comp in system.components:
            if comp.kind not in (ComponentKind.SOLVENT, ComponentKind.IONS):
                solute_indices.extend(int(item) for item in comp.atom_indices)
        box = np.asarray(system.structure.dimensions(), dtype=float)
        if not np.isfinite(box).all() or np.any(box <= 0.0):
            raise ModuleConfigError("Ion placement requires positive periodic box dimensions")
        tree = (
            cKDTree(np.mod(system.coordinates[solute_indices], box), boxsize=box)
            if solute_indices else None
        )
        eligible: list[_WaterSite] = []
        for site in sites:
            if regions and not any(lo <= site.coordinate[2] <= hi for lo, hi in regions):
                continue
            if tree is not None and tree.query(np.mod(site.coordinate, box), k=1)[0] < exclusion:
                continue
            eligible.append(site)
        return eligible

    def _select_sites(
        self,
        system: System,
        sites: list[_WaterSite],
        counts: dict[str, int],
        method: str,
        rng: np.random.Generator,
        exclusion: float,
    ) -> list[_WaterSite]:
        total_cations = sum(count for name, count in counts.items() if ion_charge(name) > 0)
        total_anions = sum(count for name, count in counts.items() if ion_charge(name) < 0)
        placement_charges = [
            ion_charge(name) for name, count in counts.items() if ion_charge(name) > 0
            for _ in range(count)
        ] + [
            ion_charge(name) for name, count in counts.items() if ion_charge(name) < 0
            for _ in range(count)
        ]
        potentials = None
        if method == "random":
            # GROMACS genion-style method: uniformly sample complete solvent
            # molecules using a reproducible seed.
            cation_order = rng.permutation(len(sites)).tolist()
            anion_order = rng.permutation(len(sites)).tolist()
        else:
            potentials = self._site_potentials(system, sites)
            cation_order = np.argsort(potentials).tolist()
            anion_order = list(reversed(cation_order))
        chosen_indices: list[int] = []
        box = np.asarray(system.structure.dimensions(), dtype=float)

        def distances_to_chosen(index: int, ignored_position: int | None = None) -> np.ndarray:
            retained = [
                chosen for position, chosen in enumerate(chosen_indices)
                if position != ignored_position
            ]
            if not retained:
                return np.empty(0, dtype=float)
            delta = np.asarray([sites[item].coordinate for item in retained]) - sites[index].coordinate
            delta -= box * np.round(delta / box)
            return np.linalg.norm(delta, axis=1)

        def take(candidate_order: list[int], needed: int) -> None:
            if needed <= 0:
                return
            target = len(chosen_indices) + needed
            for index in candidate_order:
                if index in chosen_indices:
                    continue
                distances = distances_to_chosen(index)
                if len(distances) and float(distances.min()) < exclusion:
                    continue
                chosen_indices.append(index)
                if len(chosen_indices) == target:
                    return
            raise ModuleConfigError(
                f"Could not place every ion at least {exclusion:.2f} nm apart; "
                "reduce concentration or exclusion radius"
            )

        take(cation_order, total_cations)
        take(anion_order, total_anions)

        if method == "mc" and chosen_indices:
            # True Metropolis water-site sampling. The initial configuration is
            # electrostatically favourable; trial moves replace one chosen water
            # with another eligible water and are accepted with exp(-ΔE/T).
            # Energy combines normalized solute potential with a weak screened
            # ion-ion term, while the hard periodic exclusion is always enforced.
            potential_scale = float(np.std(potentials))
            if potential_scale < 1e-8:
                potential_scale = 1.0
            scaled_potential = (potentials - float(np.median(potentials))) / potential_scale
            used = set(chosen_indices)
            n_steps = max(2000, min(20000, len(chosen_indices) * 50))

            def local_energy(position: int, site_index: int) -> float:
                charge = placement_charges[position]
                energy = charge * float(scaled_potential[site_index])
                other_positions = [
                    other for other in range(len(chosen_indices)) if other != position
                ]
                if other_positions:
                    other_indices = [chosen_indices[other] for other in other_positions]
                    delta = (
                        np.asarray([sites[item].coordinate for item in other_indices])
                        - sites[site_index].coordinate
                    )
                    delta -= box * np.round(delta / box)
                    distances = np.maximum(np.linalg.norm(delta, axis=1), exclusion)
                    other_charges = np.asarray(
                        [placement_charges[other] for other in other_positions], dtype=float
                    )
                    energy += 0.05 * float(np.sum(
                        charge * other_charges * np.exp(-distances) / distances
                    ))
                return energy

            for _step in range(n_steps):
                position = int(rng.integers(len(chosen_indices)))
                candidate = int(rng.integers(len(sites)))
                if candidate in used:
                    continue
                distances = distances_to_chosen(candidate, ignored_position=position)
                if len(distances) and float(distances.min()) < exclusion:
                    continue
                current = chosen_indices[position]
                delta_energy = (
                    local_energy(position, candidate) - local_energy(position, current)
                )
                if delta_energy <= 0.0 or float(rng.random()) < math.exp(-delta_energy):
                    used.remove(current)
                    used.add(candidate)
                    chosen_indices[position] = candidate

        return [sites[index] for index in chosen_indices]

    @staticmethod
    def _site_potentials(system: System, sites: list[_WaterSite]) -> np.ndarray:
        from gmxbuilder.modules.membrane.lipids import LipidRegistry

        centers: list[np.ndarray] = []
        charges: list[float] = []
        for comp in system.components:
            if comp.kind == ComponentKind.PROTEIN:
                seen: set[tuple[str, int, str]] = set()
                for index in comp.atom_indices:
                    i = int(index)
                    key = (
                        system.structure.resnames[i], system.structure.resids[i],
                        system.structure.chain_ids[i],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    atom_indices = [
                        int(j) for j in comp.atom_indices
                        if (system.structure.resnames[int(j)], system.structure.resids[int(j)],
                            system.structure.chain_ids[int(j)]) == key
                    ]
                    ca = next((j for j in atom_indices if system.structure.atom_names[j].strip() == "CA"), atom_indices[0])
                    charge = system.residue_formal_charge(key[0])
                    if charge:
                        centers.append(system.coordinates[ca])
                        charges.append(charge)
            elif comp.kind == ComponentKind.MEMBRANE:
                grouped: dict[tuple[str, int, str], list[int]] = {}
                for index in comp.atom_indices:
                    i = int(index)
                    key = (system.structure.resnames[i], system.structure.resids[i], system.structure.chain_ids[i])
                    grouped.setdefault(key, []).append(i)
                for (name, _rid, _chain), indices in grouped.items():
                    try:
                        charge = float(LipidRegistry.get(name.strip().upper()).charge)
                    except (KeyError, ValueError):
                        charge = 0.0
                    if charge:
                        heads = [i for i in indices if system.structure.atom_names[i].strip() in {"P", "N"}]
                        centers.append(system.coordinates[heads or indices].mean(axis=0))
                        charges.append(charge)
            elif comp.kind == ComponentKind.NUCLEIC_ACID:
                net_charge = comp.metadata.get("net_charge")
                if not comp.metadata.get("prepared") or not isinstance(
                    net_charge, (int, float)
                ) or isinstance(net_charge, bool):
                    raise ModuleConfigError(
                        "Monte Carlo ion placement requires an exact prepared "
                        "nucleic-acid topology and net charge"
                    )
                grouped: dict[tuple[int, str], list[int]] = {}
                for raw_index in comp.atom_indices:
                    index = int(raw_index)
                    key = (
                        int(system.structure.resids[index]),
                        str(system.structure.chain_ids[index]),
                    )
                    grouped.setdefault(key, []).append(index)
                if not grouped:
                    continue
                charge_per_residue = float(net_charge) / len(grouped)
                for indices in grouped.values():
                    marker = next(
                        (
                            index for index in indices
                            if system.structure.atom_names[index].strip() == "P"
                        ),
                        None,
                    )
                    centers.append(
                        system.coordinates[marker]
                        if marker is not None
                        else system.coordinates[indices].mean(axis=0)
                    )
                    charges.append(charge_per_residue)
        if not centers:
            return np.zeros(len(sites))
        center_array = np.asarray(centers)
        charge_array = np.asarray(charges)
        box = np.asarray(system.structure.dimensions(), dtype=float)
        potentials = np.empty(len(sites), dtype=float)
        # Preserve site and reduction order. Benchmarking showed Python-level
        # threading is slower for this small per-site kernel; ion selection
        # remains deterministic and serial.
        for index, site in enumerate(sites):
            delta = center_array - site.coordinate
            delta -= box * np.round(delta / box)
            distance = np.maximum(np.linalg.norm(delta, axis=1), 0.15)
            potentials[index] = np.sum(charge_array / distance)
        return potentials

    @staticmethod
    def _remove_atoms(system: System, remove_indices: list[int], waters_removed: int) -> System:
        remove = np.zeros(system.num_atoms, dtype=bool)
        remove[remove_indices] = True
        keep = ~remove
        old_to_new = np.full(system.num_atoms, -1, dtype=int)
        old_to_new[keep] = np.arange(int(keep.sum()))
        source = system.structure

        def subset(values):
            return np.asarray(values)[keep].tolist()

        structure = Structure(
            coordinates=source.coordinates[keep].copy(),
            box_vectors=source.box_vectors.copy(),
            atom_names=subset(source.atom_names),
            resnames=subset(source.resnames),
            resids=subset(source.resids),
            chain_ids=subset(source.chain_ids),
            segids=subset(source.segids),
            elements=subset(source.elements),
            occupancies=subset(source.occupancies),
            tempfactors=subset(source.tempfactors),
        )
        components: list[Component] = []
        remaining = waters_removed
        removed_set = set(remove_indices)
        for comp in system.components:
            retained = [int(i) for i in comp.atom_indices if int(i) not in removed_set]
            metadata = dict(comp.metadata)
            if comp.kind == ComponentKind.SOLVENT:
                removed_here = len(comp.atom_indices) - len(retained)
                site_count = WaterRegistry.get(str(metadata.get("water_model", system.metadata.get("water_model", "tip3p"))).lower()).n_atoms
                if removed_here % site_count:
                    raise ModuleConfigError("Ion replacement would leave a partial water molecule")
                metadata["n_molecules"] = int(metadata.get("n_molecules", 0)) - removed_here // site_count
                remaining -= removed_here // site_count
            components.append(Component(
                name=comp.name,
                kind=comp.kind,
                atom_indices=old_to_new[retained],
                metadata=metadata,
            ))
        if remaining != 0:
            raise ModuleConfigError("Water-removal accounting mismatch")
        return System(structure=structure, components=components, metadata=dict(system.metadata))

    @staticmethod
    def _ion_metrics(
        salt_counts: dict[str, int],
        neutralizing_counts: dict[str, int],
        total_counts: dict[str, int],
        concentrations: dict[str, float],
        solute_charge: float,
        waters_replaced: int,
        water_atoms_removed: int,
        config: dict,
    ) -> dict:
        ion_charge_total = sum(count * ion_charge(name) for name, count in total_counts.items())
        return {
            "salt_counts": dict(salt_counts),
            "neutralizing_counts": dict(neutralizing_counts),
            "total_counts": dict(total_counts),
            "concentrations_m": dict(concentrations),
            "solute_charge_e": float(solute_charge),
            "ion_charge_e": float(ion_charge_total),
            "final_charge_e": float(solute_charge + ion_charge_total),
            "waters_replaced": int(waters_replaced),
            "water_atoms_removed": int(water_atoms_removed),
            "placement_method": str(config.get("ion_method", "random")),
            "placement_strategy": {
                "random": "uniform_random_water_replacement",
                "replace": "periodic_electrostatic_water_replacement",
                "mc": "metropolis_water_site_sampling",
            }.get(str(config.get("ion_method", "random")).lower(), "unknown"),
            "exclusion_radius_nm": float(config.get("exclusion_radius", 0.35)),
        }
