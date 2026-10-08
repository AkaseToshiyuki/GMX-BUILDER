"""Stream actual XTC coordinates, measure slow variables and retain provenance."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from mdtraj.formats import XTCTrajectoryFile

from gmxbuilder.geometry.molecular_identity import validate_stereochemistry
from gmxbuilder.modules.membrane.lipid_ensemble import (
    orientation_population,
    orientation_summary,
)
from gmxbuilder.modules.membrane.lipid_graph import ordered_graph, whole_batch
from gmxbuilder.modules.membrane.lipid_orientation import (
    MAX_TAIL_CORE_GAP_NM,
)
from gmxbuilder.modules.membrane.lipids import LipidRegistry
from gmxbuilder.modules.membrane.local_conformations import (
    canonical_coordinates,
    definition_for,
    population_descriptors,
    validate_intrinsic_geometry,
)
from gmxbuilder.modules.membrane.local_relaxation import assess_local
from gmxbuilder.modules.membrane.local_relaxation import evidence as local_evidence
from gmxbuilder.modules.membrane.v4_measurement import assess_measured_series


class TrajectorySampler:
    def __init__(self, structure, records, resname_map, protocol, force_field, lipid_ff):
        from gmxbuilder.modules.membrane.v4_atom_selection import chemical_selection
        from gmxbuilder.modules.membrane.v4_comparison import system_composition

        self.system_composition = system_composition(structure, resname_map)
        self.protocol = protocol
        self.n_leaflet = protocol["lipids_per_leaflet"]
        self.groups = {}
        for molecule_index, (indices, upper) in enumerate(records):
            raw = str(structure.resnames[indices[0]]).strip().upper()
            name = resname_map.get(raw, raw)
            names = tuple(str(structure.atom_names[i]).strip() for i in indices)
            key = (name, names)
            self.groups.setdefault(key, []).append((indices, upper, molecule_index))
        prepared = []
        composition = {"upper": {}, "lower": {}}
        for (name, names), molecules in self.groups.items():
            lipid = LipidRegistry.get(name)
            elements, bonds = ordered_graph(name, force_field, lipid_ff, lipid.smiles, names)
            indices = np.array([v[0] for v in molecules])
            flags = np.array([v[1] for v in molecules])
            selection = chemical_selection(name, names, elements, bonds)
            prepared.append(
                {
                    "name": name,
                    "names": names,
                    "smiles": lipid.smiles,
                    "elements": elements,
                    "bonds": bonds,
                    "indices": indices,
                    "upper": flags,
                    "molecule_ids": [v[2] for v in molecules],
                    "polar": selection["polar"],
                    "tails": selection["tails"],
                    "anchor": selection["anchor"],
                    "selection": selection["record"],
                }
            )
            for label, flag in (("upper", True), ("lower", False)):
                composition[label][name] = int(np.sum(flags == flag))
        expected = {k: int(self.n_leaflet * v / 100) for k, v in protocol["composition"].items()}
        if any(counts != expected for counts in composition.values()):
            raise ValueError(f"Actual leaflet composition {composition} differs from {expected}")
        self.groups = prepared
        self.composition = composition
        targets = [group for group in self.groups if group["name"] == protocol["lipid"]]
        if len(targets) != 1:
            raise ValueError("Local-conformer analysis requires one exact target atom order")
        self.target_group = targets[0]
        water_names = {"SOL", "WAT", "TIP3", "TIP3P", "HOH", "T3P"}
        self.water_oxygens = np.array(
            [
                index
                for index, (residue, atom) in enumerate(
                    zip(structure.resnames, structure.atom_names, strict=True)
                )
                if str(residue).strip().upper() in water_names
                and str(atom).strip().upper().startswith("O")
            ],
            dtype=int,
        )

    @property
    def local_definition(self):
        return definition_for(self.target_group)

    def frames(self, path, *, include_hydration=False):
        # Raw XTC is retained at 10 ps. Structural diagnostics at 100 ps avoid
        # treating adjacent saved coordinates as independent configurations.
        with XTCTrajectoryFile(str(path)) as stream:
            while True:
                xyz, times, _steps, boxes = stream.read(n_frames=32, stride=10)
                if not len(times):
                    break
                for frame, time, box in zip(xyz, times, boxes, strict=True):
                    blocks = [
                        whole_batch(frame[g["indices"]], g["bonds"], box) for g in self.groups
                    ]
                    if include_hydration:
                        yield float(time), box, blocks, self.hydration(frame, box)
                    else:
                        yield float(time), box, blocks

    def hydration(self, frame, box):
        """Mean water-O contacts per polar atom in each target molecule."""
        from scipy.spatial import cKDTree

        from gmxbuilder.modules.membrane.local_relaxation import construction_policy
        from gmxbuilder.runtime.hardware import query_workers

        if not len(self.water_oxygens):
            raise ValueError("Local relaxation requires explicit source water oxygens")
        box = np.asarray(box, dtype=float)
        dimensions = np.diag(box)
        if np.any(dimensions <= 0) or not np.allclose(box, np.diag(dimensions), atol=1e-6):
            raise ValueError("Source hydration analysis requires the orthorhombic membrane box")
        group = self.target_group
        heads = frame[group["indices"]][:, group["polar"]]
        tree = cKDTree(np.mod(frame[self.water_oxygens], dimensions), boxsize=dimensions)
        counts = tree.query_ball_point(
            np.mod(heads.reshape(-1, 3), dimensions),
            construction_policy()["hydration_cutoff_nm"],
            return_length=True,
            workers=query_workers(heads.shape[0] * heads.shape[1]),
        )
        return counts.reshape(heads.shape[:2]).mean(axis=1)

    def metrics(self, box, blocks, *, allow_unmeasurable=False):
        from gmxbuilder.modules.membrane.lipid_equilibration import (
            BilayerThicknessError,
            head_to_head_distance,
        )

        head_z, offsets, flags, projections, cosines, tails = [], [], [], [], [], []
        molecule_names = []
        species_orientation = {}
        metrics = {
            "area_per_lipid_nm2": float(np.linalg.norm(np.cross(box[0], box[1])) / self.n_leaflet),
            "box_z_nm": float(box[2, 2]),
            "volume_nm3": float(np.linalg.det(box)),
        }
        for group, xyz in zip(self.groups, blocks, strict=True):
            head = xyz[:, group["polar"]].mean(axis=1)
            tail = xyz[:, group["tails"]].mean(axis=1)
            vector = head - tail
            cosine = vector[:, 2] / np.maximum(np.linalg.norm(vector, axis=1), 1e-12)
            sign = np.where(group["upper"], 1.0, -1.0)
            molecule_names.extend([group["name"]] * len(xyz))
            projections.extend(vector[:, 2] * sign)
            cosines.extend(cosine * sign)
            head_z.extend(xyz[:, group["anchor"], 2])
            offsets.extend(tail[:, 2] - xyz[:, group["anchor"], 2])
            flags.extend(group["upper"])
            tails.extend(xyz[:, group["tails"], 2] - xyz[:, group["anchor"], 2, None])
            # Molecular-axis P2 is explicitly not a segmental deuterium S_CD.
            for label, flag in (("upper", True), ("lower", False)):
                selected = group["upper"] == flag
                species_orientation[f"{group['name']}_{label}"] = orientation_summary(
                    vector[selected, 2] * sign[selected], cosine[selected] * sign[selected]
                )
                metrics[f"{group['name']}_{label}_axis_P2"] = float(
                    np.mean((3 * cosine[group["upper"] == flag] ** 2 - 1) / 2)
                )
        flags = np.array(flags)
        measurement_error = None
        try:
            dhh = head_to_head_distance(
                np.array(head_z), np.array(offsets), flags, float(box[2, 2])
            )
        except BilayerThicknessError as exc:
            if not allow_unmeasurable:
                raise
            measurement_error = {**exc.evidence, "reason": str(exc)}
            dhh = gap = None
        else:
            upper = np.concatenate(
                [z + dhh / 2 for z, flag in zip(tails, flags, strict=True) if flag]
            )
            lower = np.concatenate(
                [z - dhh / 2 for z, flag in zip(tails, flags, strict=True) if not flag]
            )
            gap = float(np.percentile(upper, 1) - np.percentile(lower, 99))
        metrics["head_to_head_nm"] = dhh
        host = np.array(molecule_names) != self.protocol["lipid"]
        projections, cosines = np.asarray(projections), np.asarray(cosines)
        gate_projection, gate_cosine, profile = orientation_population(
            self.protocol["lipid"], projections, cosines, projections[host], cosines[host]
        )
        gate = orientation_summary(gate_projection, gate_cosine)
        valid = bool(
            measurement_error is None
            and gate["passed"]
            and np.isfinite(gap)
            and gap <= MAX_TAIL_CORE_GAP_NM
        )
        # Species/leaflet outliers are visible without quietly inventing a new
        # acceptance threshold for only twenty guest molecules per leaflet.
        return (
            metrics,
            valid,
            {
                "oriented_fraction": gate["correct_fraction"],
                "core_gap_nm": gap,
                "mixture_head_plane_nm": dhh,
                **({"measurement_error": measurement_error} if measurement_error else {}),
                "orientation": {
                    "profile": profile,
                    "gate": gate,
                    "species_leaflet": species_orientation,
                },
            },
        )

    def analyse(self, path, work, *, expected_end_ps=None):
        with XTCTrajectoryFile(str(path)) as stream:
            if not len(stream):
                raise ValueError("V4 requires a nonempty coordinate trajectory")
            stream.seek(len(stream) - 1)
            _, final_times, _, _ = stream.read(n_frames=1)
        actual_end = float(final_times[0])
        if expected_end_ps is not None and abs(actual_end - expected_end_ps) > 0.01:
            raise ValueError("Coordinate trajectory does not reach the recorded NPT end time")
        times, rows, valid, geometry, local_rows, errors = [], [], [], [], [], []
        target_index = next(i for i, g in enumerate(self.groups) if g is self.target_group)
        for time, box, blocks, hydration in self.frames(path, include_hydration=True):
            metrics, passed, detail = self.metrics(box, blocks, allow_unmeasurable=True)
            if "measurement_error" in detail:
                errors.append(
                    {"frame_index": len(times), "time_ps": time, **detail["measurement_error"]}
                )
            times.append(time)
            rows.append(metrics)
            valid.append(passed)
            geometry.append(detail)
            local_rows.append(
                population_descriptors(
                    blocks[target_index],
                    self.local_definition,
                    self.target_group["upper"],
                    hydration,
                )
            )
        if not rows:
            raise ValueError("V4 requires a nonempty coordinate trajectory")
        # Nulls are explicit in JSON. NPZ uses NaN only as a missing-value
        # representation, accompanied by a mask; no failed frame is imputed.
        (work / "trajectory_unmeasurable_frames.json").write_text(
            json.dumps({"source_time_ps": times, "unmeasurable_frames": errors}, indent=2)
        )
        series = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
        np.savez_compressed(
            work / "trajectory_observables.npz",
            time_ps=times,
            head_to_head_measurable=np.isfinite(series["head_to_head_nm"]),
            **series,
        )
        result = assess_measured_series(times, series, errors)
        local_series = {key: np.array([row[key] for row in local_rows]) for key in local_rows[0]}
        result["local_conformations"] = local_evidence(times, local_series, self.local_definition)
        result["local_relaxation"] = assess_local(result["local_conformations"])
        np.savez_compressed(
            work / "trajectory_local_conformations.npz", time_ps=times, **local_series
        )
        result["actual_leaflet_counts"] = self.composition
        result["system_composition"] = self.system_composition
        result["geometry_frames_passed"] = int(sum(valid))
        result["geometry_frames_total"] = len(valid)
        result["structural_sampling_interval_ps"] = 100.0
        result["geometry_schema"] = "species-leaflet-screening-1"
        result["trajectory_end_ps"] = actual_end
        result["atom_selections"] = [group["selection"] for group in self.groups]
        result["observable_definitions"] = {
            "area_per_lipid_nm2": "box-xy-area-per-total-leaflet-lipid",
            "head_to_head_nm": "mixture-head-plane-separation",
            "axis_P2": "species-leaflet-molecular-axis; not segmental S_CD",
            "core_gap_nm": "mixed-head-plane-relative-tail-z-quantiles; not a local pore test",
        }
        # Retain per-frame geometry, including rejected frames, so a later
        # review can distinguish transient tilts from persistent defects.
        geometry_series = {
            key: np.asarray([detail[key] for detail in geometry], dtype=float)
            for key in ("oriented_fraction", "core_gap_nm", "mixture_head_plane_nm")
        }
        for key in geometry[0]["orientation"]["species_leaflet"]:
            geometry_series[f"oriented_fraction_{key}"] = [
                detail["orientation"]["species_leaflet"][key]["correct_fraction"]
                for detail in geometry
            ]
        np.savez_compressed(work / "trajectory_geometry.npz", time_ps=times, **geometry_series)
        return result

    def collect(self, path, analysis, staging):
        target = self.protocol["lipid"]
        local = assess_local(analysis["local_conformations"])
        start, end = local["analysis_window_ps"]
        spacing = max(100.0, local["sample_spacing_ps"])
        next_time = start
        provenance, seen = [], set()
        for time, box, blocks in self.frames(path):
            if time + 1e-3 < next_time or time > end:
                continue
            _, geometry_passed, _ = self.metrics(box, blocks, allow_unmeasurable=True)
            if not geometry_passed:
                continue
            # A target conformer is sampled from this entire membrane. A
            # damaged host must not disappear behind target-only extraction.
            for group, coordinates in zip(self.groups, blocks, strict=True):
                for xyz in coordinates:
                    validate_stereochemistry(
                        group["smiles"], group["elements"], group["bonds"], xyz
                    )
                    validate_intrinsic_geometry(xyz, group["elements"], group["bonds"])
            retained = False
            for group, coordinates in zip(self.groups, blocks, strict=True):
                if group["name"] != target:
                    continue
                for index, xyz in enumerate(coordinates):
                    xyz = canonical_coordinates(xyz, self.local_definition)
                    signature = hashlib.sha256(np.round(xyz, 5).tobytes()).hexdigest()
                    if signature in seen:
                        continue
                    seen.add(signature)
                    number = len(provenance)
                    np.savez_compressed(
                        staging / f"conf_{number:04d}.npz", coords=xyz, atom_names=group["names"]
                    )
                    provenance.append(
                        {
                            "file": f"conf_{number:04d}.npz",
                            "time_ps": time,
                            "molecule_index": group["molecule_ids"][index],
                            "leaflet": "upper" if group["upper"][index] else "lower",
                        }
                    )
                    retained = True
            if retained:
                next_time = time + spacing
        if not provenance:
            raise ValueError("No real trajectory conformers passed V4 collection gates")
        return provenance
