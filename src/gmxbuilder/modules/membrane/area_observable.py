"""Measure a bilayer's area per lipid, and say how well it is known.

The V3 library reported ``box_x * box_y / n`` of the **final frame** of a 1 ns
NPT run, gated only by ``0.75 <= ratio <= 1.35``. That is a single sample of a
quantity which fluctuates by several percent and which, from a constructed
start, drifts for tens of nanoseconds. Measured against the one converged
reference available, the V3 numbers scatter by +-10% with no consistent sign:
they are noise, not a measurement, and nothing downstream should treat them as
a target.

This module replaces that with an ordinary equilibrium measurement: discard
the drifting head of the series, block-average what remains, and report the
mean with an uncertainty and an explicit convergence verdict. A caller that
wants a number it can build on asks for ``converged`` and gets a reason when
the answer is no.

The gate that does most of the work is the effective sample count, not the
drift test. A series that is still relaxing never decorrelates -- its trend is
itself a correlation -- so it arrives with a correlation time comparable to
the whole run and is rejected for having too few independent samples. That is
the same cure stated more usefully: run longer.

Field names follow the schema the asset pipeline already used
(``relative_half_drift``, ``analysis_window_ps``) so a future generation of
assets can be read without another format change.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Schema version of the metadata block this module writes. V3 entries carry
#: no such block at all, which is how the two generations are told apart.
AREA_OBSERVABLE_SCHEMA = 2
AUTOCORRELATION_METHOD = "first-positive-sequence-single-finite-window-2"

#: A block average needs enough blocks for its own error estimate to mean
#: anything. Below this the standard error is not reported as trustworthy.
MIN_BLOCKS = 8

#: How many standard errors of its own the late-run drift may be.
#:
#: This was a fixed 1% of the mean, which is not a threshold at all: the drift
#: statistic has a noise floor set by the observable's own fluctuations and how
#: many times it decorrelates. Measured on a 100 ns POPC bilayer -- instantaneous
#: spread 2.3% of the mean, correlation time 6 ns, so about four independent
#: samples per quarter -- that floor is **1.56%**, above the threshold it was
#: being compared to. The gate could not have been satisfied at any run length,
#: and a gate that cannot pass is not a gate; it would have been read as "100 ns
#: is not enough" when the run was fine and the test was not.
#:
#: Comparing the drift to its own uncertainty is self-scaling: more independent
#: samples tighten the bound automatically, and a run without enough of them
#: fails MIN_EFFECTIVE_SAMPLES instead, which is the honest reason.
MAX_DRIFT_SIGMA = 2.0

#: Existing area-publication diagnostic threshold after autocorrelation correction.
#: This is not a universal sufficiency bound or the queue stopping policy:
#: the maintainer's queue minimum is ten effective samples in each replica.
MIN_EFFECTIVE_SAMPLES = 20.0

#: Fraction of the run that must survive equilibration detection.
#:
#: Without this the drift test is circular: the detector picks the window and
#: the test then judges that window, so any run can be declared converged by
#: discarding everything except a flat-looking tail. Measured on the four 50 ns
#: validation trajectories -- all four still contracting -- the detector kept
#: between 2.7% and 74% of the series and two of them passed the drift test on
#: the strength of it. A run whose equilibrium portion is a small minority of
#: itself has not equilibrated; it has merely been trimmed.
MIN_RETAINED_FRACTION = 0.5


@dataclass(frozen=True)
class AreaMeasurement:
    """An area per lipid with the evidence needed to decide whether to use it."""

    mean_nm2: float
    standard_error_nm2: float
    n_frames: int
    n_blocks: int
    analysis_window_ps: tuple[float, float]
    equilibration_ps: float
    retained_fraction: float
    relative_half_drift: float
    drift_sigma: float
    autocorrelation_ps: float
    effective_samples: float
    converged: bool
    rejection: str | None
    autocorrelation_method: str = AUTOCORRELATION_METHOD

    def as_metadata(self) -> dict:
        # Non-finite values are written as null, never as ``Infinity``. Python
        # emits that token happily and reads it back, but it is not JSON: a
        # reviewer's parser, or any other language, rejects the file. An
        # unmeasurable quantity is absent, which null says and inf does not.
        def finite(value: float) -> float | None:
            number = float(value)
            return number if np.isfinite(number) else None

        return {
            "schema_version": AREA_OBSERVABLE_SCHEMA,
            "autocorrelation_method": self.autocorrelation_method,
            "area_per_lipid_nm2": {
                "mean": finite(self.mean_nm2),
                "standard_error": finite(self.standard_error_nm2),
                "n_frames": self.n_frames,
                "n_blocks": self.n_blocks,
                "effective_samples": finite(self.effective_samples),
                "autocorrelation_ps": finite(self.autocorrelation_ps),
                "relative_half_drift": finite(self.relative_half_drift),
                "drift_sigma": finite(self.drift_sigma),
            },
            "analysis_window_ps": list(self.analysis_window_ps),
            "equilibration_ps": self.equilibration_ps,
            "retained_fraction": self.retained_fraction,
            "converged": self.converged,
            "rejection": self.rejection,
            "criteria": {
                "max_drift_sigma": MAX_DRIFT_SIGMA,
                "min_effective_samples": MIN_EFFECTIVE_SAMPLES,
                "min_blocks": MIN_BLOCKS,
                "min_retained_fraction": MIN_RETAINED_FRACTION,
            },
        }

    @classmethod
    def from_metadata(cls, block: dict) -> AreaMeasurement:
        """Rebuild a measurement from the block a build wrote.

        Pooling replicas otherwise means re-reading and re-analysing tens of
        gigabytes of energy files to recover numbers already computed and
        stored. Absent fields come back as NaN rather than zero, so a block
        written by an older schema cannot silently masquerade as a precise one.
        """
        area = block.get("area_per_lipid_nm2") or {}

        def number(value) -> float:
            return float(value) if isinstance(value, (int, float)) else float("nan")

        window = block.get("analysis_window_ps") or [float("nan"), float("nan")]
        return cls(
            mean_nm2=number(area.get("mean")),
            standard_error_nm2=number(area.get("standard_error")),
            n_frames=int(area.get("n_frames") or 0),
            n_blocks=int(area.get("n_blocks") or 0),
            analysis_window_ps=(number(window[0]), number(window[-1])),
            equilibration_ps=number(block.get("equilibration_ps")),
            retained_fraction=number(block.get("retained_fraction")),
            relative_half_drift=number(area.get("relative_half_drift")),
            drift_sigma=number(area.get("drift_sigma")),
            autocorrelation_ps=number(area.get("autocorrelation_ps")),
            effective_samples=number(area.get("effective_samples")),
            converged=bool(block.get("converged")),
            rejection=block.get("rejection"),
            autocorrelation_method=str(block.get("autocorrelation_method", "legacy-unversioned")),
        )


def integrated_autocorrelation(values: np.ndarray) -> float:
    """Return statistical inefficiency (1 + twice the positive-lag sum), in frames.

    Summation is windowed at the first non-positive correlation, which is the
    standard guard against integrating the noise tail of a finite series.
    Returns at least 1.0 -- an uncorrelated series still contributes one frame
    per sample.
    """
    series = np.asarray(values, dtype=float)
    n = len(series)
    if n < 4:
        return 1.0
    centred = series - series.mean()
    variance = float(centred @ centred) / n
    # Scale-relative roundoff must not create apparent correlation in a flat series.
    scale = max(abs(float(series.mean())), 1.0)
    if variance <= (scale * 1e-12) ** 2:
        return 1.0
    # Pad to at least 2*n - 1 samples so FFT correlation does not wrap circularly.
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n]
    correlation /= correlation[0]

    total = 0.0
    for lag in range(1, n):
        if correlation[lag] <= 0.0:
            break
        # The FFT gives S(lag)/S(0), already weighted by (n-lag)/n.
        # Equivalently: covariance divided by its n-lag pairs, then one
        # finite-window weight. Multiplying that weight here would apply it twice.
        total += correlation[lag]
    # Include both lag directions; the floor keeps effective samples at most n.
    return max(1.0, 1.0 + 2.0 * total)


def _detect_equilibration(values: np.ndarray) -> tuple[int, bool]:
    """Return where the analysis window starts, and whether it hit its bound.

    The series is truncated at the candidate start that leaves the largest
    number of *effective* samples: discarding a drifting head raises the
    effective count by removing correlation, while discarding too much lowers
    it by removing data, so the maximum sits at the end of equilibration. This
    is the standard automatic-equilibration criterion and it avoids hard-coding
    "the first X ns are equilibration", which is exactly the guess the V3
    protocol made and got wrong.

    The search is confined to the head of the series so that at least
    ``MIN_RETAINED_FRACTION`` survives. Two reasons, and the second is the one
    that matters: an unbounded search would (a) return windows this function's
    own caller is going to reject anyway, and (b) actively seek *degenerate*
    ones. A window short enough to contain no variation has an autocorrelation
    time of one frame and therefore the best possible effective sample count,
    so unbounded maximisation walks straight into it -- observed on a series
    whose frames repeat in blocks, where it selected a constant window and
    reported a standard error of 6e-18 for a measurement of nothing.

    The returned flag says the optimum sat on that bound: the detector wanted
    to discard more, which is itself evidence that the run is mostly
    equilibration.
    """
    n = len(values)
    if n < 4 * MIN_BLOCKS:
        return 0, False
    last_start = int(n * (1.0 - MIN_RETAINED_FRACTION))
    if last_start < 1:
        return 0, False
    best_index, best_effective = 0, -np.inf
    # A coarse scan is enough: the optimum is broad, and every candidate costs
    # one autocorrelation over the remaining series.
    for index in np.unique(np.linspace(0, last_start, 40).astype(int)):
        remaining = values[index:]
        tau = integrated_autocorrelation(remaining)
        effective = len(remaining) / tau
        if effective > best_effective:
            best_index, best_effective = int(index), effective
    return best_index, best_index >= last_start


def measure_area_per_lipid(
    times_ps: np.ndarray,
    box_x_nm: np.ndarray,
    box_y_nm: np.ndarray,
    lipids_per_leaflet: int,
) -> AreaMeasurement:
    """Measure the equilibrium area per lipid from an NPT box time series."""
    times = np.asarray(times_ps, dtype=float)
    areas = np.asarray(box_x_nm, dtype=float) * np.asarray(box_y_nm, dtype=float)
    if len(times) != len(areas):
        raise ValueError("time and box series must have equal length")
    if lipids_per_leaflet <= 0:
        raise ValueError("lipids_per_leaflet must be positive")
    if len(areas) < 2:
        raise ValueError("an area measurement needs at least two frames")

    per_lipid = areas / float(lipids_per_leaflet)
    start, detector_hit_bound = _detect_equilibration(per_lipid)
    window = per_lipid[start:]
    window_times = times[start:]

    mean = float(window.mean())

    # Drift is measured on the second half of the *whole* run, split into two
    # quarters -- deliberately not on the window the detector chose. Judging a
    # detector-selected window with a drift test is circular: the detector
    # picks the flattest tail it can find and the test then approves it. Two of
    # the four 50 ns validation trajectories passed that way while still
    # contracting at over 1% per 10 ns. The fixed window asks the physical
    # question instead: late in this run, has the area stopped moving?
    tail = per_lipid[len(per_lipid) // 2 :]
    quarter = len(tail) // 2
    measurable_drift = quarter >= 2 and mean > 0
    if measurable_drift:
        difference = float(tail[quarter:].mean() - tail[:quarter].mean())
        drift = abs(difference) / mean
        # The uncertainty of that difference, from the spread of the tail and
        # how often it decorrelates. Two quarter means, so the errors add in
        # quadrature.
        tail_tau = integrated_autocorrelation(tail)
        independent_per_quarter = max(1.0, quarter / tail_tau)
        drift_error = float(tail.std(ddof=1)) * np.sqrt(2.0 / independent_per_quarter)
        drift_sigma = abs(difference) / drift_error if drift_error > 0 else float("inf")
    else:
        drift = float("nan")
        drift_sigma = float("nan")

    frame_ps = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    tau_frames = integrated_autocorrelation(window)
    effective = len(window) / tau_frames

    # Blocks are sized to the correlation time so that block means are close to
    # independent; the error is then the ordinary standard error of those means.
    block_size = max(1, int(np.ceil(tau_frames)))
    n_blocks = len(window) // block_size
    if n_blocks >= 2:
        trimmed = window[: n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = trimmed.mean(axis=1)
        standard_error = float(block_means.std(ddof=1) / np.sqrt(n_blocks))
    else:
        n_blocks = max(n_blocks, 0)
        standard_error = float("inf")

    retained = len(window) / len(per_lipid)

    # Ordered by what the reader should do about it, not by how the statistics
    # happen to fail. A drifting series inherently has a long correlation time,
    # so "too few blocks" fires for it as well -- but reporting that would name
    # a symptom and hide the cause, which is that the bilayer is still relaxing.
    rejection = None
    if not measurable_drift:
        rejection = (
            f"{len(per_lipid)} frames is too few to measure whether the area is still "
            "moving; the run is far too short to publish an area from"
        )
    elif drift_sigma > MAX_DRIFT_SIGMA:
        rejection = (
            f"the area moves by {drift * 100:.1f}% across the second half of the run, "
            f"{drift_sigma:.1f} times the uncertainty of that difference "
            f"(limit {MAX_DRIFT_SIGMA:.0f}); the bilayer is still relaxing"
        )
    elif detector_hit_bound:
        rejection = (
            "equilibration detection wanted to discard more than "
            f"{(1.0 - MIN_RETAINED_FRACTION) * 100:.0f}% of the run; the equilibrium "
            "portion is a minority of the trajectory, so the run is too short "
            "rather than converged"
        )
    elif effective < MIN_EFFECTIVE_SAMPLES:
        # This is the gate that catches an unequilibrated run, and it is worth
        # saying why rather than only that. A series still relaxing never
        # decorrelates -- its trend *is* a correlation -- so it lands here with
        # a correlation time that is a large fraction of the run. Frames are
        # not the shortage; independent ones are, and the cure is the same
        # either way.
        rejection = (
            f"effective sample size {effective:.1f} (need {MIN_EFFECTIVE_SAMPLES:.0f}); "
            f"{len(window)} frames but a correlation time of "
            f"{tau_frames * frame_ps / 1000:.1f} ns -- the area is either fluctuating "
            "slowly or still drifting, and both need a longer run"
        )
    elif n_blocks < MIN_BLOCKS:
        rejection = (
            f"only {n_blocks} independent blocks (need {MIN_BLOCKS}); "
            "the run is short relative to the area's correlation time"
        )

    return AreaMeasurement(
        mean_nm2=mean,
        standard_error_nm2=standard_error,
        n_frames=len(window),
        n_blocks=n_blocks,
        analysis_window_ps=(float(window_times[0]), float(window_times[-1])),
        equilibration_ps=float(window_times[0] - times[0]),
        retained_fraction=retained,
        relative_half_drift=drift,
        drift_sigma=drift_sigma,
        autocorrelation_ps=tau_frames * frame_ps,
        effective_samples=effective,
        converged=rejection is None,
        rejection=rejection,
    )


_XVG_COMMENT = ("#", "@")


def box_series_from_edr(gmx: str, edr: Path, work: Path, *, timeout: int = 1800) -> tuple:
    """Extract (times_ps, box_x_nm, box_y_nm) from a GROMACS energy file."""
    output = work / f"{edr.stem}_box.xvg"
    result = subprocess.run(
        [gmx, "energy", "-f", str(edr), "-o", str(output)],
        cwd=work,
        input="Box-X\nBox-Y\n\n",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(f"gmx energy could not read {edr.name}:\n{result.stderr[-2000:]}")
    rows = [
        line.split()
        for line in output.read_text().splitlines()
        if line and line[0] not in _XVG_COMMENT
    ]
    output.unlink(missing_ok=True)
    if not rows or len(rows[0]) < 3:
        raise RuntimeError(f"{edr.name} carries no box time series")
    table = np.asarray([[float(value) for value in row[:3]] for row in rows])
    return table[:, 0], table[:, 1], table[:, 2]


def summarise(measurement: AreaMeasurement) -> str:
    """One-line human summary, for build logs."""
    verdict = "converged" if measurement.converged else f"NOT converged: {measurement.rejection}"
    return (
        f"area/lipid {measurement.mean_nm2:.4f} +- {measurement.standard_error_nm2:.4f} nm^2 "
        f"({measurement.n_frames} frames in {measurement.analysis_window_ps[0]:.0f}-"
        f"{measurement.analysis_window_ps[1]:.0f} ps, tau {measurement.autocorrelation_ps:.0f} ps, "
        f"n_eff {measurement.effective_samples:.0f}, kept "
        f"{measurement.retained_fraction * 100:.0f}%, drift "
        f"{measurement.relative_half_drift * 100:.2f}% = "
        f"{measurement.drift_sigma:.1f} sigma) -- {verdict}"
    )


__all__ = [
    "AREA_OBSERVABLE_SCHEMA",
    "AreaMeasurement",
    "MAX_REPLICA_SPREAD_SIGMA",
    "PooledAreaMeasurement",
    "box_series_from_edr",
    "integrated_autocorrelation",
    "measure_area_per_lipid",
    "pool_replicas",
    "summarise",
    "summarise_pooled",
]


@dataclass(frozen=True)
class PooledAreaMeasurement:
    """One area from several replicas, plus whether they agree with each other.

    Pooling is not just averaging. Two independent seeds of the same system
    must give the same area, so their disagreement is a measurement of the
    protocol's own error -- the thing V3 could not report, because with one
    trajectory per entry there is nothing to disagree with. An entry whose
    replicas differ by more than their combined uncertainty has not been
    measured, however long each of them ran.
    """

    mean_nm2: float
    standard_error_nm2: float
    replicas: tuple[AreaMeasurement, ...]
    spread_nm2: float
    spread_sigma: float
    effective_samples: float
    converged: bool
    rejection: str | None

    def as_metadata(self) -> dict:
        def finite(value: float) -> float | None:
            number = float(value)
            return number if np.isfinite(number) else None

        return {
            "schema_version": AREA_OBSERVABLE_SCHEMA,
            "autocorrelation_method": AUTOCORRELATION_METHOD,
            "measures": "pure-bilayer-area-per-lipid",
            "pooled_replicas": len(self.replicas),
            "area_per_lipid_nm2": {
                "mean": finite(self.mean_nm2),
                "standard_error": finite(self.standard_error_nm2),
                "effective_samples": finite(self.effective_samples),
                "replica_spread": finite(self.spread_nm2),
                "replica_spread_sigma": finite(self.spread_sigma),
            },
            "converged": self.converged,
            "rejection": self.rejection,
            "criteria": {
                "max_replica_spread_sigma": MAX_REPLICA_SPREAD_SIGMA,
                "min_effective_samples": MIN_EFFECTIVE_SAMPLES,
            },
            "replicas": [measurement.as_metadata() for measurement in self.replicas],
        }


#: How far apart replica means may be, in units of their combined error.
#:
#: Same reasoning as MAX_DRIFT_SIGMA -- a fixed percentage would be a threshold
#: on a random variable without reference to its own spread. Measured on the two
#: 100 ns POPC replicas, the means sit 0.34% apart at 0.6 sigma.
#:
#: Four rather than three because the distribution has a heavier tail than a
#: Gaussian: the per-replica error depends on an *estimated* correlation time,
#: which is sometimes estimated small. Measured over 120 pairs drawn from one
#: distribution -- pairs that agree by construction -- the false-rejection rate
#: is 5.0% at 3 sigma, 0.8% at 4 and 0.0% at 5. Across 277 entries a 3-sigma
#: gate would have sent about fourteen perfectly good ones back for a rerun.
MAX_REPLICA_SPREAD_SIGMA = 4.0


def measurement_rejection(measurement: AreaMeasurement) -> str | None:
    """Revalidate stored statistics; a serialized verdict alone is not evidence."""
    m = measurement
    if m.autocorrelation_method != AUTOCORRELATION_METHOD:
        return "obsolete autocorrelation statistics; reanalysis required"
    finite = (
        m.mean_nm2,
        m.standard_error_nm2,
        m.effective_samples,
        m.drift_sigma,
        m.retained_fraction,
        *m.analysis_window_ps,
    )
    if not all(np.isfinite(value) for value in finite):
        return "non-finite or missing area statistics"
    if m.mean_nm2 <= 0 or m.standard_error_nm2 < 0 or m.effective_samples <= 0:
        return "invalid area, uncertainty or effective sample count"
    if not m.converged:
        return m.rejection or "replica did not converge"
    if m.effective_samples < MIN_EFFECTIVE_SAMPLES:
        return f"effective sample size {m.effective_samples:.1f} (need {MIN_EFFECTIVE_SAMPLES:g})"
    if m.n_blocks < MIN_BLOCKS or m.n_frames < m.n_blocks:
        return "insufficient independent blocks or frames"
    if not 0 <= m.drift_sigma <= MAX_DRIFT_SIGMA:
        return "late-run drift exceeds the acceptance limit"
    if not MIN_RETAINED_FRACTION <= m.retained_fraction <= 1:
        return "insufficient retained trajectory fraction"
    if not 0 <= m.analysis_window_ps[0] < m.analysis_window_ps[1]:
        return "invalid analysis window"
    return None


def pool_replicas(measurements: list[AreaMeasurement]) -> PooledAreaMeasurement:
    """Combine per-replica measurements into one, and check they agree."""
    if not measurements:
        raise ValueError("pooling needs at least one replica")

    usable = [m for m in measurements if np.isfinite(m.mean_nm2)]
    if not usable:
        raise ValueError("no replica produced a finite area")

    # Weight by independent samples, not by frames: a replica that ran longer
    # but decorrelated no more often has not earned more say.
    weights = np.asarray([max(m.effective_samples, 1e-9) for m in usable], dtype=float)
    means = np.asarray([m.mean_nm2 for m in usable], dtype=float)
    mean = float(np.average(means, weights=weights))
    effective = float(weights.sum())

    errors = np.asarray(
        [m.standard_error_nm2 if np.isfinite(m.standard_error_nm2) else np.inf for m in usable]
    )
    with np.errstate(divide="ignore"):
        inverse = 1.0 / np.square(errors)
    within_run = float(np.sqrt(1.0 / inverse.sum())) if inverse.sum() > 0 else float("inf")

    if len(usable) >= 2:
        spread = float(means.max() - means.min())
        combined = float(np.sqrt(np.nansum(np.square(np.clip(errors, 0, 1e6)))))
        spread_sigma = spread / combined if combined > 0 else float("inf")
        # The scatter *between* independent seeds is the more complete error:
        # it contains everything a single run's block average cannot see,
        # including its own under-estimated correlation time. Taking the larger
        # of the two means a pair that disagrees widens the error bar it
        # publishes instead of quietly reporting a precision it has not earned.
        between_runs = float(np.std(means, ddof=1) / np.sqrt(len(means)))
        standard_error = max(within_run, between_runs)
    else:
        spread = 0.0
        spread_sigma = float("nan")
        standard_error = within_run

    rejection = None
    invalid = [reason for m in measurements if (reason := measurement_rejection(m))]
    if len(usable) != len(measurements):
        rejection = "one or more replicas have no finite area"
    elif len(usable) < 2:
        rejection = (
            "only one replica produced an area; with nothing to compare against, "
            "protocol noise and a real difference are indistinguishable -- which is "
            "exactly the V3 failure this replaces"
        )
    elif invalid:
        rejection = f"{len(invalid)} of {len(measurements)} replicas: {invalid[0]}"
    elif effective < MIN_EFFECTIVE_SAMPLES:
        rejection = (
            f"pooled effective sample size {effective:.1f} (need {MIN_EFFECTIVE_SAMPLES:.0f})"
        )
    elif spread_sigma > MAX_REPLICA_SPREAD_SIGMA:
        rejection = (
            f"replicas disagree by {spread:.4f} nm^2, {spread_sigma:.1f} times their "
            f"combined error (limit {MAX_REPLICA_SPREAD_SIGMA:.0f}); the seeds have not "
            "converged to the same bilayer"
        )

    return PooledAreaMeasurement(
        mean_nm2=mean,
        standard_error_nm2=standard_error,
        replicas=tuple(measurements),
        spread_nm2=spread,
        spread_sigma=spread_sigma,
        effective_samples=effective,
        converged=rejection is None,
        rejection=rejection,
    )


def summarise_pooled(pooled: PooledAreaMeasurement) -> str:
    verdict = "converged" if pooled.converged else f"NOT converged: {pooled.rejection}"
    return (
        f"pooled area/lipid {pooled.mean_nm2:.4f} +- {pooled.standard_error_nm2:.4f} nm^2 "
        f"from {len(pooled.replicas)} replicas (n_eff {pooled.effective_samples:.0f}, "
        f"spread {pooled.spread_nm2:.4f} = {pooled.spread_sigma:.1f} sigma) -- {verdict}"
    )
