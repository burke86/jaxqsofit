"""Run the PR benchmark used by the GitHub Actions benchmark workflow."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _time_call(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def _fetch_sdss_spectrum() -> tuple[Any, float]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.sdss import SDSS

    coord = SkyCoord(184.0307, -2.2383, unit="deg")
    xid = SDSS.query_region(coord, spectro=True, radius=5 * u.arcsec)
    if xid is None or len(xid) == 0:
        raise RuntimeError("No SDSS spectrum found near benchmark coordinate.")
    spectra = SDSS.get_spectra(matches=xid[:1])
    if not spectra:
        raise RuntimeError("SDSS returned no spectra for benchmark coordinate.")
    z = float(xid[0]["z"]) if "z" in xid.colnames else 0.1
    return spectra[0], z


def _extract_arrays(sp, z: float):
    data = sp[1].data
    lam = np.asarray(10.0 ** data["loglam"], dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    ivar = np.asarray(data["ivar"], dtype=float)
    err = np.full_like(flux, 1.0e-6, dtype=float)
    good = np.isfinite(ivar) & (ivar > 0.0)
    err[good] = 1.0 / np.sqrt(ivar[good])
    err[~np.isfinite(err)] = 1.0e-6
    err[err <= 0.0] = 1.0e-6

    # Keep the benchmark runtime bounded while preserving the line-rich region
    # used by the tutorial coordinate.
    wave_rest = lam / (1.0 + z)
    mask = (
        np.isfinite(lam)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (err > 0.0)
        & (wave_rest > 1200.0)
        & (wave_rest < 7000.0)
    )
    return lam[mask], flux[mask], err[mask]


def _fit_once(
    lam: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    z: float,
    *,
    optax_steps: int,
    optax_lr: float,
    dsps_ssp_fn: str,
) -> dict[str, Any]:
    from jaxqsofit import QSOFit, build_default_prior_config

    prior_config = build_default_prior_config(flux)
    q = QSOFit(lam=lam, flux=flux, err=err, z=z, ra=184.0307, dec=-2.2383)

    _, fit_seconds = _time_call(
        lambda: q.fit(
            deredden=False,
            fit_method="optax",
            fit_lines=True,
            decompose_host=True,
            fit_fe=False,
            fit_bc=False,
            fit_poly=True,
            plot_fig=False,
            save_fig=False,
            save_result=False,
            show_plot=False,
            prior_config=prior_config,
            dsps_ssp_fn=dsps_ssp_fn,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            verbose=False,
        ),
    )

    resid = np.asarray(q.flux, dtype=float) - np.asarray(q.model_total, dtype=float)
    sigma = np.asarray(q.err, dtype=float)
    finite = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0.0)
    wrms = float(np.sqrt(np.mean((resid[finite] / sigma[finite]) ** 2))) if np.any(finite) else float("nan")

    return {
        "fit_seconds": float(fit_seconds),
        "final_loss": float(np.asarray(q.optax_losses, dtype=float)[-1]),
        "wrms": wrms,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def run_benchmark(
    *,
    label: str,
    sha: str,
    repeats: int,
    optax_steps: int,
    optax_lr: float,
    dsps_ssp_fn: str,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    (sp, z), fetch_seconds = _time_call(_fetch_sdss_spectrum)
    (lam, flux, err), prep_seconds = _time_call(lambda: _extract_arrays(sp, z))

    runs = [
        _fit_once(
            lam,
            flux,
            err,
            z,
            optax_steps=optax_steps,
            optax_lr=optax_lr,
            dsps_ssp_fn=dsps_ssp_fn,
        )
        for _ in range(repeats)
    ]
    fit_times = [run["fit_seconds"] for run in runs]
    final_losses = [run["final_loss"] for run in runs]
    wrms_values = [run["wrms"] for run in runs]

    return {
        "label": label,
        "sha": sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pixels": int(lam.size),
        "redshift": float(z),
        "repeats": int(repeats),
        "optax_steps": int(optax_steps),
        "optax_lr": float(optax_lr),
        "fetch_seconds": float(fetch_seconds),
        "prep_seconds": float(prep_seconds),
        "fit_seconds_mean": _mean(fit_times),
        "fit_seconds_std": _stdev(fit_times),
        "fit_seconds_min": float(min(fit_times)),
        "fit_seconds_max": float(max(fit_times)),
        "total_fit_seconds": float(sum(fit_times)),
        "total_seconds": float(fetch_seconds + prep_seconds + sum(fit_times)),
        "final_loss_mean": _mean(final_losses),
        "final_loss_std": _stdev(final_losses),
        "wrms_mean": _mean(wrms_values),
        "wrms_std": _stdev(wrms_values),
        "runs": runs,
    }


def _fmt_seconds(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f} s"


def _percent_delta(candidate: float, baseline: float) -> float:
    return float(100.0 * (candidate - baseline) / baseline) if baseline else float("nan")


def render_comparison_markdown(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    workflow_url: str,
) -> str:
    fit_delta = _percent_delta(candidate["fit_seconds_mean"], baseline["fit_seconds_mean"])
    total_delta = _percent_delta(candidate["total_seconds"], baseline["total_seconds"])
    rows = [
        "<!-- jaxqsofit benchmark -->",
        "### jaxqsofit PR benchmark",
        "",
        "Benchmark input: SDSS spectrum at `SkyCoord(184.0307, -2.2383, unit=\"deg\")`.",
        "",
        "| metric | base | PR | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| commit | `{baseline['sha'][:12]}` | `{candidate['sha'][:12]}` | |",
        f"| repeats | {baseline['repeats']} | {candidate['repeats']} | |",
        f"| pixels | {baseline['pixels']} | {candidate['pixels']} | |",
        f"| redshift | {baseline['redshift']:.6g} | {candidate['redshift']:.6g} | |",
        f"| optax steps | {baseline['optax_steps']} | {candidate['optax_steps']} | |",
        "| host decomposition | enabled | enabled | |",
        f"| fit time mean +/- std | {_fmt_seconds(baseline['fit_seconds_mean'], baseline['fit_seconds_std'])} | {_fmt_seconds(candidate['fit_seconds_mean'], candidate['fit_seconds_std'])} | {fit_delta:+.2f}% |",
        f"| fit time min-max | {baseline['fit_seconds_min']:.3f}-{baseline['fit_seconds_max']:.3f} s | {candidate['fit_seconds_min']:.3f}-{candidate['fit_seconds_max']:.3f} s | |",
        f"| total job-measured time | {baseline['total_seconds']:.3f} s | {candidate['total_seconds']:.3f} s | {total_delta:+.2f}% |",
        f"| final loss mean +/- std | {baseline['final_loss_mean']:.6g} +/- {baseline['final_loss_std']:.3g} | {candidate['final_loss_mean']:.6g} +/- {candidate['final_loss_std']:.3g} | |",
        f"| residual WRMS mean +/- std | {baseline['wrms_mean']:.6g} +/- {baseline['wrms_std']:.3g} | {candidate['wrms_mean']:.6g} +/- {candidate['wrms_std']:.3g} | |",
        "",
        f"Base fetch/prep: {baseline['fetch_seconds']:.3f} s / {baseline['prep_seconds']:.3f} s.",
        f"PR fetch/prep: {candidate['fetch_seconds']:.3f} s / {candidate['prep_seconds']:.3f} s.",
        f"Run: {workflow_url}",
        "",
        "View per-run timings in the workflow artifacts.",
        "",
    ]
    return "\n".join(rows)


def render_markdown(result: dict[str, Any], *, workflow_url: str) -> str:
    return "\n".join(
        [
            "<!-- jaxqsofit benchmark -->",
            "### jaxqsofit PR benchmark",
            "",
            "Benchmark input: SDSS spectrum at `SkyCoord(184.0307, -2.2383, unit=\"deg\")`.",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| pixels | {result['pixels']} |",
            f"| redshift | {result['redshift']:.6g} |",
            f"| repeats | {result['repeats']} |",
            f"| optax steps | {result['optax_steps']} |",
            "| host decomposition | enabled |",
            f"| fetch time | {result['fetch_seconds']:.3f} s |",
            f"| prep time | {result['prep_seconds']:.3f} s |",
            f"| fit time mean +/- std | {_fmt_seconds(result['fit_seconds_mean'], result['fit_seconds_std'])} |",
            f"| fit time min-max | {result['fit_seconds_min']:.3f}-{result['fit_seconds_max']:.3f} s |",
            f"| total time | {result['total_seconds']:.3f} s |",
            f"| final loss mean +/- std | {result['final_loss_mean']:.6g} +/- {result['final_loss_std']:.3g} |",
            f"| residual WRMS mean +/- std | {result['wrms_mean']:.6g} +/- {result['wrms_std']:.3g} |",
            "",
            f"Commit: `{result['sha']}`",
            f"Run: {workflow_url}",
            "",
            "View per-run timings in the workflow artifacts.",
            "",
        ]
    )


def _workflow_url() -> str:
    workflow_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repo and run_id:
        workflow_url = f"{workflow_url}/{repo}/actions/runs/{run_id}"
    return workflow_url


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--sha", default=os.getenv("GITHUB_SHA", "local"))
    parser.add_argument("--repeats", type=int, default=int(os.getenv("JAXQSOFIT_BENCH_REPEATS", "3")))
    parser.add_argument("--dsps-ssp-fn", default="tempdata.h5")
    parser.add_argument("--optax-steps", type=int, default=int(os.getenv("JAXQSOFIT_BENCH_OPTAX_STEPS", "200")))
    parser.add_argument("--optax-lr", type=float, default=float(os.getenv("JAXQSOFIT_BENCH_OPTAX_LR", "1e-2")))


def _run_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(
        label=args.label,
        sha=args.sha,
        repeats=args.repeats,
        optax_steps=args.optax_steps,
        optax_lr=args.optax_lr,
        dsps_ssp_fn=args.dsps_ssp_fn,
    )
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "output").write_text(render_markdown(result, workflow_url=_workflow_url()))


def _compare_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_json.read_text())
    candidate = json.loads(args.candidate_json.read_text())
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "fit_seconds_delta_percent": _percent_delta(
            candidate["fit_seconds_mean"],
            baseline["fit_seconds_mean"],
        ),
        "total_seconds_delta_percent": _percent_delta(candidate["total_seconds"], baseline["total_seconds"]),
    }
    (args.output_dir / "benchmark-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (args.output_dir / "output").write_text(
        render_comparison_markdown(baseline, candidate, workflow_url=_workflow_url())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    _add_run_args(run_parser)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline-json", type=Path, required=True)
    compare_parser.add_argument("--candidate-json", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)

    argv = sys.argv[1:]
    if not argv or argv[0] not in {"run", "compare"}:
        argv = ["run", *argv]
    args = parser.parse_args(argv)

    if args.command == "compare":
        _compare_command(args)
    else:
        _run_command(args)


if __name__ == "__main__":
    main()
