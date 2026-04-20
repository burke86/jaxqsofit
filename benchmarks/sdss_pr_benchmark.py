"""Run the PR benchmark used by the GitHub Actions benchmark workflow."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS

from jaxqsofit import QSOFit, build_default_prior_config


def _time_call(fn):
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def _fetch_sdss_spectrum() -> tuple[Any, float]:
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


def run_benchmark(*, optax_steps: int, optax_lr: float, dsps_ssp_fn: str) -> dict[str, Any]:
    (sp, z), fetch_seconds = _time_call(_fetch_sdss_spectrum)
    (lam, flux, err), prep_seconds = _time_call(lambda: _extract_arrays(sp, z))

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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pixels": int(lam.size),
        "redshift": float(z),
        "optax_steps": int(optax_steps),
        "optax_lr": float(optax_lr),
        "fetch_seconds": float(fetch_seconds),
        "prep_seconds": float(prep_seconds),
        "fit_seconds": float(fit_seconds),
        "total_seconds": float(fetch_seconds + prep_seconds + fit_seconds),
        "final_loss": float(np.asarray(q.optax_losses, dtype=float)[-1]),
        "wrms": wrms,
    }


def render_markdown(result: dict[str, Any], *, sha: str, workflow_url: str) -> str:
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
            f"| optax steps | {result['optax_steps']} |",
            f"| fetch time | {result['fetch_seconds']:.3f} s |",
            f"| prep time | {result['prep_seconds']:.3f} s |",
            f"| fit time | {result['fit_seconds']:.3f} s |",
            f"| total time | {result['total_seconds']:.3f} s |",
            f"| final loss | {result['final_loss']:.6g} |",
            f"| residual WRMS | {result['wrms']:.6g} |",
            "",
            f"Commit: `{sha}`",
            f"Run: {workflow_url}",
            "",
            "View all benchmarks in the workflow artifacts.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dsps-ssp-fn", default="tempdata.h5")
    parser.add_argument("--optax-steps", type=int, default=int(os.getenv("JAXQSOFIT_BENCH_OPTAX_STEPS", "200")))
    parser.add_argument("--optax-lr", type=float, default=float(os.getenv("JAXQSOFIT_BENCH_OPTAX_LR", "1e-2")))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(
        optax_steps=args.optax_steps,
        optax_lr=args.optax_lr,
        dsps_ssp_fn=args.dsps_ssp_fn,
    )

    sha = os.getenv("GITHUB_SHA", "local")
    workflow_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if repo and run_id:
        workflow_url = f"{workflow_url}/{repo}/actions/runs/{run_id}"

    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "output").write_text(render_markdown(result, sha=sha, workflow_url=workflow_url))


if __name__ == "__main__":
    main()
