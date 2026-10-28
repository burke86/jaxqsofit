"""Gated base-vs-PR performance benchmark of a jaxqsofit SDSS spectrum fit.

The benchmark fits the SDSS spectrum from the main tutorial with a 200-step MAP
fit, then times the jit-compiled value and gradient of the fit's log density at
the MAP, the function every optimizer and sampler step evaluates.

Subcommands
-----------
fetch
    Download the spectrum once and save it, so network time stays out of the runs.
measure
    Fit and time the jaxqsofit importable in this process; write one JSON record
    with the compile time, the median time per evaluation over several trials,
    and the final MAP loss.
report
    Combine alternating base and PR ``measure`` records into a short Markdown
    comment and exit non-zero if the PR is slower than the limits.

The workflow alternates base and PR processes on one runner (base, PR, base,
PR, ...), so slow drifts of the machine affect both sides equally.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

COMMENT_MARKER = "jaxqsofit PR benchmark"
# A slowdown fails the benchmark when it is larger than this fraction and larger than
# NOISE_SIGMA times the run-to-run scatter.
MAX_SLOWDOWN = 0.10
MAX_COMPILE_SLOWDOWN = 0.25
NOISE_SIGMA = 3.0
RA, DEC = 184.0307, -2.2383


def fetch(args) -> int:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.sdss import SDSS

    matches = SDSS.query_region(SkyCoord(RA, DEC, unit="deg"), spectro=True, radius=5 * u.arcsec)
    hdul = SDSS.get_spectra(matches=matches[:1], timeout=300)[0]
    spec = hdul[1].data
    ivar = np.asarray(spec["ivar"], dtype=float)
    keep = ivar > 0
    np.savez(
        args.output,
        wave=10.0 ** np.asarray(spec["loglam"], dtype=float)[keep],
        flux=np.asarray(spec["flux"], dtype=float)[keep],
        err=1.0 / np.sqrt(ivar[keep]),
        z=float(hdul[2].data["Z"][0]),
    )
    return 0


def _fit(spectrum, dsps_ssp_fn, steps):
    """Fit as in the tutorial (host, lines, calibration polynomial) and capture the fitted model."""
    import numpyro.distributions as dist

    import jaxqsofit.core as core
    from jaxqsofit import JAXQSOFit, PriorConfig

    wave, flux, err, z = spectrum["wave"], spectrum["flux"], spectrum["err"], float(spectrum["z"])
    rest = wave / (1 + z)
    keep = (rest > 1200.0) & (rest < 7000.0)
    wave, flux, err = wave[keep], flux[keep], err[keep]

    prior_config = PriorConfig.from_spectrum(flux=flux, redshift=z)
    # gal_lgmet is absolute log10(Z), bounded by the DSPS SSP grid; keep the prior inside it.
    prior_config.host.metallicity = dist.Normal(loc=-2.3, scale=0.3)
    q = JAXQSOFit.from_arrays(lam=wave, flux=flux, err=err, z=z, ra=RA, dec=DEC)
    q.config.observation.apply_mw_deredden = False
    q.config.host.dsps_ssp_fn = dsps_ssp_fn
    q.config.continuum.fit_feii = False
    q.config.continuum.fit_balmer_continuum = False
    q.config.inference.method = "optax"
    q.config.inference.map_steps = int(steps)
    q.config.output.plot_fig = False
    q.config.output.save_fig = False
    q.config.output.save_result = False
    q.config.prior_config = prior_config

    # The MAP stage evaluates qso_fsps_joint_model(flux=..., **model_kwargs); its posterior
    # predictive is called with the same keyword arguments and the MAP point.
    captured = {}
    predictive_cls = core.Predictive

    class CapturingPredictive(predictive_cls):
        def __init__(self, model, posterior_samples=None, **kwargs):
            captured.setdefault("model", model)
            captured.setdefault("params", {k: v[0] for k, v in (posterior_samples or {}).items()})
            super().__init__(model, posterior_samples=posterior_samples, **kwargs)

        def __call__(self, rng_key, *args, **kwargs):
            captured.setdefault("kwargs", {k: v for k, v in kwargs.items() if k != "flux"})
            return super().__call__(rng_key, *args, **kwargs)

    core.Predictive = CapturingPredictive
    try:
        q.fit(verbose=False)
    finally:
        core.Predictive = predictive_cls
    if not {"model", "params", "kwargs"} <= set(captured):
        raise RuntimeError("Could not capture the fitted model from jaxqsofit.core.Predictive.")
    model_kwargs = dict(captured["kwargs"], flux=np.asarray(q.flux, dtype=float))
    model_kwargs.update(return_line_components=False, emit_deterministics=False)
    return captured["model"], model_kwargs, captured["params"], float(np.asarray(q.optax_losses)[-1])


def measure(args) -> int:
    import jax
    from numpyro.infer.util import log_density

    model, model_kwargs, params, final_loss = _fit(np.load(args.spectrum), args.dsps_ssp_fn, args.steps)
    value_and_grad = jax.jit(jax.value_and_grad(lambda p: log_density(model, (), model_kwargs, p)[0]))

    start = time.perf_counter()
    jax.block_until_ready(value_and_grad(params))
    compile_s = time.perf_counter() - start
    trial_ms = []
    for _ in range(args.trials):
        start = time.perf_counter()
        for _ in range(args.repeats):
            out = value_and_grad(params)
        jax.block_until_ready(out)
        trial_ms.append(1.0e3 * (time.perf_counter() - start) / args.repeats)

    record = {
        "label": args.label,
        "sha": args.sha,
        "compile_s": compile_s,
        "ms_per_eval": statistics.median(trial_ms),
        "trial_ms": trial_ms,
        "final_loss": final_loss,
        "jax": jax.__version__,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    return 0


def _summary(records, key):
    values = [r[key] for r in records]
    median = statistics.median(values)
    # Scaled median absolute deviation: a robust run-to-run standard deviation.
    mad = 1.4826 * statistics.median([abs(v - median) for v in values]) if len(values) > 1 else 0.0
    return median, mad


def render_report(base, pr) -> tuple[str, list[str]]:
    base_ms, base_mad = _summary(base, "ms_per_eval")
    pr_ms, pr_mad = _summary(pr, "ms_per_eval")
    base_compile, _ = _summary(base, "compile_s")
    pr_compile, _ = _summary(pr, "compile_s")
    n = min(len(base), len(pr))
    # Uncertainty of the difference of the two medians (each from n runs).
    noise_ms = (1.2533 * (base_mad**2 + pr_mad**2) ** 0.5 / n**0.5) if n > 1 else 0.0
    slowdown = pr_ms / base_ms - 1.0
    compile_slowdown = pr_compile / base_compile - 1.0

    failures = []
    if slowdown > MAX_SLOWDOWN and pr_ms - base_ms > NOISE_SIGMA * noise_ms:
        failures.append(f"value + gradient is {100 * slowdown:.0f}% slower (limit {100 * MAX_SLOWDOWN:.0f}%)")
    if compile_slowdown > MAX_COMPILE_SLOWDOWN:
        failures.append(f"compile is {100 * compile_slowdown:.0f}% slower (limit {100 * MAX_COMPILE_SLOWDOWN:.0f}%)")

    base_loss, pr_loss = base[0]["final_loss"], pr[0]["final_loss"]
    loss_change = "unchanged" if abs(pr_loss - base_loss) <= 1e-6 * max(1.0, abs(base_loss)) else f"{pr_loss - base_loss:+.4g}"
    status = "✅ no slowdown" if not failures else "❌ slower"
    lines = [
        f"### {COMMENT_MARKER}: {status}",
        "",
        f"`{base[0]['sha'][:7]}` → `{pr[0]['sha'][:7]}` · jit-compiled value and gradient of the log density of the "
        f"tutorial SDSS spectrum fit (host + lines) · {n} alternating runs per side on one runner",
        "",
        "| | base | PR | change |",
        "|---|---:|---:|---:|",
        f"| value + gradient | {base_ms:.3f} ms | {pr_ms:.3f} ms | {100 * slowdown:+.1f}% ± {100 * noise_ms / base_ms:.1f}% |",
        f"| compile | {base_compile:.2f} s | {pr_compile:.2f} s | {100 * compile_slowdown:+.0f}% |",
        f"| final MAP loss (200 steps) | {base_loss:.6g} | {pr_loss:.6g} | {loss_change} |",
        "",
        f"Fails if value + gradient is more than {100 * MAX_SLOWDOWN:.0f}% slower (and beyond {NOISE_SIGMA:.0f}σ of the "
        f"run-to-run scatter) or compile is more than {100 * MAX_COMPILE_SLOWDOWN:.0f}% slower. A changed loss means "
        "the model or the optimizer changed.",
    ]
    if failures:
        lines += ["", "**Failures**", ""] + [f"- {f}" for f in failures]
    server = os.environ.get("GITHUB_SERVER_URL")
    if server:
        lines += ["", f"[Workflow run]({server}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
                      f"{os.environ.get('GITHUB_RUN_ID', '')})"]
    return "\n".join(lines) + "\n", failures


def report(args) -> int:
    def load(pattern):
        return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]

    base, pr = load(args.base), load(args.pr)
    if not base or not pr:
        print("Missing base or PR measurements.")
        return 1
    text, failures = render_report(base, pr)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="Download the benchmark SDSS spectrum.")
    p.add_argument("--output", required=True, help="Output .npz file.")
    p.set_defaults(func=fetch)

    p = sub.add_parser("measure", help="Fit and time this process's jaxqsofit; write one JSON record.")
    p.add_argument("--spectrum", required=True, help=".npz written by fetch.")
    p.add_argument("--output", required=True)
    p.add_argument("--label", default="local")
    p.add_argument("--sha", default=os.getenv("GITHUB_SHA", "local"))
    p.add_argument("--dsps-ssp-fn", default="tempdata.h5")
    p.add_argument("--steps", type=int, default=200, help="MAP steps before timing.")
    p.add_argument("--repeats", type=int, default=100, help="Evaluations per timing trial.")
    p.add_argument("--trials", type=int, default=5)
    p.set_defaults(func=measure)

    p = sub.add_parser("report", help="Write the PR comment; exit non-zero past the limits.")
    p.add_argument("--base", required=True, help="Glob of base measure records.")
    p.add_argument("--pr", required=True, help="Glob of PR measure records.")
    p.add_argument("--output", required=True)
    p.set_defaults(func=report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
