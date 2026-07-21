"""End-to-end mock benchmark for continuum/reddening NUTS geometry.

Runs the public :class:`JAXQSOFit` path, including MAP initialization, on a
narrow-line mock and a broad+narrow-line mock.  The polynomial cases use the
same residual-curvature basis constructed by a normal production fit.

Run with::

    conda run -n jaxcpu python benchmarks/continuum_geometry_benchmark.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from numpyro.diagnostics import effective_sample_size

from jaxqsofit import JAXQSOFit, PriorConfig


C_KMS = 299792.458


def _gaussian_velocity(wave, center, fwhm_kms, peak):
    sigma_ln = fwhm_kms / (2.354820045 * C_KMS)
    return peak * np.exp(-0.5 * (np.log(wave / center) / sigma_ln) ** 2)


def make_mock(*, broad, seed=2026, n_pixels=320):
    rng = np.random.default_rng(seed + int(broad))
    wave = np.linspace(4300.0, 5400.0, n_pixels)
    pivot = 4800.0
    intrinsic = 1.0 * (wave / pivot) ** -1.25
    attenuation = 10.0 ** (-0.4 * 0.08 * (wave / 2500.0) ** -1.2)
    mean = intrinsic * attenuation
    mean += _gaussian_velocity(wave, 4862.68, 500.0, 0.35)
    if broad:
        mean += _gaussian_velocity(wave, 4862.68, 4500.0, 0.22)
    error = np.full_like(wave, 0.025)
    return wave, mean + rng.normal(0.0, error), error


def summarize(q, max_tree_depth):
    mcmc = q.numpyro_mcmc
    extra = mcmc.get_extra_fields()
    steps = np.asarray(extra["num_steps"], dtype=int)
    samples = mcmc.get_samples(group_by_chain=True)
    ess = []
    for value in samples.values():
        if np.asarray(value).ndim >= 2:
            values = np.ravel(np.asarray(effective_sample_size(np.asarray(value))))
            ess.extend(values[np.isfinite(values)])
    return {
        "step_size": float(np.asarray(mcmc.last_state.adapt_state.step_size)),
        "median_steps": float(np.median(steps)),
        "p90_steps": float(np.percentile(steps, 90)),
        "max_depth_fraction": float(np.mean(steps >= 2**max_tree_depth - 1)),
        "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "median_ess": float(np.median(ess)) if ess else float("nan"),
    }


def run_case(
    *, broad, polynomial, reddening, warmup, samples, map_steps, max_tree_depth,
    line_block_dense_mass=False, residualize_reddening=False, seed=0,
):
    wave, flux, error = make_mock(broad=broad)
    prior = PriorConfig.from_spectrum(flux=flux, redshift=0.0, pl_pivot=4800.0)
    q = JAXQSOFit.from_arrays(lam=wave, flux=flux, err=error, z=0.0, ra=0.0, dec=0.0)
    q.config.observation.apply_mw_deredden = False
    q.config.spectroscopy.resolving_power = 2000.0
    q.resolving_power = 2000.0
    q.config.host.enabled = False
    q.config.lines.enabled = True
    q.config.lines.use_broad_lines = bool(broad)
    q.config.lines.use_narrow_lines = True
    q.config.continuum.fit_feii = False
    q.config.continuum.fit_balmer_continuum = False
    q.config.continuum.fit_polynomial_tilt = bool(polynomial)
    q.config.continuum.fit_reddening = bool(reddening)
    q.config.continuum.polynomial_order = 2
    q.config.inference.method = "optax+nuts"
    q.config.inference.map_steps = int(map_steps)
    q.config.inference.learning_rate = 1e-2
    q.config.inference.num_warmup = int(warmup)
    q.config.inference.num_samples = int(samples)
    q.config.inference.num_chains = 1
    q.config.inference.target_accept_prob = 0.9
    q.config.inference.dense_mass = False
    q.config.inference.line_block_dense_mass = bool(line_block_dense_mass)
    q.config.inference.standardize_active_priors = True
    q.config.inference.max_tree_depth = int(max_tree_depth)
    q.config.inference.random_seed = int(seed)
    q.config.output.plot_fig = False
    q.config.output.save_fig = False
    q.config.output.save_result = False
    q.config.output.show_plot = False
    prior._model_priors["residualize_reddening_geometry"] = bool(
        residualize_reddening
    )
    q.config.prior_config = prior
    q.fit(verbose=False)
    return summarize(q, max_tree_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--map-steps", type=int, default=300)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--residualize-reddening", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    output = {}
    for broad in (False, True):
        mock_name = "broad_narrow" if broad else "narrow"
        for polynomial, reddening in ((False, False), (False, True), (True, False), (True, True)):
            name = f"{mock_name}:poly={polynomial}:reddening={reddening}"
            output[name] = run_case(
                broad=broad,
                polynomial=polynomial,
                reddening=reddening,
                warmup=args.warmup,
                samples=args.samples,
                map_steps=args.map_steps,
                max_tree_depth=args.max_tree_depth,
                residualize_reddening=args.residualize_reddening,
            )
            print(json.dumps({name: output[name]}, indent=2), flush=True)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
