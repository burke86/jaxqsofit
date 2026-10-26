"""Batched MAP/Laplace fits of many spectra with one compiled program.

:meth:`JAXQSOFit.fit` compiles and optimizes one spectrum at a time. When many
fitters share a configuration, :func:`fit_batch` stacks their model inputs and runs
the optax MAP (and optionally a Laplace approximation) under ``jax.vmap`` with
:mod:`jaxsedfit.batched`.

Two things make the model structure depend on each spectrum and are removed first:

* Pixel counts differ: spectra are padded with masked pixels (``pixel_mask``) that
  continue the log-wavelength grid.
* The tied-line table depends on rest-frame coverage: fitters are split into
  redshift windows that share one table spanning the window's coverage, and lines a
  spectrum does not cover get zero amplitude (``tied_line_meta["line_mask"]``),
  constrained by their priors only.

The batched fit therefore uses a slightly different line table than a single
:meth:`JAXQSOFit.fit` (tied groups that straddle the coverage edge keep all their
members), and its stage-1 continuum MAP masks the strong-line windows instead of
removing those pixels.
"""
from __future__ import annotations

import ctypes
import gc
import json
import sys
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive

import jaxsedfit.preload as _jaxsedfit_preload
from jaxsedfit.batched import BatchedInputs, make_batched_fitter
from jaxsedfit.config import serialize_config
from jaxsedfit.spectral_model import tied_line_coverage_mask

from .core import (
    JAXQSOFit,
    _build_line_init_values,
    _materialize_prior_config,
    _optax_stage1_init_values,
    _stage1_continuum_keep_mask,
)
from .custom_components import (
    inject_default_custom_component_priors,
    inject_default_custom_line_component_priors,
)
from .model import (
    _balmer_static_terms_jax,
    _extract_line_table_from_prior_config,
    build_tied_line_meta_from_linelist,
    qso_fsps_joint_model,
)

_BATCH_METHODS = {"optax": "map", "map": "map", "laplace": "laplace"}
# Flags the stage-1 continuum MAP switches off, as in JAXQSOFit.run_fsps_optax_fit.
_STAGE1_FLAGS = dict(use_lines=False, fit_fe=False, fit_bc=False, fit_poly=False, fit_poly_order=2)


def _clear_per_spectrum_caches() -> None:
    """Drop jaxsedfit caches that gain an entry for every spectrum.

    The host SSP basis is cached by the content of the rest-frame wavelength grid,
    which differs for every spectrum (about 35 MB each), and the projection caches
    are keyed by redshift. Built model inputs keep their own references.
    """
    _jaxsedfit_preload.clear_per_object_caches()
    _jaxsedfit_preload._HOST_BASIS_CACHE.clear()


def _release_memory() -> None:
    """Free a fitted chunk and hand the memory back to the OS.

    The chunk's inputs contain reference cycles, so they are only freed by the cyclic
    garbage collector; on Linux, glibc also keeps freed memory unless asked to trim
    it. As in ``jaxsedfit.fit_batch``, memory otherwise grew by about a chunk per chunk
    on CI runners.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def _batch_model(*, pixel_mask, stage1_pixel_mask, stage1=False, **kwargs):
    """``qso_fsps_joint_model`` with the stage-1 continuum switch used by the batch."""
    if stage1:
        kwargs.update(_STAGE1_FLAGS)
        pixel_mask = stage1_pixel_mask
    return qso_fsps_joint_model(pixel_mask=pixel_mask, **kwargs)


def _signature(q: JAXQSOFit, prep: dict[str, Any]) -> str:
    """Everything a batch must share: settings and the prior structure, not data."""
    cfg = q.config
    data = {
        name: serialize_config(getattr(cfg, name))
        for name in ("preprocessing", "continuum", "agn", "bal", "host", "lines", "inference")
    }
    run = prep["run_kwargs"]
    data["prior_structure"] = sorted((str(k), type(v).__name__) for k, v in run["prior_config"].items())
    data["resolution"] = [cfg.spectroscopy.resolving_power is None, bool(q.apply_instrumental_resolution)]
    data["custom"] = [c.name for c in (*run["custom_components"], *run["custom_line_components"])]
    return json.dumps(data, sort_keys=True, default=repr)


def _model_inputs(q: JAXQSOFit, prep: dict[str, Any], table_range, n_pixels: int | None = None) -> dict[str, Any]:
    """Keyword inputs of ``qso_fsps_joint_model`` for one prepared fitter.

    Parameters
    ----------
    q : JAXQSOFit
        Fitter after :meth:`JAXQSOFit._prepare_fit`.
    prep : dict
        Output of :meth:`JAXQSOFit._prepare_fit`.
    table_range : tuple of float
        Rest-frame range [A] that fixes the tied-line table.
    n_pixels : int, optional
        Pad the spectrum to this many pixels; None leaves it unpadded.
    """
    run = prep["run_kwargs"]
    wave, flux, err = (np.asarray(x, dtype=float) for x in (q.wave, q.flux, q.err))
    prior_config = _materialize_prior_config(run["prior_config"])
    prior_config = inject_default_custom_component_priors(prior_config, flux, run["custom_components"])
    prior_config = inject_default_custom_line_component_priors(prior_config, flux, run["custom_line_components"])
    line_table = _extract_line_table_from_prior_config(prior_config)
    if line_table is None:
        raise ValueError("fit_batch needs a line table in the prior configuration.")
    meta = dict(build_tied_line_meta_from_linelist(
        line_table,
        np.asarray(table_range, dtype=float),
        pool_narrow_centroids=bool(prior_config.get("pool_narrow_centroids", True)),
    ))
    # The single-object table is built from this spectrum's own wavelength range.
    meta["line_mask"] = np.asarray(tied_line_coverage_mask(meta, (wave.min(), wave.max())))

    n_real = wave.size
    n_pad = 0 if n_pixels is None else int(n_pixels) - n_real
    if n_pad < 0:
        raise ValueError(f"Spectrum has {n_real} pixels, more than n_pixels={n_pixels}.")
    stage1_keep = _stage1_continuum_keep_mask(wave)
    if n_pad:
        step = float(np.median(np.diff(np.log(wave))))
        wave = np.concatenate([wave, wave[-1] * np.exp(step * np.arange(1, n_pad + 1))])
        flux = np.concatenate([flux, np.zeros(n_pad)])
        # Huge errors also drop padding out of the error-weighted reddening projection.
        err = np.concatenate([err, np.full(n_pad, 1.0e30)])
        stage1_keep = np.concatenate([stage1_keep, np.zeros(n_pad, dtype=bool)])
    pixel_mask = np.arange(wave.size) < n_real

    fsps_grid = q._build_fsps_grid_for_fit(
        wave=wave,
        age_grid_gyr=run["age_grid_gyr"],
        logzsol_grid=run["logzsol_grid"],
        dsps_ssp_fn=run["dsps_ssp_fn"],
        decompose_host=run["decompose_host"],
        z_qso=q.z,
    )
    balmer_bb, balmer_tau, balmer_edge = _balmer_static_terms_jax(jnp.asarray(wave), balmer_te=15000.0)
    return dict(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors=prior_config.get("conti_priors", {}),
        tied_line_meta=meta,
        fsps_grid=fsps_grid,
        fe_uv_wave=q.fe_uv_wave,
        fe_uv_flux=q.fe_uv_flux,
        fe_op_wave=q.fe_op_wave,
        fe_op_flux=q.fe_op_flux,
        fe_uv_flux_on_wave=np.interp(wave, q.fe_uv_wave, np.maximum(q.fe_uv_flux, 0.0), left=0.0, right=0.0),
        fe_op_flux_on_wave=np.interp(wave, q.fe_op_wave, np.maximum(q.fe_op_flux, 0.0), left=0.0, right=0.0),
        balmer_bb_shape=np.asarray(balmer_bb, dtype=float),
        balmer_tau_shape=np.asarray(balmer_tau, dtype=float),
        balmer_below_edge=np.asarray(balmer_edge, dtype=bool),
        use_lines=run["use_lines"],
        prior_config=prior_config,
        decompose_host=run["decompose_host"],
        fit_pl=run["fit_pl"],
        fit_fe=run["fit_fe"],
        fit_bc=run["fit_bc"],
        fit_poly=run["fit_poly"],
        fit_reddening=run["fit_reddening"],
        fit_poly_order=run["fit_poly_order"],
        z_qso=float(q.z),
        return_line_components=False,
        emit_deterministics=False,
        custom_components=run["custom_components"],
        custom_line_components=run["custom_line_components"],
        pixel_mask=pixel_mask,
        stage1_pixel_mask=stage1_keep,
    )


def _init_values(q: JAXQSOFit, prep: dict[str, Any], inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Stage-1 continuum and line initial values, as in ``run_fsps_optax_fit``."""
    run = prep["run_kwargs"]
    values = _optax_stage1_init_values(
        inputs["prior_config"],
        np.asarray(q.flux, dtype=float),
        inputs["fsps_grid"],
        decompose_host=run["decompose_host"],
        fit_pl=run["fit_pl"],
        fit_reddening=run["fit_reddening"],
    )
    values.update(_build_line_init_values(inputs["tied_line_meta"], inputs["prior_config"], use_lines=run["use_lines"]))
    return {k: np.asarray(v, dtype=float) for k, v in values.items()}


def _redshift_windows(fitters, indices, redshift_window):
    """Split ``indices`` (sorted by redshift) into windows of ln(1 + z) width."""
    lnz = {i: np.log1p(max(float(fitters[i].z), 0.0)) for i in indices}
    order = sorted(indices, key=lambda i: lnz[i])
    windows, start = [], 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and (redshift_window is None or lnz[order[stop]] - lnz[order[start]] <= redshift_window):
            stop += 1
        windows.append(order[start:stop])
        start = stop
    return windows


def fit_batch(
    fitters: Sequence[JAXQSOFit],
    method: str = "map",
    *,
    chunk_size: int = 16,
    redshift_window: float | None = 0.25,
    num_samples: int = 400,
    seed: int | None = None,
    progress_bar: bool = True,
):
    """Fit many configured spectra with vmapped MAP or Laplace inference.

    Fitters are grouped by everything in their configuration except the data
    (spectrum, redshift, sky position and data-scaled prior values), then split
    into redshift windows that share one tied-line table. Each group compiles one
    program and fits ``chunk_size`` spectra at a time. MAP settings (steps,
    learning rate, seed) come from each group's ``config.inference``; the MAP is
    staged like ``method="optax"``.

    Parameters
    ----------
    fitters : sequence of JAXQSOFit
        Configured fitters, e.g. from :meth:`JAXQSOFit.from_arrays`.
    method : {"map", "laplace"}
        ``"map"`` (``"optax"`` is accepted) or a Laplace approximation at the MAP.
    chunk_size : int
        Spectra per compiled batch.
    redshift_window : float or None
        Widest ln(1 + z) spread of spectra sharing a line table; None gives each
        configuration group one table.
    num_samples : int
        Laplace draws per spectrum.
    seed : int, optional
        Base random seed; defaults to the first fitter's ``inference.random_seed``.
    progress_bar : bool
        Show a progress bar over chunks.

    Returns
    -------
    list of FitResult
        One result per fitter, in input order. Each fitter is also left in its
        fitted state (posterior products, ``optax_map_point``), as after
        :meth:`JAXQSOFit.fit`.
    """
    fitters = list(fitters)
    if not fitters:
        raise ValueError("fit_batch needs at least one fitter.")
    method = str(method).lower()
    if method not in _BATCH_METHODS:
        raise ValueError("fit_batch supports method 'map' (or 'optax') and 'laplace'.")
    kind = _BATCH_METHODS[method]
    preps = [q._prepare_fit(verbose=False) for q in fitters]
    if any(p["run_kwargs"]["use_psf_phot"] for p in preps):
        raise NotImplementedError("fit_batch does not support PSF photometry yet; fit those spectra with JAXQSOFit.fit.")
    base_key = jax.random.PRNGKey(int(fitters[0].config.inference.random_seed if seed is None else seed))

    groups: dict[str, list[int]] = {}
    for i, (q, prep) in enumerate(zip(fitters, preps)):
        groups.setdefault(_signature(q, prep), []).append(i)
    windows = [w for indices in groups.values() for w in _redshift_windows(fitters, indices, redshift_window)]
    progress = None
    if progress_bar:
        from tqdm.auto import tqdm

        progress = tqdm(total=sum(-(-len(w) // chunk_size) for w in windows), desc=f"fit_batch ({kind})", unit="chunk")

    results: list[Any] = [None] * len(fitters)
    for window in windows:
        table_range = (min(float(np.min(fitters[i].wave)) for i in window),
                       max(float(np.max(fitters[i].wave)) for i in window))
        n_pixels = max(len(fitters[i].wave) for i in window)
        # Stage split of JAXQSOFit.run_fsps_optax_fit.
        steps = int(preps[window[0]]["optax_steps"])
        stage1_steps = max(100, steps // 3)

        def new_fitter(inputs):
            layout = BatchedInputs(_batch_model, inputs)
            return layout, make_batched_fitter(
                layout,
                method=kind,
                steps=max(100, steps - stage1_steps),
                learning_rate=float(preps[window[0]]["optax_lr"]),
                staged=True,
                staged_steps=stage1_steps,
                num_samples=int(num_samples),
                continuum_model_kwargs={"stage1": True},
            )

        layout = fit = None
        for start in range(0, len(window), chunk_size):
            chunk = window[start:start + chunk_size]
            padded = list(chunk) + [chunk[-1]] * (chunk_size - len(chunk))
            inputs = [_model_inputs(fitters[i], preps[i], table_range, n_pixels) for i in padded]
            _clear_per_spectrum_caches()
            if layout is None:
                layout, fit = new_fitter(inputs)
            try:
                arrays, redshift = layout.stack(inputs)
            except ValueError:
                # A leaf that was constant in the first chunk varies here: recalibrate.
                layout, fit = new_fitter(inputs)
                arrays, redshift = layout.stack(inputs)
            init = [_init_values(fitters[i], preps[i], x) for i, x in zip(padded, inputs)]
            init_values = {k: np.stack([v[k] for v in init]) for k in init[0]}
            keys = jnp.stack([jax.random.fold_in(base_key, i) for i in padded])
            out = fit(arrays, redshift, keys, init_values=init_values)
            for j, i in enumerate(chunk):
                samples = {site: np.asarray(v[j]) for site, v in out["samples"].items()}
                map_point = {site: np.asarray(v[j]) for site, v in out["map"].items()}
                results[i] = _load_result(fitters[i], preps[i], table_range, samples, map_point, kind,
                                          float(out["loss"][j]), out.get("hessian_min_eigenvalue", [None] * len(padded))[j])
            del out, arrays, inputs, init, init_values
            _clear_per_spectrum_caches()
            _release_memory()
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return results


def _load_result(q, prep, table_range, samples, map_point, kind, loss, min_eigenvalue):
    """Leave ``q`` in its fitted state from batch draws, as ``run_fsps_optax_fit`` does."""
    run = prep["run_kwargs"]
    inputs = _model_inputs(q, prep, table_range)  # unpadded, same line table
    inputs.pop("stage1_pixel_mask")
    inputs.pop("pixel_mask")
    for key in ("flux", "return_line_components", "emit_deterministics"):
        inputs.pop(key)
    pred = Predictive(
        qso_fsps_joint_model,
        posterior_samples={k: jnp.asarray(v) for k, v in samples.items()},
        return_sites=q._predictive_return_sites(
            custom_components=run["custom_components"],
            custom_line_components=run["custom_line_components"],
        ),
    )
    pred_out = pred(jax.random.PRNGKey(int(run["random_seed"])), flux=None, **inputs)
    q._clear_nuts_run_state()
    q.numpyro_mcmc = None
    q.svi = q.svi_state = q.svi_params = None
    q.tied_line_meta = inputs["tied_line_meta"]
    q.optax_map_point = map_point
    q.optax_losses = np.asarray([loss])
    q.batch_diagnostics = {"loss": loss, "hessian_min_eigenvalue": None if min_eigenvalue is None else float(min_eigenvalue)}
    q._consume_posterior_outputs(
        samples=samples,
        pred_out=pred_out,
        fsps_grid=inputs["fsps_grid"],
        tied_line_meta=inputs["tied_line_meta"],
        use_lines=run["use_lines"],
        decompose_host=run["decompose_host"],
    )
    return q._make_result(method="optax" if kind == "map" else "laplace", path=None, figure=None)
