#!/usr/bin/env python
"""Profile the core jaxqsofit likelihood.

This is a manual diagnostic benchmark, not a CI acceptance test. It uses a
synthetic rest-frame spectrum and reports JIT-compiled log-density and
value+gradient timings with compilation excluded.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_cache_root = Path(tempfile.gettempdir()) / "jaxqsofit-profile-cache"
_mpl_cache = _cache_root / "matplotlib"
try:
    _cache_root.mkdir(parents=True, exist_ok=True)
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
except OSError:
    pass

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density

jax.config.update("jax_enable_x64", True)

from jaxqsofit.defaults import build_default_prior_config
from jaxqsofit.model import (
    _balmer_static_terms_jax,
    _normalize_template_flux,
    build_fsps_template_grid,
    build_tied_line_meta_from_linelist,
    qso_fsps_joint_model,
)


def _block(x: Any) -> None:
    jax.tree_util.tree_map(
        lambda y: y.block_until_ready() if hasattr(y, "block_until_ready") else y,
        x,
    )


def _bench(fn, arg: dict[str, Any], repeats: int) -> float:
    _block(fn(arg))
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(arg)
    _block(out)
    return (time.perf_counter() - t0) / max(repeats, 1) * 1e3


def _make_data(n_wave: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = np.exp(np.linspace(np.log(3500.0), np.log(8500.0), n_wave))
    continuum = 55.0 * (wave / 5100.0) ** -0.6
    host = 25.0 * np.exp(-0.5 * ((wave - 5200.0) / 1200.0) ** 2)
    ha = 70.0 * np.exp(-0.5 * ((wave - 6564.61) / 45.0) ** 2)
    hb = 30.0 * np.exp(-0.5 * ((wave - 4862.68) / 35.0) ** 2)
    oiii = 45.0 * np.exp(-0.5 * ((wave - 5008.24) / 6.0) ** 2)
    flux = continuum + host + ha + hb + oiii
    err = np.full_like(wave, 2.0)
    return wave, flux, err


def _materialize_prior(
    flux: np.ndarray,
    *,
    host_sfh_model: str,
    convolution_method: str,
    include_elg_narrow_lines: bool,
    include_high_ionization_lines: bool,
) -> dict[str, Any]:
    prior = build_default_prior_config(
        flux,
        include_elg_narrow_lines=include_elg_narrow_lines,
        include_high_ionization_lines=include_high_ionization_lines,
    )
    if hasattr(prior, "to_mapping"):
        prior = prior.to_mapping()
    prior = dict(prior)
    prior["host_sfh_model"] = host_sfh_model
    prior["convolution_method"] = convolution_method
    prior["resolving_power"] = None
    prior["student_t_df"] = 5.0

    if host_sfh_model == "delayed":
        prior["log_host_aperture_scale"] = {"dist": "Delta", "value": 0.0}
        prior["mass_metallicity_relation"] = {"enabled": False}
    return prior


def _load_fe_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = REPO_ROOT / "src" / "jaxqsofit" / "data"
    fe_uv = np.genfromtxt(data_dir / "fe_uv.txt")
    fe_op = np.genfromtxt(data_dir / "fe_optical.txt")

    fe_uv_wave = 10 ** fe_uv[:, 0]
    fe_uv_flux = _normalize_template_flux(np.maximum(fe_uv[:, 1], 0.0), target_amp=1.0)

    fe_op_wave_all = 10 ** fe_op[:, 0]
    fe_op_flux_all = _normalize_template_flux(np.maximum(fe_op[:, 1], 0.0), target_amp=1.0)
    optical = (fe_op_wave_all > 3686.0) & (fe_op_wave_all < 7484.0)
    fe_op_wave = fe_op_wave_all[optical]
    fe_op_flux = fe_op_flux_all[optical]
    return fe_uv_wave, fe_uv_flux, fe_op_wave, fe_op_flux


def _resolve_dsps_ssp_fn(path: str) -> str:
    dsps_path = Path(path)
    if not dsps_path.is_absolute():
        dsps_path = REPO_ROOT / dsps_path
    return str(dsps_path)


def _make_model_kwargs(args: argparse.Namespace, host_sfh_model: str) -> dict[str, Any]:
    wave, flux, err = _make_data(args.n_wave)
    prior = _materialize_prior(
        flux,
        host_sfh_model=host_sfh_model,
        convolution_method=args.convolution,
        include_elg_narrow_lines=args.include_elg_narrow_lines,
        include_high_ionization_lines=args.include_high_ionization_lines,
    )

    tied_line_meta = build_tied_line_meta_from_linelist(prior["line"]["table"], wave)
    fsps_grid = build_fsps_template_grid(
        wave,
        age_grid_gyr=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
        dsps_ssp_fn=_resolve_dsps_ssp_fn(args.dsps_ssp_fn),
        z_qso=args.z_qso,
    )
    fe_uv_wave, fe_uv_flux, fe_op_wave, fe_op_flux = _load_fe_templates()

    kwargs = dict(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta=tied_line_meta,
        fsps_grid=fsps_grid,
        fe_uv_wave=fe_uv_wave,
        fe_uv_flux=fe_uv_flux,
        fe_op_wave=fe_op_wave,
        fe_op_flux=fe_op_flux,
        prior_config=prior,
        z_qso=args.z_qso,
        psf_mags=None,
        psf_mag_errs=None,
        psf_filter_curves=None,
        use_psf_phot=False,
        return_line_components=False,
        emit_deterministics=False,
        custom_components=(),
        custom_line_components=(),
    )

    if not args.no_cache:
        balmer_bb_shape, balmer_tau_shape, balmer_below_edge = _balmer_static_terms_jax(
            jnp.asarray(wave),
            balmer_te=15000.0,
        )
        kwargs.update(
            fe_uv_flux_on_wave=np.interp(wave, fe_uv_wave, fe_uv_flux, left=0.0, right=0.0),
            fe_op_flux_on_wave=np.interp(wave, fe_op_wave, fe_op_flux, left=0.0, right=0.0),
            balmer_bb_shape=np.asarray(balmer_bb_shape),
            balmer_tau_shape=np.asarray(balmer_tau_shape),
            balmer_below_edge=np.asarray(balmer_below_edge),
        )
    return kwargs


def _initial_params(kwargs: dict[str, Any], seed_value: int) -> dict[str, Any]:
    tr = trace(seed(qso_fsps_joint_model, jax.random.PRNGKey(seed_value))).get_trace(**kwargs)
    return {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _profile_variant(
    name: str,
    base_kwargs: dict[str, Any],
    *,
    repeats: int,
    grad_repeats: int,
    seed_value: int,
    **switches: Any,
) -> tuple[str, int, float, float]:
    kwargs = dict(base_kwargs)
    kwargs.update(switches)
    params = _initial_params(kwargs, seed_value)

    def ld(p):
        val, _ = log_density(qso_fsps_joint_model, (), kwargs, p)
        return val

    jit_ld = jax.jit(ld)
    jit_vg = jax.jit(jax.value_and_grad(ld))
    forward_ms = _bench(jit_ld, params, repeats=repeats)
    grad_ms = _bench(jit_vg, params, repeats=grad_repeats)
    return name, len(params), forward_ms, grad_ms


def _active_shape_summary(base_kwargs: dict[str, Any]) -> str:
    tied = base_kwargs["tied_line_meta"]
    fsps_grid = base_kwargs["fsps_grid"]
    line_count = int(tied["n_lines"])
    v_groups = int(tied["n_vgroups"])
    w_groups = int(tied["n_wgroups"])
    f_groups = int(tied["n_fgroups"])
    template_matrix = tuple(np.asarray(fsps_grid.templates).shape)
    return (
        f"n_wave={len(base_kwargs['wave'])}, n_lines={line_count}, "
        f"groups(v/w/f)=({v_groups}/{w_groups}/{f_groups}), "
        f"fsps_template_matrix={template_matrix}"
    )


def _host_modes(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return ("delayed", "flexible")
    return (selection,)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wave", type=int, default=3690, help="Number of synthetic spectral pixels.")
    parser.add_argument(
        "--host-sfh",
        choices=("delayed", "flexible", "both"),
        default="both",
        help="Host SFH mode to profile.",
    )
    parser.add_argument("--repeats", type=int, default=30, help="Forward log-density timing repeats.")
    parser.add_argument(
        "--grad-repeats",
        type=int,
        default=15,
        help="Value+gradient timing repeats.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed used to draw initial latent values.")
    parser.add_argument("--z-qso", type=float, default=0.1, help="Synthetic source redshift.")
    parser.add_argument(
        "--dsps-ssp-fn",
        default="tempdata.h5",
        help="DSPS SSP template file. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--convolution",
        choices=("fft", "direct"),
        default="fft",
        help="Fe/Balmer broadening convolution backend.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable static Fe/Balmer interpolation caches for comparison.",
    )
    parser.add_argument(
        "--include-elg-narrow-lines",
        action="store_true",
        help="Include ELG narrow lines in the active line table.",
    )
    parser.add_argument(
        "--include-high-ionization-lines",
        action="store_true",
        help="Include high-ionization lines in the active line table.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"backend={jax.default_backend()} jax={jax.__version__}")
    print("timings exclude JIT compilation; units are milliseconds per call")
    print(
        f"convolution={args.convolution} static_cache={'off' if args.no_cache else 'on'} "
        f"repeats={args.repeats} grad_repeats={args.grad_repeats}"
    )

    variants = (
        (
            "full lines+host+Fe+Balmer+poly+red",
            dict(
                use_lines=True,
                decompose_host=True,
                fit_pl=True,
                fit_fe=True,
                fit_bc=True,
                fit_poly=True,
                fit_reddening=True,
            ),
        ),
        (
            "no lines",
            dict(
                use_lines=False,
                decompose_host=True,
                fit_pl=True,
                fit_fe=True,
                fit_bc=True,
                fit_poly=True,
                fit_reddening=True,
            ),
        ),
        (
            "no Fe/Balmer",
            dict(
                use_lines=True,
                decompose_host=True,
                fit_pl=True,
                fit_fe=False,
                fit_bc=False,
                fit_poly=True,
                fit_reddening=True,
            ),
        ),
        (
            "no host",
            dict(
                use_lines=True,
                decompose_host=False,
                fit_pl=True,
                fit_fe=True,
                fit_bc=True,
                fit_poly=True,
                fit_reddening=True,
            ),
        ),
        (
            "continuum+likelihood only",
            dict(
                use_lines=False,
                decompose_host=False,
                fit_pl=True,
                fit_fe=False,
                fit_bc=False,
                fit_poly=False,
                fit_reddening=False,
            ),
        ),
    )

    for host_sfh in _host_modes(args.host_sfh):
        print(f"\nHost SFH: {host_sfh}")
        base_kwargs = _make_model_kwargs(args, host_sfh)
        print(_active_shape_summary(base_kwargs))
        print(f"{'variant':36s} {'params':>7s} {'forward':>12s} {'value+grad':>12s}")
        print("-" * 72)
        for name, switches in variants:
            result = _profile_variant(
                name,
                base_kwargs,
                repeats=args.repeats,
                grad_repeats=args.grad_repeats,
                seed_value=args.seed,
                **switches,
            )
            variant, n_params, forward_ms, grad_ms = result
            print(f"{variant:36s} {n_params:7d} {forward_ms:12.3f} {grad_ms:12.3f}")


if __name__ == "__main__":
    main()
