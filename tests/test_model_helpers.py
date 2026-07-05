import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace
from types import SimpleNamespace
from jaxsedfit.host import HostBasisJax

import jaxqsofit.model as model_mod
from jaxqsofit.defaults import _build_default_prior_config as build_default_prior_config
from jaxqsofit.custom_components import make_custom_component
from jaxqsofit.model import (
    _delayed_sfh_host_spectrum,
    _fe_template_component,
    _flexible_host_raw_weight_locs,
    _host_redshift_prior_params,
    _smc_like_reddening_jax,
    _smooth_bounded_affine,
    _extract_line_table_from_prior_config,
    _luminosity_distance_cm_jax,
    _shift_and_broaden_single_spectrum_lnlam,
    build_host_template_grid,
    build_fsps_template_grid,
    build_tied_line_metadata,
    build_tied_line_meta_from_linelist,
    gaussian_bal_optical_depth_component,
    negative_bal_component,
    quasar_spectral_model,
    qso_fsps_joint_model,
    reconstruct_spectral_components,
    reconstruct_posterior_components,
)


def test_extract_line_table_from_prior_config_uses_canonical_layout():
    table = [{'lambda': 5008.24, 'linename': 'OIII5007', 'compname': 'Hb', 'ngauss': 1, 'inisca': 1.0, 'minsca': 0.0, 'maxsca': 1e3, 'inisig': 1e-3, 'minsig': 1e-4, 'maxsig': 1e-2, 'voff': 0.01, 'vindex': 1, 'windex': 1, 'findex': 1, 'fvalue': 1.0}]

    assert _extract_line_table_from_prior_config({'line': {'table': table}}) is table
    assert _extract_line_table_from_prior_config({'line_priors': table}) is None
    assert _extract_line_table_from_prior_config({'line_table': table}) is None
    assert _extract_line_table_from_prior_config({'line': {'priors': table}}) is None


def test_package_enables_jax_x64_explicitly():
    assert jax.config.jax_enable_x64 is True


def test_gaussian_bal_optical_depth_component_uses_velocity_fwhm_width():
    line_lambda = 1549.06
    v_out = 6000.0
    fwhm_kms = 8000.0
    center = line_lambda * (1.0 - v_out / model_mod.C_KMS)
    sigma_lambda = center * (fwhm_kms / 2.354820045) / model_mod.C_KMS
    wave = jnp.asarray([center, center + sigma_lambda])
    tau = gaussian_bal_optical_depth_component(
        wave,
        {
            "v_out": jnp.asarray(v_out),
            "fwhm_kms": jnp.asarray(fwhm_kms),
            "tau_peak": jnp.asarray(1.0),
            "shape_power": jnp.asarray(2.0),
        },
        {"line_lambda": line_lambda},
    )

    np.testing.assert_allclose(np.asarray(tau), np.asarray([1.0, np.exp(-0.5)]), rtol=1e-6)


def test_public_model_names_delegate_to_legacy_implementations(monkeypatch):
    calls = {}

    def _legacy_model(*args, **kwargs):
        calls["model"] = (args, kwargs)
        return "model"

    def _reconstruct(*args, **kwargs):
        calls["reconstruct"] = (args, kwargs)
        return "components"

    def _grid(*args, **kwargs):
        calls["grid"] = (args, kwargs)
        return "grid"

    def _line_meta(*args, **kwargs):
        calls["line_meta"] = (args, kwargs)
        return "line_meta"

    def _bal(*args, **kwargs):
        calls["bal"] = (args, kwargs)
        return "bal"

    monkeypatch.setattr(model_mod, "qso_fsps_joint_model", _legacy_model)
    monkeypatch.setattr(model_mod, "reconstruct_posterior_components", _reconstruct)
    monkeypatch.setattr(model_mod, "build_fsps_template_grid", _grid)
    monkeypatch.setattr(model_mod, "build_tied_line_meta_from_linelist", _line_meta)
    monkeypatch.setattr(model_mod, "negative_gaussian_bal_component", _bal)

    assert quasar_spectral_model("wave", fit_pl=True) == "model"
    assert reconstruct_spectral_components("samples", n_draws=2) == "components"
    assert build_host_template_grid(dsps_ssp_fn="tempdata.h5") == "grid"
    assert build_tied_line_metadata([], "wave") == "line_meta"
    assert negative_bal_component("wave", params={}, metadata={}) == "bal"
    assert calls["model"][0] == ("wave",)
    assert calls["model"][1]["fit_pl"] is True
    assert calls["reconstruct"][1]["n_draws"] == 2
    assert calls["grid"][1]["dsps_ssp_fn"] == "tempdata.h5"
    assert calls["line_meta"][0] == ([], "wave")
    assert calls["bal"][1]["metadata"] == {}


def test_qso_model_honors_numpyro_prior_distribution_families():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    cfg = build_default_prior_config(flux).to_mapping()
    cfg["PL_slope"] = {"dist": "TruncatedNormal", "loc": -1.5, "scale": 0.2, "low": -2.5, "high": -0.5}
    cfg["log_Fe_uv_norm"] = {"dist": "Normal", "loc": np.log(0.01), "scale": 0.1}
    model = substitute(
        seed(qso_fsps_joint_model, jax.random.PRNGKey(0)),
        data={
            "PL_norm": np.array(1.0),
            "PL_slope": np.array(-1.5),
            "log_Fe_uv_norm": np.array(np.log(0.01)),
            "log_Fe_op_over_uv": np.array(0.0),
            "Fe_uv_FWHM": np.array(3000.0),
            "Fe_op_FWHM": np.array(3000.0),
            "Fe_uv_shift": np.array(0.0),
            "Fe_op_shift": np.array(0.0),
            "frac_jitter": np.array(0.0),
            "add_jitter": np.array(0.0),
        },
    )
    tr = trace(model).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.ones(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=False,
        fit_pl=True,
        fit_fe=True,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
    )

    assert tr["PL_slope"]["fn"].__class__.__name__ in {"TruncatedNormal", "TwoSidedTruncatedDistribution"}
    assert tr["log_Fe_uv_norm"]["type"] == "sample"
    assert tr["Fe_uv_norm"]["type"] == "deterministic"
    np.testing.assert_allclose(np.asarray(tr["Fe_uv_norm"]["value"]), 0.01)

    cfg["log_Fe_uv_norm"] = {"dist": "LogNormal", "loc": np.log(0.01), "scale": 0.1}
    model = substitute(
        seed(qso_fsps_joint_model, jax.random.PRNGKey(1)),
        data={
            "PL_norm": np.array(1.0),
            "PL_slope": np.array(-1.5),
            "Fe_uv_norm": np.array(0.01),
            "log_Fe_op_over_uv": np.array(0.0),
            "Fe_uv_FWHM": np.array(3000.0),
            "Fe_op_FWHM": np.array(3000.0),
            "Fe_uv_shift": np.array(0.0),
            "Fe_op_shift": np.array(0.0),
            "frac_jitter": np.array(0.0),
            "add_jitter": np.array(0.0),
        },
    )
    tr = trace(model).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.ones(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=False,
        fit_pl=True,
        fit_fe=True,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
    )

    assert "log_Fe_uv_norm" not in tr
    assert tr["Fe_uv_norm"]["type"] == "sample"
    assert tr["Fe_uv_norm"]["fn"].__class__.__name__ == "LogNormal"


def test_reddening_a2500_is_sampled_in_log_space_and_exposed():
    wave = np.linspace(2000.0, 3000.0, 8)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    prior_config = build_default_prior_config(flux).to_mapping()
    prior_config["PL_pivot"] = 2500.0
    prior_config["poly_pivot"] = 2500.0
    fsps_grid = SimpleNamespace(templates=np.zeros((wave.size, 1)))
    tied_line_meta = build_tied_line_meta_from_linelist([], wave)

    model_trace = trace(seed(qso_fsps_joint_model, 0)).get_trace(
        wave,
        flux,
        err,
        None,
        tied_line_meta,
        fsps_grid,
        np.array([2000.0, 3000.0]),
        np.zeros(2),
        np.array([2000.0, 3000.0]),
        np.zeros(2),
        use_lines=False,
        prior_config=prior_config,
        decompose_host=False,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=True,
    )

    assert model_trace["log_reddening_a2500"]["type"] == "sample"
    assert model_trace["reddening_a2500"]["type"] == "deterministic"


def test_fe_template_component_smoothly_bounds_fwhm_below_template_base():
    wave = jnp.linspace(1900.0, 3100.0, 256)
    wave_template = jnp.linspace(2000.0, 3000.0, 128)
    flux_template = jnp.exp(-0.5 * ((wave_template - 2500.0) / 80.0) ** 2)

    def component_sum(fwhm_kms):
        return jnp.sum(
            _fe_template_component(
                wave,
                wave_template,
                flux_template,
                norm=1.0,
                fwhm_kms=fwhm_kms,
                shift_frac=0.0,
                base_fwhm_kms=2000.0,
            )
        )

    component = _fe_template_component(
        wave,
        wave_template,
        flux_template,
        norm=1.0,
        fwhm_kms=1500.0,
        shift_frac=0.0,
        base_fwhm_kms=2000.0,
    )
    grad = jax.grad(component_sum)(1500.0)

    assert bool(jnp.all(jnp.isfinite(component)))
    assert float(jnp.max(component)) > 0.0
    assert bool(jnp.isfinite(grad))


def test_shift_and_broaden_uses_wide_kernel_for_broad_components():
    n_pix = 4097
    dln = 1e-4
    lnwave = jnp.log(5000.0) + dln * (jnp.arange(n_pix) - n_pix // 2)
    spectrum = jnp.zeros(n_pix).at[n_pix // 2].set(1.0)
    sigma_kms = 200.0 * dln * model_mod.C_KMS

    broadened = _shift_and_broaden_single_spectrum_lnlam(
        lnwave,
        spectrum,
        v_kms=0.0,
        sigma_kms=sigma_kms,
    )

    assert bool(jnp.all(jnp.isfinite(broadened)))
    assert float(jnp.sum(broadened)) > 0.99
    assert float(jnp.max(broadened)) < 0.003


def test_smc_like_reddening_parameter_is_a2500_magnitude():
    atten = _smc_like_reddening_jax(jnp.asarray([2500.0]), a_uv=1.0)

    assert np.isclose(float(atten[0]), 10.0 ** -0.4)


def test_build_fsps_template_grid_can_reuse_fit_template_norms(monkeypatch):
    ssp_wave = np.linspace(1000.0, 5000.0, 10)

    class _SSPData:
        pass

    _SSPData.ssp_lgmet = np.array([0.0])
    _SSPData.ssp_lg_age_gyr = np.array([0.0])
    _SSPData.ssp_wave = ssp_wave
    _SSPData.ssp_flux = (1.0 + 0.001 * (ssp_wave - 1000.0))[None, None, :]

    monkeypatch.setattr(model_mod, "load_ssp_templates", lambda fn: _SSPData())

    fit_grid = build_fsps_template_grid(
        wave_out=np.linspace(3000.0, 5000.0, 20),
        age_grid_gyr=[1.0],
        logzsol_grid=[0.0],
        dsps_ssp_fn="unused.h5",
        build_physical_host_basis=False,
    )
    fit_norms = [meta["norm"] for meta in fit_grid.template_meta]
    recon_grid = build_fsps_template_grid(
        wave_out=np.linspace(2000.0, 5000.0, 20),
        age_grid_gyr=[1.0],
        logzsol_grid=[0.0],
        dsps_ssp_fn="unused.h5",
        build_physical_host_basis=False,
        template_norms=fit_norms,
    )

    assert recon_grid.template_meta[0]["norm"] == fit_norms[0]
    assert not np.isclose(np.nanmedian(np.abs(recon_grid.templates[:, 0])), 1.0)


def test_reconstruct_requires_fit_grid_pivots_for_grid_dependent_terms():
    wave = np.linspace(2000.0, 3600.0, 8)
    samples = {
        "PL_norm": np.array([1.0]),
        "PL_slope": np.array([0.0]),
        "fsps_weights": np.zeros((1, 1)),
    }

    try:
        reconstruct_posterior_components(
            wave,
            samples,
            pred_out=None,
            age_grid_gyr=[1.0],
            logzsol_grid=[0.0],
            dsps_ssp_fn="unused.h5",
            prior_config={},
            fit_poly=False,
            fit_poly_order=0,
            fit_reddening=False,
            fe_uv_wave=np.array([2000.0, 3600.0]),
            fe_uv_flux=np.zeros(2),
            fe_op_wave=np.array([2000.0, 3600.0]),
            fe_op_flux=np.zeros(2),
            decompose_host=False,
        )
    except ValueError as exc:
        assert "PL_pivot" in str(exc)
    else:
        raise AssertionError("missing PL_pivot should fail reconstruction with PL samples")

    samples = {
        "PL_norm": np.array([0.0]),
        "poly_c1": np.array([0.1]),
        "fsps_weights": np.zeros((1, 1)),
    }
    try:
        reconstruct_posterior_components(
            wave,
            samples,
            pred_out=None,
            age_grid_gyr=[1.0],
            logzsol_grid=[0.0],
            dsps_ssp_fn="unused.h5",
            prior_config={"PL_pivot": 2500.0},
            fit_poly=True,
            fit_poly_order=1,
            fit_reddening=False,
            fe_uv_wave=np.array([2000.0, 3600.0]),
            fe_uv_flux=np.zeros(2),
            fe_op_wave=np.array([2000.0, 3600.0]),
            fe_op_flux=np.zeros(2),
            decompose_host=False,
        )
    except ValueError as exc:
        assert "poly_pivot" in str(exc)
    else:
        raise AssertionError("missing poly_pivot should fail reconstruction with polynomial samples")


def test_reconstruct_reddening_applies_to_nuclear_continuum_components():
    wave = np.linspace(2000.0, 3600.0, 80)
    samples = {
        "PL_norm": np.array([10.0]),
        "PL_slope": np.array([0.0]),
        "Fe_uv_norm": np.array([5.0]),
        "log_Fe_op_over_uv": np.array([0.0]),
        "Fe_uv_FWHM": np.array([3000.0]),
        "Fe_op_FWHM": np.array([3000.0]),
        "Fe_uv_shift": np.array([0.0]),
        "Fe_op_shift": np.array([0.0]),
        "Balmer_norm": np.array([3.0]),
        "Balmer_Tau": np.array([0.5]),
        "Balmer_vel": np.array([3000.0]),
        "reddening_a2500": np.array([1.0]),
        "fsps_weights": np.zeros((1, 1)),
        "gal_v_kms": np.array([0.0]),
        "gal_sigma_kms": np.array([100.0]),
    }
    prior_config = {"PL_pivot": 2500.0, "reddening_uv_ref": 2500.0, "reddening_alpha": 1.2}
    fe_wave = np.linspace(1900.0, 3700.0, 80)
    fe_flux = np.ones_like(fe_wave)

    plain = reconstruct_posterior_components(
        wave,
        samples,
        pred_out=None,
        age_grid_gyr=[1.0],
        logzsol_grid=[0.0],
        dsps_ssp_fn="unused.h5",
        prior_config=prior_config,
        fit_poly=False,
        fit_poly_order=0,
        fit_reddening=False,
        fe_uv_wave=fe_wave,
        fe_uv_flux=fe_flux,
        fe_op_wave=fe_wave,
        fe_op_flux=fe_flux,
        decompose_host=False,
    )
    reddened = reconstruct_posterior_components(
        wave,
        samples,
        pred_out=None,
        age_grid_gyr=[1.0],
        logzsol_grid=[0.0],
        dsps_ssp_fn="unused.h5",
        prior_config=prior_config,
        fit_poly=False,
        fit_poly_order=0,
        fit_reddening=True,
        fe_uv_wave=fe_wave,
        fe_uv_flux=fe_flux,
        fe_op_wave=fe_wave,
        fe_op_flux=fe_flux,
        decompose_host=False,
    )
    idx = int(np.argmin(np.abs(wave - 2500.0)))
    expected = float(_smc_like_reddening_jax(jnp.asarray([wave[idx]]), a_uv=1.0)[0])

    for key in ("PL", "Fe_uv", "Fe_op", "Balmer_cont"):
        ratio = reddened["draws"][key][0, idx] / plain["draws"][key][0, idx]
        assert np.isclose(ratio, expected, rtol=1e-4)


def test_build_tied_line_meta_from_linelist_minimal():
    line_table = [
        {
            'lambda': 5008.24,
            'linename': 'OIII5007',
            'compname': 'Hb',
            'ngauss': 1,
            'inisca': 1.0,
            'minsca': 0.0,
            'maxsca': 1e3,
            'inisig': 1e-3,
            'minsig': 1e-4,
            'maxsig': 1e-2,
            'voff': 0.01,
            'vindex': 1,
            'windex': 1,
            'findex': 1,
            'fvalue': 1.0,
        },
        {
            'lambda': 4960.30,
            'linename': 'OIII4959',
            'compname': 'Hb',
            'ngauss': 1,
            'inisca': 0.3,
            'minsca': 0.0,
            'maxsca': 1e3,
            'inisig': 1e-3,
            'minsig': 1e-4,
            'maxsig': 1e-2,
            'voff': 0.01,
            'vindex': 1,
            'windex': 1,
            'findex': 1,
            'fvalue': 0.33,
        },
    ]
    wave = np.linspace(4800.0, 5100.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta['n_lines'] == 2
    assert meta['n_vgroups'] >= 1
    assert meta['n_wgroups'] >= 1
    assert meta['n_fgroups'] >= 1
    assert len(meta['names']) == 2
    assert np.all(np.isfinite(meta['line_lambda']))
    for key in ("vgroup_jax", "wgroup_jax", "fgroup_jax", "flux_ratio_jax", "broad_mask_jax"):
        assert key in meta


def test_build_tied_line_meta_uses_fvalue_when_inisca_is_zero():
    line_table = [
        {
            "lambda": 6564.61,
            "linename": "Ha_br",
            "compname": "Ha",
            "ngauss": 1,
            "inisca": 0.0,
            "minsca": 0.0,
            "maxsca": 1e3,
            "inisig": 5e-3,
            "minsig": 1e-3,
            "maxsig": 5e-2,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 0.05,
        }
    ]
    wave = np.linspace(6400.0, 6800.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta["n_fgroups"] == 1
    assert np.allclose(meta["amp_init_group"], [0.05])


def test_build_tied_line_meta_moves_amp_init_off_lower_boundary():
    line_table = [
        {
            "lambda": 6564.61,
            "linename": "Ha_br",
            "compname": "Ha",
            "ngauss": 1,
            "inisca": 0.0,
            "minsca": 1.0e-4,
            "maxsca": 0.12,
            "inisig": 5e-3,
            "minsig": 1e-3,
            "maxsig": 5e-2,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0e-4,
        }
    ]
    wave = np.linspace(6400.0, 6800.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta["amp_init_group"][0] > meta["amp_min_group"][0]
    assert meta["amp_init_group"][0] < meta["amp_max_group"][0]


def test_build_tied_line_meta_includes_lines_with_broad_wings_overlapping_window():
    base = {
        "compname": "edge",
        "ngauss": 1,
        "inisca": 1.0,
        "minsca": 0.0,
        "maxsca": 1e3,
        "inisig": 0.001,
        "minsig": 0.0005,
        "voff": 0.005,
        "vindex": 0,
        "windex": 0,
        "findex": 0,
        "fvalue": 1.0,
    }
    line_table = [
        {
            **base,
            "lambda": 1450.0,
            "linename": "broad_edge",
            "maxsig": 0.02,
        },
        {
            **base,
            "lambda": 1450.0,
            "linename": "narrow_edge",
            "maxsig": 0.001,
        },
    ]
    wave = np.linspace(1500.0, 1700.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta["names"] == ["broad_edge_1"]
    assert meta["n_lines"] == 1


def test_build_tied_line_meta_uses_voff_as_log_wavelength_offset():
    line_table = [
        {
            "lambda": 1549.06,
            "linename": "CIV_br",
            "compname": "CIV",
            "ngauss": 1,
            "inisca": 1.0,
            "minsca": 0.0,
            "maxsca": 1e3,
            "inisig": 0.01,
            "minsig": 0.001,
            "maxsig": 0.05,
            "voff": 0.015,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        }
    ]
    wave = np.linspace(1500.0, 1700.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta["n_vgroups"] == 1
    assert np.allclose(meta["dmu_min_group"], [-0.015])
    assert np.allclose(meta["dmu_max_group"], [0.015])


def test_build_tied_line_meta_expands_ngauss_into_independent_groups():
    line_table = [
        {
            "lambda": 1549.06,
            "linename": "CIV_br",
            "compname": "CIV",
            "ngauss": 3,
            "inisca": 1.0,
            "minsca": 0.0,
            "maxsca": 1e3,
            "inisig": 0.01,
            "minsig": 0.001,
            "maxsig": 0.05,
            "voff": 0.015,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        }
    ]
    wave = np.linspace(1500.0, 1700.0, 200)

    meta = build_tied_line_meta_from_linelist(line_table, wave)

    assert meta["n_lines"] == 3
    assert meta["n_vgroups"] == 3
    assert meta["n_wgroups"] == 3
    assert meta["n_fgroups"] == 3
    assert len(set(meta["compnames"])) == 3
    assert np.allclose(meta["flux_ratio"], np.ones(3))


def test_qso_fsps_joint_model_reports_log_lambda_llambda_requested_continuum_luminosities():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux).to_mapping()
    cfg["host_sfh_model"] = "flexible"

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(1.0),
        "PL_norm": np.array(5.0e6),
        "PL_slope": np.array(0.0),
        "tau_host": np.array(1.0),
        "fsps_weights_raw": np.array([0.0]),
        "gal_v_kms": np.array(0.0),
        "log_gal_sigma_kms": np.log(100.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=1.0,
    )

    for wave_label in ("1350", "2500", "3000", "5100"):
        site_name = f"log_lambda_Llambda_{wave_label}_agn"
        assert site_name in tr
        assert np.isfinite(float(tr[site_name]["value"]))


def test_flexible_host_age_prior_prefers_old_template_logits():
    class _Grid:
        templates = np.zeros((8, 3), dtype=float)
        template_meta = [
            {"tage_gyr": 0.1, "logzsol": 0.0},
            {"tage_gyr": 1.0, "logzsol": 0.0},
            {"tage_gyr": 10.0, "logzsol": 0.0},
        ]

    cfg = {
        "raw_w": {"dist": "Normal", "loc": -0.5, "scale": 1.0},
        "host_template_age_prior": {
            "type": "prefer_old",
            "pivot_gyr": 1.0,
            "strength": 1.0,
        },
    }

    loc = np.asarray(_flexible_host_raw_weight_locs(_Grid(), cfg, 3), dtype=float)

    assert loc[0] < loc[1] < loc[2]
    assert np.allclose(loc, [-1.5, -0.5, 0.5])


def test_flexible_host_age_prior_can_be_disabled():
    class _Grid:
        templates = np.zeros((8, 2), dtype=float)
        template_meta = [
            {"tage_gyr": 0.1, "logzsol": 0.0},
            {"tage_gyr": 10.0, "logzsol": 0.0},
        ]

    cfg = {
        "raw_w": {"dist": "Normal", "loc": -0.5, "scale": 1.0},
        "host_template_age_prior": {"enabled": False},
    }

    loc = np.asarray(_flexible_host_raw_weight_locs(_Grid(), cfg, 2), dtype=float)

    assert np.allclose(loc, [-0.5, -0.5])


def test_qso_fsps_joint_model_supports_delayed_sfh_host_with_mzr():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux).to_mapping()
    cfg["host_sfh_model"] = "delayed"
    cfg["mass_metallicity_relation"] = {
        "enabled": True,
        "pivot_mass": 10.0,
        "pivot_logzsol": -0.1,
        "slope": 0.25,
        "scale": 0.3,
    }

    class _Grid:
        templates = np.column_stack(
            [
                np.ones(wave.size),
                np.linspace(0.8, 1.2, wave.size),
                np.linspace(1.2, 0.8, wave.size),
                np.full(wave.size, 0.7),
            ]
        )
        template_meta = [
            {"tage_gyr": 0.1, "logzsol": -0.5},
            {"tage_gyr": 1.0, "logzsol": -0.5},
            {"tage_gyr": 0.1, "logzsol": 0.0},
            {"tage_gyr": 1.0, "logzsol": 0.0},
        ]
        age_grid_gyr = np.array([0.1, 1.0])
        logzsol_grid = np.array([-0.5, 0.0])

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(1.0),
        "PL_norm": np.array(1.0),
        "PL_slope": np.array(0.0),
        "log_stellar_mass": np.array(10.0),
        "log_sfh_age_gyr": np.log(1.0),
        "log_sfh_tau_over_age": np.log(0.5),
        "gal_lgmet": np.array(-0.1),
        "log_gal_lgmet_scatter": np.log(0.2),
        "gal_v_kms": np.array(0.0),
        "log_gal_sigma_kms": np.log(100.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
    )

    weights = np.asarray(tr["fsps_weights_frac"]["value"])
    assert weights.shape == (4,)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0.0)
    assert "host_amp" in tr
    assert "mass_metallicity_relation_prior" in tr
    assert "mass_metallicity_relation_logprior" in tr
    assert np.isfinite(float(tr["mass_metallicity_relation_logprior"]["value"]))
    assert np.isfinite(float(tr["sfh_age_gyr"]["value"]))
    assert np.isfinite(float(tr["sfh_tau_gyr"]["value"]))


def test_delayed_sfh_host_uses_physical_stellar_mass_scaling():
    wave = np.linspace(4000.0, 4100.0, 16)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux).to_mapping()
    cfg["host_sfh_model"] = "delayed"
    cfg["mass_metallicity_relation"] = {"enabled": False}
    cfg["log_host_aperture_scale"] = {"dist": "Normal", "loc": 0.0, "scale": 0.1}

    class _Grid:
        templates = np.zeros((wave.size, 4), dtype=float)
        template_meta = [
            {"tage_gyr": 0.1, "logzsol": -0.5, "dsps_lg_age_gyr": -1.0, "dsps_lgmet": -1.0},
            {"tage_gyr": 1.0, "logzsol": -0.5, "dsps_lg_age_gyr": 0.0, "dsps_lgmet": -1.0},
            {"tage_gyr": 0.1, "logzsol": 0.0, "dsps_lg_age_gyr": -1.0, "dsps_lgmet": -0.5},
            {"tage_gyr": 1.0, "logzsol": 0.0, "dsps_lg_age_gyr": 0.0, "dsps_lgmet": -0.5},
        ]
        age_grid_gyr = np.array([0.1, 1.0])
        logzsol_grid = np.array([-0.5, 0.0])
        host_basis_jax = HostBasisJax(
            ssp_lgmet=jnp.array([-1.0, -0.5, 0.0], dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.log10(jnp.array([0.1, 0.5, 1.0], dtype=jnp.float64)),
            rest_llambda=jnp.ones((3, 3, wave.size), dtype=jnp.float64),
            surviving_frac_by_age=jnp.ones((3,), dtype=jnp.float64),
            n_ly_per_msun=jnp.zeros((3, 3), dtype=jnp.float64),
            ly_lum_per_msun=jnp.zeros((3, 3), dtype=jnp.float64),
            gal_t_table=jnp.geomspace(0.01, 1.2, 16),
        )
        t_obs_gyr = 1.2

    base_params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(0.0),
        "PL_norm": np.array(0.0),
        "PL_slope": np.array(0.0),
        "log_sfh_age_gyr": np.log(1.0),
        "log_sfh_tau_over_age": np.log(0.5),
        "gal_lgmet": np.array(-0.5),
        "log_gal_lgmet_scatter": np.log(0.2),
        "log_host_aperture_scale": np.array(0.0),
        "gal_v_kms": np.array(0.0),
        "log_gal_sigma_kms": np.log(1.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }

    def _host_for_mass(log_stellar_mass):
        params = dict(base_params, log_stellar_mass=np.array(log_stellar_mass))
        tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
            wave=wave,
            flux=flux,
            err=err,
            conti_priors={},
            tied_line_meta={"n_lines": 0},
            fsps_grid=_Grid(),
            fe_uv_wave=np.array([4000.0, 4100.0]),
            fe_uv_flux=np.zeros(2),
            fe_op_wave=np.array([4000.0, 4100.0]),
            fe_op_flux=np.zeros(2),
            use_lines=False,
            prior_config=cfg,
            decompose_host=True,
            fit_pl=False,
            fit_fe=False,
            fit_bc=False,
            fit_poly=False,
            fit_reddening=False,
            z_qso=0.1,
        )
        assert "log_host_amp" not in tr
        assert "host_amp" in tr
        return np.asarray(tr["gal_model_intrinsic"]["value"], dtype=float)

    host_low = _host_for_mass(8.0)
    host_high = _host_for_mass(10.0)

    mask = host_low > 0.0
    assert np.any(mask)
    assert np.allclose(host_high[mask] / host_low[mask], 100.0, rtol=1e-6)


def test_psf_host_fraction_applies_to_total_host_before_fiber_aperture_scale():
    wave = np.linspace(4000.0, 4100.0, 16)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux).to_mapping()
    cfg["host_sfh_model"] = "delayed"
    cfg["mass_metallicity_relation"] = {"enabled": False}
    cfg["log_host_aperture_scale"] = {"dist": "Delta", "value": np.log(0.25)}

    class _Grid:
        templates = np.zeros((wave.size, 4), dtype=float)
        template_meta = [
            {"tage_gyr": 0.1, "logzsol": -0.5, "dsps_lg_age_gyr": -1.0, "dsps_lgmet": -1.0},
            {"tage_gyr": 1.0, "logzsol": -0.5, "dsps_lg_age_gyr": 0.0, "dsps_lgmet": -1.0},
            {"tage_gyr": 0.1, "logzsol": 0.0, "dsps_lg_age_gyr": -1.0, "dsps_lgmet": -0.5},
            {"tage_gyr": 1.0, "logzsol": 0.0, "dsps_lg_age_gyr": 0.0, "dsps_lgmet": -0.5},
        ]
        age_grid_gyr = np.array([0.1, 1.0])
        logzsol_grid = np.array([-0.5, 0.0])
        host_basis_jax = HostBasisJax(
            ssp_lgmet=jnp.array([-1.0, -0.5, 0.0], dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.log10(jnp.array([0.1, 0.5, 1.0], dtype=jnp.float64)),
            rest_llambda=jnp.ones((3, 3, wave.size), dtype=jnp.float64),
            surviving_frac_by_age=jnp.ones((3,), dtype=jnp.float64),
            n_ly_per_msun=jnp.zeros((3, 3), dtype=jnp.float64),
            ly_lum_per_msun=jnp.zeros((3, 3), dtype=jnp.float64),
            gal_t_table=jnp.geomspace(0.01, 1.2, 16),
        )
        t_obs_gyr = 1.2

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(0.0),
        "PL_norm": np.array(0.0),
        "PL_slope": np.array(0.0),
        "log_stellar_mass": np.array(10.0),
        "log_sfh_age_gyr": np.log(1.0),
        "log_sfh_tau_over_age": np.log(0.5),
        "gal_lgmet": np.array(-0.5),
        "log_gal_lgmet_scatter": np.log(0.2),
        "gal_v_kms": np.array(0.0),
        "log_gal_sigma_kms": np.log(1.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
        "delta_m_psf_raw": np.array(0.0),
        "eta_psf_raw": np.array(0.4),
        "sigma_phot_extra": np.array(0.0),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([4000.0, 4100.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([4000.0, 4100.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=False,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=0.1,
        use_psf_phot=True,
        psf_mags=np.array([20.0]),
        psf_mag_errs=np.array([0.1]),
        psf_filter_curves={"trans": np.ones((1, wave.size), dtype=float)},
    )

    gal_total = np.asarray(tr["gal_model_total"]["value"], dtype=float)
    gal_fiber = np.asarray(tr["gal_model"]["value"], dtype=float)
    gal_psf = np.asarray(tr["gal_model_psf"]["value"], dtype=float)

    assert np.allclose(gal_fiber, 0.25 * gal_total)
    assert np.allclose(gal_psf, 0.4 * gal_total)


def test_delayed_sfh_host_accepts_jitted_redshift_tracer():
    wave = np.linspace(4000.0, 4100.0, 16)
    flux = np.ones_like(wave)
    cfg = build_default_prior_config(flux).to_mapping()
    cfg["host_sfh_model"] = "delayed"
    cfg["mass_metallicity_relation"] = {"enabled": False}
    cfg["z_qso"] = 0.1
    cfg["log_host_aperture_scale"] = {"dist": "Delta", "value": 0.0}

    class _Grid:
        templates = np.zeros((wave.size, 4), dtype=float)
        template_meta = [
            {"tage_gyr": 0.1, "logzsol": -0.5, "dsps_lg_age_gyr": -1.0, "dsps_lgmet": -1.0},
            {"tage_gyr": 1.0, "logzsol": -0.5, "dsps_lg_age_gyr": 0.0, "dsps_lgmet": -1.0},
            {"tage_gyr": 0.1, "logzsol": 0.0, "dsps_lg_age_gyr": -1.0, "dsps_lgmet": -0.5},
            {"tage_gyr": 1.0, "logzsol": 0.0, "dsps_lg_age_gyr": 0.0, "dsps_lgmet": -0.5},
        ]
        age_grid_gyr = np.array([0.1, 1.0])
        logzsol_grid = np.array([-0.5, 0.0])
        host_basis_jax = HostBasisJax(
            ssp_lgmet=jnp.array([-1.0, -0.5, 0.0], dtype=jnp.float64),
            ssp_lg_age_gyr=jnp.log10(jnp.array([0.1, 0.5, 1.0], dtype=jnp.float64)),
            rest_llambda=jnp.ones((3, 3, wave.size), dtype=jnp.float64),
            surviving_frac_by_age=jnp.ones((3,), dtype=jnp.float64),
            n_ly_per_msun=jnp.zeros((3, 3), dtype=jnp.float64),
            ly_lum_per_msun=jnp.zeros((3, 3), dtype=jnp.float64),
            gal_t_table=jnp.geomspace(0.01, 1.2, 16),
        )
        t_obs_gyr = 1.2

    params = {
        "log_stellar_mass": np.array(10.0),
        "log_sfh_age_gyr": np.log(1.0),
        "log_sfh_tau_over_age": np.log(0.5),
        "gal_lgmet": np.array(-0.5),
        "log_gal_lgmet_scatter": np.log(0.2),
    }

    def _host_sum(z_qso):
        wrapped = substitute(seed(_delayed_sfh_host_spectrum, jax.random.PRNGKey(0)), data=params)
        gal_intrinsic, _, _ = wrapped(_Grid(), cfg, jnp.asarray(1.0), z_qso)
        return jnp.sum(gal_intrinsic)

    value = jax.jit(_host_sum)(jnp.asarray(0.1))
    assert np.isfinite(float(value))
    assert float(value) > 0.0


def test_qso_fsps_joint_model_fast_line_path_matches_component_split():
    wave = np.linspace(4800.0, 5100.0, 64)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)
    line_table = [
        {
            "lambda": 4862.68,
            "linename": "Hb_br",
            "compname": "Hb",
            "ngauss": 1,
            "inisca": 1.0,
            "minsca": 0.0,
            "maxsca": 10.0,
            "inisig": 0.01,
            "minsig": 0.001,
            "maxsig": 0.05,
            "voff": 0.01,
            "vindex": 0,
            "windex": 0,
            "findex": 0,
            "fvalue": 1.0,
        },
        {
            "lambda": 5008.24,
            "linename": "OIII5007",
            "compname": "Hb",
            "ngauss": 1,
            "inisca": 0.5,
            "minsca": 0.0,
            "maxsca": 10.0,
            "inisig": 0.003,
            "minsig": 0.001,
            "maxsig": 0.02,
            "voff": 0.01,
            "vindex": 1,
            "windex": 1,
            "findex": 1,
            "fvalue": 1.0,
        },
    ]
    tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    params = {
        "cont_norm": np.array(1.0),
        "line_dmu_group_std": np.zeros(tied_line_meta["n_vgroups"]),
        "line_log_broad_fwhm_std": np.array(0.0),
        "line_log_narrow_fwhm_std": np.array(0.0),
        "line_log_fwhm_delta_group_std": np.zeros(tied_line_meta["n_wgroups"]),
        "line_amp_group": tied_line_meta["amp_init_group"],
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }

    def _trace(return_line_components, emit_deterministics=True):
        return trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
            wave=wave,
            flux=flux,
            err=err,
            conti_priors={},
            tied_line_meta=tied_line_meta,
            fsps_grid=_Grid(),
            fe_uv_wave=np.array([2000.0, 6000.0]),
            fe_uv_flux=np.zeros(2),
            fe_op_wave=np.array([2000.0, 6000.0]),
            fe_op_flux=np.zeros(2),
            use_lines=True,
            prior_config=cfg,
            decompose_host=False,
            fit_pl=False,
            fit_fe=False,
            fit_bc=False,
            fit_poly=False,
            fit_reddening=False,
            return_line_components=return_line_components,
            emit_deterministics=emit_deterministics,
        )

    tr_split = _trace(True)
    tr_fast = _trace(False)

    assert np.allclose(tr_fast["line_model"]["value"], tr_split["line_model"]["value"])
    assert np.allclose(tr_fast["model"]["value"], tr_split["model"]["value"])
    assert np.allclose(tr_fast["line_model_broad"]["value"], 0.0)
    assert np.allclose(tr_fast["line_model_narrow"]["value"], 0.0)

    tr_fit = _trace(False, emit_deterministics=False)
    assert "obs" in tr_fit
    assert "model" not in tr_fit
    assert "line_model" not in tr_fit


def test_smooth_bounded_affine_preserves_init_and_bound_gradients():
    value0 = _smooth_bounded_affine(
        jnp.asarray(0.0),
        loc=jnp.asarray(0.5),
        scale=jnp.asarray(0.2),
        low=jnp.asarray(0.0),
        high=jnp.asarray(1.0),
    )
    np.testing.assert_allclose(np.asarray(value0), 0.5, rtol=1.0e-12, atol=1.0e-12)

    grad_hi = jax.grad(
        lambda eps: _smooth_bounded_affine(
            eps,
            loc=jnp.asarray(0.5),
            scale=jnp.asarray(0.2),
            low=jnp.asarray(0.0),
            high=jnp.asarray(1.0),
        )
    )(jnp.asarray(10.0))
    grad_lo = jax.grad(
        lambda eps: _smooth_bounded_affine(
            eps,
            loc=jnp.asarray(0.5),
            scale=jnp.asarray(0.2),
            low=jnp.asarray(0.0),
            high=jnp.asarray(1.0),
        )
    )(jnp.asarray(-10.0))

    assert float(grad_hi) > 0.0
    assert float(grad_lo) > 0.0


def test_qso_fsps_joint_model_skips_disabled_fe_and_balmer(monkeypatch):
    wave = np.linspace(2000.0, 6000.0, 16)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)
    tied_line_meta = build_tied_line_meta_from_linelist([], wave)

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("disabled Fe/Balmer component was evaluated")

    monkeypatch.setattr(model_mod, "_fe_template_component", _raise_if_called)
    monkeypatch.setattr(model_mod, "_balmer_continuum_jax", _raise_if_called)

    params = {
        "cont_norm": np.array(1.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta=tied_line_meta,
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=False,
        fit_pl=False,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        return_line_components=False,
        emit_deterministics=False,
    )

    assert "obs" in tr


def test_luminosity_distance_cm_jax_is_finite_and_vectorizable():
    z = jnp.asarray([0.1, 1.0, 2.0])
    d_l = _luminosity_distance_cm_jax(z)

    assert d_l.shape == (3,)
    assert np.all(np.isfinite(np.asarray(d_l)))
    assert np.all(np.asarray(d_l) > 0.0)
    assert np.all(np.diff(np.asarray(d_l)) > 0.0)


def test_qso_fsps_joint_model_custom_component_returns_jax_array():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux)

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    comp = make_custom_component(
        name="const_term",
        parameter_priors={"c0": {"dist": "Normal", "loc": 0.0, "scale": 1.0}},
        evaluate=lambda wave, params, metadata: jnp.zeros_like(wave) + params["c0"],
    )

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(1.0),
        "PL_norm": np.array(5.0e6),
        "PL_slope": np.array(0.0),
        "tau_host": np.array(1.0),
        "fsps_weights_raw": np.array([0.0]),
        "gal_v_kms": np.array(0.0),
        "log_gal_sigma_kms": np.log(100.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
        "custom_const_term_c0": np.array(0.5),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=jnp.asarray(1.0),
        custom_components=[comp],
    )

    value = tr["custom_const_term_model"]["value"]
    assert isinstance(value, jax.Array)
    assert np.all(np.isfinite(np.asarray(value)))


def test_host_redshift_prior_params_shift_negative_at_high_z():
    cfg = {
        "host_redshift_prior": {
            "enabled": True,
            "z_mid": 1.0,
            "width": 0.2,
            "lowz_loc_offset": 0.0,
            "highz_loc_offset": -8.0,
            "lowz_scale_mult": 1.0,
            "highz_scale_mult": 0.05,
            "lowz_df": 3.0,
            "highz_df": 20.0,
        }
    }
    w_low, offset_low, scale_low, df_low = _host_redshift_prior_params(cfg, 0.2)
    w_mid, offset_mid, scale_mid, df_mid = _host_redshift_prior_params(cfg, 1.0)
    w_high, offset_high, scale_high, df_high = _host_redshift_prior_params(cfg, 2.0)

    assert float(w_low) < 0.1
    assert np.isclose(float(offset_low), 0.0, atol=0.15)
    assert np.isclose(float(scale_low), 1.0, atol=0.1)
    assert np.isclose(float(df_low), 3.0, atol=1.0)
    assert np.isclose(float(w_mid), 0.5, atol=1e-6)
    assert np.isclose(float(offset_mid), -4.0, atol=1e-6)
    assert np.isclose(float(scale_mid), 0.525, atol=1e-6)
    assert np.isclose(float(df_mid), 11.5, atol=1e-6)
    assert float(w_high) > 0.99
    assert np.isclose(float(offset_high), -7.95, atol=0.05)
    assert np.isclose(float(scale_high), 0.056, atol=0.02)
    assert np.isclose(float(df_high), 19.9, atol=0.2)


def test_host_redshift_prior_params_disable_restores_zero_offset():
    w, offset, scale_mult, df_eff = _host_redshift_prior_params({"host_redshift_prior": {"enabled": False}}, 2.0)
    assert float(w) == 0.0
    assert float(offset) == 0.0
    assert float(scale_mult) == 1.0
    assert df_eff is None


def test_host_redshift_prior_penalizes_same_host_more_at_high_z():
    cfg = build_default_prior_config(np.array([1.0, 2.0, 3.0], dtype=float)).to_mapping()
    cfg["host_redshift_prior"]["enabled"] = True
    _, offset_low, scale_low, df_low = _host_redshift_prior_params(cfg, 0.2)
    _, offset_high, scale_high, df_high = _host_redshift_prior_params(cfg, 2.0)

    x = jnp.asarray(1.5)
    logp_low = dist.StudentT(df=float(df_low), loc=float(offset_low), scale=float(scale_low)).log_prob(x)
    logp_high = dist.StudentT(df=float(df_high), loc=float(offset_high), scale=float(scale_high)).log_prob(x)

    assert float(logp_high) < float(logp_low)


def test_qso_fsps_joint_model_derives_host_fraction_diagnostics():
    wave = np.linspace(2000.0, 6000.0, 32)
    flux = np.ones_like(wave)
    err = np.full_like(wave, 0.1)
    cfg = build_default_prior_config(flux).to_mapping()
    cfg["host_sfh_model"] = "flexible"

    class _Grid:
        templates = np.zeros((wave.size, 1), dtype=float)
        template_meta = [{"tage_gyr": 1.0, "logzsol": 0.0}]

    params = {
        "cont_norm": np.array(1.0),
        "log_frac_host": np.array(-1.0),
        "PL_norm": np.array(5.0e6),
        "PL_slope": np.array(0.0),
        "tau_host": np.array(1.0),
        "fsps_weights_raw": np.array([0.0]),
        "gal_v_kms": np.array(0.0),
        "log_gal_sigma_kms": np.log(100.0),
        "frac_jitter": np.array(0.0),
        "add_jitter": np.array(0.0),
    }
    tr = trace(substitute(seed(qso_fsps_joint_model, jax.random.PRNGKey(0)), data=params)).get_trace(
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={"n_lines": 0},
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 6000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 6000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=False,
        prior_config=cfg,
        decompose_host=True,
        fit_pl=True,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        z_qso=2.0,
    )

    assert "host_amp" in tr
    assert "frac_host" in tr
    assert "log_frac_host" in tr
    assert tr["host_amp"]["type"] == "deterministic"
    assert tr["log_frac_host"]["type"] == "sample"
