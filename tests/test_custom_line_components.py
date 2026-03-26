import numpy as np
import jax
from numpyro.infer import Predictive

import jaxqsofit.model as modelmod
from jaxqsofit import make_custom_line_component
from jaxqsofit.custom_components import (
    custom_line_component_site_names,
    inject_default_custom_line_component_priors,
)


def test_custom_line_component_prior_injection_and_site_names():
    comps = (
        make_custom_line_component(
            name="exp_wing",
            parameter_priors={
                "amp": {"dist": "LogNormal", "loc": 0.0, "scale": 0.5},
                "tau": {"dist": "LogNormal", "loc": np.log(500.0), "scale": 0.3},
            },
            evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["amp"],
            line_kind="broad",
        ),
    )
    cfg = inject_default_custom_line_component_priors({}, np.array([1.0, 2.0, 3.0]), comps)

    assert "custom_line_exp_wing_amp" in cfg
    assert "custom_line_exp_wing_tau" in cfg
    assert custom_line_component_site_names(comps) == ["custom_line_exp_wing_model"]


def test_custom_line_components_add_to_broad_and_narrow_models():
    wave = np.linspace(2000.0, 3000.0, 5)
    flux = np.zeros_like(wave)
    err = np.ones_like(wave)

    class _Grid:
        templates = np.zeros((5, 1), dtype=float)

    broad_comp = make_custom_line_component(
        name="exp_broad",
        parameter_priors={"amp": {"dist": "Normal", "loc": 0.0, "scale": 1.0}},
        evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["amp"],
        line_kind="broad",
    )
    narrow_comp = make_custom_line_component(
        name="exp_narrow",
        parameter_priors={"amp": {"dist": "Normal", "loc": 0.0, "scale": 1.0}},
        evaluate=lambda wave, params, metadata: np.zeros_like(np.asarray(wave)) + params["amp"],
        line_kind="narrow",
    )

    prior_config = {
        "log_cont_norm": {"dist": "LogNormal", "loc": 0.0, "scale": 0.1},
        "PL_slope": {"dist": "Normal", "loc": 0.0, "scale": 0.1},
        "frac_jitter": {"dist": "HalfNormal", "scale": 0.1},
        "add_jitter": {"dist": "HalfNormal", "scale": 0.1},
        "custom_line_exp_broad_amp": {"dist": "Normal", "loc": 2.0, "scale": 1e-6},
        "custom_line_exp_narrow_amp": {"dist": "Normal", "loc": 3.0, "scale": 1e-6},
    }

    out = Predictive(
        modelmod.qso_fsps_joint_model,
        posterior_samples={},
        num_samples=1,
        return_sites=[
            "line_model_broad",
            "line_model_narrow",
            "line_model",
            "custom_line_exp_broad_model",
            "custom_line_exp_narrow_model",
        ],
    )(
        jax.random.PRNGKey(0),
        wave=wave,
        flux=flux,
        err=err,
        conti_priors={},
        tied_line_meta={
            "n_lines": 0,
            "n_vgroups": 0,
            "n_wgroups": 0,
            "n_fgroups": 0,
            "ln_lambda0": np.array([], dtype=float),
            "vgroup": np.array([], dtype=int),
            "wgroup": np.array([], dtype=int),
            "fgroup": np.array([], dtype=int),
            "flux_ratio": np.array([], dtype=float),
            "dmu_init_group": np.array([], dtype=float),
            "dmu_min_group": np.array([], dtype=float),
            "dmu_max_group": np.array([], dtype=float),
            "sig_init_group": np.array([], dtype=float),
            "sig_min_group": np.array([], dtype=float),
            "sig_max_group": np.array([], dtype=float),
            "amp_init_group": np.array([], dtype=float),
            "amp_min_group": np.array([], dtype=float),
            "amp_max_group": np.array([], dtype=float),
            "names": [],
            "compnames": [],
            "line_lambda": np.array([], dtype=float),
        },
        fsps_grid=_Grid(),
        fe_uv_wave=np.array([2000.0, 3000.0]),
        fe_uv_flux=np.zeros(2),
        fe_op_wave=np.array([2000.0, 3000.0]),
        fe_op_flux=np.zeros(2),
        use_lines=True,
        prior_config=prior_config,
        decompose_host=False,
        fit_pl=False,
        fit_fe=False,
        fit_bc=False,
        fit_poly=False,
        fit_reddening=False,
        fit_poly_order=0,
        custom_components=(),
        custom_line_components=(broad_comp, narrow_comp),
    )

    assert np.allclose(out["custom_line_exp_broad_model"][0], 2.0)
    assert np.allclose(out["custom_line_exp_narrow_model"][0], 3.0)
    assert np.allclose(out["line_model_broad"][0], 2.0)
    assert np.allclose(out["line_model_narrow"][0], 3.0)
    assert np.allclose(out["line_model"][0], 5.0)
