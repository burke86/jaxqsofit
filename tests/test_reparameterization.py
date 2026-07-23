import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.util import log_density

from jaxqsofit.reparameterization import (
    NORMAL_LOGNORMAL_STANDARDIZATION,
    NormalLogNormalStandardizeReparam,
    normal_lognormal_standardization_reparam,
    standardized_prior_site,
)


def _standardizable_prior_model(prior):
    return numpyro.sample(
        "physical",
        prior,
        infer={
            NORMAL_LOGNORMAL_STANDARDIZATION: {
                "auxiliary_name": standardized_prior_site("physical"),
            }
        },
    )


def test_normal_lognormal_standardization_is_exact_and_maps_initial_values():
    wrapped = reparam(
        _standardizable_prior_model,
        config=normal_lognormal_standardization_reparam,
    )
    standardized = np.asarray(-0.4)
    priors = (
        dist.Normal(2.0, 3.0),
        dist.LogNormal(np.log(2.0), 0.7),
    )

    for prior in priors:
        unconstrained = prior.loc + prior.scale * standardized
        physical = (
            np.exp(np.asarray(unconstrained))
            if isinstance(prior, dist.LogNormal)
            else np.asarray(unconstrained)
        )
        original_density, _ = log_density(
            _standardizable_prior_model,
            (),
            {"prior": prior},
            {"physical": physical},
        )
        standardized_density, standardized_trace = log_density(
            wrapped,
            (),
            {"prior": prior},
            {"physical_std": standardized},
        )
        log_abs_det = np.log(float(prior.scale))
        if isinstance(prior, dist.LogNormal):
            log_abs_det += np.log(float(physical))
        np.testing.assert_allclose(
            standardized_density,
            original_density + log_abs_det,
            rtol=1.0e-12,
        )
        assert standardized_trace["physical"]["type"] == "deterministic"
        np.testing.assert_allclose(
            standardized_trace["physical"]["value"],
            physical,
            rtol=1.0e-12,
        )
        reparameterizer = NormalLogNormalStandardizeReparam("physical_std")
        np.testing.assert_allclose(
            reparameterizer.transform_initial_value(prior, physical),
            standardized,
            rtol=1.0e-12,
        )
