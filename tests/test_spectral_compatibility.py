"""Compatibility coverage for spectral modules now implemented by jaxsedfit."""

import jaxqsofit.components as legacy_components
import jaxqsofit.config as legacy_config
import jaxqsofit.defaults as legacy_defaults
import jaxqsofit.geometry as legacy_geometry
import jaxqsofit.model as legacy_model
import jaxqsofit.reparameterization as legacy_reparameterization
import jaxsedfit.spectral_components as spectral_components
import jaxsedfit.spectral_config as spectral_config
import jaxsedfit.spectral_defaults as spectral_defaults
import jaxsedfit.spectral_geometry as spectral_geometry
import jaxsedfit.spectral_model as spectral_model
import jaxsedfit.spectral_reparameterization as spectral_reparameterization
from jaxsedfit.spectroscopy import (
    SpectralComponentConfig,
    evaluate_joint_spectral_components,
)


def test_legacy_spectral_modules_are_true_aliases():
    assert legacy_components is spectral_components
    assert legacy_config is spectral_config
    assert legacy_defaults is spectral_defaults
    assert legacy_geometry is spectral_geometry
    assert legacy_model is spectral_model
    assert legacy_reparameterization is spectral_reparameterization


def test_public_spectral_api_matches_legacy_exports():
    assert SpectralComponentConfig is legacy_components.SpectralComponentConfig
    assert (
        evaluate_joint_spectral_components
        is legacy_components.evaluate_joint_spectral_components
    )
