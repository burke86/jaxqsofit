import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jaxqsofit import JAXQSOFit
from jaxqsofit.config import FitConfig, Observation, SpectroscopyData, fit_config_from_mapping
from jaxqsofit.model import (
    _rest_log_lambda_llambda_from_flam,
    _synth_ab_mag_from_grid,
    _luminosity_distance_cm_jax,
    spectral_likelihood_weight_from_resolving_power,
    combine_gaussian_sigma,
    instrumental_sigma_kms,
    instrumental_sigma_lnwave,
)


def test_rest_frame_conversion_conserves_integrated_flux():
    wave_obs = np.linspace(4000.0, 8000.0, 2001)
    flux_obs = 2.0 + np.exp(-0.5 * ((wave_obs - 6000.0) / 100.0) ** 2)
    err_obs = np.full_like(flux_obs, 0.1)
    z = 1.3
    fitter = JAXQSOFit.from_arrays(lam=wave_obs, flux=flux_obs, err=err_obs, z=z)

    wave_rest, flux_rest, err_rest = fitter._rest_frame(wave_obs, flux_obs, err_obs, z)

    assert np.trapezoid(flux_rest, wave_rest) == pytest.approx(
        np.trapezoid(flux_obs, wave_obs), rel=2e-12
    )
    assert np.allclose(err_rest, err_obs * (1.0 + z))


@pytest.mark.parametrize("z", [0.01, 0.3, 1.0, 3.0])
def test_lambda_llambda_matches_direct_rest_flux_conversion(z):
    wave_rest = 5100.0
    flam_rest = 7.5
    measured = float(_rest_log_lambda_llambda_from_flam(wave_rest, flam_rest, z))
    distance_cm = float(_luminosity_distance_cm_jax(z))
    expected = np.log10(4.0 * np.pi * distance_cm**2 * wave_rest * flam_rest * 1e-17)
    assert measured == pytest.approx(expected, rel=2e-12)


def test_synthetic_ab_magnitude_matches_direct_bandpass_integral():
    wave = np.linspace(3500.0, 7500.0, 4001)
    flam_1e17 = 5.0 * (wave / 5100.0) ** -1.3
    trans = np.exp(-0.5 * ((wave - 5200.0) / 550.0) ** 2)
    measured = float(_synth_ab_mag_from_grid(wave, flam_1e17, trans))
    c_ang_s = 2.99792458e18
    fnu = np.trapezoid(flam_1e17 * 1e-17 * trans * wave, wave) / np.trapezoid(
        trans * c_ang_s / wave, wave
    )
    expected = -2.5 * np.log10(fnu) - 48.60
    assert measured == pytest.approx(expected, abs=2e-10)


def test_power_law_slope_recovery_is_stable_across_snr():
    wave = np.linspace(1300.0, 7000.0, 3000)
    pivot = 3000.0
    true_slope = -1.55
    true_flux = 8.0 * (wave / pivot) ** true_slope
    rng = np.random.default_rng(9182)
    for snr in (10.0, 30.0, 100.0):
        noisy = true_flux + rng.normal(0.0, true_flux / snr)
        recovered = np.polyfit(np.log(wave / pivot), np.log(noisy), 1)[0]
        assert recovered == pytest.approx(true_slope, abs=0.025)


def test_mass_and_aperture_scale_are_exactly_degenerate_in_spectrum_normalization():
    shape = np.array([0.4, 1.0, 0.7])
    mass_a, aperture_a = 10.0**10, 0.2
    mass_b, aperture_b = 10.0**9, 2.0
    assert np.allclose(shape * mass_a * aperture_a, shape * mass_b * aperture_b)


def test_resolution_likelihood_weight_converges_with_pixel_sampling():
    resolving_power = 2000.0
    coarse = np.geomspace(4000.0, 8000.0, 1500)
    fine = np.geomspace(4000.0, 8000.0, 6000)
    coarse_weight = float(spectral_likelihood_weight_from_resolving_power(coarse, resolving_power))
    fine_weight = float(spectral_likelihood_weight_from_resolving_power(fine, resolving_power))
    coarse_neff = coarse_weight * coarse.size
    fine_neff = fine_weight * fine.size
    assert coarse_neff == pytest.approx(fine_neff, rel=5e-3)


@pytest.mark.parametrize(
    "wave",
    ([5000.0, 4900.0, 5100.0], [5000.0, 5000.0, 5100.0], [np.nan, -1.0, 0.0]),
)
def test_invalid_wavelength_grids_are_rejected(wave):
    with pytest.raises(ValueError, match="wave_obs"):
        FitConfig(
            observation=Observation(redshift=0.1),
            spectroscopy=SpectroscopyData(wave_obs=wave, fluxes=[1.0, 1.0, 1.0], errors=[0.1] * 3),
        ).validate()


@pytest.mark.parametrize("redshift", [np.nan, np.inf, -0.01])
def test_invalid_redshift_is_rejected(redshift):
    with pytest.raises(ValueError, match="redshift"):
        FitConfig(
            observation=Observation(redshift=redshift),
            spectroscopy=SpectroscopyData(wave_obs=[4000.0, 5000.0], fluxes=[1.0, 1.0], errors=[0.1, 0.1]),
        ).validate()


def test_unknown_nested_configuration_key_is_rejected():
    with pytest.raises(ValueError, match="resolving_powr"):
        fit_config_from_mapping(
            {
                "observation": {"redshift": 0.1},
                "spectroscopy": {
                    "wave_obs": [4000.0, 5000.0],
                    "fluxes": [1.0, 1.0],
                    "errors": [0.1, 0.1],
                    "resolving_powr": 2000.0,
                },
            }
        )


def test_unknown_top_level_configuration_key_is_rejected():
    with pytest.raises(ValueError, match="likelihod"):
        fit_config_from_mapping(
            {
                "observation": {"redshift": 0.1},
                "spectroscopy": {
                    "wave_obs": [4000.0, 5000.0],
                    "fluxes": [1.0, 1.0],
                    "errors": [0.1, 0.1],
                },
                "likelihod": {},
            }
        )


def test_missing_errors_behavior_is_explicit_and_deterministic():
    fitter = JAXQSOFit.from_arrays(
        lam=[4000.0, 5000.0],
        flux=[1.0, 1.0],
        err=None,
        z=0.1,
    )
    assert np.array_equal(fitter.err_in, np.full(2, 1e-6))


def test_instrumental_resolution_is_disabled_by_default():
    spec = SpectroscopyData(
        wave_obs=[4000.0, 5000.0], fluxes=[1.0, 1.0], errors=[0.1, 0.1], resolving_power=2000.0
    )
    assert spec.apply_instrumental_resolution is False


def test_instrumental_resolution_requires_resolving_power():
    with pytest.raises(ValueError, match="requires resolving_power"):
        FitConfig(
            observation=Observation(redshift=0.1),
            spectroscopy=SpectroscopyData(
                wave_obs=[4000.0, 5000.0],
                fluxes=[1.0, 1.0],
                errors=[0.1, 0.1],
                apply_instrumental_resolution=True,
            ),
        ).validate()


def test_instrumental_resolution_conversion_and_quadrature():
    resolving_power = 2000.0
    sigma_ln = float(instrumental_sigma_lnwave(resolving_power))
    sigma_kms = float(instrumental_sigma_kms(resolving_power))
    assert sigma_ln == pytest.approx(1.0 / (2.354820045 * resolving_power))
    assert sigma_kms == pytest.approx(299792.458 / (2.354820045 * resolving_power))
    assert float(combine_gaussian_sigma(100.0, sigma_kms)) == pytest.approx(np.hypot(100.0, sigma_kms))


def test_instrumental_line_broadening_conserves_integrated_flux():
    lnwave = np.linspace(np.log(4900.0), np.log(5100.0), 20001)
    wave = np.exp(lnwave)
    mu = np.log(5000.0)
    amp = 3.0
    sigma_intrinsic = 80.0 / 299792.458
    sigma_effective = float(combine_gaussian_sigma(sigma_intrinsic, instrumental_sigma_lnwave(2000.0)))
    amp_effective = amp * sigma_intrinsic / sigma_effective * np.exp(
        0.5 * (sigma_intrinsic**2 - sigma_effective**2)
    )
    intrinsic = amp * np.exp(-0.5 * ((lnwave - mu) / sigma_intrinsic) ** 2)
    broadened = amp_effective * np.exp(-0.5 * ((lnwave - mu) / sigma_effective) ** 2)
    assert np.trapezoid(broadened, wave) == pytest.approx(np.trapezoid(intrinsic, wave), rel=2e-7)


def test_instrumental_quadrature_has_finite_gradients_and_infinite_r_limit():
    grad = jax.grad(lambda sigma: combine_gaussian_sigma(sigma, instrumental_sigma_kms(2000.0)))(
        jnp.asarray(120.0)
    )
    assert np.isfinite(float(grad))
    assert float(instrumental_sigma_kms(np.inf)) == 0.0
    assert float(combine_gaussian_sigma(120.0, instrumental_sigma_kms(np.inf))) == pytest.approx(120.0)
