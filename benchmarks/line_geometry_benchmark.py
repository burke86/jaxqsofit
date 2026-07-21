"""Benchmark emission-line parameterizations on small synthetic spectra.

This is a manual geometry diagnostic, not a supported jaxqsofit model or a CI
performance test. Every parameterization in this module is experimental. It compares
peak-amplitude coordinates with integrated-flux coordinates and compares an
independent two-Gaussian broad profile with ordered widths plus total flux and
a mixing fraction.

Run with::

    conda run -n jaxcpu python benchmarks/line_geometry_benchmark.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_enable_x64", True)

SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class ExperimentalGeometryVariant:
    """Declarative benchmark configuration; not a supported production model."""

    n_broad: int
    amplitude: str = "peak"

    def __post_init__(self):
        if self.n_broad not in {0, 1, 2}:
            raise ValueError("n_broad must be 0, 1, or 2")
        if self.amplitude not in {"peak", "unit_norm"}:
            raise ValueError("amplitude must be 'peak' or 'unit_norm'")


def gaussian_peak(velocity, center, sigma, peak):
    return peak * jnp.exp(-0.5 * ((velocity - center) / sigma) ** 2)


def gaussian_flux(velocity, center, sigma, integrated_flux):
    peak = integrated_flux / (SQRT_2PI * sigma)
    return gaussian_peak(velocity, center, sigma, peak)


def continuum(velocity, level, slope):
    return level + slope * velocity / 10_000.0


def gaussian_unit_noise_norm(velocity, center, sigma, error):
    """Gaussian profile with unit norm in the likelihood's noise metric."""
    profile = jnp.exp(-0.5 * ((velocity - center) / sigma) ** 2)
    return profile / jnp.sqrt(jnp.sum(jnp.square(profile / error)))


def joint_broad_noise_basis(velocity, center_1, sigma_1, center_2, sigma_2, error):
    """Ordered Gram--Schmidt basis for two broad profiles in noise space."""
    profile_1 = gaussian_peak(velocity, center_1, sigma_1, 1.0) / error
    profile_2 = gaussian_peak(velocity, center_2, sigma_2, 1.0) / error
    basis_1 = profile_1 / jnp.linalg.norm(profile_1)
    residual_2 = profile_2 - basis_1 * jnp.dot(basis_1, profile_2)
    basis_2 = residual_2 / jnp.linalg.norm(residual_2)
    # Multiplication by error maps the orthonormal noise-space basis back to flux.
    return error * basis_1, error * basis_2


def normal_std(name, loc, scale):
    value = loc + scale * numpyro.sample(f"{name}_std", dist.Normal(0.0, 1.0))
    return numpyro.deterministic(name, value)


def lognormal_std(name, log_loc, log_scale):
    log_value = log_loc + log_scale * numpyro.sample(f"{name}_std", dist.Normal(0.0, 1.0))
    return numpyro.deterministic(name, jnp.exp(log_value))


def build_experimental_geometry_model(
    variant: ExperimentalGeometryVariant,
) -> Callable:
    """Build a benchmark-only model that is explicitly outside the public API."""

    def model(velocity, flux=None, error=0.03):
        level = normal_std("continuum", 1.0, 0.2)
        slope = normal_std("slope", 0.0, 0.1)
        center_name = (
            "narrow_center"
            if variant.n_broad == 0
            else ("line_center" if variant.n_broad == 1 else "systemic_center")
        )
        systemic_center = normal_std(
            center_name,
            0.0,
            250.0,
        )
        narrow_sigma = lognormal_std(
            "narrow_sigma",
            np.log(220.0),
            0.55 if variant.n_broad == 0 else 0.45,
        )

        broad_sigmas = []
        broad_centers = []
        if variant.n_broad == 1:
            broad_sigmas.append(
                lognormal_std("broad_sigma", np.log(1800.0), 0.7)
            )
            broad_centers.append(systemic_center)
        elif variant.n_broad == 2:
            broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
            broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
            broad_centers.extend(
                [
                    systemic_center + broad_offset - 0.5 * broad_relative,
                    systemic_center + broad_offset + 0.5 * broad_relative,
                ]
            )
            low = np.log(600.0)
            high = np.log(6000.0)
            target = np.log(np.asarray([1400.0, 3600.0]))
            gaps = np.diff(np.concatenate(([low], target, [high])))
            logit_loc = np.log(gaps[:-1] / gaps[-1])
            logits_std = numpyro.sample(
                "broad_width_logits_std",
                dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1),
            )
            logits = jnp.asarray(logit_loc) + 0.5 * logits_std
            spacings = jax.nn.softmax(jnp.concatenate([logits, jnp.zeros(1)]))
            ordered = low + (high - low) * jnp.cumsum(spacings[:-1])
            broad_sigmas.extend(
                [
                    numpyro.deterministic("broad_sigma_1", jnp.exp(ordered[0])),
                    numpyro.deterministic("broad_sigma_2", jnp.exp(ordered[1])),
                ]
            )

        reference_sigmas = [220.0]
        reference_peaks = [0.45]
        if variant.n_broad == 1:
            reference_sigmas.append(1800.0)
            reference_peaks.append(0.20)
        elif variant.n_broad == 2:
            reference_sigmas.extend([1400.0, 3600.0])
            reference_peaks.extend([0.18, 0.10])
        sigmas = [narrow_sigma, *broad_sigmas]
        centers = [systemic_center, *broad_centers]

        mean = continuum(velocity, level, slope)
        for index, (center, sigma, reference_sigma, reference_peak) in enumerate(
            zip(centers, sigmas, reference_sigmas, reference_peaks)
        ):
            label = "narrow" if index == 0 else "broad"
            suffix = "" if variant.n_broad == 1 or index == 0 else f"_{index}"
            log_scale = 0.8 if variant.n_broad == 0 else (0.7 if index == 0 else 0.9)
            if variant.amplitude == "peak":
                amplitude = lognormal_std(
                    f"{label}_peak{suffix}", np.log(reference_peak), log_scale
                )
                component = gaussian_peak(velocity, center, sigma, amplitude)
            else:
                reference_profile = gaussian_peak(
                    velocity, 0.0, reference_sigma, 1.0
                )
                reference_norm = jnp.sqrt(
                    jnp.sum(jnp.square(reference_profile / error))
                )
                amplitude = lognormal_std(
                    f"{label}_unit_norm_amplitude{suffix}",
                    jnp.log(reference_peak * reference_norm),
                    log_scale,
                )
                component = amplitude * gaussian_unit_noise_norm(
                    velocity, center, sigma, error
                )
            mean += component
        numpyro.sample("obs", dist.Normal(mean, error), obs=flux)

    model.__name__ = (
        f"experimental_geometry_{variant.n_broad}_broad_{variant.amplitude}"
    )
    return model


def narrow_peak_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("narrow_center", 0.0, 250.0)
    sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.55)
    peak = lognormal_std("narrow_peak", np.log(0.45), 0.8)
    mean = continuum(velocity, level, slope) + gaussian_peak(velocity, center, sigma, peak)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def narrow_flux_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("narrow_center", 0.0, 250.0)
    sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.55)
    line_flux = lognormal_std(
        "narrow_integrated_flux",
        np.log(0.45 * 220.0 * SQRT_2PI),
        0.8,
    )
    mean = continuum(velocity, level, slope) + gaussian_flux(velocity, center, sigma, line_flux)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def narrow_unit_norm_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("narrow_center", 0.0, 250.0)
    sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.55)
    reference_norm = jnp.sqrt(
        jnp.sum(jnp.square(gaussian_peak(velocity, 0.0, 220.0, 1.0) / error))
    )
    amplitude = lognormal_std(
        "narrow_unit_norm_amplitude", jnp.log(0.45 * reference_norm), 0.8
    )
    mean = continuum(velocity, level, slope)
    mean += amplitude * gaussian_unit_noise_norm(velocity, center, sigma, error)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def one_broad_narrow_independent_model(velocity, flux=None, error=0.03):
    """Experimental peak-amplitude baseline with independent widths."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    broad_sigma = lognormal_std("broad_sigma", np.log(1800.0), 0.7)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)
    broad_peak = lognormal_std("broad_peak", np.log(0.20), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, center, broad_sigma, broad_peak)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def one_broad_narrow_unit_norm_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    broad_sigma = lognormal_std("broad_sigma", np.log(1800.0), 0.7)
    narrow_reference_norm = jnp.sqrt(
        jnp.sum(jnp.square(gaussian_peak(velocity, 0.0, 220.0, 1.0) / error))
    )
    broad_reference_norm = jnp.sqrt(
        jnp.sum(jnp.square(gaussian_peak(velocity, 0.0, 1800.0, 1.0) / error))
    )
    narrow_amplitude = lognormal_std(
        "narrow_unit_norm_amplitude", jnp.log(0.45 * narrow_reference_norm), 0.7
    )
    broad_amplitude = lognormal_std(
        "broad_unit_norm_amplitude", jnp.log(0.20 * broad_reference_norm), 0.9
    )
    mean = continuum(velocity, level, slope)
    mean += narrow_amplitude * gaussian_unit_noise_norm(
        velocity, center, narrow_sigma, error
    )
    mean += broad_amplitude * gaussian_unit_noise_norm(
        velocity, center, broad_sigma, error
    )
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def one_broad_narrow_ordered_flux_model(velocity, flux=None, error=0.03):
    """Integrated fluxes with a positive broad/narrow log-width gap."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    log_width_ratio = normal_std("broad_log_width_ratio", np.log(1800.0 / 220.0), 0.4)
    broad_sigma = narrow_sigma * jnp.exp(jax.nn.softplus(log_width_ratio))
    total_flux = lognormal_std("total_integrated_flux", np.log(1150.0), 0.75)
    narrow_fraction_logit = normal_std("narrow_fraction_logit", -1.0, 1.0)
    narrow_fraction = jax.nn.sigmoid(narrow_fraction_logit)
    mean = continuum(velocity, level, slope)
    mean += gaussian_flux(velocity, center, narrow_sigma, total_flux * narrow_fraction)
    mean += gaussian_flux(velocity, center, broad_sigma, total_flux * (1.0 - narrow_fraction))
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def one_broad_narrow_separate_flux_model(velocity, flux=None, error=0.03):
    """Independent integrated fluxes and independently sampled widths."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    broad_sigma = lognormal_std("broad_sigma", np.log(1800.0), 0.7)
    narrow_flux = lognormal_std("narrow_integrated_flux", np.log(248.0), 0.7)
    broad_flux = lognormal_std("broad_integrated_flux", np.log(902.0), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_flux(velocity, center, narrow_sigma, narrow_flux)
    mean += gaussian_flux(velocity, center, broad_sigma, broad_flux)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_independent_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)
    # Deliberately exchangeable priors expose the component-label symmetry.
    broad_sigma_1 = lognormal_std("broad_sigma_1", np.log(2300.0), 0.7)
    broad_sigma_2 = lognormal_std("broad_sigma_2", np.log(2300.0), 0.7)
    broad_peak_1 = lognormal_std("broad_peak_1", np.log(0.14), 0.9)
    broad_peak_2 = lognormal_std("broad_peak_2", np.log(0.14), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, center, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, center, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_independent_flux_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    broad_sigma_1 = lognormal_std("broad_sigma_1", np.log(2300.0), 0.7)
    broad_sigma_2 = lognormal_std("broad_sigma_2", np.log(2300.0), 0.7)
    narrow_flux = lognormal_std("narrow_integrated_flux", np.log(248.0), 0.7)
    broad_flux_1 = lognormal_std("broad_integrated_flux_1", np.log(777.0), 0.9)
    broad_flux_2 = lognormal_std("broad_integrated_flux_2", np.log(777.0), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_flux(velocity, center, narrow_sigma, narrow_flux)
    mean += gaussian_flux(velocity, center, broad_sigma_1, broad_flux_1)
    mean += gaussian_flux(velocity, center, broad_sigma_2, broad_flux_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_ordered_peak_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)
    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap",
        np.log(np.log(3600.0 / 1400.0)),
        0.45,
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))
    broad_peak_1 = lognormal_std("broad_peak_1", np.log(0.18), 0.9)
    broad_peak_2 = lognormal_std("broad_peak_2", np.log(0.10), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, center, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, center, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_ordered_peak_independent_centers_model(velocity, flux=None, error=0.03):
    """Experimental ordered widths with independent component centroids."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    narrow_center = normal_std("narrow_center", 0.0, 250.0)
    broad_center_1 = normal_std("broad_center_1", 0.0, 250.0)
    broad_center_2 = normal_std("broad_center_2", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)
    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap", np.log(np.log(3600.0 / 1400.0)), 0.45
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))
    broad_peak_1 = lognormal_std("broad_peak_1", np.log(0.18), 0.9)
    broad_peak_2 = lognormal_std("broad_peak_2", np.log(0.10), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, narrow_center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, broad_center_1, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, broad_center_2, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_ordered_peak_hierarchical_centers_model(velocity, flux=None, error=0.03):
    """Ordered widths with systemic, broad, and zero-sum relative offsets."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)
    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap", np.log(np.log(3600.0 / 1400.0)), 0.45
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))
    broad_peak_1 = lognormal_std("broad_peak_1", np.log(0.18), 0.9)
    broad_peak_2 = lognormal_std("broad_peak_2", np.log(0.10), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, systemic_center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, broad_center_1, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, broad_center_2, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_softmax_peak_hierarchical_centers_model(
    velocity, flux=None, error=0.03
):
    """Experimental softmax-spacing width coordinates."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)

    low = np.log(600.0)
    high = np.log(6000.0)
    target = np.log(np.asarray([1400.0, 3600.0]))
    gaps = np.diff(np.concatenate(([low], target, [high])))
    logit_loc = np.log(gaps[:-1] / gaps[-1])
    logits_std = numpyro.sample(
        "broad_width_logits_std", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
    )
    logits = jnp.asarray(logit_loc) + 0.5 * logits_std
    spacings = jax.nn.softmax(jnp.concatenate([logits, jnp.zeros(1)]))
    ordered_log_sigma = low + (high - low) * jnp.cumsum(spacings[:-1])
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(ordered_log_sigma[0]))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(ordered_log_sigma[1]))

    broad_peak_1 = lognormal_std("broad_peak_1", np.log(0.18), 0.9)
    broad_peak_2 = lognormal_std("broad_peak_2", np.log(0.10), 0.9)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, systemic_center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, broad_center_1, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, broad_center_2, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_softmax_flux_hierarchical_centers_model(
    velocity, flux=None, error=0.03
):
    """Matched softmax widths with separate integrated broad-line fluxes."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)

    low = np.log(600.0)
    high = np.log(6000.0)
    target = np.log(np.asarray([1400.0, 3600.0]))
    gaps = np.diff(np.concatenate(([low], target, [high])))
    logit_loc = np.log(gaps[:-1] / gaps[-1])
    logits_std = numpyro.sample(
        "broad_width_logits_std", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
    )
    logits = jnp.asarray(logit_loc) + 0.5 * logits_std
    spacings = jax.nn.softmax(jnp.concatenate([logits, jnp.zeros(1)]))
    ordered_log_sigma = low + (high - low) * jnp.cumsum(spacings[:-1])
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(ordered_log_sigma[0]))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(ordered_log_sigma[1]))

    broad_flux_1 = lognormal_std(
        "broad_integrated_flux_1", np.log(0.18 * 1400.0 * SQRT_2PI), 0.9
    )
    broad_flux_2 = lognormal_std(
        "broad_integrated_flux_2", np.log(0.10 * 3600.0 * SQRT_2PI), 0.9
    )
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, systemic_center, narrow_sigma, narrow_peak)
    mean += gaussian_flux(velocity, broad_center_1, broad_sigma_1, broad_flux_1)
    mean += gaussian_flux(velocity, broad_center_2, broad_sigma_2, broad_flux_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_softmax_unit_norm_hierarchical_centers_model(
    velocity, flux=None, error=0.03
):
    """Experimental ordered widths with noise-normalized amplitudes."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)

    low = np.log(600.0)
    high = np.log(6000.0)
    target = np.log(np.asarray([1400.0, 3600.0]))
    gaps = np.diff(np.concatenate(([low], target, [high])))
    logit_loc = np.log(gaps[:-1] / gaps[-1])
    logits_std = numpyro.sample(
        "broad_width_logits_std", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
    )
    logits = jnp.asarray(logit_loc) + 0.5 * logits_std
    spacings = jax.nn.softmax(jnp.concatenate([logits, jnp.zeros(1)]))
    ordered_log_sigma = low + (high - low) * jnp.cumsum(spacings[:-1])
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(ordered_log_sigma[0]))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(ordered_log_sigma[1]))

    def reference_norm(sigma):
        profile = gaussian_peak(velocity, 0.0, sigma, 1.0)
        return jnp.sqrt(jnp.sum(jnp.square(profile / error)))

    narrow_amplitude = lognormal_std(
        "narrow_unit_norm_amplitude",
        jnp.log(0.45 * reference_norm(220.0)),
        0.7,
    )
    broad_amplitude_1 = lognormal_std(
        "broad_unit_norm_amplitude_1",
        jnp.log(0.18 * reference_norm(1400.0)),
        0.9,
    )
    broad_amplitude_2 = lognormal_std(
        "broad_unit_norm_amplitude_2",
        jnp.log(0.10 * reference_norm(3600.0)),
        0.9,
    )
    mean = continuum(velocity, level, slope)
    mean += narrow_amplitude * gaussian_unit_noise_norm(
        velocity, systemic_center, narrow_sigma, error
    )
    mean += broad_amplitude_1 * gaussian_unit_noise_norm(
        velocity, broad_center_1, broad_sigma_1, error
    )
    mean += broad_amplitude_2 * gaussian_unit_noise_norm(
        velocity, broad_center_2, broad_sigma_2, error
    )
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_softmax_joint_orthogonal_hierarchical_centers_model(
    velocity, flux=None, error=0.03
):
    """Ordered widths with a dynamically orthogonal joint broad basis."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)

    low = np.log(600.0)
    high = np.log(6000.0)
    target = np.log(np.asarray([1400.0, 3600.0]))
    gaps = np.diff(np.concatenate(([low], target, [high])))
    logit_loc = np.log(gaps[:-1] / gaps[-1])
    logits_std = numpyro.sample(
        "broad_width_logits_std", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
    )
    logits = jnp.asarray(logit_loc) + 0.5 * logits_std
    spacings = jax.nn.softmax(jnp.concatenate([logits, jnp.zeros(1)]))
    ordered_log_sigma = low + (high - low) * jnp.cumsum(spacings[:-1])
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(ordered_log_sigma[0]))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(ordered_log_sigma[1]))

    narrow_reference_norm = jnp.sqrt(
        jnp.sum(jnp.square(gaussian_peak(velocity, 0.0, 220.0, 1.0) / error))
    )
    narrow_amplitude = lognormal_std(
        "narrow_unit_norm_amplitude",
        jnp.log(0.45 * narrow_reference_norm),
        0.7,
    )

    reference_basis_1, reference_basis_2 = joint_broad_noise_basis(
        velocity, 0.0, 1400.0, 0.0, 3600.0, error
    )
    reference_signal = gaussian_peak(velocity, 0.0, 1400.0, 0.18)
    reference_signal += gaussian_peak(velocity, 0.0, 3600.0, 0.10)
    reference_coeff_1 = jnp.dot(reference_basis_1 / error, reference_signal / error)
    reference_coeff_2 = jnp.dot(reference_basis_2 / error, reference_signal / error)
    broad_coeff_1 = lognormal_std(
        "broad_joint_orthogonal_amplitude_1", jnp.log(reference_coeff_1), 0.9
    )
    broad_coeff_2 = lognormal_std(
        "broad_joint_orthogonal_amplitude_2", jnp.log(reference_coeff_2), 0.9
    )
    broad_basis_1, broad_basis_2 = joint_broad_noise_basis(
        velocity,
        broad_center_1,
        broad_sigma_1,
        broad_center_2,
        broad_sigma_2,
        error,
    )
    mean = continuum(velocity, level, slope)
    mean += narrow_amplitude * gaussian_unit_noise_norm(
        velocity, systemic_center, narrow_sigma, error
    )
    mean += broad_coeff_1 * broad_basis_1 + broad_coeff_2 * broad_basis_2
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_ordered_relative_peak_hierarchical_centers_model(
    velocity, flux=None, error=0.03
):
    """Ordered broad profile with peak heights relative to local continuum."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_contrast = lognormal_std("narrow_peak_contrast", np.log(0.45), 0.7)
    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap", np.log(np.log(3600.0 / 1400.0)), 0.45
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))
    broad_contrast_1 = lognormal_std("broad_peak_contrast_1", np.log(0.18), 0.9)
    broad_contrast_2 = lognormal_std("broad_peak_contrast_2", np.log(0.10), 0.9)
    narrow_peak = narrow_contrast * continuum(systemic_center, level, slope)
    broad_peak_1 = broad_contrast_1 * continuum(broad_center_1, level, slope)
    broad_peak_2 = broad_contrast_2 * continuum(broad_center_2, level, slope)
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, systemic_center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, broad_center_1, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, broad_center_2, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_ordered_ew_hierarchical_centers_model(
    velocity, flux=None, error=0.03
):
    """Exact reparameterization of peak priors using rest-frame equivalent widths."""
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap", np.log(np.log(3600.0 / 1400.0)), 0.45
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))

    centers = (systemic_center, broad_center_1, broad_center_2)
    sigmas = (narrow_sigma, broad_sigma_1, broad_sigma_2)
    peak_locs = (np.log(0.45), np.log(0.18), np.log(0.10))
    peak_scales = (0.7, 0.9, 0.9)
    reference_sigmas = (220.0, 1400.0, 3600.0)
    labels = ("narrow", "broad_1", "broad_2")
    peaks = []
    standard_normal = dist.Normal(0.0, 1.0)
    for label, center, sigma, peak_loc, peak_scale, reference_sigma in zip(
        labels, centers, sigmas, peak_locs, peak_scales, reference_sigmas, strict=True
    ):
        # EW = integrated line flux / local continuum.  The correction factor
        # below preserves the baseline model's independent log-peak prior.
        ew_loc = peak_loc + np.log(SQRT_2PI * reference_sigma)
        ew_std = numpyro.sample(f"{label}_ew_std", dist.Normal(0.0, 1.0))
        log_ew = ew_loc + peak_scale * ew_std
        local_continuum = continuum(center, level, slope)
        log_peak = log_ew + jnp.log(local_continuum) - jnp.log(SQRT_2PI * sigma)
        numpyro.factor(
            f"{label}_physical_peak_prior",
            dist.Normal(peak_loc, peak_scale).log_prob(log_peak)
            - standard_normal.log_prob(ew_std),
        )
        numpyro.deterministic(f"{label}_equivalent_width", jnp.exp(log_ew))
        peaks.append(jnp.exp(log_peak))

    narrow_peak = numpyro.deterministic("narrow_peak", peaks[0])
    broad_peak_1 = numpyro.deterministic("broad_peak_1", peaks[1])
    broad_peak_2 = numpyro.deterministic("broad_peak_2", peaks[2])
    mean = continuum(velocity, level, slope)
    mean += gaussian_peak(velocity, systemic_center, narrow_sigma, narrow_peak)
    mean += gaussian_peak(velocity, broad_center_1, broad_sigma_1, broad_peak_1)
    mean += gaussian_peak(velocity, broad_center_2, broad_sigma_2, broad_peak_2)
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_fixed_orthogonal_hierarchical_model(velocity, flux=None, error=0.03):
    """Exact fixed-coordinate transform orthogonalizing lines to level/slope."""
    systemic_center = normal_std("systemic_center", 0.0, 250.0)
    broad_offset = normal_std("broad_center_offset", 0.0, 250.0)
    broad_relative = normal_std("broad_relative_offset", 0.0, 250.0)
    broad_center_1 = systemic_center + broad_offset - 0.5 * broad_relative
    broad_center_2 = systemic_center + broad_offset + 0.5 * broad_relative
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_peak = lognormal_std("narrow_peak", np.log(0.45), 0.7)
    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap", np.log(np.log(3600.0 / 1400.0)), 0.45
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))
    broad_peak_1 = lognormal_std("broad_peak_1", np.log(0.18), 0.9)
    broad_peak_2 = lognormal_std("broad_peak_2", np.log(0.10), 0.9)

    continuum_design = jnp.stack(
        [jnp.ones_like(velocity), velocity / 10_000.0], axis=1
    )
    reference_profiles = jnp.stack(
        [
            gaussian_peak(velocity, 70.0, 220.0, 1.0),
            gaussian_peak(velocity, 70.0, 1400.0, 1.0),
            gaussian_peak(velocity, 70.0, 3600.0, 1.0),
        ],
        axis=1,
    )
    weight = jnp.full_like(velocity, 1.0 / error**2)
    gram = continuum_design.T @ (weight[:, None] * continuum_design)
    projection = jnp.linalg.solve(
        gram,
        continuum_design.T @ (weight[:, None] * reference_profiles),
    )
    line_profiles = jnp.stack(
        [
            gaussian_peak(velocity, systemic_center, narrow_sigma, 1.0),
            gaussian_peak(velocity, broad_center_1, broad_sigma_1, 1.0),
            gaussian_peak(velocity, broad_center_2, broad_sigma_2, 1.0),
        ],
        axis=1,
    )
    amplitudes = jnp.stack([narrow_peak, broad_peak_1, broad_peak_2])
    orthogonal_profiles = line_profiles - continuum_design @ projection

    physical_loc = jnp.asarray([1.0, 0.0])
    physical_scale = jnp.asarray([0.2, 0.1])
    reference_amplitudes = jnp.asarray([0.45, 0.18, 0.10])
    transformed_loc = physical_loc + projection @ reference_amplitudes
    continuum_std = numpyro.sample(
        "continuum_orth_std", dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)
    )
    transformed_coefficients = transformed_loc + physical_scale * continuum_std
    physical_coefficients = transformed_coefficients - projection @ amplitudes
    standard_normal = dist.Normal(0.0, 1.0)
    physical_prior = dist.Normal(physical_loc, physical_scale)
    numpyro.factor(
        "physical_continuum_prior",
        jnp.sum(physical_prior.log_prob(physical_coefficients))
        - jnp.sum(standard_normal.log_prob(continuum_std)),
    )
    numpyro.deterministic("continuum", physical_coefficients[0])
    numpyro.deterministic("slope", physical_coefficients[1])
    mean = continuum_design @ transformed_coefficients + orthogonal_profiles @ amplitudes
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def broad_narrow_ordered_flux_model(velocity, flux=None, error=0.03):
    level = normal_std("continuum", 1.0, 0.2)
    slope = normal_std("slope", 0.0, 0.1)
    center = normal_std("line_center", 0.0, 250.0)
    narrow_sigma = lognormal_std("narrow_sigma", np.log(220.0), 0.45)
    narrow_flux = lognormal_std(
        "narrow_integrated_flux",
        np.log(0.45 * 220.0 * SQRT_2PI),
        0.7,
    )

    log_sigma_mid = normal_std("broad_log_sigma_mid", np.log(2300.0), 0.45)
    log_sigma_gap = normal_std(
        "broad_log_sigma_gap",
        np.log(np.log(3600.0 / 1400.0)),
        0.45,
    )
    half_gap = 0.5 * jnp.exp(log_sigma_gap)
    broad_sigma_1 = numpyro.deterministic("broad_sigma_1", jnp.exp(log_sigma_mid - half_gap))
    broad_sigma_2 = numpyro.deterministic("broad_sigma_2", jnp.exp(log_sigma_mid + half_gap))
    total_broad_flux = lognormal_std(
        "broad_integrated_flux",
        np.log(1550.0),
        0.9,
    )
    mix_logit = normal_std("broad_mix_logit", 0.0, 1.25)
    mix = jax.nn.sigmoid(mix_logit)

    mean = continuum(velocity, level, slope)
    mean += gaussian_flux(velocity, center, narrow_sigma, narrow_flux)
    mean += gaussian_flux(velocity, center, broad_sigma_1, total_broad_flux * mix)
    mean += gaussian_flux(velocity, center, broad_sigma_2, total_broad_flux * (1.0 - mix))
    numpyro.sample("obs", dist.Normal(mean, error), obs=flux)


def make_mocks(seed=2026, n_pixels=240, error=0.03):
    rng = np.random.default_rng(seed)
    velocity = np.linspace(-12_000.0, 12_000.0, n_pixels)
    narrow_mean = 1.0 + 0.025 * velocity / 10_000.0
    narrow_mean += np.asarray(gaussian_peak(velocity, 70.0, 220.0, 0.45))

    one_broad_mean = 1.0 + 0.025 * velocity / 10_000.0
    one_broad_mean += np.asarray(gaussian_peak(velocity, 70.0, 220.0, 0.45))
    one_broad_mean += np.asarray(gaussian_peak(velocity, 70.0, 1800.0, 0.20))

    two_broad_mean = 1.0 + 0.025 * velocity / 10_000.0
    two_broad_mean += np.asarray(gaussian_peak(velocity, 70.0, 220.0, 0.45))
    two_broad_mean += np.asarray(gaussian_peak(velocity, 70.0, 1400.0, 0.18))
    two_broad_mean += np.asarray(gaussian_peak(velocity, 70.0, 3600.0, 0.10))
    return (
        velocity,
        narrow_mean + rng.normal(0.0, error, n_pixels),
        one_broad_mean + rng.normal(0.0, error, n_pixels),
        two_broad_mean + rng.normal(0.0, error, n_pixels),
    )


def make_two_broad_width_mock(velocity, sigma_1, sigma_2, *, seed, error=0.03):
    """Return the same two-broad-component mock at a different width scale."""
    rng = np.random.default_rng(seed)
    mean = 1.0 + 0.025 * velocity / 10_000.0
    mean += np.asarray(gaussian_peak(velocity, 70.0, 220.0, 0.45))
    mean += np.asarray(gaussian_peak(velocity, 70.0, sigma_1, 0.18))
    mean += np.asarray(gaussian_peak(velocity, 70.0, sigma_2, 0.10))
    return mean + rng.normal(0.0, error, velocity.size)


def make_matched_noisy_mock(kind, strength=1.0, error=0.03):
    """Build a seed-dependent noisy mock shared by competing coordinates."""
    def generate(seed):
        rng = np.random.default_rng(seed)
        velocity = np.linspace(-12_000.0, 12_000.0, 240)
        mean = 1.0 + 0.025 * velocity / 10_000.0
        mean += strength * np.asarray(
            gaussian_peak(velocity, 70.0, 220.0, 0.45)
        )
        if kind in {"one_broad", "two_broad"}:
            sigma, peak = (1800.0, 0.20) if kind == "one_broad" else (1400.0, 0.18)
            mean += strength * np.asarray(
                gaussian_peak(velocity, 70.0, sigma, peak)
            )
        if kind == "two_broad":
            mean += strength * np.asarray(
                gaussian_peak(velocity, 70.0, 3600.0, 0.10)
            )
        return mean + rng.normal(0.0, error, velocity.size)

    return generate


def summarize_run(mcmc, max_tree_depth):
    extra = mcmc.get_extra_fields()
    steps = np.asarray(extra["num_steps"], dtype=int)
    samples = mcmc.get_samples(group_by_chain=True)
    ess_values = []
    for value in samples.values():
        ess_values.extend(np.ravel(np.asarray(effective_sample_size(np.asarray(value)))))
    summary = {
        "step_size": float(np.asarray(mcmc.last_state.adapt_state.step_size)),
        "median_leapfrog_steps": float(np.median(steps)),
        "p90_leapfrog_steps": float(np.percentile(steps, 90)),
        "max_depth_fraction": float(np.mean(steps >= 2**max_tree_depth - 1)),
        "mean_accept_prob": float(np.mean(np.asarray(extra["accept_prob"]))),
        "divergences": int(np.sum(np.asarray(extra["diverging"]))),
        "min_ess": float(np.min(ess_values)),
        "median_ess": float(np.median(ess_values)),
    }
    if "broad_sigma_1" in samples and "broad_sigma_2" in samples:
        sigma_1 = np.asarray(samples["broad_sigma_1"])
        sigma_2 = np.asarray(samples["broad_sigma_2"])
        summary["broad_1_wider_fraction"] = float(np.mean(sigma_1 > sigma_2))
        summary["broad_sigma_1_mean"] = float(np.mean(sigma_1))
        summary["broad_sigma_2_mean"] = float(np.mean(sigma_2))
    amplitude_names = [
        name
        for name in (
            "narrow_peak",
            "broad_peak",
            "broad_peak_1",
            "broad_peak_2",
            "broad_integrated_flux_1",
            "broad_integrated_flux_2",
            "narrow_unit_norm_amplitude",
            "broad_unit_norm_amplitude",
            "broad_unit_norm_amplitude_1",
            "broad_unit_norm_amplitude_2",
            "broad_joint_orthogonal_amplitude_1",
            "broad_joint_orthogonal_amplitude_2",
            "narrow_peak_contrast",
            "broad_peak_contrast_1",
            "broad_peak_contrast_2",
        )
        if name in samples
    ]
    if "continuum" in samples and amplitude_names:
        continuum_draws = np.ravel(np.asarray(samples["continuum"]))
        correlations = [
            abs(
                float(
                    np.corrcoef(
                        continuum_draws,
                        np.ravel(np.asarray(samples[name])),
                    )[0, 1]
                )
            )
            for name in amplitude_names
        ]
        summary["median_abs_continuum_amplitude_corr"] = float(
            np.median(correlations)
        )
        summary["max_abs_continuum_amplitude_corr"] = float(np.max(correlations))
    width_amplitude_pairs = [
        ("narrow_sigma", "narrow_peak"),
        ("narrow_sigma", "narrow_unit_norm_amplitude"),
        ("broad_sigma", "broad_peak"),
        ("broad_sigma", "broad_unit_norm_amplitude"),
        ("broad_sigma_1", "broad_peak_1"),
        ("broad_sigma_1", "broad_unit_norm_amplitude_1"),
        ("broad_sigma_2", "broad_peak_2"),
        ("broad_sigma_2", "broad_unit_norm_amplitude_2"),
        ("broad_sigma_1", "broad_joint_orthogonal_amplitude_1"),
        ("broad_sigma_1", "broad_joint_orthogonal_amplitude_2"),
        ("broad_sigma_2", "broad_joint_orthogonal_amplitude_1"),
        ("broad_sigma_2", "broad_joint_orthogonal_amplitude_2"),
    ]
    width_amplitude_correlations = [
        abs(
            float(
                np.corrcoef(
                    np.ravel(np.asarray(samples[width_name])),
                    np.ravel(np.asarray(samples[amplitude_name])),
                )[0, 1]
            )
        )
        for width_name, amplitude_name in width_amplitude_pairs
        if width_name in samples and amplitude_name in samples
    ]
    if width_amplitude_correlations:
        summary["median_abs_width_amplitude_corr"] = float(
            np.median(width_amplitude_correlations)
        )
        summary["max_abs_width_amplitude_corr"] = float(
            np.max(width_amplitude_correlations)
        )
    return summary


def run_one(model: Callable, velocity, flux, *, seed, warmup, samples, max_tree_depth, dense_mass):
    kernel = NUTS(model, dense_mass=dense_mass, target_accept_prob=0.9, max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        velocity=jnp.asarray(velocity),
        flux=jnp.asarray(flux),
        error=0.03,
        extra_fields=("num_steps", "accept_prob", "diverging"),
    )
    return summarize_run(mcmc, max_tree_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3, help="Number of independent NUTS seeds")
    parser.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--case", action="append", help="Run only the named case (repeatable)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    velocity, narrow_flux, one_broad_flux, two_broad_flux = make_mocks()
    two_broad_narrow_flux = make_two_broad_width_mock(
        velocity, 900.0, 2200.0, seed=2027
    )
    two_broad_wide_flux = make_two_broad_width_mock(
        velocity, 2800.0, 5600.0, seed=2028
    )
    declarative_models = {
        (n_broad, amplitude): build_experimental_geometry_model(
            ExperimentalGeometryVariant(n_broad=n_broad, amplitude=amplitude)
        )
        for n_broad in (0, 1, 2)
        for amplitude in ("peak", "unit_norm")
    }
    cases = {
        "matched_narrow_peak_strong": (
            declarative_models[(0, "peak")],
            make_matched_noisy_mock("narrow"),
        ),
        "matched_narrow_unit_norm_strong": (
            declarative_models[(0, "unit_norm")],
            make_matched_noisy_mock("narrow"),
        ),
        "matched_one_broad_peak_strong": (
            declarative_models[(1, "peak")],
            make_matched_noisy_mock("one_broad"),
        ),
        "matched_one_broad_unit_norm_strong": (
            declarative_models[(1, "unit_norm")],
            make_matched_noisy_mock("one_broad"),
        ),
        "matched_two_broad_peak_strong": (
            declarative_models[(2, "peak")],
            make_matched_noisy_mock("two_broad"),
        ),
        "matched_two_broad_unit_norm_strong": (
            declarative_models[(2, "unit_norm")],
            make_matched_noisy_mock("two_broad"),
        ),
        "matched_two_broad_joint_orthogonal_strong": (
            broad_narrow_softmax_joint_orthogonal_hierarchical_centers_model,
            make_matched_noisy_mock("two_broad"),
        ),
        "matched_narrow_peak_weak": (
            declarative_models[(0, "peak")],
            make_matched_noisy_mock("narrow", strength=0.2),
        ),
        "matched_narrow_unit_norm_weak": (
            declarative_models[(0, "unit_norm")],
            make_matched_noisy_mock("narrow", strength=0.2),
        ),
        "matched_one_broad_peak_weak": (
            declarative_models[(1, "peak")],
            make_matched_noisy_mock("one_broad", strength=0.2),
        ),
        "matched_one_broad_unit_norm_weak": (
            declarative_models[(1, "unit_norm")],
            make_matched_noisy_mock("one_broad", strength=0.2),
        ),
        "matched_two_broad_peak_weak": (
            declarative_models[(2, "peak")],
            make_matched_noisy_mock("two_broad", strength=0.2),
        ),
        "matched_two_broad_unit_norm_weak": (
            declarative_models[(2, "unit_norm")],
            make_matched_noisy_mock("two_broad", strength=0.2),
        ),
        "matched_two_broad_joint_orthogonal_weak": (
            broad_narrow_softmax_joint_orthogonal_hierarchical_centers_model,
            make_matched_noisy_mock("two_broad", strength=0.2),
        ),
        "narrow_experimental_peak_baseline": (narrow_peak_model, narrow_flux),
        "narrow_integrated_flux": (narrow_flux_model, narrow_flux),
        "one_broad_experimental_independent_peak": (one_broad_narrow_independent_model, one_broad_flux),
        "one_broad_separate_integrated_flux": (one_broad_narrow_separate_flux_model, one_broad_flux),
        "one_broad_ordered_total_flux": (one_broad_narrow_ordered_flux_model, one_broad_flux),
        "two_broad_experimental_independent_peak": (broad_narrow_independent_model, two_broad_flux),
        "two_broad_separate_integrated_flux": (broad_narrow_independent_flux_model, two_broad_flux),
        "two_broad_ordered_peak": (broad_narrow_ordered_peak_model, two_broad_flux),
        "two_broad_ordered_peak_independent_centers": (
            broad_narrow_ordered_peak_independent_centers_model,
            two_broad_flux,
        ),
        "two_broad_ordered_peak_hierarchical_centers": (
            broad_narrow_ordered_peak_hierarchical_centers_model,
            two_broad_flux,
        ),
        "two_broad_softmax_peak_hierarchical_centers": (
            broad_narrow_softmax_peak_hierarchical_centers_model,
            two_broad_flux,
        ),
        "two_broad_softmax_flux_hierarchical_centers": (
            broad_narrow_softmax_flux_hierarchical_centers_model,
            two_broad_flux,
        ),
        "two_broad_narrow_softmax": (
            broad_narrow_softmax_peak_hierarchical_centers_model,
            two_broad_narrow_flux,
        ),
        "two_broad_narrow_mean_gap": (
            broad_narrow_ordered_peak_hierarchical_centers_model,
            two_broad_narrow_flux,
        ),
        "two_broad_narrow_softmax_flux": (
            broad_narrow_softmax_flux_hierarchical_centers_model,
            two_broad_narrow_flux,
        ),
        "two_broad_wide_softmax": (
            broad_narrow_softmax_peak_hierarchical_centers_model,
            two_broad_wide_flux,
        ),
        "two_broad_wide_mean_gap": (
            broad_narrow_ordered_peak_hierarchical_centers_model,
            two_broad_wide_flux,
        ),
        "two_broad_wide_softmax_flux": (
            broad_narrow_softmax_flux_hierarchical_centers_model,
            two_broad_wide_flux,
        ),
        "two_broad_ordered_relative_peak_hierarchical_centers": (
            broad_narrow_ordered_relative_peak_hierarchical_centers_model,
            two_broad_flux,
        ),
        "two_broad_ordered_ew_hierarchical_centers": (
            broad_narrow_ordered_ew_hierarchical_centers_model,
            two_broad_flux,
        ),
        "two_broad_fixed_orthogonal_hierarchical": (
            broad_narrow_fixed_orthogonal_hierarchical_model,
            two_broad_flux,
        ),
        "two_broad_ordered_total_flux": (broad_narrow_ordered_flux_model, two_broad_flux),
    }
    if args.case:
        missing = sorted(set(args.case) - set(cases))
        if missing:
            parser.error(f"unknown case(s): {', '.join(missing)}")
        cases = {name: cases[name] for name in args.case}
    output = {}
    for name, (model, observed_flux) in cases.items():
        runs = [
            run_one(
                model,
                velocity,
                observed_flux(10_000 + seed) if callable(observed_flux) else observed_flux,
                seed=100 + seed,
                warmup=args.warmup,
                samples=args.samples,
                max_tree_depth=args.max_tree_depth,
                dense_mass=args.dense_mass,
            )
            for seed in range(args.seeds)
        ]
        output[name] = {
            "runs": runs,
            "median": {key: float(np.median([run[key] for run in runs])) for key in runs[0]},
        }
        print(json.dumps({name: output[name]["median"]}, indent=2), flush=True)

    if not args.summary_only:
        print("\nFULL_RESULTS")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
