from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import numpyro.distributions as dist


@dataclass(frozen=True)
class ErrorScaledHalfNormalPrior:
    """Half-normal prior whose scale is resolved from the observed errors."""

    scale_multiplier: float


@dataclass
class Observation:
    """Observation-level metadata for one quasar spectrum."""

    object_id: str = "result"
    redshift: float = 0.0
    ra: float | None = None
    dec: float | None = None
    apply_mw_deredden: bool = True


@dataclass
class SpectroscopyData:
    """Observed spectral measurements on an observed-frame wavelength grid.

    ``mask`` is an optional Boolean keep-mask: ``True`` includes a pixel in the
    fit and ``False`` rejects it. It is combined with the automatic finite-flux,
    finite-wavelength, and positive-finite-error mask before preprocessing.
    """

    wave_obs: Sequence[float]
    fluxes: Sequence[float]
    errors: Sequence[float] | float | None = None
    resolving_power: float | None = None
    apply_instrumental_resolution: bool = False
    mask: Sequence[bool] | None = None

    def validate(self) -> None:
        """Validate spectroscopy payload lengths and numerical domain."""
        n = len(self.wave_obs)
        if n < 2:
            raise ValueError("Spectroscopy requires at least two pixels.")
        if len(self.fluxes) != n:
            raise ValueError("Spectroscopy fluxes must have the same length as wave_obs.")
        if self.errors is not None and not np.isscalar(self.errors) and len(self.errors) != n:
            raise ValueError("Spectroscopy errors must be scalar, None, or match wave_obs length.")
        if self.mask is not None and len(self.mask) != n:
            raise ValueError("spectroscopy mask must match wave_obs length.")
        if self.apply_instrumental_resolution and self.resolving_power is None:
            raise ValueError("apply_instrumental_resolution=True requires resolving_power.")
        if self.resolving_power is not None:
            resolving_power = float(self.resolving_power)
            if not np.isfinite(resolving_power) or resolving_power <= 0.0:
                raise ValueError("resolving_power must be finite and positive when provided.")
        wave = np.asarray(self.wave_obs, dtype=float)
        valid_wave = wave[np.isfinite(wave) & (wave > 0.0)]
        if valid_wave.size < 2:
            raise ValueError("wave_obs must contain at least two finite positive wavelengths.")
        if np.any(np.diff(valid_wave) <= 0.0):
            raise ValueError("wave_obs must be strictly increasing.")


@dataclass
class PSFPhotometryData:
    """Optional PSF-aperture photometry used for spectral recalibration.

    JAXQSOFit is a spectral fitter, so these data are only used as an extra
    calibration constraint on the fitted spectrum. Use bands whose transmission
    curves overlap the observed spectral wavelength coverage. For full joint
    spectrum + broadband SED modeling, use ``jaxsedfit`` instead.
    """

    magnitudes: Sequence[float]
    magnitude_errors: Sequence[float]
    filter_names: Sequence[str] = ("u", "g", "r", "i", "z")

    def validate(self) -> None:
        """Validate PSF photometry vector lengths."""
        n = len(self.magnitudes)
        if len(self.magnitude_errors) != n or len(self.filter_names) != n:
            raise ValueError("PSF magnitudes, errors, and filter_names must have the same length.")


@dataclass
class PreprocessingConfig:
    """Spectrum preprocessing options applied before fitting."""

    wave_range: tuple[float, float] | None = None
    wave_mask: Sequence[Sequence[float]] | None = None
    mask_lya_forest: bool = True


@dataclass
class BALConfig:
    """Built-in BAL absorption component configuration."""

    enabled: bool = False
    tau_scale: float = 0.25
    covering_loc: float = 0.15
    covering_scale: float = 0.12
    covering_high: float = 0.70
    fwhm_kms_loc: float = 8000.0
    fwhm_kms_scale: float = 2500.0
    fwhm_kms_low: float = 2000.0
    fwhm_kms_high: float = 15000.0


@dataclass
class ContinuumConfig:
    """Continuum and spectral component switches.

    Important
    ---------
    Intrinsic AGN reddening is **enabled by default** (``fit_reddening=True``).
    One fitted E(B-V) screen attenuates the power-law continuum, UV/optical
    Fe II, Balmer continuum, and custom nuclear continuum components. It does
    not attenuate emission lines or host-galaxy starlight. Milky Way foreground
    dereddening is separate and is controlled by
    :attr:`Observation.apply_mw_deredden`.

    For stable NUTS geometry, the power-law is sampled by default in apparent
    (post-attenuation) normalization and slope coordinates. The constant and
    linear parts of the reddening curve are absorbed into those coordinates;
    E(B-V) controls the remaining curvature. This is an exact coordinate
    transformation and does not change the physical model or priors.

    The polynomial is a multiplicative residual flux-calibration term applied
    to the complete model spectrum. ``polynomial_order`` is its highest
    residual-curvature degree. The constant and linear directions are omitted
    because they duplicate the power-law normalization and slope;
    consequently, values below two add no polynomial coefficients.
    """

    fit_power_law: bool = True
    fit_feii: bool = True
    fit_balmer_continuum: bool = False
    fit_polynomial_tilt: bool = True
    fit_reddening: bool = True
    polynomial_order: int = 2
    broadening_convolution: str = "fft"

    def __post_init__(self) -> None:
        method = str(self.broadening_convolution).lower()
        if method not in {"fft", "direct"}:
            raise ValueError("ContinuumConfig.broadening_convolution must be 'fft' or 'direct'.")
        self.broadening_convolution = method


@dataclass
class HostConfig:
    """Host-galaxy spectral decomposition configuration."""

    enabled: bool = True
    sfh_model: str = "delayed"
    dsps_ssp_fn: str = "tempdata.h5"
    age_grid_gyr: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    logzsol_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.2)


@dataclass
class LineConfig:
    """Emission-line model configuration.

    Narrow-line centroid behavior
    -----------------------------
    The default ``pool_narrow_centroids=True`` pools narrow cores into three
    exact kinematic families: low-ionization/systemic, high-ionization, and
    coronal. Every line within a family shares one centroid and one FWHM,
    including lines in different complexes. There are no per-complex residual
    centroid or width offsets and no wavelength-calibration-error parameters.
    The low-ionization velocity has a zero-centered Normal prior with default
    scale 250 km/s. The high-ionization velocity is offset from it with a
    default 150 km/s scale, and the coronal velocity is offset from the
    high-ionization velocity with a default 250 km/s scale.

    Broad components and components identified as wings or outflows do not
    participate in global NLR pooling; their existing centroid models are
    retained. Explicit centroid ties defined by positive ``vindex`` values in
    the line table are also retained.

    Set ``pool_narrow_centroids=False`` to restore the line-table kinematics:
    otherwise untied narrow groups receive independent bounded centroids and
    widths, while explicit ``vindex`` and ``windex`` ties remain intact.

    ``include_elg_narrow_lines`` and ``include_high_ionization_lines`` append
    the corresponding built-in optional rows to the active line table. These
    are model-component switches, not prior-construction options, and apply
    whether the fit uses automatically generated or user-supplied priors.
    """

    enabled: bool = True
    use_broad_lines: bool = True
    use_narrow_lines: bool = True
    pool_narrow_centroids: bool = True
    include_elg_narrow_lines: bool = False
    include_high_ionization_lines: bool = False
    custom_components: Sequence[Any] | None = None
    custom_line_components: Sequence[Any] | None = None


@dataclass
class AGNConfig:
    """High-level AGN spectral-type presets.

    The ``agn_type`` flag is intentionally a preset layer over explicit continuum,
    host, and line switches. Call :meth:`FitConfig.apply_agn_type_defaults`
    or :meth:`FitConfig.set_agn_type` to apply it, then override individual
    component switches as needed.
    """

    agn_type: int = 1

    def __post_init__(self) -> None:
        self.agn_type = int(self.agn_type)
        if self.agn_type not in {1, 2}:
            raise ValueError("AGNConfig.agn_type must be 1 or 2.")


@dataclass
class InferenceConfig:
    """Inference defaults for Optax and NUTS.

    ``random_seed`` controls stochastic inference and posterior prediction so
    repeated fits with the same configuration are reproducible.

    NUTS standardizes active prior coordinates by default, which puts latent
    parameters with different physical units on comparable scales.  It uses a
    diagonal mass matrix globally and learns dense blocks only within emission-
    line complexes.  Those complexes contain strongly correlated amplitudes,
    widths, and centroids, while a fully dense matrix across the entire spectral
    model is expensive and poorly estimated by typical warmup lengths.  This
    block structure captures the important local correlations without coupling
    unrelated continuum, host, and line parameters.
    """

    method: str = "optax+nuts"
    random_seed: int = 0
    map_steps: int = 2000
    learning_rate: float = 1.0e-2
    num_warmup: int = 250
    num_samples: int = 250
    num_chains: int = 1
    target_accept_prob: float = 0.85
    dense_mass: bool = False
    line_block_dense_mass: bool = True
    standardize_active_priors: bool = True
    max_tree_depth: int = 8
    plot_init: bool = False


def _scalar_or_list(value: Any) -> Any:
    """Convert scalar array-like distribution parameters into plain Python values.


    Parameters
    ----------
    value : object
        value value.
    """
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return arr.tolist()


def _numpyro_distribution_to_mapping(value: Any) -> dict[str, Any] | None:
    """Convert supported NumPyro distributions into the model prior schema.

    Parameters
    ----------
    value : object
        value value.
    """
    module = getattr(value.__class__, "__module__", "")
    if not module.startswith("numpyro.distributions"):
        return None

    name = value.__class__.__name__
    if name in {"Normal", "LogNormal"}:
        return {
            "dist": name,
            "loc": _scalar_or_list(value.loc),
            "scale": _scalar_or_list(value.scale),
        }
    if name == "TruncatedNormal":
        return {
            "dist": name,
            "loc": _scalar_or_list(value.loc),
            "scale": _scalar_or_list(value.scale),
            "low": _scalar_or_list(value.low),
            "high": _scalar_or_list(value.high),
        }
    if name == "TwoSidedTruncatedDistribution":
        base = value.base_dist
        if base.__class__.__name__ == "Normal":
            return {
                "dist": "TruncatedNormal",
                "loc": _scalar_or_list(base.loc),
                "scale": _scalar_or_list(base.scale),
                "low": _scalar_or_list(value.low),
                "high": _scalar_or_list(value.high),
            }
    if name == "HalfNormal":
        return {"dist": name, "scale": _scalar_or_list(value.scale)}
    if name == "StudentT":
        return {
            "dist": name,
            "df": _scalar_or_list(value.df),
            "loc": _scalar_or_list(value.loc),
            "scale": _scalar_or_list(value.scale),
        }
    if name == "Uniform":
        return {
            "dist": name,
            "low": _scalar_or_list(value.low),
            "high": _scalar_or_list(value.high),
        }
    if name == "Exponential":
        rate = _scalar_or_list(value.rate)
        return {"dist": name, "scale": 1.0 / rate if np.isscalar(rate) else (1.0 / np.asarray(rate)).tolist()}
    raise TypeError(f"Unsupported NumPyro prior distribution: {name}")


def _mapping_to_numpyro_distribution(value: Mapping[str, Any]) -> dist.Distribution:
    """Restore a NumPyro distribution from a legacy serialized mapping."""
    family = str(value.get("dist", value.get("family", ""))).lower()
    if family in {"normal", "gaussian"}:
        return dist.Normal(value.get("loc", 0.0), value.get("scale", 1.0))
    if family in {"truncatednormal", "truncated_normal", "truncnormal", "truncnorm"}:
        return dist.TruncatedNormal(
            value.get("loc", 0.0), value.get("scale", 1.0),
            low=value.get("low", -np.inf), high=value.get("high", np.inf),
        )
    if family in {"lognormal", "log-normal", "log_normal"}:
        return dist.LogNormal(value.get("loc", 0.0), value.get("scale", 1.0))
    if family in {"halfnormal", "half_normal"}:
        return dist.HalfNormal(value.get("scale", 1.0))
    if family in {"student_t", "studentt", "t"}:
        return dist.StudentT(value.get("df", 5.0), value.get("loc", 0.0), value.get("scale", 1.0))
    if family in {"uniform", "flat"}:
        return dist.Uniform(value.get("low", 0.0), value.get("high", 1.0))
    if family in {"exponential", "exp"}:
        return dist.Exponential(1.0 / value.get("scale", 1.0))
    raise TypeError(f"Unsupported serialized prior distribution: {family!r}")


def _prior_to_mapping(value: Any) -> dist.Distribution:
    """Return the canonical in-memory NumPyro distribution.

    Parameters
    ----------
    value : object
        value value.
    """
    if isinstance(value, dist.Distribution):
        return value
    if isinstance(value, Mapping):
        return _mapping_to_numpyro_distribution(value)
    raise TypeError("Prior fields must be supported numpyro.distributions objects.")


@dataclass
class PowerLawPriorConfig:
    """Semantic power-law prior options."""

    slope: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert power-law prior settings into model-site keys."""
        out: dict[str, Any] = {}
        if self.slope is not None:
            out["PL_slope"] = _prior_to_mapping(self.slope)
        return out


@dataclass
class OutputConfig:
    """Plotting and persistence defaults."""

    output_path: str | None = None
    save_name: str | None = None
    save_result: bool = True
    plot_fig: bool = True
    plot_init: bool = False
    save_fig: bool = True
    show_plot: bool = False


@dataclass
class ContinuumPriorConfig:
    """Semantic continuum-prior options."""

    power_law_pivot: float | None = None
    polynomial_pivot: float | None = None
    output_wavelengths: Sequence[float] | None = None
    powerlaw: PowerLawPriorConfig = field(default_factory=PowerLawPriorConfig)

    def __post_init__(self) -> None:
        """Normalize nested continuum prior sections passed as mappings."""
        if isinstance(self.powerlaw, Mapping):
            self.powerlaw = PowerLawPriorConfig(
                **{k: v for k, v in self.powerlaw.items() if k in PowerLawPriorConfig.__dataclass_fields__}
            )

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic continuum prior settings into model-site keys."""
        out: dict[str, Any] = {}
        out.update(self.powerlaw.to_mapping())
        if self.power_law_pivot is not None:
            out["PL_pivot"] = float(self.power_law_pivot)
        if self.polynomial_pivot is not None:
            out["poly_pivot"] = float(self.polynomial_pivot)
        if self.output_wavelengths is not None:
            out["out_params"] = {"cont_lum_waves": list(self.output_wavelengths)}
        return out


@dataclass
class HostPriorConfig:
    """Semantic host-galaxy prior options."""

    redshift_weight_enabled: bool | None = None
    fraction: Any | None = None
    stellar_mass: Any | None = None
    aperture_scale: Any | None = None
    sfh_age_gyr: Any | None = None
    sfh_tau_over_age: Any | None = None
    metallicity: Any | None = None
    metallicity_scatter: Any | None = None
    template_age_prior: Mapping[str, Any] | None = None
    sfh_model: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic host prior settings into model-site keys."""
        out: dict[str, Any] = {}
        if self.fraction is not None:
            out["log_frac_host"] = _prior_to_mapping(self.fraction)
        if self.stellar_mass is not None:
            out["log_stellar_mass"] = _prior_to_mapping(self.stellar_mass)
        if self.aperture_scale is not None:
            out["log_host_aperture_scale"] = _prior_to_mapping(self.aperture_scale)
        if self.sfh_age_gyr is not None:
            out["log_sfh_age_gyr"] = _prior_to_mapping(self.sfh_age_gyr)
        if self.sfh_tau_over_age is not None:
            out["log_sfh_tau_over_age"] = _prior_to_mapping(self.sfh_tau_over_age)
        if self.metallicity is not None:
            out["gal_lgmet"] = _prior_to_mapping(self.metallicity)
        if self.metallicity_scatter is not None:
            out["log_gal_lgmet_scatter"] = _prior_to_mapping(self.metallicity_scatter)
        if self.template_age_prior is not None:
            out["host_template_age_prior"] = dict(self.template_age_prior)
        if self.sfh_model is not None:
            out["host_sfh_model"] = str(self.sfh_model)
        if self.redshift_weight_enabled is not None:
            host_z = dict(out.get("host_redshift_prior", {}))
            host_z["enabled"] = bool(self.redshift_weight_enabled)
            out["host_redshift_prior"] = host_z
        return out


@dataclass
class LinePriorConfig:
    """Semantic emission-line prior options.

    ``extra_amp_scale_mult`` regularizes redundant broad-line components.  The
    first Gaussian in each multi-Gaussian broad line keeps the usual amplitude
    prior; later Gaussians are pulled toward zero with a scale equal to this
    multiplier times the first component's initial amplitude.
    """

    table: Sequence[Mapping[str, Any]] | None = None
    dmu_scale_mult: float | None = None
    sig_scale_mult: float | None = None
    amp_scale_mult: float | None = None
    extra_amp_scale_mult: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic emission-line prior settings into model-site keys."""
        out: dict[str, Any] = {}
        if self.table is not None:
            out["line"] = {"table": list(self.table)}
        if self.dmu_scale_mult is not None:
            out["line_dmu_scale_mult"] = float(self.dmu_scale_mult)
        if self.sig_scale_mult is not None:
            out["line_sig_scale_mult"] = float(self.sig_scale_mult)
        if self.amp_scale_mult is not None:
            out["line_amp_scale_mult"] = float(self.amp_scale_mult)
        if self.extra_amp_scale_mult is not None:
            out["line_extra_amp_scale_mult"] = float(self.extra_amp_scale_mult)
        return out


@dataclass
class FeIIPriorConfig:
    """Semantic Fe II prior options.

    The UV and optical templates have independent amplitudes but share their
    velocity ``fwhm`` and ``shift``.  ``uv_fwhm`` and ``optical_fwhm`` remain
    accepted as legacy aliases; when both are supplied, ``uv_fwhm`` defines
    the shared-width prior.
    """

    uv_norm: Any | None = None
    op_over_uv: Any | None = None
    fwhm: Any | None = None
    shift: Any | None = None
    uv_fwhm: Any | None = None
    optical_fwhm: Any | None = None
    fractional_error: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic Fe II prior settings into model-site keys."""
        out: dict[str, Any] = {}
        if self.uv_norm is not None:
            out["log_Fe_uv_norm"] = _prior_to_mapping(self.uv_norm)
        if self.op_over_uv is not None:
            out["log_Fe_op_over_uv"] = _prior_to_mapping(self.op_over_uv)
        shared_fwhm = self.fwhm
        if shared_fwhm is None:
            shared_fwhm = self.uv_fwhm
        if shared_fwhm is None:
            shared_fwhm = self.optical_fwhm
        if shared_fwhm is not None:
            out["log_Fe_FWHM"] = _prior_to_mapping(shared_fwhm)
        if self.shift is not None:
            out["Fe_shift"] = _prior_to_mapping(self.shift)
        if self.fractional_error is not None:
            out["frac_fe_jitter"] = _prior_to_mapping(self.fractional_error)
        return out


@dataclass
class PSFPriorConfig:
    """Semantic PSF recalibration prior options."""

    def to_mapping(self) -> dict[str, Any]:
        """Return PSF recalibration prior settings."""
        return {}


@dataclass
class PriorConfig:
    """Object-oriented prior configuration for a quasar spectral fit.

    Parameters
    ----------
    continuum : ContinuumPriorConfig or mapping, optional
        Priors and fixed settings for the power-law continuum, polynomial
        pivot, and requested continuum-luminosity outputs.
    host : HostPriorConfig or mapping, optional
        Priors controlling the host-galaxy fraction, stellar population,
        aperture scale, metallicity, and host SFH behavior.
    lines : LinePriorConfig or mapping, optional
        Emission-line table and scale multipliers for tied line positions,
        widths, and amplitudes.
    feii : FeIIPriorConfig or mapping, optional
        Fe II normalization, optical/UV ratio, and broadening priors.
    psf : PSFPriorConfig or mapping, optional
        Priors for PSF-photometry recalibration terms.
    student_t_df : float, optional
        Degrees of freedom for the spectral Student-t likelihood.
    """

    continuum: ContinuumPriorConfig = field(default_factory=ContinuumPriorConfig)
    host: HostPriorConfig = field(default_factory=HostPriorConfig)
    lines: LinePriorConfig = field(default_factory=LinePriorConfig)
    feii: FeIIPriorConfig = field(default_factory=FeIIPriorConfig)
    psf: PSFPriorConfig = field(default_factory=PSFPriorConfig)
    student_t_df: float | None = None
    _model_priors: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize nested prior sections passed as mappings."""
        self.continuum = _coerce_dataclass(ContinuumPriorConfig, self.continuum)
        self.host = _coerce_dataclass(HostPriorConfig, self.host)
        self.lines = _coerce_dataclass(LinePriorConfig, self.lines)
        self.feii = _coerce_dataclass(FeIIPriorConfig, self.feii)
        self.psf = _coerce_dataclass(PSFPriorConfig, self.psf)
        self._model_priors = dict(self._model_priors)

    @classmethod
    def _from_model_priors(cls, model_priors: Mapping[str, Any]) -> "PriorConfig":
        """Build a PriorConfig from the low-level default-prior payload.

        Parameters
        ----------
        model_priors : mapping
            Low-level prior dictionary produced by the internal default-prior
            builder. This preserves defaults while still allowing semantic
            section overrides through the public ``PriorConfig`` API.
        """
        out = cls()
        out._model_priors = dict(model_priors)
        return out

    @classmethod
    def from_spectrum(
        cls,
        flux: Sequence[float],
        redshift: float | None = None,
        *,
        line_config: Mapping[str, Any] | None = None,
        pl_pivot: float | None = None,
    ) -> "PriorConfig":
        """Build default priors from an observed spectrum flux scale.

        The low-level model prior builder expects rest-frame flux density. This
        constructor accepts observed-frame flux and redshift, applying the
        standard ``flux_rest = flux * (1 + redshift)`` conversion when a
        redshift is provided.

        Parameters
        ----------
        flux : sequence of float
            Observed-frame spectral flux density used to set scale-aware
            default priors.
        redshift : float, optional
            Source redshift. When supplied, the flux scale is converted to the
            rest-frame convention expected by the low-level prior builder.
        line_config : mapping, optional
            Optional line-table or line-prior settings passed to the default
            prior builder.
        pl_pivot : float, optional
            Rest-frame wavelength pivot for the power-law continuum prior.
        """
        from .defaults import _build_default_prior_config

        flux_arr = np.asarray(flux, dtype=float)
        flux_for_priors = flux_arr * (1.0 + float(redshift)) if redshift is not None else flux_arr
        return _build_default_prior_config(
            flux_for_priors,
            line_config=None if line_config is None else dict(line_config),
            pl_pivot=pl_pivot,
        )

    @property
    def powerlaw(self) -> PowerLawPriorConfig:
        """Semantic power-law prior section."""
        return self.continuum.powerlaw

    @powerlaw.setter
    def powerlaw(self, value: PowerLawPriorConfig | Mapping[str, Any]) -> None:
        """Set the semantic power-law prior section.

        Parameters
        ----------
        value : PowerLawPriorConfig or mapping
            Replacement power-law prior section.
        """
        self.continuum.powerlaw = _coerce_dataclass(PowerLawPriorConfig, value)

    @property
    def fe(self) -> FeIIPriorConfig:
        """Alias for the Fe II prior section."""
        return self.feii

    @fe.setter
    def fe(self, value: FeIIPriorConfig | Mapping[str, Any]) -> None:
        """Set the Fe II prior section through the shorter alias.

        Parameters
        ----------
        value : FeIIPriorConfig or mapping
            Replacement Fe II prior section.
        """
        self.feii = _coerce_dataclass(FeIIPriorConfig, value)

    def to_mapping(self) -> dict[str, Any]:
        """Return flat model-site keys with distributions kept as objects."""
        out: dict[str, Any] = dict(self._model_priors)
        out.update(self.continuum.to_mapping())
        out.update(self.host.to_mapping())
        out.update(self.lines.to_mapping())
        out.update(self.feii.to_mapping())
        out.update(self.psf.to_mapping())
        if self.student_t_df is not None:
            out["student_t_df"] = float(self.student_t_df)
        return out


@dataclass
class FitConfig:
    """Top-level configuration bundle for one JAXQSOFit spectral fit.

    Parameters
    ----------
    observation : Observation or mapping
        Source metadata such as redshift, object identifier, sky coordinates,
        and Milky Way dereddening behavior.
    spectroscopy : SpectroscopyData or mapping
        Observed spectrum, uncertainties, masks, and optional resolution
        metadata.
    psf_photometry : PSFPhotometryData or mapping, optional
        Optional PSF-aperture magnitudes used as an extra spectral
        recalibration constraint.
    preprocessing : PreprocessingConfig or mapping, optional
        Wavelength trimming, manual masks, and Ly-alpha forest masking options.
    continuum : ContinuumConfig or mapping, optional
        Switches for the power law, Fe II, Balmer continuum, reddening,
        polynomial tilt, and convolution method.
    agn : AGNConfig or mapping, optional
        High-level AGN type preset.
    bal : BALConfig or mapping, optional
        Built-in broad absorption line component settings.
    host : HostConfig or mapping, optional
        Host-galaxy decomposition settings and SSP grid choices.
    lines : LineConfig or mapping, optional
        Emission-line switches and optional custom components.
    inference : InferenceConfig or mapping, optional
        Optax and NUTS controls, including warmup, samples, dense mass, and
        maximum tree depth.
    output : OutputConfig or mapping, optional
        Plotting and persistence behavior.
    prior_config : PriorConfig or mapping, optional
        Semantic or low-level priors consumed by the NumPyro model.
    """

    observation: Observation
    spectroscopy: SpectroscopyData
    psf_photometry: PSFPhotometryData | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    continuum: ContinuumConfig = field(default_factory=ContinuumConfig)
    agn: AGNConfig = field(default_factory=AGNConfig)
    bal: BALConfig = field(default_factory=BALConfig)
    host: HostConfig = field(default_factory=HostConfig)
    lines: LineConfig = field(default_factory=LineConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prior_config: PriorConfig | None = None

    def __post_init__(self) -> None:
        """Coerce mapping-style nested configs into dataclass objects."""
        if not isinstance(self.observation, Observation):
            self.observation = _coerce_dataclass(Observation, self.observation)
        if not isinstance(self.spectroscopy, SpectroscopyData):
            self.spectroscopy = _coerce_dataclass(SpectroscopyData, self.spectroscopy)
        if self.psf_photometry is not None and not isinstance(self.psf_photometry, PSFPhotometryData):
            self.psf_photometry = _coerce_dataclass(PSFPhotometryData, self.psf_photometry)
        if not isinstance(self.preprocessing, PreprocessingConfig):
            self.preprocessing = _coerce_dataclass(PreprocessingConfig, self.preprocessing)
        if not isinstance(self.continuum, ContinuumConfig):
            self.continuum = _coerce_dataclass(ContinuumConfig, self.continuum)
        if not isinstance(self.agn, AGNConfig):
            self.agn = _coerce_dataclass(AGNConfig, self.agn)
        if not isinstance(self.bal, BALConfig):
            self.bal = _coerce_dataclass(BALConfig, self.bal)
        if not isinstance(self.host, HostConfig):
            self.host = _coerce_dataclass(HostConfig, self.host)
        if not isinstance(self.lines, LineConfig):
            self.lines = _coerce_dataclass(LineConfig, self.lines)
        if not isinstance(self.inference, InferenceConfig):
            self.inference = _coerce_dataclass(InferenceConfig, self.inference)
        if not isinstance(self.output, OutputConfig):
            self.output = _coerce_dataclass(OutputConfig, self.output)
        if self.prior_config is not None:
            self.prior_config = _coerce_prior_config(self.prior_config)

    def apply_agn_type_defaults(self) -> None:
        """Apply coherent component defaults for ``agn.agn_type``.

        Type 1 is the broad-line AGN preset. Type 2 is the narrow-line,
        host-dominated preset. This mutates explicit component switches, so call
        it before any manual overrides that should win over the preset.
        """

        agn_type = int(self.agn.agn_type)
        if agn_type == 1:
            self.lines.enabled = True
            self.lines.use_broad_lines = True
            self.lines.use_narrow_lines = True
            self.continuum.fit_feii = True
            self.continuum.fit_balmer_continuum = True
            self.host.enabled = True
        elif agn_type == 2:
            self.lines.enabled = True
            self.lines.use_broad_lines = False
            self.lines.use_narrow_lines = True
            self.continuum.fit_feii = False
            self.continuum.fit_balmer_continuum = False
            self.host.enabled = True
        else:
            raise ValueError("agn.agn_type must be 1 or 2.")

    def set_agn_type(self, agn_type: int) -> None:
        """Set ``agn.agn_type`` and apply the corresponding component defaults.

        Parameters
        ----------
        agn_type : {1, 2}
            AGN spectral-type preset. Type 1 enables broad-line AGN defaults;
            type 2 enables narrow-line, host-dominated defaults.
        """

        self.agn = AGNConfig(agn_type=int(agn_type))
        self.apply_agn_type_defaults()

    def validate(self) -> None:
        """Validate required nested data payloads."""
        if not np.isfinite(float(self.observation.redshift)) or float(self.observation.redshift) < 0.0:
            raise ValueError("observation.redshift must be finite and non-negative.")
        self.spectroscopy.validate()
        if self.psf_photometry is not None:
            self.psf_photometry.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass tree into a plain dictionary."""
        return serialize_config(self)


def _coerce_dataclass(cls, value: Any):
    """Convert an existing instance or mapping into the requested dataclass.

    Parameters
    ----------
    value : object
        value value.
    """
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown {cls.__name__} field(s): {names}")
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in value:
                kwargs[field_name] = value[field_name]
        return cls(**kwargs)
    raise TypeError(f"Cannot coerce {type(value)!r} to {cls.__name__}")


def _coerce_prior_config(value: Any) -> PriorConfig:
    """Coerce structured prior mappings into :class:`PriorConfig`.

    Parameters
    ----------
    value : object
        value value.
    """
    if isinstance(value, PriorConfig):
        return value
    if value is None:
        return PriorConfig()
    if not isinstance(value, Mapping):
        return _coerce_dataclass(PriorConfig, value)
    data = dict(value)
    nested_keys = {"continuum", "host", "lines", "feii", "psf", "student_t_df", "_model_priors"}
    if any(key in data for key in nested_keys):
        cfg = PriorConfig(
            continuum=_coerce_dataclass(ContinuumPriorConfig, data.get("continuum", {})),
            host=_coerce_dataclass(HostPriorConfig, data.get("host", {})),
            lines=_coerce_dataclass(LinePriorConfig, data.get("lines", {})),
            feii=_coerce_dataclass(FeIIPriorConfig, data.get("feii", {})),
            psf=_coerce_dataclass(PSFPriorConfig, data.get("psf", {})),
            student_t_df=data.get("student_t_df"),
        )
        cfg._model_priors = dict(data.get("_model_priors", {}))
        return cfg
    raise ValueError("prior_config mappings must use structured PriorConfig sections.")


def fit_config_from_mapping(data: Mapping[str, Any]) -> FitConfig:
    """Build a validated FitConfig from a nested mapping.

    Parameters
    ----------
    data : mapping
        Nested configuration dictionary, typically loaded from JSON/YAML or a
        serialized ``FitConfig``.
    """

    allowed = set(FitConfig.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"Unknown FitConfig field(s): {names}")

    psf_raw = data.get("psf_photometry")
    psf_obj = None if psf_raw is None else _coerce_dataclass(PSFPhotometryData, psf_raw)
    cfg = FitConfig(
        observation=_coerce_dataclass(Observation, data.get("observation", {})),
        spectroscopy=_coerce_dataclass(SpectroscopyData, data["spectroscopy"]),
        psf_photometry=psf_obj,
        preprocessing=_coerce_dataclass(PreprocessingConfig, data.get("preprocessing", {})),
        continuum=_coerce_dataclass(ContinuumConfig, data.get("continuum", {})),
        agn=_coerce_dataclass(AGNConfig, data.get("agn", {})),
        bal=_coerce_dataclass(BALConfig, data.get("bal", {})),
        host=_coerce_dataclass(HostConfig, data.get("host", {})),
        lines=_coerce_dataclass(LineConfig, data.get("lines", {})),
        inference=_coerce_dataclass(InferenceConfig, data.get("inference", {})),
        output=_coerce_dataclass(OutputConfig, data.get("output", {})),
        prior_config=None if data.get("prior_config") is None else _coerce_prior_config(data.get("prior_config", {})),
    )
    cfg.validate()
    return cfg


def serialize_config(value: Any) -> Any:
    """Convert config-like objects into JSON-serializable Python values.


    Parameters
    ----------
    value : object
        Dataclass, mapping, sequence, NumPy array, NumPyro distribution, or
        scalar value to convert into JSON-compatible containers.
    """

    prior = _numpyro_distribution_to_mapping(value)
    if prior is not None:
        return serialize_config(prior)
    if is_dataclass(value):
        return {k: serialize_config(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: serialize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
