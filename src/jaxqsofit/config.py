from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np


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
    """Observed spectral measurements on an observed-frame wavelength grid."""

    wave_obs: Sequence[float]
    fluxes: Sequence[float]
    errors: Sequence[float] | float | None = None
    wavelength_dispersion: Sequence[float] | None = None
    resolving_power: float | None = None
    mask: Sequence[bool] | None = None

    def validate(self) -> None:
        """Validate spectroscopy payload array lengths."""
        n = len(self.wave_obs)
        if len(self.fluxes) != n:
            raise ValueError("Spectroscopy fluxes must have the same length as wave_obs.")
        if self.errors is not None and not np.isscalar(self.errors) and len(self.errors) != n:
            raise ValueError("Spectroscopy errors must be scalar, None, or match wave_obs length.")
        if self.wavelength_dispersion is not None and len(self.wavelength_dispersion) != n:
            raise ValueError("wavelength_dispersion must match wave_obs length.")
        if self.mask is not None and len(self.mask) != n:
            raise ValueError("spectroscopy mask must match wave_obs length.")


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
    """Continuum and spectral component switches."""

    fit_power_law: bool = True
    fit_feii: bool = True
    fit_balmer_continuum: bool = False
    fit_polynomial_tilt: bool = True
    fit_reddening: bool = True
    polynomial_order: int = 2


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
    """Emission-line model configuration."""

    enabled: bool = True
    custom_components: Sequence[Any] | None = None
    custom_line_components: Sequence[Any] | None = None


@dataclass
class InferenceConfig:
    """Inference defaults for Optax and NUTS."""

    method: str = "optax+nuts"
    map_steps: int = 600
    learning_rate: float = 1.0e-2
    num_warmup: int = 50
    num_samples: int = 50
    num_chains: int = 1
    target_accept_prob: float = 0.9
    plot_init: bool = False


def _scalar_or_list(value: Any) -> Any:
    """Convert scalar array-like distribution parameters into plain Python values."""
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    return arr.tolist()


def _numpyro_distribution_to_mapping(value: Any) -> dict[str, Any] | None:
    """Convert supported NumPyro distributions into the model prior schema."""
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


def _prior_to_mapping(value: Any) -> Any:
    """Convert public prior specs to low-level mappings."""
    if isinstance(value, Mapping):
        return dict(value)
    prior = _numpyro_distribution_to_mapping(value)
    if prior is not None:
        return prior
    raise TypeError("Prior fields must be mappings or supported numpyro.distributions objects.")


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
    """Semantic emission-line prior options."""

    table: Sequence[Mapping[str, Any]] | None = None
    dmu_scale_mult: float | None = None
    sig_scale_mult: float | None = None
    amp_scale_mult: float | None = None

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
        return out


@dataclass
class FeIIPriorConfig:
    """Semantic Fe II prior options."""

    uv_norm: Any | None = None
    op_over_uv: Any | None = None
    uv_fwhm: Any | None = None
    optical_fwhm: Any | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic Fe II prior settings into model-site keys."""
        out: dict[str, Any] = {}
        if self.uv_norm is not None:
            out["log_Fe_uv_norm"] = _prior_to_mapping(self.uv_norm)
        if self.op_over_uv is not None:
            out["log_Fe_op_over_uv"] = _prior_to_mapping(self.op_over_uv)
        if self.uv_fwhm is not None:
            out["log_Fe_uv_FWHM"] = _prior_to_mapping(self.uv_fwhm)
        if self.optical_fwhm is not None:
            out["log_Fe_op_FWHM"] = _prior_to_mapping(self.optical_fwhm)
        return out


@dataclass
class PSFPriorConfig:
    """Semantic PSF recalibration prior options."""

    def to_mapping(self) -> dict[str, Any]:
        """Return PSF recalibration prior settings."""
        return {}


@dataclass
class PriorConfig:
    """Object-oriented prior configuration."""

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
        """Build a PriorConfig from the low-level default-prior payload."""
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
        include_elg_narrow_lines: bool = False,
        include_high_ionization_lines: bool = False,
        pl_pivot: float | None = None,
    ) -> "PriorConfig":
        """Build default priors from an observed spectrum flux scale.

        ``build_default_prior_config`` expects rest-frame flux density. This
        constructor accepts observed-frame flux and redshift, applying the
        standard ``flux_rest = flux * (1 + redshift)`` conversion when a
        redshift is provided.
        """
        from .defaults import build_default_prior_config

        flux_arr = np.asarray(flux, dtype=float)
        flux_for_priors = flux_arr * (1.0 + float(redshift)) if redshift is not None else flux_arr
        return build_default_prior_config(
            flux_for_priors,
            line_config=None if line_config is None else dict(line_config),
            include_elg_narrow_lines=include_elg_narrow_lines,
            include_high_ionization_lines=include_high_ionization_lines,
            pl_pivot=pl_pivot,
        )

    @property
    def powerlaw(self) -> PowerLawPriorConfig:
        """Semantic power-law prior section."""
        return self.continuum.powerlaw

    @powerlaw.setter
    def powerlaw(self, value: PowerLawPriorConfig | Mapping[str, Any]) -> None:
        """Set the semantic power-law prior section."""
        self.continuum.powerlaw = _coerce_dataclass(PowerLawPriorConfig, value)

    @property
    def fe(self) -> FeIIPriorConfig:
        """Alias for the Fe II prior section."""
        return self.feii

    @fe.setter
    def fe(self, value: FeIIPriorConfig | Mapping[str, Any]) -> None:
        """Set the Fe II prior section through the shorter alias."""
        self.feii = _coerce_dataclass(FeIIPriorConfig, value)

    def to_mapping(self) -> dict[str, Any]:
        """Return the flat prior mapping consumed by low-level model code."""
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
    """Top-level configuration bundle for one JAXQSOFit spectral fit."""

    observation: Observation
    spectroscopy: SpectroscopyData
    psf_photometry: PSFPhotometryData | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    continuum: ContinuumConfig = field(default_factory=ContinuumConfig)
    bal: BALConfig = field(default_factory=BALConfig)
    host: HostConfig = field(default_factory=HostConfig)
    lines: LineConfig = field(default_factory=LineConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prior_config: PriorConfig | None = None

    def __post_init__(self) -> None:
        """Coerce mapping-style nested configs into dataclass objects."""
        if not isinstance(self.bal, BALConfig):
            self.bal = _coerce_dataclass(BALConfig, self.bal)
        if self.prior_config is not None:
            self.prior_config = _coerce_prior_config(self.prior_config)

    def validate(self) -> None:
        """Validate required nested data payloads."""
        self.spectroscopy.validate()
        if self.psf_photometry is not None:
            self.psf_photometry.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass tree into a plain dictionary."""
        return serialize_config(self)


def _coerce_dataclass(cls, value: Any):
    """Convert an existing instance or mapping into the requested dataclass."""
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in value:
                kwargs[field_name] = value[field_name]
        return cls(**kwargs)
    raise TypeError(f"Cannot coerce {type(value)!r} to {cls.__name__}")


def _coerce_prior_config(value: Any) -> PriorConfig:
    """Coerce structured prior mappings into :class:`PriorConfig`."""
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
    """Build a validated FitConfig from a nested mapping."""

    psf_raw = data.get("psf_photometry")
    psf_obj = None if psf_raw is None else _coerce_dataclass(PSFPhotometryData, psf_raw)
    cfg = FitConfig(
        observation=_coerce_dataclass(Observation, data.get("observation", {})),
        spectroscopy=_coerce_dataclass(SpectroscopyData, data["spectroscopy"]),
        psf_photometry=psf_obj,
        preprocessing=_coerce_dataclass(PreprocessingConfig, data.get("preprocessing", {})),
        continuum=_coerce_dataclass(ContinuumConfig, data.get("continuum", {})),
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
    """Convert config-like objects into JSON-serializable Python values."""

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
