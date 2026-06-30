from __future__ import annotations

from collections.abc import MutableMapping
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


@dataclass
class ContinuumConfig:
    """Continuum and spectral component switches."""

    fit_power_law: bool = True
    fit_feii: bool = True
    fit_balmer_continuum: bool = False
    bal: BALConfig = field(default_factory=BALConfig)
    fit_polynomial_tilt: bool = True
    fit_reddening: bool = True
    polynomial_order: int = 2

    def __post_init__(self) -> None:
        """Coerce mapping-style BAL config into :class:`BALConfig`."""
        if not isinstance(self.bal, BALConfig):
            self.bal = _coerce_dataclass(BALConfig, self.bal)


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


@dataclass
class OutputConfig:
    """Plotting and persistence defaults."""

    output_path: str | None = None
    save_name: str | None = None
    save_result: bool = True
    plot_fig: bool = True
    save_fig: bool = True
    show_plot: bool = False


@dataclass
class ContinuumPriorConfig:
    """Semantic continuum-prior options."""

    power_law_pivot: float | None = None
    polynomial_pivot: float | None = None
    output_wavelengths: Sequence[float] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic continuum prior settings into model-site keys."""
        out = dict(self.overrides)
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
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic host prior settings into model-site keys."""
        out = dict(self.overrides)
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
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic emission-line prior settings into model-site keys."""
        out = dict(self.overrides)
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

    uv_fwhm: Mapping[str, Any] | None = None
    optical_fwhm: Mapping[str, Any] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Convert semantic Fe II prior settings into model-site keys."""
        out = dict(self.overrides)
        if self.uv_fwhm is not None:
            out["log_Fe_uv_FWHM"] = dict(self.uv_fwhm)
        if self.optical_fwhm is not None:
            out["log_Fe_op_FWHM"] = dict(self.optical_fwhm)
        return out


@dataclass
class PSFPriorConfig:
    """Semantic PSF recalibration prior options."""

    overrides: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Return low-level PSF recalibration prior overrides."""
        return dict(self.overrides)


@dataclass
class PriorConfig(MutableMapping[str, Any]):
    """Object-oriented prior configuration with dict-compatible overrides."""

    continuum: ContinuumPriorConfig = field(default_factory=ContinuumPriorConfig)
    host: HostPriorConfig = field(default_factory=HostPriorConfig)
    lines: LinePriorConfig = field(default_factory=LinePriorConfig)
    feii: FeIIPriorConfig = field(default_factory=FeIIPriorConfig)
    psf: PSFPriorConfig = field(default_factory=PSFPriorConfig)
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Return the flat prior mapping consumed by low-level model code."""
        out: dict[str, Any] = dict(self.overrides)
        out.update(self.continuum.to_mapping())
        out.update(self.host.to_mapping())
        out.update(self.lines.to_mapping())
        out.update(self.feii.to_mapping())
        out.update(self.psf.to_mapping())
        return out

    def __getitem__(self, key: str) -> Any:
        """Return one prior entry from the flat mapping view."""
        return self.to_mapping()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Store a low-level prior override by model-site key."""
        self.overrides[str(key)] = value

    def __delitem__(self, key: str) -> None:
        """Remove a low-level override entry."""
        if key in self.overrides:
            del self.overrides[key]
            return
        raise KeyError(key)

    def __iter__(self):
        """Iterate over keys in the flat mapping view."""
        return iter(self.to_mapping())

    def __len__(self) -> int:
        """Return the number of entries in the flat mapping view."""
        return len(self.to_mapping())

    def __contains__(self, key: object) -> bool:
        """Return True when a key exists in the flat mapping view."""
        return key in self.to_mapping()

    def get(self, key: str, default: Any = None) -> Any:
        """Return a prior entry or default from the flat mapping view."""
        return self.to_mapping().get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set and return a low-level override when the key is absent."""
        if key not in self:
            self[key] = default
        return self[key]


@dataclass
class FitConfig:
    """Top-level configuration bundle for one JAXQSOFit spectral fit."""

    observation: Observation
    spectroscopy: SpectroscopyData
    psf_photometry: PSFPhotometryData | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    continuum: ContinuumConfig = field(default_factory=ContinuumConfig)
    host: HostConfig = field(default_factory=HostConfig)
    lines: LineConfig = field(default_factory=LineConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prior_config: PriorConfig | None = None

    def __post_init__(self) -> None:
        """Coerce mapping-style prior configs into :class:`PriorConfig`."""
        if self.prior_config is not None:
            self.prior_config = _coerce_prior_config(self.prior_config)

    def validate(self) -> None:
        """Validate required nested data payloads."""
        self.spectroscopy.validate()
        if self.psf_photometry is not None:
            self.psf_photometry.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass tree into a plain dictionary."""
        return asdict(self)


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
    """Coerce flat and nested prior mappings into :class:`PriorConfig`."""
    if isinstance(value, PriorConfig):
        return value
    if value is None:
        return PriorConfig()
    if not isinstance(value, Mapping):
        return _coerce_dataclass(PriorConfig, value)
    data = dict(value)
    nested_keys = {"continuum", "host", "lines", "feii", "psf", "overrides"}
    if any(key in data for key in nested_keys):
        cfg = PriorConfig(
            continuum=_coerce_dataclass(ContinuumPriorConfig, data.get("continuum", {})),
            host=_coerce_dataclass(HostPriorConfig, data.get("host", {})),
            lines=_coerce_dataclass(LinePriorConfig, data.get("lines", {})),
            feii=_coerce_dataclass(FeIIPriorConfig, data.get("feii", {})),
            psf=_coerce_dataclass(PSFPriorConfig, data.get("psf", {})),
            overrides=dict(data.get("overrides", {})),
        )
        for key, item in data.items():
            if key not in nested_keys:
                cfg.overrides[key] = item
        return cfg
    return PriorConfig(overrides=data)


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

    if is_dataclass(value):
        return {k: serialize_config(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: serialize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
