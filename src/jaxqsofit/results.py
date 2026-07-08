from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def median_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return posterior medians for every value in a sample-like mapping.

    Parameters
    ----------
    values : object
        values value.
    """
    out: dict[str, Any] = {}
    for key, value in (values or {}).items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = arr.item()
        elif arr.size == 0:
            out[key] = arr
        else:
            out[key] = np.nanmedian(arr, axis=0)
    return out


@dataclass
class _PosteriorState:
    """Internal mutable posterior state produced by a jaxqsofit run."""

    method: str | None = None
    samples: Mapping[str, Any] | None = None
    predictive: Mapping[str, Any] | None = None
    bands: Mapping[str, Any] | None = None
    path: Path | None = None
    figure: Any = None
    trace_figure: Any = None
    corner_figure: Any = None
    hydrated: bool = False
    resumed_from_samples: bool = False


@dataclass
class PredictionResult:
    """Dict-like posterior prediction or reconstruction result."""

    data: Mapping[str, Any]
    fitter: Any
    _median: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def median(self) -> dict[str, Any]:
        """Median predictive values over the leading posterior axis."""
        if self._median is None:
            self._median = median_mapping(self.data)
        return self._median

    def __getitem__(self, key: str) -> Any:
        """__getitem__ helper.

        Parameters
        ----------
        key : object
            key value.
        """
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """get helper.

        Parameters
        ----------
        key : object
            key value.
        default : object
            default value.
        """
        return self.data.get(key, default)


@dataclass
class FitResult:
    """High-level result object returned by ``jaxqsofit`` fits."""

    fitter: Any
    samples: Mapping[str, Any] | None
    median: Mapping[str, Any]
    method: str
    summary: Mapping[str, Any] | None = None
    path: Path | None = None
    figure: Any = None
    _state: _PosteriorState | None = field(default=None, repr=False, compare=False)

    def predict(self, **kwargs) -> PredictionResult:
        """Reconstruct posterior spectral components for this fit.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to
            :meth:`jaxqsofit.JAXQSOFit.reconstruct_posterior_spectrum`, such as
            wavelength-grid overrides or posterior draw selection options.
        """
        if self._state is not None:
            kwargs.setdefault("_state", self._state)
        return PredictionResult(self.fitter.reconstruct_posterior_spectrum(**kwargs), fitter=self.fitter)

    def save(self, path: str | Path | None = None, **kwargs) -> Path:
        """Save the result with the fitter's native persistence format.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination path for the posterior bundle. If omitted, the fitter's
            configured output path and object name are used.
        **kwargs : dict
            Additional persistence options forwarded to the fitter's native
            save method.
        """
        if self._state is not None:
            kwargs.setdefault("_state", self._state)
        self.path = Path(self.fitter.save(path, **kwargs))
        if self._state is not None:
            self._state.path = self.path
        return self.path

    def plot_corner(self, **kwargs):
        """Plot posterior samples with the fitter's corner-plot helper.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to :meth:`jaxqsofit.JAXQSOFit.plot_corner`,
            such as parameter selection, labels, output path, and display
            options.
        """
        return self.fitter.plot_corner(**kwargs)

    def plot_trace(self, **kwargs):
        """Plot posterior samples with the fitter's trace-plot helper.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to :meth:`jaxqsofit.JAXQSOFit.plot_trace`,
            such as parameter selection, output path, and display options.
        """
        return self.fitter.plot_trace(**kwargs)
