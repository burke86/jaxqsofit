"""Plotting helpers for :class:`jaxqsofit.JAXQSOFit`.

The public functions in this module take a fitted ``JAXQSOFit`` instance as
first argument. ``JAXQSOFit`` keeps method wrappers around them so notebook code
can continue to call ``fitter.plot_spectrum()`` and related methods.
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    _filter_wave_to_angstrom_array,
    _filter_wave_to_angstrom_scalar,
    _get_sdss_filters,
)

__all__ = [
    "posterior_series",
    "filter_half_width_angstrom",
    "plot_filter_metadata",
    "style_axis",
    "synthetic_photometry_for_plot",
    "observed_photometry_for_plot",
    "plot_trace",
    "plot_corner",
    "plot_mcmc_diagnostics",
    "plot_initialization",
    "plot_spectrum",
    "plot_fig",
]


def _plot_line_component_as_broad(label):
    """Return True for line profiles drawn with broad-component styling."""
    name = str(label).lower()
    base = name.rsplit('_', 1)[0] if name.rsplit('_', 1)[-1].isdigit() else name
    return (
        base.endswith('_br')
        or ('_br' in base)
        or base.endswith('w')
    )


def posterior_series(fitter, param_names=None, max_vector_elems=2):
    """Flatten posterior samples into labeled 1D series for diagnostics.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing ``numpyro_samples``.
    param_names : list[str] | str | None, optional
        Parameter selector. Use ``'all'`` for all posterior keys.
    max_vector_elems : int or None, optional
        Maximum number of vector elements to expand per key.
    """
    if not hasattr(fitter, 'numpyro_samples') or fitter.numpyro_samples is None:
        return []

    samples = fitter.numpyro_samples
    if param_names == 'all':
        param_names = sorted(samples.keys())
    elif param_names is None:
        param_names = [
            'cont_norm', 'host_amp', 'log_frac_host', 'PL_norm', 'PL_slope',
            'Fe_uv_norm', 'log_Fe_op_over_uv',
            'Fe_FWHM', 'Fe_shift',
            'Balmer_norm', 'Balmer_Tau', 'Balmer_vel',
            'gal_v_kms', 'log_gal_sigma_kms', 'gal_sigma_kms',
            'frac_jitter', 'add_jitter',
        ]

    out = []
    for name in param_names:
        if name not in samples:
            continue
        arr = np.asarray(samples[name])
        if arr.ndim == 1:
            out.append((name, arr))
        elif arr.ndim >= 2:
            arr2 = arr.reshape(arr.shape[0], -1)
            nflat = arr2.shape[1]
            if max_vector_elems is None or int(max_vector_elems) < 0:
                ncomp = nflat
            else:
                ncomp = min(nflat, int(max_vector_elems))
            for i in range(ncomp):
                out.append((f'{name}[{i}]', arr2[:, i]))
    return out


def filter_half_width_angstrom(filt):
    """Return an approximate half-width for a photometric filter.

    Parameters
    ----------
    filt : object
        Filter-like object with ``wave`` and ``transmission`` arrays.
    """
    filt_wave = _filter_wave_to_angstrom_array(filt.wave)
    filt_trans = np.asarray(filt.transmission, dtype=float)
    support = filt_wave[filt_trans > 0.01 * np.nanmax(filt_trans)]
    if support.size >= 2:
        return 0.5 * float(support.max() - support.min())
    return 0.0

def plot_filter_metadata(fitter, bands):
    """Return plotting metadata arrays for the requested photometric bands.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object used for filter width helper methods.
    bands : sequence[str]
        Photometric band names.
    """
    filters = _get_sdss_filters()
    filt_list = [filters.get(str(band)) for band in bands]
    valid = np.asarray([filt is not None for filt in filt_list], dtype=bool)
    filt_list = [filt for filt in filt_list if filt is not None]
    if len(filt_list) == 0:
        return valid, np.array([], dtype=float), np.array([], dtype=float)
    eff_wave_obs = np.asarray(
        [_filter_wave_to_angstrom_scalar(filt.effective_wavelength) for filt in filt_list],
        dtype=float,
    )
    half_width_obs = np.asarray(
        [fitter._filter_half_width_angstrom(filt) for filt in filt_list],
        dtype=float,
    )
    return valid, eff_wave_obs, half_width_obs

def style_axis(ax, spine_lw=1.5):
    """Apply consistent axis styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    spine_lw : float, optional
        Line width for all spines.
    """
    ax.grid(False)
    ax.tick_params(
        which='both',
        direction='in',
        top=True,
        right=True,
        length=6,
        width=1.2,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)


def plot_initialization(
    *,
    wave,
    flux,
    model,
    host,
    powerlaw,
    line,
    redchi2,
    stage_name,
    model_label,
    show_plot=True,
):
    """Plot an Optax initialization model using the final spectrum-plot style.

    Parameters
    ----------
    wave : array-like
        Rest-frame wavelength grid.
    flux : array-like
        Observed flux density on ``wave``.
    model : array-like
        Initialization model flux density on ``wave``.
    host : array-like
        Host-galaxy component on ``wave``.
    powerlaw : array-like
        Power-law continuum component on ``wave``.
    line : array-like
        Emission-line component on ``wave``.
    redchi2 : float
        Reduced chi-square value for the initialization model. Kept for
        bookkeeping; it is not shown in the plot title.
    stage_name : str
        Title text for the initialization stage.
    model_label : str
        Legend label for the total initialization model.
    show_plot : bool, optional
        If True, call ``plt.show()`` before returning the figure.
    """
    from .mplstyle import use_style

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    model = np.asarray(model, dtype=float)
    host = np.asarray(host, dtype=float)
    powerlaw = np.asarray(powerlaw, dtype=float)
    line = np.asarray(line, dtype=float)
    finite_flux = flux[np.isfinite(flux)]

    with use_style():
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        fig, (ax, axr) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(15, 8),
            gridspec_kw={"height_ratios": [4.0, 1.2], "hspace": 0.05},
        )
        flux_ref = float(np.nanpercentile(np.abs(finite_flux), 95)) if finite_flux.size else 1.0
        comp_floor = max(1e-8, 0.005 * flux_ref)

        def _show_component(arr):
            """_show_component helper.

            Parameters
            ----------
            arr : object
                arr value.
            """
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            return arr.size > 0 and float(np.nanmax(np.abs(arr))) >= comp_floor

        ax.plot(wave, flux, color="k", lw=1.0, alpha=1.0, label="data", zorder=2, rasterized=True)
        ax.plot(wave, model, color="b", lw=1.8, label=model_label, zorder=6, rasterized=True)
        if _show_component(host):
            ax.plot(wave, host, color="purple", lw=1.8, label="host galaxy", zorder=4, rasterized=True)
        else:
            ax.plot(wave, host, color="purple", lw=1.8, zorder=4, rasterized=True)
        if _show_component(powerlaw):
            ax.plot(wave, powerlaw, color="orange", lw=1.5, label="power law", zorder=5, rasterized=True)
        else:
            ax.plot(wave, powerlaw, color="orange", lw=1.5, zorder=5, rasterized=True)
        if _show_component(line):
            ax.plot(wave, line, color="lightskyblue", lw=1.5, label="lines", zorder=5, rasterized=True)

        ax.set_xlim(float(np.nanmin(wave)), float(np.nanmax(wave)))
        y_arrays = [arr[np.isfinite(arr)] for arr in (flux, model, host, powerlaw, line)]
        y_arrays = [arr for arr in y_arrays if arr.size > 0]
        if y_arrays:
            yplot = np.concatenate(y_arrays)
            ymin = float(np.nanpercentile(yplot, 0.5))
            ymax = float(np.nanpercentile(yplot, 99.9))
            if np.isfinite(ymin) and np.isfinite(ymax):
                pad = 0.10 * (ymax - ymin) if ymax > ymin else max(0.1 * abs(ymax), 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)

        broad_line_markers = [
            ("Ly$\\alpha$", 1215.67),
            ("CIV", 1549.06),
            ("CIII]", 1908.73),
            ("MgII", 2798.75),
            ("H$\\beta$", 4862.68),
            ("H$\\alpha$", 6564.61),
        ]
        xlo, xhi = ax.get_xlim()
        y_top = ax.get_ylim()[1]
        text_x_offset = 0.01 * (xhi - xlo)
        for label, lam0 in broad_line_markers:
            if xlo <= lam0 <= xhi:
                ax.axvline(lam0, color="gray", ls="--", lw=0.8, alpha=0.35, zorder=1)
                ax.text(
                    lam0 - text_x_offset,
                    y_top * 0.985,
                    label,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=12,
                    color="dimgray",
                    alpha=0.9,
                    zorder=7,
                )

        ax.set_ylabel(r"$f_{\lambda}$ (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=20)
        ax.set_title(stage_name)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=12, ncol=2)

        resid = flux - model
        axr.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
        axr.plot(wave, resid, color="gray", lw=1.0, ls=":", alpha=0.9, rasterized=True)
        r = resid[np.isfinite(resid)]
        if r.size > 0:
            rlim = np.nanpercentile(np.abs(r), 99)
            if np.isfinite(rlim) and rlim > 0:
                axr.set_ylim(-1.15 * rlim, 1.15 * rlim)
        axr.set_ylabel("resid", fontsize=20)
        axr.set_xlabel("Rest Wavelength (Å)", fontsize=20)
        style_axis(ax)
        style_axis(axr)
        if show_plot:
            plt.show()
    return fig


def synthetic_photometry_for_plot(fitter, model_attr='model_total'):
    """Return rest-frame synthetic photometry points for plotting, if available.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing PSF photometry metadata and model arrays.
    model_attr : str, optional
        Name of the model array attribute to project through the filters.
    """
    if not bool(getattr(fitter, 'use_psf_phot', False)):
        return None
    bands = list(getattr(fitter, 'psf_bands', []) or [])
    mag_errs = np.asarray(getattr(fitter, 'psf_mag_errs', []), dtype=float)
    if len(bands) == 0 or mag_errs.size != len(bands):
        return None
    if not hasattr(fitter, 'wave') or len(getattr(fitter, 'wave', [])) == 0:
        return None

    model_rf = np.asarray(getattr(fitter, model_attr, []), dtype=float)
    if model_rf.size != len(fitter.wave) or not np.any(np.isfinite(model_rf)):
        return None

    wave_rf = np.asarray(fitter.wave, dtype=float)
    z = float(getattr(fitter, 'z', 0.0))
    wave_obs = wave_rf * (1.0 + z)
    flam_obs = model_rf / max(1.0 + z, 1e-8)
    filters = _get_sdss_filters()
    c_ang_s = 2.99792458e18

    x_rf, xerr_rf, y_rf, yerr_rf = [], [], [], []
    for band, mag_err in zip(bands, mag_errs):
        filt = filters.get(str(band))
        if filt is None or not np.isfinite(mag_err) or mag_err <= 0:
            continue

        filt_wave = _filter_wave_to_angstrom_array(filt.wave)
        filt_trans = np.asarray(filt.transmission, dtype=float)
        trans = np.interp(wave_obs, filt_wave, filt_trans, left=0.0, right=0.0)
        trans = np.clip(trans, 0.0, None)
        flam_obs_cgs = 1e-17 * flam_obs
        num = float(np.trapezoid(flam_obs_cgs * trans * wave_obs, wave_obs))
        den = float(np.trapezoid(trans * c_ang_s / np.clip(wave_obs, 1e-8, None), wave_obs))
        if not np.isfinite(num) or not np.isfinite(den) or num <= 0 or den <= 0:
            continue

        mag_syn = -2.5 * np.log10(num / den) - 48.60
        if not np.isfinite(mag_syn):
            continue

        eff_wave_obs = _filter_wave_to_angstrom_scalar(filt.effective_wavelength)
        support = filt_wave[filt_trans > 0.01 * np.nanmax(filt_trans)]
        if support.size >= 2:
            half_width_obs = 0.5 * float(support.max() - support.min())
        else:
            half_width_obs = 0.0

        fnu_obs = 10.0 ** (-0.4 * (mag_syn + 48.60))
        flam_band_obs_cgs = fnu_obs * c_ang_s / max(eff_wave_obs ** 2, 1e-30)
        flam_band_obs = flam_band_obs_cgs / 1e-17
        flam_band_rf = flam_band_obs * (1.0 + z)
        flam_band_rf_err = flam_band_rf * (0.4 * np.log(10.0) * float(mag_err))

        x_rf.append(eff_wave_obs / (1.0 + z))
        xerr_rf.append(half_width_obs / (1.0 + z))
        y_rf.append(flam_band_rf)
        yerr_rf.append(flam_band_rf_err)

    if len(x_rf) == 0:
        return None
    return (
        np.asarray(x_rf, dtype=float),
        np.asarray(xerr_rf, dtype=float),
        np.asarray(y_rf, dtype=float),
        np.asarray(yerr_rf, dtype=float),
    )

def observed_photometry_for_plot(fitter):
    """Return rest-frame observed PSF photometry points for plotting, if available.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing PSF photometry metadata.
    """
    if not bool(getattr(fitter, 'use_psf_phot', False)):
        return None
    bands = list(getattr(fitter, 'psf_bands', []) or [])
    mags = np.asarray(getattr(fitter, 'psf_mags', []), dtype=float)
    mag_errs = np.asarray(getattr(fitter, 'psf_mag_errs', []), dtype=float)
    if len(bands) == 0 or mags.size != len(bands) or mag_errs.size != len(bands):
        return None

    z = float(getattr(fitter, 'z', 0.0))
    c_ang_s = 2.99792458e18
    filter_valid, eff_wave_obs, half_width_obs = fitter._plot_filter_metadata(bands)
    phot_valid = np.isfinite(mags) & np.isfinite(mag_errs) & (mag_errs > 0)
    valid = filter_valid & phot_valid
    if not np.any(valid):
        return None

    mags = mags[valid]
    mag_errs = mag_errs[valid]
    eff_wave_obs = eff_wave_obs[phot_valid[filter_valid]]
    half_width_obs = half_width_obs[phot_valid[filter_valid]]

    fnu_obs = 10.0 ** (-0.4 * (mags + 48.60))
    flam_band_obs_cgs = fnu_obs * c_ang_s / np.clip(eff_wave_obs ** 2, 1e-30, None)
    flam_band_obs = flam_band_obs_cgs / 1e-17
    flam_band_rf = flam_band_obs * (1.0 + z)
    flam_band_rf_err = flam_band_rf * (0.4 * np.log(10.0) * mag_errs)

    return (
        eff_wave_obs / (1.0 + z),
        half_width_obs / (1.0 + z),
        flam_band_rf,
        flam_band_rf_err,
    )


def plot_trace(
    fitter,
    param_names=None,
    max_vector_elems=2,
    save_fig_path=None,
    save_fig_name=None,
    show_plot=False,
):
    """Plot posterior trace series for selected parameters.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing posterior samples.
    param_names : list[str] | str | None, optional
        Parameter selector. Use ``'all'`` to include all posterior keys.
    max_vector_elems : int or None, optional
        Maximum number of vector elements to expand per key.
    save_fig_path : str or None, optional
        Output directory when saving figures. If ``None``, uses ``fitter.output_path``
        (or ``'.'`` when unset).
    save_fig_name : str or None, optional
        Output filename override.
    show_plot : bool, optional
        If True, display the figure interactively with ``plt.show()``.
        Defaults to False so diagnostics are safe to call in headless terminals.
    """
    series = fitter._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
    if len(series) == 0:
        return None

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(1.2 * n, 3)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, vals) in zip(axes, series):
        ax.plot(np.arange(len(vals)), vals, color='black', lw=0.7)
        ax.set_ylabel(label, fontsize=9)
        fitter._style_axis(ax)
    axes[-1].set_xlabel('Sample', fontsize=10)
    fig.tight_layout()
    if show_plot:
        plt.show()
    if fitter.save_fig:
        out_name = f'{fitter.filename}_trace.pdf' if save_fig_name is None else save_fig_name
        save_dir = fitter.output_path if save_fig_path is None else save_fig_path
        if save_dir is None:
            save_dir = '.'
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, out_name)
        fig.savefig(out_file)
        print(f"Saved trace plot: {out_file}")
        plt.close(fig)
    fitter.trace_fig = fig
    return fig

def plot_corner(
    fitter,
    param_names=None,
    max_vector_elems=2,
    bins=30,
    max_points=5000,
    save_fig_path=None,
    save_fig_name=None,
    show_plot=False,
):
    """Plot posterior projections with ``corner.corner``.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing posterior samples.
    param_names : list[str] | str | None, optional
        Parameter selector. Use ``'all'`` to include all posterior keys.
    max_vector_elems : int or None, optional
        Maximum number of vector elements to expand per key.
    bins : int, optional
        Histogram bin count.
    max_points : int, optional
        Maximum posterior draws to plot (subsampled if needed).
    save_fig_path : str or None, optional
        Output directory when saving figures. If ``None``, uses ``fitter.output_path``
        (or ``'.'`` when unset).
    save_fig_name : str or None, optional
        Output filename override.
    show_plot : bool, optional
        If True, display the figure interactively with ``plt.show()``.
        Defaults to False so diagnostics are safe to call in headless terminals.
    """
    series = fitter._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
    if len(series) == 0:
        return None
    try:
        import corner
    except ImportError as exc:
        raise ImportError("plot_corner requires the 'corner' package to be installed.") from exc

    labels = [s[0] for s in series]
    data = np.column_stack([s[1] for s in series])
    if data.shape[0] > int(max_points):
        idx = np.linspace(0, data.shape[0] - 1, int(max_points), dtype=int)
        data = data[idx]

    fig = corner.corner(
        data,
        labels=labels,
        bins=bins,
        show_titles=True,
        color="black",
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        plot_contours=True,
        hist2d_kwargs={"bins": max(8, int(bins // 2)), "levels": [0.393, 0.865, 0.989]},
        fill_contours=False,
        no_fill_contours=True,
        smooth=0.8,
        smooth1d=0.8,
        max_n_ticks=3,
        quiet=True,
        labelpad=0.3,
        label_kwargs={"fontsize": 9},
        title_kwargs={"fontsize": 9},
        use_math_text=False,
        title_fmt=".3g",
    )
    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=8)
        fitter._style_axis(ax)
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.06, hspace=0.06)
    if show_plot:
        plt.show()
    if fitter.save_fig:
        out_name = f'{fitter.filename}_corner.pdf' if save_fig_name is None else save_fig_name
        save_dir = fitter.output_path if save_fig_path is None else save_fig_path
        if save_dir is None:
            save_dir = '.'
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, out_name)
        fig.savefig(out_file)
        print(f"Saved corner plot: {out_file}")
        plt.close(fig)
    fitter.corner_fig = fig
    return fig

def plot_mcmc_diagnostics(fitter, do_trace=True, do_corner=True,
                          param_names=None,
                          max_vector_elems=2,
                          corner_bins=30, corner_max_points=2000,
                          save_fig_path=None,
                          show_plot=False):
    """Plot trace and/or corner diagnostics in a single convenience call.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing posterior samples.
    do_trace : bool, optional
        If True, render trace plot.
    do_corner : bool, optional
        If True, render corner plot.
    param_names : list[str] | str | None
        Parameter selector shared by both trace and corner plots.
        Use ``'all'`` to include all posterior parameters.
    max_vector_elems : int or None, optional
        Maximum number of vector elements to expand per key.
    corner_bins : int, optional
        Histogram bin count for corner plot.
    corner_max_points : int, optional
        Maximum posterior draws to use in corner plot.
    save_fig_path : str or None, optional
        Output directory when saving figures. If ``None``, uses ``fitter.output_path``
        (or ``'.'`` when unset).
    show_plot : bool, optional
        If True, display each enabled diagnostics figure interactively with
        ``plt.show()``. Defaults to False so diagnostics are safe to call in
        headless terminals.
    """
    if do_trace:
        fitter.plot_trace(
            param_names=param_names,
            max_vector_elems=max_vector_elems,
            save_fig_path=save_fig_path,
            show_plot=show_plot,
        )
    if do_corner:
        fitter.plot_corner(
            param_names=param_names,
            max_vector_elems=max_vector_elems,
            bins=corner_bins,
            max_points=corner_max_points,
            save_fig_path=save_fig_path,
            show_plot=show_plot,
        )

def plot_spectrum(fitter, **kwargs):
    """Plot the fitted spectrum, model components, and residuals.

    This is the preferred public plotting method. It delegates to
    :meth:`plot_fig`, which remains available for compatibility with older
    notebooks.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing model and component arrays.
    **kwargs
        Keyword arguments forwarded to :func:`plot_fig`.
    """
    return fitter.plot_fig(**kwargs)

def plot_fig(fitter, save_fig_path=None, broad_fwhm=1200, plot_legend=True, ylims=None, plot_residual=True, show_title=True,
             plot_1sigma=True, sigma_alpha=0.12, show_plot=True, plot_psf_space=False, plot_intrinsic_powerlaw=False):
    """Plot data, model components, line decomposition, and residuals.

    Parameters
    ----------
    fitter : JAXQSOFit
        Fitted object containing model and component arrays.
    save_fig_path : str or None, optional
        Output directory when saving figures. If ``None``, uses ``fitter.output_path``
        (or ``'.'`` when unset).
    broad_fwhm : float, optional
        Reserved broad-line threshold (kept for compatibility).
    plot_legend : bool, optional
        If True, draw a legend.
    ylims : tuple[float, float] or None, optional
        Optional y-axis limits for the spectrum panel.
    plot_residual : bool, optional
        If True, draw residual panel below the spectrum.
    show_title : bool, optional
        Reserved title toggle kept for compatibility.
    plot_1sigma : bool, optional
        If True, draw 16-84% posterior bands for available components.
    sigma_alpha : float, optional
        Alpha transparency of 1-sigma bands.
    show_plot : bool, optional
        If True, call ``plt.show()``. If False, figure is created/saved without display.
    plot_psf_space : bool, optional
        If True, plot the PSF-space model/components. In this mode the
        residual panel is suppressed because the observed spectrum remains
        on the fiber scale.
    plot_intrinsic_powerlaw : bool, optional
        If True, overlay the intrinsic AGN power law before multiplicative
        polynomial tilt and additive edge corrections.
    """
    if bool(getattr(fitter, "_resumed_from_samples", False)):
        fitter._ensure_hydrated_from_samples()
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    psf_total_model = np.asarray(getattr(fitter, 'psf_model', []), dtype=float)
    use_psf_space = bool(plot_psf_space) and psf_total_model.size == len(getattr(fitter, 'wave', [])) and np.any(np.isfinite(psf_total_model))
    residual_enabled = bool(plot_residual) and not use_psf_space

    if residual_enabled:
        fig, (ax, ax_resid) = plt.subplots(
            2,
            1,
            figsize=(15, 8),
            sharex=True,
            gridspec_kw={'height_ratios': [4.0, 1.2], 'hspace': 0.05},
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax_resid = None

    flux_ref = float(np.nanpercentile(np.abs(fitter.flux[np.isfinite(fitter.flux)]), 95)) if np.any(np.isfinite(fitter.flux)) else 1.0
    comp_floor = max(1e-8, 0.005 * flux_ref)
    psf_scale = float(getattr(fitter, 'scale_psf', np.nan))
    psf_scale = psf_scale if np.isfinite(psf_scale) else 1.0

    total_model_plot = fitter.psf_model if use_psf_space else fitter.model_total
    host_plot = fitter.host_psf if use_psf_space else fitter.host
    pl_plot = psf_scale * fitter.f_pl_model if use_psf_space else fitter.f_pl_model
    pl_intrinsic = np.asarray(getattr(fitter, 'f_pl_model_intrinsic', []), dtype=float)
    pl_intrinsic_plot = psf_scale * pl_intrinsic if use_psf_space and pl_intrinsic.size == len(fitter.wave) else pl_intrinsic
    fe_total_model = psf_scale * (fitter.f_fe_mgii_model + fitter.f_fe_balmer_model) if use_psf_space else (fitter.f_fe_mgii_model + fitter.f_fe_balmer_model)
    bc_plot = psf_scale * fitter.f_bc_model if use_psf_space else fitter.f_bc_model
    line_plot = fitter.line_psf if use_psf_space else fitter.f_line_model
    total_model_label = 'total model (PSF)' if use_psf_space else 'total model'
    host_label = 'host galaxy (PSF)' if use_psf_space else 'host galaxy'
    powerlaw_label = 'power law (PSF)' if use_psf_space else 'power law'
    intrinsic_powerlaw_label = 'intrinsic power law (PSF)' if use_psf_space else 'intrinsic power law'
    fe_label = 'Fe II (PSF)' if use_psf_space else 'Fe II'
    bc_label = 'Balmer continuum (PSF)' if use_psf_space else 'Balmer continuum'
    line_label = 'total lines (PSF)' if use_psf_space else 'total lines'
    custom_component_colors = [
        'darkorange',
        'crimson',
        'slateblue',
        'seagreen',
        'saddlebrown',
        'deeppink',
    ]
    custom_components = list(getattr(fitter, 'custom_components', {}).items())
    custom_line_components = list(getattr(fitter, 'custom_line_components', {}).items())

    def _is_bal_component_name(name):
        """Return True for fitted BAL custom components.

        Parameters
        ----------
        name : object
            name value.
        """
        return str(name).startswith('bal_')

    def _custom_component_color(name, idx):
        """Return the plotting color for one custom component.

        Parameters
        ----------
        name : object
            name value.
        idx : object
            idx value.
        """
        if _is_bal_component_name(name):
            return 'red'
        return custom_component_colors[idx % len(custom_component_colors)]

    def _show_component(arr):
        """Return True when a component has finite amplitude worth plotting.

        Parameters
        ----------
        arr : object
            arr value.
        """
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return False
        return float(np.nanmax(np.abs(arr))) >= comp_floor

    def _finite_component_values(*arrays):
        """Collect finite values from one or more component arrays.


        Parameters
        ----------
        *arrays : tuple
            Additional positional arguments.
        """
        vals = []
        for arr in arrays:
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                vals.append(arr)
        return vals

    pred_bands = getattr(fitter, "pred_bands_psf" if use_psf_space else "pred_bands", None) or {}
    if plot_1sigma and pred_bands:
        band_colors = {
            'total_model': 'b',
            'host': 'purple',
            'PL': 'orange',
            'FeII': 'teal',
            'Balmer_cont': 'y',
            'lines': 'lightskyblue',
        }
        for key, color in band_colors.items():
            if key not in pred_bands:
                continue
            lo, hi = pred_bands[key]
            if len(lo) == len(fitter.wave) and _show_component(0.5 * (np.asarray(lo) + np.asarray(hi))):
                ax.fill_between(
                    fitter.wave,
                    lo,
                    hi,
                    color=color,
                    alpha=sigma_alpha,
                    linewidth=0,
                    zorder=0,
                    rasterized=True,
                )
        for idx, (name, model) in enumerate(custom_components + custom_line_components):
            if name not in pred_bands:
                continue
            lo, hi = pred_bands[name]
            if len(lo) != len(fitter.wave) or not _show_component(model):
                continue
            ax.fill_between(
                fitter.wave,
                lo,
                hi,
                color=_custom_component_color(name, idx),
                alpha=sigma_alpha,
                linewidth=0,
                zorder=0,
                rasterized=True,
            )
    if bool(plot_intrinsic_powerlaw) and pred_bands and not use_psf_space:
        if 'PL_intrinsic' in pred_bands:
            lo, hi = pred_bands['PL_intrinsic']
            if len(lo) == len(fitter.wave) and _show_component(0.5 * (np.asarray(lo) + np.asarray(hi))):
                ax.fill_between(
                    fitter.wave,
                    lo,
                    hi,
                    color='darkorange',
                    alpha=sigma_alpha,
                    linewidth=0,
                    zorder=0,
                    rasterized=True,
                )

    ax.plot(
        fitter.wave_prereduced,
        fitter.flux_prereduced,
        color='k' if not use_psf_space else 'gray',
        lw=1,
        label='data' if not use_psf_space else 'fiber data',
        zorder=2,
        alpha=1.0 if not use_psf_space else 0.6,
        rasterized=True,
    )
    ax.plot(fitter.wave, total_model_plot, color='b', lw=1.8, label=total_model_label, zorder=6, rasterized=True)
    if _show_component(host_plot):
        ax.plot(fitter.wave, host_plot, color='purple', lw=1.8, label=host_label, zorder=4, rasterized=True)
    else:
        ax.plot(fitter.wave, host_plot, color='purple', lw=1.8, zorder=4, rasterized=True)
    if _show_component(pl_plot):
        ax.plot(fitter.wave, pl_plot, color='orange', lw=1.5, label=powerlaw_label, zorder=5, rasterized=True)
    else:
        ax.plot(fitter.wave, pl_plot, color='orange', lw=1.5, zorder=5, rasterized=True)
    if bool(plot_intrinsic_powerlaw) and pl_intrinsic_plot.size == len(fitter.wave):
        if _show_component(pl_intrinsic_plot):
            ax.plot(
                fitter.wave,
                pl_intrinsic_plot,
                color='darkorange',
                lw=1.4,
                ls='--',
                label=intrinsic_powerlaw_label,
                zorder=5,
                rasterized=True,
            )
        else:
            ax.plot(fitter.wave, pl_intrinsic_plot, color='darkorange', lw=1.4, ls='--', zorder=5, rasterized=True)
    if _show_component(fe_total_model):
        ax.plot(fitter.wave, fe_total_model, color='teal', lw=1.2, label=fe_label, zorder=5, rasterized=True)
    else:
        ax.plot(fitter.wave, fe_total_model, color='teal', lw=1.2, zorder=5, rasterized=True)
    if _show_component(bc_plot):
        ax.plot(fitter.wave, bc_plot, color='y', lw=1.2, label=bc_label, zorder=5, rasterized=True)
    else:
        ax.plot(fitter.wave, bc_plot, color='y', lw=1.2, zorder=5, rasterized=True)
    custom_component_zorder = 4.8
    bal_legend_drawn = False
    for idx, (name, model) in enumerate(custom_components + custom_line_components):
        color = _custom_component_color(name, idx)
        is_bal_component = _is_bal_component_name(name)
        if is_bal_component:
            label = 'BAL'
        else:
            label = name.replace('_', ' ')
        if _show_component(model):
            draw_label = label
            if is_bal_component and bal_legend_drawn:
                draw_label = None
            ax.plot(
                fitter.wave,
                model,
                color=color,
                lw=1.4,
                label=draw_label,
                zorder=custom_component_zorder,
                rasterized=True,
            )
            if is_bal_component and draw_label is not None:
                bal_legend_drawn = True
        else:
            ax.plot(fitter.wave, model, color=color, lw=1.4, zorder=custom_component_zorder, rasterized=True)
    if len(line_plot) == len(fitter.wave):
        if _show_component(line_plot):
            ax.plot(
                fitter.wave,
                line_plot,
                color='lightskyblue',
                lw=1.5,
                label=line_label,
                zorder=5,
                rasterized=True,
            )
        else:
            ax.plot(
                fitter.wave,
                line_plot,
                color='lightskyblue',
                lw=1.5,
                label=line_label,
                zorder=5,
                rasterized=True,
            )

    obs_phot_points = fitter._observed_photometry_for_plot()
    if obs_phot_points is not None:
        phot_x, phot_xerr, phot_y, phot_yerr = obs_phot_points
        ax.errorbar(
            phot_x,
            phot_y,
            yerr=phot_yerr,
            xerr=phot_xerr,
            fmt='s',
            ms=7,
            color='black',
            mfc='white',
            mec='black',
            mew=0.8,
            elinewidth=1.0,
            capsize=3,
            alpha=0.95,
            zorder=8,
            label='PSF photometry',
        )

    syn_phot_points = fitter._synthetic_photometry_for_plot(
        model_attr='psf_model' if use_psf_space else 'model_total'
    )
    if syn_phot_points is not None:
        phot_x, phot_xerr, phot_y, phot_yerr = syn_phot_points
        ax.errorbar(
            phot_x,
            phot_y,
            yerr=phot_yerr,
            xerr=phot_xerr,
            fmt='o',
            ms=7,
            color='crimson',
            mec='white',
            mew=0.8,
            elinewidth=1.0,
            capsize=3,
            alpha=0.95,
            zorder=8,
            label='synthetic photometry',
        )

    # Plot posterior-median individual Gaussian line-component profiles.
    if (hasattr(fitter, 'line_component_amp_median')
            and hasattr(fitter, 'line_component_mu_median')
            and hasattr(fitter, 'line_component_sig_median')
            and hasattr(fitter, 'tied_line_meta')
            and len(fitter.line_component_amp_median) > 0):
        comp_labels = fitter.tied_line_meta.get('names', [''] * len(fitter.line_component_amp_median))
        profiles = np.asarray(
            getattr(
                fitter,
                'line_component_profiles_psf' if use_psf_space else 'line_component_profiles',
                np.empty((0, len(fitter.wave))),
            ),
            dtype=float,
        )
        if profiles.ndim == 2 and profiles.shape[1] == len(fitter.wave):
            drew_broad_label = False
            drew_narrow_label = False
            show_line_leg = _show_component(line_plot)
            n_profiles = min(profiles.shape[0], len(comp_labels))
            for i in range(n_profiles):
                prof = profiles[i]
                cname = str(comp_labels[i]).lower()
                is_broad = _plot_line_component_as_broad(cname)
                if is_broad:
                    lbl = 'broad components' if (show_line_leg and not drew_broad_label) else None
                    ax.plot(
                        fitter.wave,
                        prof,
                        color='red',
                        lw=0.7,
                        alpha=0.35,
                        zorder=3,
                        label=lbl,
                        rasterized=True,
                    )
                    drew_broad_label = True
                else:
                    lbl = 'narrow components' if (show_line_leg and not drew_narrow_label) else None
                    ax.plot(
                        fitter.wave,
                        prof,
                        color='green',
                        lw=0.7,
                        alpha=0.25,
                        zorder=3,
                        label=lbl,
                        rasterized=True,
                    )
                    drew_narrow_label = True

    ax.set_xlim(fitter.wave.min(), fitter.wave.max())
    if ylims is None:
        robust_data_ylim = []
        for data_arr in (fitter.flux_prereduced, fitter.flux):
            data_arr = np.asarray(data_arr, dtype=float)
            finite = data_arr[np.isfinite(data_arr)]
            if finite.size == 0:
                continue
            lo, hi = np.nanpercentile(finite, [0.5, 99.9])
            med = float(np.nanmedian(finite))
            mad = float(1.4826 * np.nanmedian(np.abs(finite - med)))
            if np.isfinite(mad) and mad > 0.0:
                lo = max(float(lo), med - 10.0 * mad)
                hi = min(float(hi), med + 10.0 * mad)
            else:
                lo = hi = med
            robust_data_ylim.append(np.asarray([lo, hi], dtype=float))

        component_ylim_arrays = [
            total_model_plot,
            host_plot,
            pl_plot,
            fe_total_model,
            bc_plot,
            line_plot,
            *[model for _, model in custom_components],
            *[model for _, model in custom_line_components],
        ]
        if plot_1sigma and pred_bands:
            for key in ('total_model', 'host', 'PL', 'FeII', 'Balmer_cont', 'lines'):
                if key not in pred_bands:
                    continue
                lo, hi = pred_bands[key]
                if len(lo) != len(fitter.wave) or len(hi) != len(fitter.wave):
                    continue
                mid = 0.5 * (np.asarray(lo, dtype=float) + np.asarray(hi, dtype=float))
                if _show_component(mid):
                    component_ylim_arrays.extend([lo, hi])
            for name, model in custom_components + custom_line_components:
                if name not in pred_bands or not _show_component(model):
                    continue
                lo, hi = pred_bands[name]
                if len(lo) == len(fitter.wave) and len(hi) == len(fitter.wave):
                    component_ylim_arrays.extend([lo, hi])
        yvals = _finite_component_values(
            *robust_data_ylim,
            *component_ylim_arrays,
        )
        if yvals:
            yplot = np.concatenate(yvals)
            ymin = float(np.nanmin(yplot))
            ymax = float(np.nanmax(yplot))
            if np.isfinite(ymin) and np.isfinite(ymax):
                if ymax > ymin:
                    pad = 0.10 * (ymax - ymin)
                else:
                    pad = max(0.1 * abs(ymax), 1.0)
                ax.set_ylim(float(ymin - pad), float(ymax + pad))
    else:
        ax.set_ylim(ylims[0], ylims[1])

    # Mark common broad-line AGN transitions on the spectrum panel.
    broad_line_markers = [
        ("Ly$\\alpha$", 1215.67),
        ("CIV", 1549.06),
        ("CIII]", 1908.73),
        ("MgII", 2798.75),
        ("H$\\beta$", 4862.68),
        ("H$\\alpha$", 6564.61),
    ]
    xlo, xhi = ax.get_xlim()
    y_top = ax.get_ylim()[1]
    text_x_offset = 0.01 * (xhi - xlo)
    for label, lam0 in broad_line_markers:
        if xlo <= lam0 <= xhi:
            ax.axvline(lam0, color="gray", ls="--", lw=0.8, alpha=0.35, zorder=1)
            ax.text(
                lam0 - text_x_offset,
                y_top * 0.985,
                label,
                rotation=90,
                va="top",
                ha="center",
                fontsize=12,
                color="dimgray",
                alpha=0.9,
                zorder=7,
            )

    if residual_enabled and len(total_model_plot) == len(fitter.wave) and ax_resid is not None:
        resid = fitter.flux - total_model_plot
        ax_resid.plot(
            fitter.wave,
            resid,
            color='gray',
            ls='dotted',
            lw=1.0,
            zorder=2,
            rasterized=True,
        )
        ax_resid.axhline(0.0, color='k', ls='--', lw=0.8, zorder=1)
        r = resid[np.isfinite(resid)]
        if r.size > 0:
            rlim = np.nanpercentile(np.abs(r), 99)
            if np.isfinite(rlim) and rlim > 0:
                ax_resid.set_ylim(-1.15 * rlim, 1.15 * rlim)
        ax_resid.set_ylabel('resid', fontsize=20)
        fitter._style_axis(ax_resid)

    if residual_enabled and ax_resid is not None:
        ax_resid.set_xlabel('Rest Wavelength (Å)', fontsize=20)
    else:
        ax.set_xlabel('Rest Wavelength (Å)', fontsize=20)
    ax.set_ylabel(r'$f_{\lambda}$ (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=20)
    fitter._style_axis(ax)
    if plot_legend:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=12, ncol=2)
    if show_plot:
        plt.show()
    if fitter.save_fig:
        save_dir = fitter.output_path if save_fig_path is None else save_fig_path
        if save_dir is None:
            save_dir = '.'
        os.makedirs(save_dir, exist_ok=True)
        out_file = os.path.join(save_dir, fitter.filename + '.pdf')
        fig.savefig(out_file, dpi=150)
        print(f"Saved spectrum plot: {out_file}")
        plt.close(fig)
    fitter.fig = fig
    return
