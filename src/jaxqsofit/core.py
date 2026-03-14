from __future__ import annotations

import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

import jax
import jax.numpy as jnp
import optax
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import optax_to_numpyro

from .defaults import build_default_prior_config
from .model import (
    C_KMS,
    _extract_line_table_from_prior_config,
    _get_sfd_query,
    _normalize_template_flux,
    _np_to_jnp,
    build_fsps_template_grid,
    build_tied_line_meta_from_linelist,
    qso_fsps_joint_model,
    reconstruct_posterior_components,
    unred,
)

class QSOFit:
    def __init__(self, lam, flux, err=None, z=0.0, ra=-999, dec=-999, filename=None, output_path=None,
                 wdisp=None):
        """Initialize a spectral fitting object with observed-frame inputs.

        Parameters
        ----------
        lam : array-like
            Observed-frame wavelength array in Angstrom.
        flux : array-like
            Observed-frame flux density array.
        err : array-like or float or None, optional
            Per-pixel 1-sigma uncertainty. If ``None``, a default of ``1e-6``
            is used.
        z : float, optional
            Source redshift.
        ra, dec : float, optional
            Sky coordinates in degrees for Galactic dereddening.
        filename : str or None, optional
            Basename used for saving result tables/figures. If ``None``,
            a filename is auto-generated from ``ra`` and ``dec``.
        output_path : str or None, optional
            Output directory for saved artifacts.
        wdisp : array-like or None, optional
            Optional wavelength dispersion vector (stored only).

        Notes
        -----
        Providing per-pixel `err` is strongly recommended for robust inference.
        If `err` is not provided, a small default uncertainty (`1e-6`) is used;
        in that case, fitted intrinsic-scatter terms absorb much of the noise model.

        Use keyword arguments to avoid ambiguity (for example `z=...`).
        """
        self.lam_in = np.asarray(lam, dtype=np.float64)
        self.flux_in = np.asarray(flux, dtype=np.float64)
        if err is None:
            self.err_in = np.full_like(self.flux_in, 1e-6, dtype=np.float64)
        else:
            err_arr = np.asarray(err, dtype=np.float64)
            if err_arr.ndim == 0:
                self.err_in = np.full_like(self.flux_in, float(err_arr), dtype=np.float64)
            else:
                self.err_in = err_arr
        self.z = z
        self.wdisp = wdisp
        self.ra = ra
        self.dec = dec
        self.install_path = os.path.dirname(os.path.abspath(__file__))
        self.output_path = output_path
        self.filename = self._resolve_filename(filename=filename, ra=ra, dec=dec)

    @staticmethod
    def _resolve_filename(filename=None, ra=-999, dec=-999):
        """Resolve a filesystem-safe basename for outputs."""
        if filename is not None and str(filename).strip() != "":
            return str(filename).strip()
        try:
            ra_f = float(ra)
            dec_f = float(dec)
        except Exception:
            return "result"
        if np.isfinite(ra_f) and np.isfinite(dec_f) and (ra_f != -999) and (dec_f != -999):
            return f"ra{ra_f:.5f}_dec{dec_f:.5f}"
        return "result"

    def _posterior_bundle_path(self, save_name=None, save_path=None):
        """Return the default on-disk path for a saved posterior bundle."""
        out_name = f"{self.filename}_samples.pkl" if save_name is None else save_name
        out_dir = self.output_path if save_path is None else save_path
        if out_dir is None:
            out_dir = '.'
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, out_name)

    @staticmethod
    def _serialize_for_pickle(value):
        """Recursively convert model state into pickle-friendly Python objects."""
        if isinstance(value, dict):
            return {k: QSOFit._serialize_for_pickle(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(QSOFit._serialize_for_pickle(v) for v in value)
        if isinstance(value, (np.ndarray, np.generic)):
            return np.asarray(value)
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return np.asarray(value)
        return value

    def save_posterior_bundle(self, save_name=None, save_path=None):
        """Persist the fitted object state needed to reload posterior results."""
        state = {}
        excluded = {"numpyro_mcmc", "svi", "svi_state"}
        for key, value in self.__dict__.items():
            if key in excluded:
                continue
            state[key] = self._serialize_for_pickle(value)

        out_file = self._posterior_bundle_path(save_name=save_name, save_path=save_path)
        with open(out_file, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved posterior bundle: {out_file}")
        return out_file

    @classmethod
    def load_from_samples(
        cls,
        filename,
        output_path=None,
        save_name=None,
        plot_fig=True,
        plot_diagnostics=True,
        kwargs_plot=None,
        diagnostics_kwargs=None,
    ):
        """Load a saved posterior bundle, optionally recreate figures, and return a QSOFit object."""
        resolved_name = cls._resolve_filename(filename=filename)
        bundle_name = f"{resolved_name}_samples.pkl" if save_name is None else save_name
        bundle_dir = '.' if output_path is None else output_path
        bundle_path = os.path.join(bundle_dir, bundle_name)
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Posterior bundle not found: {bundle_path}")

        with open(bundle_path, "rb") as fh:
            state = pickle.load(fh)

        obj = cls(
            lam=state["lam_in"],
            flux=state["flux_in"],
            err=state.get("err_in"),
            z=state.get("z", 0.0),
            ra=state.get("ra", -999),
            dec=state.get("dec", -999),
            filename=state.get("filename", resolved_name),
            output_path=output_path if output_path is not None else state.get("output_path"),
            wdisp=state.get("wdisp"),
        )
        obj.__dict__.update(state)
        obj.install_path = os.path.dirname(os.path.abspath(__file__))

        if plot_fig:
            plot_kwargs = {} if kwargs_plot is None else dict(kwargs_plot)
            if "show_plot" not in plot_kwargs:
                plot_kwargs["show_plot"] = False
            obj.plot_fig(**plot_kwargs)
        if plot_diagnostics:
            diag_kwargs = {} if diagnostics_kwargs is None else dict(diagnostics_kwargs)
            obj.plot_mcmc_diagnostics(**diag_kwargs)
        return obj

    def fit(self, name=None, deredden=True,
            wave_range=None, wave_mask=None, save_fits_name=None,
            fit_lines=True, save_result=True, plot_fig=True, save_fig=True,
            show_plot=False,
            decompose_host=True,
            fit_pl=True,
            fit_fe=True,
            fit_bc=False,
            fit_poly=True,
            fit_poly_order=2,
            fit_poly_edge_flex=True,
            mask_lya_forest=True,
            fit_method='jaxopt+nuts',
            verbose=True,
            fsps_age_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
            fsps_logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
            prior_config=None,
            dsps_ssp_fn='tempdata.h5',
            nuts_warmup=50,
            nuts_samples=50,
            nuts_chains=1,
            nuts_target_accept=0.9,
            optax_steps=600,
            optax_lr=1e-2,
            kwargs_plot=None):
        """Run end-to-end preprocessing, fitting, and optional plotting/saving.

        Parameters
        ----------
        name : str or None, optional
            Optional override basename used for figure/output naming.
        deredden : bool, optional
            If True, apply Galactic dereddening with dustmaps SFD.
        wave_range : tuple[float, float] or None, optional
            Rest-frame wavelength range to keep.
        wave_mask : array-like or None, optional
            Rest-frame wavelength intervals to mask.
        save_fits_name : str or None, optional
            Basename for saved result table.
        fit_lines : bool, optional
            Enable/disable emission-line model.
        save_result : bool, optional
            If True, write summary results to CSV.
        plot_fig : bool, optional
            If True, render decomposition figure.
        save_fig : bool, optional
            If True, save rendered figures.
        show_plot : bool, optional
            If True, call ``plt.show()`` for the decomposition figure.
            Default is ``False`` to avoid interactive pop-ups in notebook/pipeline runs.
        decompose_host : bool, optional
            Enable/disable host SPS decomposition.
        fit_pl : bool, optional
            Enable/disable AGN power-law component.
        fit_fe : bool, optional
            Enable/disable FeII components.
        fit_bc : bool, optional
            Enable/disable Balmer continuum.
        fit_poly : bool, optional
            Enable/disable multiplicative polynomial tilt.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt. Uses
            coefficients ``poly_c1`` through ``poly_cN``.
        fit_poly_edge_flex : bool, optional
            If True, enable localized blue/red exponential edge correction
            terms within the polynomial model.
        mask_lya_forest : bool, optional
            If True, mask pixels with rest-frame wavelength below Ly-alpha
            (1215.67 Angstrom) before fitting.
        fit_method : {'nuts', 'optax', 'optax+nuts'}, optional
            Fitting backend.
        verbose : bool, optional
            Verbose optimizer output where applicable.
        fsps_age_grid : sequence of float, optional
            SSP age grid in Gyr.
        fsps_logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary. If None, defaults are auto-built.
        dsps_ssp_fn : str, optional
            Path to DSPS SSP HDF5 template file.
        nuts_warmup, nuts_samples : int, optional
            NUTS warmup and posterior sample counts.
        nuts_chains : int, optional
            Number of MCMC chains.
        nuts_target_accept : float, optional
            Target accept probability for NUTS.
        optax_steps : int, optional
            Number of SVI/Optax warm-start steps.
        optax_lr : float, optional
            Learning rate for Optax Adam.
        kwargs_plot : dict or None, optional
            Extra keyword arguments passed to :meth:`plot_fig`.
        """

        if kwargs_plot is None:
            kwargs_plot = {}
        if 'show_plot' not in kwargs_plot:
            kwargs_plot['show_plot'] = show_plot

        # Persist fit configuration so posterior reconstructions can be built on
        # alternate wavelength grids after fitting.
        self._fit_deredden = bool(deredden)
        self._fit_decompose_host = bool(decompose_host)
        self._fit_fit_lines = bool(fit_lines)
        self._fit_fit_pl = bool(fit_pl)
        self._fit_fit_fe = bool(fit_fe)
        self._fit_fit_bc = bool(fit_bc)
        self._fit_fit_poly = bool(fit_poly)
        self._fit_fit_poly_order = int(fit_poly_order)
        self._fit_fit_poly_edge_flex = bool(fit_poly_edge_flex)
        self._fit_mask_lya_forest = bool(mask_lya_forest)
        self._fit_method = str(fit_method)
        self._fit_fsps_age_grid = tuple(fsps_age_grid)
        self._fit_fsps_logzsol_grid = tuple(fsps_logzsol_grid)
        self._fit_prior_config = prior_config
        self._fit_dsps_ssp_fn = str(dsps_ssp_fn)

        self.wave_range = wave_range
        self.wave_mask = wave_mask
        self.linefit = fit_lines
        self.save_fig = save_fig
        self.verbose = verbose
        if name is not None and str(name).strip() != "":
            self.filename = str(name).strip()
        prior_config_input = prior_config
        prior_config = {} if prior_config is None else prior_config
        out_params = prior_config.get('out_params', {})
        self.L_conti_wave = np.asarray(out_params.get('cont_loc', []), dtype=float)

        data_dir = os.path.join(self.install_path, 'data')
        self.fe_uv = np.genfromtxt(os.path.join(data_dir, 'fe_uv.txt'))
        self.fe_op = np.genfromtxt(os.path.join(data_dir, 'fe_optical.txt'))

        self.fe_uv_wave = 10 ** self.fe_uv[:, 0]
        # Normalize non-negative template amplitudes to O(1) so Fe norms are in data-flux units.
        self.fe_uv_flux = _normalize_template_flux(np.maximum(self.fe_uv[:, 1], 0.0), target_amp=1.0)

        fe_op_wave = 10 ** self.fe_op[:, 0]
        fe_op_flux = _normalize_template_flux(np.maximum(self.fe_op[:, 1], 0.0), target_amp=1.0)
        m = (fe_op_wave > 3686.) & (fe_op_wave < 7484.)
        self.fe_op_wave = fe_op_wave[m]
        self.fe_op_flux = fe_op_flux[m]

        if save_fits_name is None:
            save_fits_name = self.filename

        ind_gooderror = np.where((self.err_in > 0) & np.isfinite(self.err_in) & (self.flux_in != 0) & np.isfinite(self.flux_in), True, False)
        self.err = self.err_in[ind_gooderror]
        self.flux = self.flux_in[ind_gooderror]
        self.lam = self.lam_in[ind_gooderror]

        if prior_config_input is None:
            prior_config = build_default_prior_config(self.flux)
        self._fit_prior_config = prior_config

        if wave_range is not None:
            self._wave_trim(self.lam, self.flux, self.err, self.z)
        if wave_mask is not None:
            self._wave_msk(self.lam, self.flux, self.err, self.z)
        if mask_lya_forest:
            self._mask_lya_forest(self.lam, self.flux, self.err, self.z)
        if deredden:
            self._de_redden(self.lam, self.flux, self.err, self.ra, self.dec)

        self._rest_frame(self.lam, self.flux, self.err, self.z)
        self._calculate_sn(self.wave, self.flux)
        self._orignial_spec(self.wave, self.flux, self.err)

        if fit_method == 'nuts':
            self.run_fsps_numpyro_fit(
                num_warmup=nuts_warmup,
                num_samples=nuts_samples,
                num_chains=nuts_chains,
                target_accept_prob=nuts_target_accept,
                age_grid_gyr=fsps_age_grid,
                logzsol_grid=fsps_logzsol_grid,
                prior_config=prior_config,
                dsps_ssp_fn=dsps_ssp_fn,
                use_lines=fit_lines,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=fit_fe,
                fit_bc=fit_bc,
                fit_poly=fit_poly,
                fit_poly_order=fit_poly_order,
                fit_poly_edge_flex=fit_poly_edge_flex,
            )
        elif fit_method == 'optax':
            self.run_fsps_optax_fit(
                num_steps=optax_steps,
                learning_rate=optax_lr,
                age_grid_gyr=fsps_age_grid,
                logzsol_grid=fsps_logzsol_grid,
                prior_config=prior_config,
                dsps_ssp_fn=dsps_ssp_fn,
                use_lines=fit_lines,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=fit_fe,
                fit_bc=fit_bc,
                fit_poly=fit_poly,
                fit_poly_order=fit_poly_order,
                fit_poly_edge_flex=fit_poly_edge_flex,
            )
        elif fit_method == 'optax+nuts':
            self.run_fsps_optax_nuts_fit(
                optax_steps=optax_steps,
                optax_learning_rate=optax_lr,
                num_warmup=nuts_warmup,
                num_samples=nuts_samples,
                num_chains=nuts_chains,
                target_accept_prob=nuts_target_accept,
                age_grid_gyr=fsps_age_grid,
                logzsol_grid=fsps_logzsol_grid,
                prior_config=prior_config,
                dsps_ssp_fn=dsps_ssp_fn,
                use_lines=fit_lines,
                decompose_host=decompose_host,
                fit_pl=fit_pl,
                fit_fe=fit_fe,
                fit_bc=fit_bc,
                fit_poly=fit_poly,
                fit_poly_order=fit_poly_order,
                fit_poly_edge_flex=fit_poly_edge_flex,
            )
        else:
            raise ValueError(f"Unknown fit_method='{fit_method}'. Use 'nuts', 'optax', or 'optax+nuts'.")

        if save_result:
            self.save_result(self.conti_result, self.conti_result_type, self.conti_result_name,
                             self.line_result, self.line_result_type, self.line_result_name,
                             save_fits_name)
            self.save_posterior_bundle()
        if plot_fig:
            self.plot_fig(**kwargs_plot)

    def run_fsps_numpyro_fit(self, num_warmup=500, num_samples=1000, num_chains=1,
                             target_accept_prob=0.9,
                             age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                             logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                             prior_config=None,
                             dsps_ssp_fn='tempdata.h5',
                             use_lines=True,
                             decompose_host=True,
                             fit_pl=True,
                             fit_fe=True,
                             fit_bc=True,
                             fit_poly=False,
                             fit_poly_order=2,
                             fit_poly_edge_flex=True,
                             init_values=None):
        """Fit the full model using NUTS MCMC and store posterior summaries.

        Parameters
        ----------
        num_warmup, num_samples : int, optional
            MCMC warmup and posterior sample counts.
        num_chains : int, optional
            Number of MCMC chains.
        target_accept_prob : float, optional
            Target acceptance probability for NUTS.
        age_grid_gyr : sequence of float, optional
            SSP age grid in Gyr.
        logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary for model blocks.
        dsps_ssp_fn : str, optional
            DSPS SSP template HDF5 path.
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_poly_edge_flex : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        init_values : dict or None, optional
            Optional initial values for ``init_to_value``.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        if prior_config is None:
            prior_config = build_default_prior_config(flux)
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None:
            raise ValueError(
                "fit_lines=True requires line priors/table in prior_config. "
                "Pass prior_config['line_priors'] (or prior_config['line']['table'])."
            )

        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)
        else:
            tied_line_meta = {
                'n_lines': 0,
                'n_vgroups': 0,
                'n_wgroups': 0,
                'n_fgroups': 0,
                'ln_lambda0': _np_to_jnp(np.array([], dtype=float)),
                'vgroup': np.array([], dtype=int),
                'wgroup': np.array([], dtype=int),
                'fgroup': np.array([], dtype=int),
                'flux_ratio': np.array([], dtype=float),
                'dmu_init_group': np.array([], dtype=float),
                'dmu_min_group': np.array([], dtype=float),
                'dmu_max_group': np.array([], dtype=float),
                'sig_init_group': np.array([], dtype=float),
                'sig_min_group': np.array([], dtype=float),
                'sig_max_group': np.array([], dtype=float),
                'amp_init_group': np.array([], dtype=float),
                'amp_min_group': np.array([], dtype=float),
                'amp_max_group': np.array([], dtype=float),
                'names': [],
                'compnames': [],
                'line_lambda': np.array([], dtype=float),
            }
        fsps_grid = build_fsps_template_grid(
            wave_out=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
        )
        self.tied_line_meta = tied_line_meta

        init_vals = {'gal_v_kms': 0.0, 'gal_sigma_kms': 150.0} if init_values is None else init_values
        init_strategy = init_to_value(values=init_vals)
        kernel = NUTS(qso_fsps_joint_model, init_strategy=init_strategy, target_accept_prob=target_accept_prob, dense_mass=True, max_tree_depth=8)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=True, jit_model_args=False)
        rng_key = jax.random.PRNGKey(0)
        mcmc.run(
            rng_key,
            wave=wave,
            flux=flux,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
        )
        samples = mcmc.get_samples()

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples=samples,
            return_sites=['f_pl_model', 'f_fe_mgii_model', 'f_fe_balmer_model', 'f_bc_model', 'f_poly_model',
                          'agn_model', 'gal_model', 'line_model', 'continuum_model', 'model',
                          'fsps_weights', 'line_amp_per_component', 'line_mu_per_component', 'line_sig_per_component'],
        )
        pred_out = pred(
            rng_key,
            wave=wave,
            flux=None,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
        )

        self.numpyro_mcmc = mcmc
        self._consume_posterior_outputs(
            samples=samples,
            pred_out=pred_out,
            fsps_grid=fsps_grid,
            tied_line_meta=tied_line_meta,
            use_lines=use_lines,
            decompose_host=decompose_host,
        )

    def run_fsps_optax_fit(self, num_steps=2000, learning_rate=1e-2,
                           age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                           logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                           prior_config=None,
                           dsps_ssp_fn='tempdata.h5',
                           use_lines=True,
                           decompose_host=True,
                           fit_pl=True,
                           fit_fe=True,
                           fit_bc=True,
                           fit_poly=False,
                           fit_poly_order=2,
                           fit_poly_edge_flex=True):
        """Fit a MAP approximation using staged SVI with an Optax optimizer.

        Parameters
        ----------
        num_steps : int, optional
            Total SVI steps across all stages.
        learning_rate : float, optional
            Adam learning rate.
        age_grid_gyr : sequence of float, optional
            SSP age grid in Gyr.
        logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary for model blocks.
        dsps_ssp_fn : str, optional
            DSPS SSP template HDF5 path.
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_poly_edge_flex : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        """
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        err = np.asarray(self.err, dtype=float)

        if prior_config is None:
            prior_config = build_default_prior_config(flux)
        conti_priors = prior_config.get('conti_priors', {})
        line_table = _extract_line_table_from_prior_config(prior_config)

        if use_lines and line_table is None:
            raise ValueError(
                "fit_lines=True requires line priors/table in prior_config. "
                "Pass prior_config['line_priors'] (or prior_config['line']['table'])."
            )

        if line_table is not None:
            tied_line_meta = build_tied_line_meta_from_linelist(line_table, wave)
        else:
            tied_line_meta = {
                'n_lines': 0,
                'n_vgroups': 0,
                'n_wgroups': 0,
                'n_fgroups': 0,
                'ln_lambda0': _np_to_jnp(np.array([], dtype=float)),
                'vgroup': np.array([], dtype=int),
                'wgroup': np.array([], dtype=int),
                'fgroup': np.array([], dtype=int),
                'flux_ratio': np.array([], dtype=float),
                'dmu_init_group': np.array([], dtype=float),
                'dmu_min_group': np.array([], dtype=float),
                'dmu_max_group': np.array([], dtype=float),
                'sig_init_group': np.array([], dtype=float),
                'sig_min_group': np.array([], dtype=float),
                'sig_max_group': np.array([], dtype=float),
                'amp_init_group': np.array([], dtype=float),
                'amp_min_group': np.array([], dtype=float),
                'amp_max_group': np.array([], dtype=float),
                'names': [],
                'compnames': [],
                'line_lambda': np.array([], dtype=float),
            }
        fsps_grid = build_fsps_template_grid(
            wave_out=wave,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            dsps_ssp_fn=dsps_ssp_fn,
        )
        self.tied_line_meta = tied_line_meta

        def _run_svi(guide, steps, use_lines_i, fit_pl_i, fit_fe_i, fit_bc_i, fit_poly_i, fit_poly_order_i, fit_poly_edge_flex_i, decompose_host_i):
            """Run an SVI stage and return optimizer state/results."""
            optimizer = optax_to_numpyro(optax.adam(learning_rate))
            svi = SVI(qso_fsps_joint_model, guide, optimizer, loss=Trace_ELBO())
            key = jax.random.PRNGKey(0)
            result = svi.run(
                key,
                int(steps),
                wave=wave,
                flux=flux,
                err=err,
                conti_priors=conti_priors,
                tied_line_meta=tied_line_meta,
                fsps_grid=fsps_grid,
                fe_uv_wave=self.fe_uv_wave,
                fe_uv_flux=self.fe_uv_flux,
                fe_op_wave=self.fe_op_wave,
                fe_op_flux=self.fe_op_flux,
                use_lines=use_lines_i,
                prior_config=prior_config,
                decompose_host=decompose_host_i,
                fit_pl=fit_pl_i,
                fit_fe=fit_fe_i,
                fit_bc=fit_bc_i,
                fit_poly=fit_poly_i,
                fit_poly_order=fit_poly_order_i,
                fit_poly_edge_flex=fit_poly_edge_flex_i,
                progress_bar=self.verbose,
            )
            return svi, result

        # Stage 1: warm start on simpler landscape (continuum/host only).
        n1 = max(100, int(num_steps // 3))
        guide1 = AutoDelta(
            qso_fsps_joint_model,
            init_loc_fn=init_to_value(values={'gal_v_kms': 0.0, 'gal_sigma_kms': 150.0}),
        )
        svi1, res1 = _run_svi(
            guide1,
            n1,
            use_lines_i=False,
            fit_pl_i=fit_pl,
            fit_fe_i=False,
            fit_bc_i=False,
            fit_poly_i=False,
            fit_poly_order_i=2,
            fit_poly_edge_flex_i=False,
            decompose_host_i=decompose_host,
        )
        map1 = guide1.median(res1.params)

        # Stage 2: full model initialized from stage-1 MAP for overlapping parameters.
        n2 = max(100, int(num_steps - n1))
        guide2 = AutoDelta(
            qso_fsps_joint_model,
            init_loc_fn=init_to_value(values=map1),
        )
        svi, res2 = _run_svi(
            guide2,
            n2,
            use_lines_i=use_lines,
            fit_pl_i=fit_pl,
            fit_fe_i=fit_fe,
            fit_bc_i=fit_bc,
            fit_poly_i=fit_poly,
            fit_poly_order_i=fit_poly_order,
            fit_poly_edge_flex_i=fit_poly_edge_flex,
            decompose_host_i=decompose_host,
        )

        svi_state = res2.state
        svi_params = res2.params
        losses = np.concatenate([np.asarray(res1.losses), np.asarray(res2.losses)])
        map_point = guide2.median(svi_params)
        samples = {k: np.asarray(v)[None, ...] for k, v in map_point.items()}
        rng_key = jax.random.PRNGKey(0)

        pred = Predictive(
            qso_fsps_joint_model,
            posterior_samples={k: jnp.asarray(v) for k, v in samples.items()},
            return_sites=['f_pl_model', 'f_fe_mgii_model', 'f_fe_balmer_model', 'f_bc_model', 'f_poly_model',
                          'agn_model', 'gal_model', 'line_model', 'continuum_model', 'model',
                          'fsps_weights', 'line_amp_per_component', 'line_mu_per_component', 'line_sig_per_component'],
        )
        pred_out = pred(
            rng_key,
            wave=wave,
            flux=None,
            err=err,
            conti_priors=conti_priors,
            tied_line_meta=tied_line_meta,
            fsps_grid=fsps_grid,
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            use_lines=use_lines,
            prior_config=prior_config,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
        )

        self.numpyro_mcmc = None
        self.svi = svi
        self.svi_state = svi_state
        self.svi_params = svi_params
        self.optax_losses = losses
        self.optax_map_point = map_point
        self._consume_posterior_outputs(
            samples=samples,
            pred_out=pred_out,
            fsps_grid=fsps_grid,
            tied_line_meta=tied_line_meta,
            use_lines=use_lines,
            decompose_host=decompose_host,
        )

    def run_fsps_optax_nuts_fit(self, optax_steps=2000, optax_learning_rate=1e-2,
                                num_warmup=500, num_samples=1000, num_chains=1,
                                target_accept_prob=0.9,
                                age_grid_gyr=(0.1, 0.3, 1.0, 3.0, 10.0),
                                logzsol_grid=(-1.0, -0.5, 0.0, 0.2),
                                prior_config=None,
                                dsps_ssp_fn='tempdata.h5',
                                use_lines=True,
                                decompose_host=True,
                                fit_pl=True,
                                fit_fe=True,
                                fit_bc=True,
                                fit_poly=False,
                                fit_poly_order=2,
                                fit_poly_edge_flex=True):
        """Warm-start with Optax MAP, then run NUTS as final inference.

        Parameters
        ----------
        optax_steps : int, optional
            Number of SVI/Optax warm-start steps.
        optax_learning_rate : float, optional
            Learning rate for SVI warm-start.
        num_warmup, num_samples : int, optional
            NUTS warmup and posterior sample counts.
        num_chains : int, optional
            Number of MCMC chains.
        target_accept_prob : float, optional
            Target acceptance probability for NUTS.
        age_grid_gyr : sequence of float, optional
            SSP age grid in Gyr.
        logzsol_grid : sequence of float, optional
            SSP metallicity grid in log(Z/Zsun).
        prior_config : dict or None, optional
            Prior/config dictionary for model blocks.
        dsps_ssp_fn : str, optional
            DSPS SSP template HDF5 path.
        use_lines, decompose_host, fit_pl, fit_fe, fit_bc, fit_poly, fit_poly_edge_flex : bool, optional
            Component toggles for model blocks.
        fit_poly_order : int, optional
            Polynomial order for the multiplicative continuum tilt.
        """
        self.run_fsps_optax_fit(
            num_steps=optax_steps,
            learning_rate=optax_learning_rate,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            prior_config=prior_config,
            dsps_ssp_fn=dsps_ssp_fn,
            use_lines=use_lines,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
        )
        init_values = getattr(self, 'optax_map_point', None)
        self.run_fsps_numpyro_fit(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            age_grid_gyr=age_grid_gyr,
            logzsol_grid=logzsol_grid,
            prior_config=prior_config,
            dsps_ssp_fn=dsps_ssp_fn,
            use_lines=use_lines,
            decompose_host=decompose_host,
            fit_pl=fit_pl,
            fit_fe=fit_fe,
            fit_bc=fit_bc,
            fit_poly=fit_poly,
            fit_poly_order=fit_poly_order,
            fit_poly_edge_flex=fit_poly_edge_flex,
            init_values=init_values,
        )

    def _consume_posterior_outputs(self, samples, pred_out, fsps_grid, tied_line_meta, use_lines, decompose_host):
        """Populate model components, uncertainty bands, and summary tables.

        Parameters
        ----------
        samples : dict
            Posterior samples keyed by parameter name.
        pred_out : dict
            Posterior predictive outputs from ``Predictive``.
        fsps_grid : FSPSTemplateGrid
            Host SSP template grid metadata.
        tied_line_meta : dict
            Emission-line grouping metadata.
        use_lines : bool
            Whether line model was enabled.
        decompose_host : bool
            Whether host model was enabled.
        """
        flux = np.asarray(self.flux, dtype=float)
        self.numpyro_samples = samples
        self.fsps_grid = fsps_grid
        self.pred_out = pred_out
        self._pred_host_draws = np.asarray(pred_out['gal_model'])
        self._pred_bc_draws = np.asarray(pred_out['f_bc_model'])
        self._pred_cont_draws = np.asarray(pred_out['continuum_model'])
        self._pred_total_draws = np.asarray(pred_out['model'])
        self._pred_line_draws = np.asarray(pred_out['line_model'])

        self.f_pl_model = np.median(np.asarray(pred_out['f_pl_model']), axis=0)
        self.f_fe_mgii_model = np.median(np.asarray(pred_out['f_fe_mgii_model']), axis=0)
        self.f_fe_balmer_model = np.median(np.asarray(pred_out['f_fe_balmer_model']), axis=0)
        self.f_bc_model = np.median(np.asarray(pred_out['f_bc_model']), axis=0)
        self.f_poly_model = np.median(np.asarray(pred_out['f_poly_model']), axis=0)
        self.qso = np.median(np.asarray(pred_out['agn_model']), axis=0)
        self.host = np.median(np.asarray(pred_out['gal_model']), axis=0)
        self.f_line_model = np.median(np.asarray(pred_out['line_model']), axis=0)
        self.f_conti_model = np.median(np.asarray(pred_out['continuum_model']), axis=0)
        self.model_total = np.median(np.asarray(pred_out['model']), axis=0)
        self.fsps_weights_median = np.median(np.asarray(pred_out['fsps_weights']), axis=0)
        self.line_flux = flux - self.f_conti_model
        self.decomposed = True

        def _band(x):
            """Compute 16th/84th percentile uncertainty band across samples."""
            a = np.asarray(x)
            return np.percentile(a, 16, axis=0), np.percentile(a, 84, axis=0)

        cont_plus_lines = np.asarray(pred_out['continuum_model']) + np.asarray(pred_out['line_model'])
        fe_total = np.asarray(pred_out['f_fe_mgii_model']) + np.asarray(pred_out['f_fe_balmer_model'])
        self.pred_bands = {
            'total_model': _band(pred_out['model']),
            'host': _band(pred_out['gal_model']),
            'PL': _band(pred_out['f_pl_model']),
            'FeII': _band(fe_total),
            'Balmer_cont': _band(pred_out['f_bc_model']),
            'continuum': _band(pred_out['continuum_model']),
            'lines': _band(pred_out['line_model']),
            'conti_plus_lines': _band(cont_plus_lines),
        }
        if self.verbose:
            print("max data        :", np.nanmax(self.flux))
            print("max total model :", np.nanmax(self.model_total))
            print("max PL          :", np.nanmax(self.f_pl_model))
            print("max host        :", np.nanmax(self.host))
            print("max FeII UV     :", np.nanmax(self.f_fe_mgii_model))
            print("max FeII opt    :", np.nanmax(self.f_fe_balmer_model))
            print("max Balmer cont :", np.nanmax(self.f_bc_model))
            print("max lines       :", np.nanmax(self.f_line_model))

        if decompose_host and 'gal_v_kms' in samples and 'gal_sigma_kms' in samples:
            gal_v = float(np.median(np.asarray(samples['gal_v_kms'])))
            gal_v_err = float(np.std(np.asarray(samples['gal_v_kms'])))
            gal_sig = float(np.median(np.asarray(samples['gal_sigma_kms'])))
            gal_sig_err = float(np.std(np.asarray(samples['gal_sigma_kms'])))
        else:
            gal_v, gal_v_err, gal_sig, gal_sig_err = 0.0, 0.0, 0.0, 0.0

        ages = np.array([m['tage_gyr'] for m in fsps_grid.template_meta], dtype=float)
        mets = np.array([m['logzsol'] for m in fsps_grid.template_meta], dtype=float)
        wsum = np.sum(self.fsps_weights_median)
        age_weighted = float(np.sum(self.fsps_weights_median * ages) / wsum) if wsum > 0 else -1.0
        metal_weighted = float(np.sum(self.fsps_weights_median * mets) / wsum) if wsum > 0 else -99.0

        self.frac_host_4200 = self._host_fraction_at_wave(4200.0)
        self.frac_host_5100 = self._host_fraction_at_wave(5100.0)
        self.frac_host_2500 = self._host_fraction_at_wave(2500.0)
        self.frac_bc_2500 = self._bc_fraction_at_wave(2500.0)

        n_samp = int(np.asarray(next(iter(samples.values()))).shape[0]) if len(samples) > 0 else 1
        if 'cont_norm' in samples:
            cont_samp = np.asarray(samples['cont_norm'])
            if decompose_host and 'log_frac_host' in samples:
                frac_host_samp = 1.0 / (1.0 + np.exp(-np.asarray(samples['log_frac_host'])))
                agn_samp = cont_samp * (1.0 - frac_host_samp)
            else:
                agn_samp = cont_samp
            pl_norm_samp = agn_samp
        elif 'PL_norm' in samples:
            pl_norm_samp = np.asarray(samples['PL_norm'])
        else:
            pl_norm_samp = np.full((n_samp,), np.nan)
        if 'PL_slope' in samples:
            pl_slope_med = float(np.nanmedian(np.asarray(samples['PL_slope'])))
            pl_slope_err = float(np.nanstd(np.asarray(samples['PL_slope'])))
        else:
            pl_slope_med = np.nan
            pl_slope_err = np.nan
        self.conti_result = np.array([
            self.ra, self.dec, str(self.filename), self.z,
            self.SN_ratio_conti,
            float(np.nanmedian(pl_norm_samp)), float(np.nanstd(pl_norm_samp)),
            pl_slope_med, pl_slope_err,
            gal_sig, gal_sig_err, gal_v, gal_v_err,
            self.frac_host_4200, self.frac_host_5100, self.frac_host_2500, self.frac_bc_2500,
            age_weighted, metal_weighted,
        ], dtype=object)
        self.conti_result_type = np.array([
            'float', 'float', 'str', 'float', 'float',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float', 'float', 'float'
        ], dtype=object)
        self.conti_result_name = np.array([
            'ra', 'dec', 'filename', 'redshift', 'SN_ratio_conti',
            'PL_norm', 'PL_norm_err', 'PL_slope', 'PL_slope_err',
            'sigma', 'sigma_err', 'v_off', 'v_off_err',
            'frac_host_4200', 'frac_host_5100', 'frac_host_2500', 'frac_bc_2500',
            'fsps_age_weighted_gyr', 'fsps_logzsol_weighted'
        ], dtype=object)

        if use_lines and tied_line_meta['n_lines'] > 0:
            amp_comp = np.asarray(pred_out['line_amp_per_component'])
            mu_comp = np.asarray(pred_out['line_mu_per_component'])
            sig_comp = np.asarray(pred_out['line_sig_per_component'])

            amp_med = np.median(amp_comp, axis=0)
            amp_err = np.std(amp_comp, axis=0)
            mu_med = np.median(mu_comp, axis=0)
            mu_err = np.std(mu_comp, axis=0)
            sig_med = np.median(sig_comp, axis=0)
            sig_err = np.std(sig_comp, axis=0)

            vals, names, types = [], [], []
            for i, nm in enumerate(tied_line_meta['names']):
                vals.extend([amp_med[i], amp_err[i], mu_med[i], mu_err[i], sig_med[i], sig_err[i]])
                names.extend([f'{nm}_scale', f'{nm}_scale_err', f'{nm}_centerwave', f'{nm}_centerwave_err', f'{nm}_sigma', f'{nm}_sigma_err'])
                types.extend(['float'] * 6)

            self.line_result = np.array(vals, dtype=object)
            self.line_result_type = np.array(types, dtype=object)
            self.line_result_name = np.array(names, dtype=object)
            self.gauss_result = self.line_result
            self.gauss_result_name = self.line_result_name
            self.line_component_amp_median = amp_med
            self.line_component_mu_median = mu_med
            self.line_component_sig_median = sig_med
        else:
            self.line_result = np.array([])
            self.line_result_type = np.array([])
            self.line_result_name = np.array([])
            self.gauss_result = np.array([])
            self.gauss_result_name = np.array([])
            self.line_component_amp_median = np.array([])
            self.line_component_mu_median = np.array([])
            self.line_component_sig_median = np.array([])

    def _wave_trim(self, lam, flux, err, z):
        """Apply rest-frame wavelength range trimming.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Redshift used for rest-frame conversion.
        """
        ind_trim = np.where((lam / (1 + z) > self.wave_range[0]) & (lam / (1 + z) < self.wave_range[1]), True, False)
        self.lam, self.flux, self.err = lam[ind_trim], flux[ind_trim], err[ind_trim]
        if len(self.lam) < 100:
            raise RuntimeError('No enough pixels in the input wave_range!')
        return self.lam, self.flux, self.err

    def _wave_msk(self, lam, flux, err, z):
        """Mask user-provided rest-frame wavelength intervals.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Redshift used for rest-frame conversion.
        """
        for msk in range(len(self.wave_mask)):
            ind_not_mask = ~np.where((lam / (1 + z) > self.wave_mask[msk, 0]) & (lam / (1 + z) < self.wave_mask[msk, 1]), True, False)
            self.lam, self.flux, self.err = lam[ind_not_mask], flux[ind_not_mask], err[ind_not_mask]
            lam, flux, err = self.lam, self.flux, self.err
        return self.lam, self.flux, self.err

    def _mask_lya_forest(self, lam, flux, err, z, lya_rest=1215.67):
        """Mask observed pixels blueward of rest-frame Ly-alpha.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Redshift used for rest-frame conversion.
        lya_rest : float, optional
            Rest-frame Ly-alpha cutoff in Angstrom.
        """
        keep = (lam / (1 + z)) >= float(lya_rest)
        self.lam, self.flux, self.err = lam[keep], flux[keep], err[keep]
        if len(self.lam) < 10:
            raise RuntimeError('Not enough pixels after Ly-alpha forest masking.')
        return self.lam, self.flux, self.err

    def _de_redden(self, lam, flux, err, ra, dec):
        """Correct observed flux/error for Galactic extinction using dustmaps.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        ra, dec : float
            Sky coordinates in degrees.
        """
        sfd_query = _get_sfd_query()
        coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame='icrs')
        ebv = float(np.asarray(sfd_query(coord)))
        zero_flux = np.where(flux == 0, True, False)
        flux[zero_flux] = 1e-10
        flux_unred = unred(lam, flux, ebv)
        err_unred = err * flux_unred / flux
        flux_unred[zero_flux] = 0
        self.flux = flux_unred
        self.err = err_unred
        return self.flux

    def _rest_frame(self, lam, flux, err, z):
        """Convert observed-frame spectra to rest-frame convention.

        Parameters
        ----------
        lam, flux, err : ndarray
            Observed-frame wavelength, flux, and uncertainty arrays.
        z : float
            Source redshift.
        """
        self.wave = lam / (1 + z)
        self.flux = flux * (1 + z)
        self.err = err * (1 + z)
        return self.wave, self.flux, self.err

    def _orignial_spec(self, wave, flux, err):
        """Cache the pre-modeling spectrum for plotting/debugging.

        Parameters
        ----------
        wave, flux, err : ndarray
            Rest-frame wavelength, flux, and uncertainty arrays.
        """
        self.wave_prereduced = wave
        self.flux_prereduced = flux
        self.err_prereduced = err

    def _calculate_sn(self, wave, flux, alter=True):
        """Estimate continuum S/N from standard windows or robust fallback.

        Parameters
        ----------
        wave, flux : ndarray
            Rest-frame wavelength and flux arrays.
        alter : bool, optional
            If True and standard windows are unavailable, use robust fallback.
        """
        ind5100 = np.where((wave > 5080) & (wave < 5130), True, False)
        ind3000 = np.where((wave > 3000) & (wave < 3050), True, False)
        ind1350 = np.where((wave > 1325) & (wave < 1375), True, False)
        if np.all(np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) < 10):
            if alter is False:
                self.SN_ratio_conti = -1.
                return self.SN_ratio_conti
            input_data = np.array(flux)
            input_data = np.array(input_data[np.where(input_data != 0.0)])
            n = len(input_data)
            if n > 4:
                signal = np.median(input_data)
                noise = 0.6052697 * np.median(np.abs(2.0 * input_data[2:n - 2] - input_data[0:n - 4] - input_data[4:n]))
                self.SN_ratio_conti = float(signal / noise)
            else:
                self.SN_ratio_conti = -1.
        else:
            tmp_SN = np.array([flux[ind5100].mean() / flux[ind5100].std(), flux[ind3000].mean() / flux[ind3000].std(), flux[ind1350].mean() / flux[ind1350].std()])
            tmp_SN = tmp_SN[np.array([np.sum(ind5100), np.sum(ind3000), np.sum(ind1350)]) > 10]
            self.SN_ratio_conti = np.nanmean(tmp_SN) if not np.all(np.isnan(tmp_SN)) else -1.
        return self.SN_ratio_conti

    def _host_fraction_at_wave(self, w0):
        """Return host/continuum flux fraction at wavelength ``w0``.

        Parameters
        ----------
        w0 : float
            Rest-frame wavelength in Angstrom.
        """
        return self._component_fraction_at_wave(self.host, w0)

    def _bc_fraction_at_wave(self, w0):
        """Return Balmer-continuum/continuum flux fraction at wavelength ``w0``.

        Parameters
        ----------
        w0 : float
            Rest-frame wavelength in Angstrom.
        """
        return self._component_fraction_at_wave(self.f_bc_model, w0)

    def _component_fraction_at_wave(self, component, w0):
        """Return component fraction relative to fitted continuum at ``w0``.

        Parameters
        ----------
        component : ndarray
            Component flux array evaluated on ``self.wave``.
        w0 : float
            Rest-frame wavelength in Angstrom.
        """
        if len(self.wave) == 0:
            return -1.
        comp = np.interp(w0, self.wave, component, left=np.nan, right=np.nan)
        total = np.interp(w0, self.wave, self.f_conti_model, left=np.nan, right=np.nan)
        if not np.isfinite(comp) or not np.isfinite(total) or total == 0:
            return -1.
        return float(comp / total)

    def reconstruct_posterior_spectrum(
        self,
        wave_out=None,
        wave_min=2500.0,
        wave_max=None,
        n_draws=None,
        return_components=True,
    ):
        """Rebuild posterior component draws on a requested rest-frame grid.

        Parameters
        ----------
        wave_out : array-like or None, optional
            Explicit rest-frame wavelength grid. If ``None``, build a grid from
            ``min(wave_min, self.wave.min())`` to ``wave_max or self.wave.max()``
            using the median native wavelength spacing.
        wave_min, wave_max : float or None, optional
            Bounds for the auto-generated grid when ``wave_out`` is ``None``.
        n_draws : int or None, optional
            If provided, use at most the first ``n_draws`` posterior samples.
        return_components : bool, optional
            If True, include per-component draws and medians in the return value.
        """
        if not hasattr(self, 'numpyro_samples') or not hasattr(self, 'fsps_grid'):
            raise RuntimeError("No posterior samples available. Run fit() first.")
        if not hasattr(self, 'wave') or len(self.wave) < 2:
            raise RuntimeError("No fitted rest-frame wavelength grid available.")

        wave_native = np.asarray(self.wave, dtype=float)
        dw = float(np.nanmedian(np.diff(wave_native)))
        if not np.isfinite(dw) or dw <= 0:
            raise RuntimeError("Unable to infer wavelength spacing for reconstruction.")

        if wave_out is None:
            wmin = min(float(wave_min), float(np.nanmin(wave_native)))
            wmax = float(np.nanmax(wave_native) if wave_max is None else wave_max)
            if wmin >= wmax:
                raise ValueError("Requested reconstruction grid has non-positive span.")
            wave_out = np.arange(wmin, wmax + 0.5 * dw, dw, dtype=float)
        else:
            wave_out = np.asarray(wave_out, dtype=float)

        if wave_out.ndim != 1 or wave_out.size < 2 or not np.all(np.isfinite(wave_out)):
            raise ValueError("wave_out must be a finite 1D wavelength grid.")

        prior_config = getattr(self, '_fit_prior_config', None)
        if prior_config is None:
            prior_config = build_default_prior_config(np.asarray(self.flux, dtype=float))
        return reconstruct_posterior_components(
            wave_out=wave_out,
            samples=self.numpyro_samples,
            pred_out=getattr(self, 'pred_out', None),
            age_grid_gyr=getattr(self, '_fit_fsps_age_grid', self.fsps_grid.age_grid_gyr),
            logzsol_grid=getattr(self, '_fit_fsps_logzsol_grid', self.fsps_grid.logzsol_grid),
            dsps_ssp_fn=getattr(self, '_fit_dsps_ssp_fn', 'tempdata.h5'),
            prior_config=prior_config,
            fit_poly=bool(getattr(self, '_fit_fit_poly', False)),
            fit_poly_order=int(getattr(self, '_fit_fit_poly_order', 2)),
            fit_poly_edge_flex=bool(getattr(self, '_fit_fit_poly_edge_flex', False)),
            fe_uv_wave=self.fe_uv_wave,
            fe_uv_flux=self.fe_uv_flux,
            fe_op_wave=self.fe_op_wave,
            fe_op_flux=self.fe_op_flux,
            n_draws=n_draws,
            return_components=return_components,
        )

    def component_fraction_at_wave(self, component='host', wave0=2500.0, reference='continuum', reconstruct=False, n_draws=None):
        """Return component/reference flux fraction at a requested wavelength.

        Parameters
        ----------
        component, reference : str, optional
            Component names. Supported reconstructed names are ``host``, ``PL``,
            ``Fe_uv``, ``Fe_op``, ``Balmer_cont``, ``edge_additive``, and ``continuum``.
        wave0 : float, optional
            Rest-frame wavelength in Angstrom.
        reconstruct : bool, optional
            If True, rebuild posterior components on a grid that reaches ``wave0``.
            Returns ``(median, err)`` from the posterior draws.
        n_draws : int or None, optional
            Maximum number of posterior draws to use in the reconstruction.
        """
        if not reconstruct:
            component_map = {
                'host': getattr(self, 'host', None),
                'Balmer_cont': getattr(self, 'f_bc_model', None),
                'continuum': getattr(self, 'f_conti_model', None),
            }
            comp_arr = component_map.get(component)
            ref_arr = component_map.get(reference, getattr(self, 'f_conti_model', None))
            if comp_arr is None or ref_arr is None or len(self.wave) == 0:
                return -1.0, np.nan
            comp = np.interp(wave0, self.wave, comp_arr, left=np.nan, right=np.nan)
            ref = np.interp(wave0, self.wave, ref_arr, left=np.nan, right=np.nan)
            if not np.isfinite(comp) or not np.isfinite(ref) or ref == 0:
                return -1.0, np.nan
            return float(comp / ref), np.nan

        recon = self.reconstruct_posterior_spectrum(wave_min=min(float(wave0), float(np.nanmin(self.wave))), n_draws=n_draws)
        wave = recon['wave']
        idx = int(np.argmin(np.abs(wave - float(wave0))))
        if component not in recon['draws'] or reference not in recon['draws']:
            raise ValueError(f"Unknown reconstructed component/reference: {component}, {reference}")
        num = np.asarray(recon['draws'][component], dtype=float)[:, idx]
        den = np.asarray(recon['draws'][reference], dtype=float)[:, idx]
        frac = np.divide(num, den, out=np.full_like(num, np.nan), where=np.isfinite(den) & (den != 0))
        good = np.isfinite(frac)
        if not np.any(good):
            return np.nan, np.nan
        p16, p50, p84 = np.percentile(frac[good], [16.0, 50.0, 84.0])
        return float(p50), float(0.5 * (p84 - p16))

    def _line_profile_from_params(
        self,
        line_key: str,
        amp: np.ndarray,
        mu: np.ndarray,
        sig: np.ndarray,
    ) -> np.ndarray:
        """Build a line profile from explicit Gaussian parameter arrays.

        Parameters
        ----------
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        amp, mu, sig : ndarray
            Gaussian amplitudes, centers (ln lambda), and widths.
        """
        if not hasattr(self, 'wave') or len(self.wave) == 0:
            return np.array([], dtype=float)
        if not hasattr(self, 'tied_line_meta'):
            return np.zeros_like(self.wave, dtype=float)

        names = np.asarray(self.tied_line_meta.get('names', []))
        amp = np.asarray(amp, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sig = np.asarray(sig, dtype=float)
        if names.size == 0 or amp.size == 0 or mu.size == 0 or sig.size == 0:
            return np.zeros_like(self.wave, dtype=float)

        keep = np.array([str(n).startswith(f'{line_key}_') for n in names], dtype=bool)
        if not np.any(keep):
            return np.zeros_like(self.wave, dtype=float)

        lnw = np.log(np.asarray(self.wave, dtype=float))
        prof = np.zeros_like(lnw)
        for a, m, s in zip(amp[keep], mu[keep], sig[keep]):
            if np.isfinite(a) and np.isfinite(m) and np.isfinite(s) and s > 0:
                prof += a * np.exp(-0.5 * ((lnw - m) / s) ** 2)
        return prof

    def line_profile_from_components(self, line_key: str) -> np.ndarray:
        """Build a line-only profile from posterior-median Gaussian components.

        Parameters
        ----------
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        """
        if not hasattr(self, 'line_component_amp_median'):
            return np.zeros_like(self.wave, dtype=float)
        return self._line_profile_from_params(
            line_key=line_key,
            amp=np.asarray(getattr(self, 'line_component_amp_median', []), dtype=float),
            mu=np.asarray(getattr(self, 'line_component_mu_median', []), dtype=float),
            sig=np.asarray(getattr(self, 'line_component_sig_median', []), dtype=float),
        )

    def line_profile_from_draw(self, draw_index: int, line_key: str) -> np.ndarray:
        """Build a line-only profile for one posterior draw index.

        Parameters
        ----------
        draw_index : int
            Posterior draw index.
        line_key : str
            Line-name prefix (for example ``'Hb_br'``).
        """
        if not hasattr(self, 'pred_out') or self.pred_out is None:
            return np.zeros_like(self.wave, dtype=float)
        if 'line_amp_per_component' not in self.pred_out:
            return np.zeros_like(self.wave, dtype=float)

        amp_draws = np.asarray(self.pred_out['line_amp_per_component'])
        mu_draws = np.asarray(self.pred_out['line_mu_per_component'])
        sig_draws = np.asarray(self.pred_out['line_sig_per_component'])
        if amp_draws.ndim != 2 or mu_draws.ndim != 2 or sig_draws.ndim != 2:
            return np.zeros_like(self.wave, dtype=float)

        idx = int(draw_index)
        if idx < 0 or idx >= amp_draws.shape[0]:
            raise IndexError(f'draw_index {idx} is out of bounds for {amp_draws.shape[0]} posterior draws')

        return self._line_profile_from_params(
            line_key=line_key,
            amp=amp_draws[idx],
            mu=mu_draws[idx],
            sig=sig_draws[idx],
        )

    def line_props(self, profile: np.ndarray, wave: np.ndarray | None = None) -> tuple[float, float]:
        """Return ``(fwhm_kms, integrated_area)`` from a line profile.

        Parameters
        ----------
        profile : ndarray
            Line profile values.
        wave : ndarray or None, optional
            Wavelength array. If ``None``, ``self.wave`` is used.
        """
        p = np.asarray(profile, dtype=float)
        w = np.asarray(self.wave if wave is None else wave, dtype=float)
        if p.size == 0 or w.size == 0 or p.size != w.size:
            return np.nan, np.nan
        if not np.any(np.isfinite(p)) or np.nanmax(p) <= 0:
            return np.nan, np.nan

        ipeak = int(np.nanargmax(p))
        peak_lam = w[ipeak]
        half = 0.5 * p[ipeak]
        idx = np.where(p >= half)[0]
        area = float(np.trapezoid(np.clip(p, 0.0, None), w))
        if idx.size < 2 or not np.isfinite(peak_lam) or peak_lam <= 0:
            return np.nan, area

        fwhm_a = w[idx[-1]] - w[idx[0]]
        fwhm_kms = C_KMS * fwhm_a / peak_lam
        return float(fwhm_kms), area

    def line_props_from_profile(self, wave: np.ndarray, profile: np.ndarray) -> tuple[float, float]:
        """Compatibility wrapper for :meth:`line_props`.

        Parameters
        ----------
        wave : ndarray
            Wavelength array.
        profile : ndarray
            Line profile values.
        """
        return self.line_props(profile=profile, wave=wave)

    def save_result(self, conti_result, conti_result_type, conti_result_name, line_result, line_result_type, line_result_name, save_fits_name):
        """Write continuum+line summary table to a pandas CSV file.

        Parameters
        ----------
        conti_result, line_result : ndarray
            Continuum and line result values.
        conti_result_type, line_result_type : ndarray
            Legacy dtype tags (stored but not enforced).
        conti_result_name, line_result_name : ndarray
            Column names for continuum and line outputs.
        save_fits_name : str
            Output basename for CSV.
        """
        self.all_result = np.concatenate([conti_result, line_result])
        self.all_result_type = np.concatenate([conti_result_type, line_result_type])
        self.all_result_name = np.concatenate([conti_result_name, line_result_name])
        df = pd.DataFrame([self.all_result], columns=self.all_result_name)
        out_dir = self.output_path if self.output_path is not None else '.'
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, save_fits_name + '.csv')
        df.to_csv(out_file, index=False)
        print(f"Saved results table: {out_file}")
        return

    def _posterior_series(self, param_names=None, max_vector_elems=2):
        """Flatten posterior samples into labeled 1D series for diagnostics.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` for all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        """
        if not hasattr(self, 'numpyro_samples') or self.numpyro_samples is None:
            return []

        samples = self.numpyro_samples
        if param_names == 'all':
            param_names = sorted(samples.keys())
        elif param_names is None:
            param_names = [
                'cont_norm', 'log_frac_host', 'PL_slope', 'Fe_uv_norm', 'Fe_op_norm',
                'Balmer_norm', 'Balmer_Te', 'Balmer_Tau',
                'gal_v_kms', 'gal_sigma_kms',
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

    @staticmethod
    def _style_axis(ax, spine_lw=1.5):
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

    def plot_trace(self, param_names=None, max_vector_elems=2, save_fig_path=None, save_fig_name=None):
        """Plot posterior trace series for selected parameters.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` to include all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        save_fig_name : str or None, optional
            Output filename override.
        """
        series = self._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
        if len(series) == 0:
            return None

        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(2.2 * n, 4)), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (label, vals) in zip(axes, series):
            ax.plot(np.arange(len(vals)), vals, color='tab:blue', lw=0.8)
            ax.set_ylabel(label, fontsize=9)
            self._style_axis(ax)
        axes[-1].set_xlabel('Sample', fontsize=10)
        fig.tight_layout()
        plt.show()
        if self.save_fig:
            out_name = f'{self.filename}_trace.pdf' if save_fig_name is None else save_fig_name
            save_dir = self.output_path if save_fig_path is None else save_fig_path
            if save_dir is None:
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, out_name)
            fig.savefig(out_file)
            print(f"Saved trace plot: {out_file}")
            plt.close(fig)
        self.trace_fig = fig
        return fig

    def plot_corner(self, param_names=None, max_vector_elems=2, bins=30, max_points=2000, save_fig_path=None, save_fig_name=None):
        """Plot a simple corner-style posterior projection matrix.

        Parameters
        ----------
        param_names : list[str] | str | None, optional
            Parameter selector. Use ``'all'`` to include all posterior keys.
        max_vector_elems : int or None, optional
            Maximum number of vector elements to expand per key.
        bins : int, optional
            Histogram bin count.
        max_points : int, optional
            Maximum posterior draws to plot (subsampled if needed).
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        save_fig_name : str or None, optional
            Output filename override.
        """
        series = self._posterior_series(param_names=param_names, max_vector_elems=max_vector_elems)
        if len(series) == 0:
            return None

        labels = [s[0] for s in series]
        data = np.column_stack([s[1] for s in series])
        if data.shape[0] > int(max_points):
            idx = np.linspace(0, data.shape[0] - 1, int(max_points), dtype=int)
            data = data[idx]
        ndim = data.shape[1]

        fig, axes = plt.subplots(ndim, ndim, figsize=(2.2 * ndim, 2.2 * ndim))
        fig.subplots_adjust(wspace=0.03, hspace=0.03)
        for i in range(ndim):
            for j in range(ndim):
                ax = axes[i, j]
                if i < j:
                    ax.axis('off')
                    continue
                if i == j:
                    ax.hist(data[:, j], bins=bins, color='tab:blue', alpha=0.75)
                else:
                    ax.hist2d(data[:, j], data[:, i], bins=bins, cmap='Blues', cmin=1)
                if i == ndim - 1:
                    ax.set_xlabel(labels[j], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=8)
                else:
                    ax.set_yticklabels([])
                self._style_axis(ax)
        fig.tight_layout(pad=0.35)
        plt.show()
        if self.save_fig:
            out_name = f'{self.filename}_corner.pdf' if save_fig_name is None else save_fig_name
            save_dir = self.output_path if save_fig_path is None else save_fig_path
            if save_dir is None:
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, out_name)
            fig.savefig(out_file)
            print(f"Saved corner plot: {out_file}")
            plt.close(fig)
        self.corner_fig = fig
        return fig

    def plot_mcmc_diagnostics(self, do_trace=True, do_corner=True,
                              param_names=None,
                              max_vector_elems=2,
                              corner_bins=30, corner_max_points=2000,
                              save_fig_path=None):
        """Plot trace and/or corner diagnostics in a single convenience call.

        Parameters
        ----------
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
            Output directory when saving figures. If ``None``, uses ``self.output_path``
            (or ``'.'`` when unset).
        """
        if do_trace:
            self.plot_trace(
                param_names=param_names,
                max_vector_elems=max_vector_elems,
                save_fig_path=save_fig_path,
            )
        if do_corner:
            self.plot_corner(
                param_names=param_names,
                max_vector_elems=max_vector_elems,
                bins=corner_bins,
                max_points=corner_max_points,
                save_fig_path=save_fig_path,
            )

    def plot_fig(self, save_fig_path=None, broad_fwhm=1200, plot_legend=True, ylims=None, plot_residual=True, show_title=True,
                 plot_1sigma=True, sigma_alpha=0.12, show_plot=True):
        """Plot data, model components, line decomposition, and residuals.

        Parameters
        ----------
        save_fig_path : str or None, optional
            Output directory when saving figures. If ``None``, uses ``self.output_path``
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
        """
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        if plot_residual:
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

        flux_ref = float(np.nanpercentile(np.abs(self.flux[np.isfinite(self.flux)]), 95)) if np.any(np.isfinite(self.flux)) else 1.0
        comp_floor = max(1e-8, 0.005 * flux_ref)

        def _show_component(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return False
            return float(np.nanmax(np.abs(arr))) >= comp_floor

        if plot_1sigma and hasattr(self, 'pred_bands'):
            band_colors = {
                'total_model': 'b',
                'host': 'purple',
                'PL': 'orange',
                'FeII': 'teal',
                'Balmer_cont': 'y',
                'lines': 'lightskyblue',
            }
            for key, color in band_colors.items():
                if key not in self.pred_bands:
                    continue
                lo, hi = self.pred_bands[key]
                if len(lo) == len(self.wave) and _show_component(0.5 * (np.asarray(lo) + np.asarray(hi))):
                    ax.fill_between(
                        self.wave,
                        lo,
                        hi,
                        color=color,
                        alpha=sigma_alpha,
                        linewidth=0,
                        zorder=0,
                        rasterized=True,
                    )

        ax.plot(
            self.wave_prereduced,
            self.flux_prereduced,
            'k',
            lw=1,
            label='data',
            zorder=2,
            rasterized=True,
        )
        ax.plot(self.wave, self.model_total, color='b', lw=1.8, label='total model', zorder=6, rasterized=True)
        if _show_component(self.host):
            ax.plot(self.wave, self.host, color='purple', lw=1.8, label='host', zorder=4, rasterized=True)
        else:
            ax.plot(self.wave, self.host, color='purple', lw=1.8, zorder=4, rasterized=True)
        if _show_component(self.f_pl_model):
            ax.plot(self.wave, self.f_pl_model, color='orange', lw=1.5, label='PL', zorder=5, rasterized=True)
        else:
            ax.plot(self.wave, self.f_pl_model, color='orange', lw=1.5, zorder=5, rasterized=True)
        fe_total_model = self.f_fe_mgii_model + self.f_fe_balmer_model
        if _show_component(fe_total_model):
            ax.plot(self.wave, fe_total_model, color='teal', lw=1.2, label='FeII', zorder=5, rasterized=True)
        else:
            ax.plot(self.wave, fe_total_model, color='teal', lw=1.2, zorder=5, rasterized=True)
        if _show_component(self.f_bc_model):
            ax.plot(self.wave, self.f_bc_model, color='y', lw=1.2, label='Balmer cont.', zorder=5, rasterized=True)
        else:
            ax.plot(self.wave, self.f_bc_model, color='y', lw=1.2, zorder=5, rasterized=True)
        if len(self.f_line_model) == len(self.wave):
            if _show_component(self.f_line_model):
                ax.plot(
                    self.wave,
                    self.f_line_model,
                    color='lightskyblue',
                    lw=1.5,
                    label='lines',
                    zorder=5,
                    rasterized=True,
                )
            else:
                ax.plot(
                    self.wave,
                    self.f_line_model,
                    color='lightskyblue',
                    lw=1.5,
                    zorder=5,
                    rasterized=True,
                )

        # Plot individual Gaussian line components: broad (*_br) in red, narrow in green.
        if (hasattr(self, 'line_component_amp_median')
                and hasattr(self, 'line_component_mu_median')
                and hasattr(self, 'line_component_sig_median')
                and hasattr(self, 'tied_line_meta')
                and len(self.line_component_amp_median) > 0):
            lnwave = np.log(self.wave)
            comp_labels = self.tied_line_meta.get('names', [''] * len(self.line_component_amp_median))
            drew_broad_label = False
            drew_narrow_label = False
            show_line_leg = _show_component(self.f_line_model)
            for i in range(len(self.line_component_amp_median)):
                amp = float(self.line_component_amp_median[i])
                mu = float(self.line_component_mu_median[i])
                sig = float(self.line_component_sig_median[i])
                if not np.isfinite(amp) or not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
                    continue
                prof = amp * np.exp(-0.5 * ((lnwave - mu) / sig) ** 2)
                # Keep component plotting consistent with polynomial correction if enabled.
                if hasattr(self, 'f_poly_model') and len(self.f_poly_model) == len(prof):
                    prof = prof * self.f_poly_model
                cname = str(comp_labels[i]).lower()
                is_broad = cname.endswith('_br') or ('_br' in cname)
                if is_broad:
                    lbl = 'broad comps' if (show_line_leg and not drew_broad_label) else None
                    ax.plot(
                        self.wave,
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
                    lbl = 'narrow comps' if (show_line_leg and not drew_narrow_label) else None
                    ax.plot(
                        self.wave,
                        prof,
                        color='green',
                        lw=0.7,
                        alpha=0.25,
                        zorder=3,
                        label=lbl,
                        rasterized=True,
                    )
                    drew_narrow_label = True

        ax.set_xlim(self.wave.min(), self.wave.max())
        if ylims is None:
            yplot = np.concatenate([
                self.flux[np.isfinite(self.flux)],
                self.model_total[np.isfinite(self.model_total)]
            ])
            if yplot.size > 0:
                y1, y2 = np.nanpercentile(yplot, [1, 99])
                if np.isfinite(y1) and np.isfinite(y2) and y2 > y1:
                    pad = 0.15 * (y2 - y1)
                    ax.set_ylim(0, y2 + pad)
        else:
            ax.set_ylim(0, ylims[1])

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
        for label, lam0 in broad_line_markers:
            if xlo <= lam0 <= xhi:
                ax.axvline(lam0, color="gray", ls="--", lw=0.8, alpha=0.35, zorder=1)
                ax.text(
                    lam0,
                    y_top * 0.985,
                    label,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=9,
                    color="dimgray",
                    alpha=0.9,
                    zorder=7,
                )

        if plot_residual and len(self.model_total) == len(self.wave) and ax_resid is not None:
            resid = self.flux - self.model_total
            ax_resid.plot(
                self.wave,
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
            self._style_axis(ax_resid)

        if plot_residual and ax_resid is not None:
            ax_resid.set_xlabel(r'Rest Wavelength ($\AA$)', fontsize=20)
        else:
            ax.set_xlabel(r'Rest Wavelength ($\AA$)', fontsize=20)
        ax.set_ylabel(r'$f_{\lambda}\ (10^{-17}\ \mathrm{erg}\ \mathrm{s}^{-1}\ \mathrm{cm}^{-2}\ \AA^{-1})$', fontsize=20)
        self._style_axis(ax)
        if plot_legend:
            ax.legend(frameon=True, framealpha=0.9, edgecolor='0.3', fontsize=9, ncol=2)
        if show_plot:
            plt.show()
        if self.save_fig:
            save_dir = self.output_path if save_fig_path is None else save_fig_path
            if save_dir is None:
                save_dir = '.'
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, self.filename + '.pdf')
            fig.savefig(out_file, dpi=150)
            print(f"Saved spectrum plot: {out_file}")
            plt.close(fig)
        self.fig = fig
        return
