Quickstart
==========

``jaxqsofit`` is configured through a single :class:`jaxqsofit.FitConfig` object.
It groups the observation metadata, the spectrum, the continuum, host and line
options, and the inference, output and prior settings. Build the config, pass it
to :class:`jaxqsofit.JAXQSOFit`, and call :meth:`jaxqsofit.JAXQSOFit.fit`.

Fit one spectrum
----------------

``wave_obs`` is the observed-frame wavelength in Å. ``fluxes`` and ``errors`` are
:math:`f_\lambda` in units of
:math:`10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}`, as in SDSS spectra.

.. code-block:: python

   import numpy as np
   from jaxqsofit import (
       ContinuumConfig,
       FitConfig,
       HostConfig,
       InferenceConfig,
       JAXQSOFit,
       Observation,
       OutputConfig,
       SpectroscopyData,
   )

   # Example arrays
   lam = np.linspace(3800.0, 9200.0, 2000)
   flux = 50.0 + 0.002 * (lam - 6000.0)
   err = np.full_like(flux, 0.5)

   cfg = FitConfig(
       # Synthetic spectrum: no sky position, so no Milky Way dereddening
       observation=Observation(object_id="demo", redshift=0.1, apply_mw_deredden=False),
       spectroscopy=SpectroscopyData(
           wave_obs=lam,
           fluxes=flux,
           errors=err,
           resolving_power=2000.0,
           apply_instrumental_resolution=True,
       ),
       continuum=ContinuumConfig(fit_feii=True, fit_balmer_continuum=True),
       host=HostConfig(enabled=True, dsps_ssp_fn="tempdata.h5"),
       inference=InferenceConfig(method="optax+nuts"),
       output=OutputConfig(save_result=False, plot_fig=True),
   )

   q = JAXQSOFit(cfg)
   result = q.fit()

``dsps_ssp_fn`` must point to a valid DSPS SSP HDF5 file (see :doc:`installation`);
a missing file is the most common setup failure. For real spectra, give the
target's ``ra`` and ``dec`` to :class:`jaxqsofit.Observation`: Milky Way
dereddening is on by default and needs them. For a fast fit, use
``method="optax"`` (a MAP) or ``"laplace"`` (a MAP with Gaussian uncertainties);
:doc:`inference` compares the methods.

Look at the result
------------------

``result.spectrum`` holds the posterior draws of the fitted components and of every
emission line, with explicit units:

.. code-block:: python

   hb = result.spectrum.line_groups["Hb_br"]        # all broad H-beta components
   np.percentile(hb.total_flux_erg_s_cm2, [16, 50, 84])

   result.spectrum.lines["Hb_br_1"].fwhm_kms        # one Gaussian component
   result.spectrum.host_flambda_1e17                # host model, (draw, pixel)

   q.plot_fig()                                     # data, components, residuals
   result.plot_corner(show_plot=False)

:doc:`results` describes the results interface, line measurements and saving.

Next steps
----------

* :doc:`tutorials`: worked examples with SDSS spectra, a composite quasar, Fe II,
  host-dominated spectra, PSF recalibration, custom components and BALs.
* :doc:`model`: the continuum, host, BAL and custom components, and instrumental
  resolution.
* :doc:`lines`: the emission-line table and its ties.
* :doc:`likelihood`: the error model, flux calibration and priors.
* :doc:`inference` and :doc:`batch`: choosing a fitting method, and fitting
  catalogs.
* :doc:`pyqsofit`: how ``jaxqsofit`` differs from PyQSOFit.
