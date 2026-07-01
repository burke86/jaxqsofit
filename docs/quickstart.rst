Quickstart
==========

``jaxqsofit`` is configured through a single :class:`jaxqsofit.FitConfig`
object. The top-level config groups the observation metadata, spectroscopy
arrays, continuum and host-galaxy options, inference settings, output behavior,
and optional prior configuration. Build the config first, pass it to
:class:`jaxqsofit.JAXQSOFit`, and then call :meth:`jaxqsofit.JAXQSOFit.fit`.

Minimal fitting example
-----------------------

The observed wavelength array passed as ``wave_obs`` should be in Angstroms.
The ``fluxes`` and ``errors`` arrays should be in units of
:math:`10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{\AA}^{-1}\,\mathrm{cm}^{-2}`.

.. code-block:: python

   import numpy as np
   from jaxqsofit import (
       ContinuumConfig,
       HostConfig,
       InferenceConfig,
       JAXQSOFit,
       Observation,
       OutputConfig,
       FitConfig,
       SpectroscopyData,
   )

   # Example arrays
   lam = np.linspace(3800.0, 9200.0, 2000)
   flux = 50.0 + 0.002 * (lam - 6000.0)
   err = np.full_like(flux, 0.5)
   z = 0.1

   cfg = FitConfig(
       observation=Observation(object_id='demo', redshift=z),
       spectroscopy=SpectroscopyData(wave_obs=lam, fluxes=flux, errors=err),
       continuum=ContinuumConfig(fit_feii=True, fit_balmer_continuum=True),
       host=HostConfig(enabled=True, dsps_ssp_fn='tempdata.h5'),
       inference=InferenceConfig(method='nuts'),
       output=OutputConfig(save_result=False, plot_fig=True),
   )
   q = JAXQSOFit(cfg)
   result = q.fit()
   result.samples
   result.plot_corner(show_plot=False)

Fast mode
---------

For a fast MAP-style fit, use:

.. code-block:: python

   q.config.inference.method = 'optax'
   q.config.inference.map_steps = 1500
   q.config.inference.learning_rate = 1e-2
   result = q.fit()
   result.save("fit_outputs")

When ``save_result=True`` or :meth:`jaxqsofit.FitResult.save` is used,
``jaxqsofit`` writes an HDF5 posterior bundle named ``<object_id>_samples.h5``.

Hybrid mode
-----------

Warm-start with Optax, then run NUTS:

.. code-block:: python

   q.config.inference.method = 'optax+nuts'
   q.config.inference.map_steps = 800
   q.config.inference.num_warmup = 200
   q.config.inference.num_samples = 400
   result = q.fit()
   components = result.predict(n_draws=200)
