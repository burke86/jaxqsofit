Spectral model
==============

The model spectrum is the sum of an AGN continuum, Fe II and Balmer-continuum
emission, a host galaxy and Gaussian emission lines (see :doc:`lines`), optionally
with BAL absorption and custom components, all fitted jointly.

AGN type presets
----------------

AGN type presets choose a coherent set of components:

.. code-block:: python

   from jaxqsofit import AGNConfig

   cfg.agn = AGNConfig(agn_type=1)
   cfg.apply_agn_type_defaults()

``agn_type=1``
   Broad-line AGN: broad and narrow lines, Fe II, the Balmer continuum and the
   host are enabled.

``agn_type=2``
   Narrow-line or host-dominated: the built-in broad line components are removed;
   narrow lines and the host stay on; Fe II and the Balmer continuum are off.

The preset changes ordinary component switches, so apply it before any manual
overrides that should win:

.. code-block:: python

   cfg.set_agn_type(2)
   cfg.continuum.fit_feii = True  # explicit override after the preset

AGN continuum
-------------

:class:`jaxqsofit.ContinuumConfig` switches the non-line AGN components:

.. code-block:: python

   cfg.continuum.fit_power_law = True          # default
   cfg.continuum.fit_feii = True               # default
   cfg.continuum.fit_balmer_continuum = False  # default
   cfg.continuum.fit_reddening = True          # default
   cfg.continuum.fit_polynomial_tilt = True    # default; see likelihood

``fit_feii`` and ``fit_balmer_continuum``
   UV/optical Fe II templates and the Balmer continuum. They help when the spectrum
   covers those features with enough signal to constrain them. Leave them off for
   quick tests, narrow wavelength coverage, or low signal-to-noise spectra.

``fit_reddening``
   Intrinsic AGN reddening, **on by default**. One fitted E(B-V) screen attenuates
   the power law, Fe II, the Balmer continuum and custom nuclear continua; it does
   not attenuate emission lines or the host. Milky Way dereddening is separate
   (``Observation.apply_mw_deredden``). The power law is sampled in its apparent,
   post-attenuation normalization and slope, an exact reparameterization that
   improves sampling without changing the model.

Host galaxy
-----------

:class:`jaxqsofit.HostConfig` controls the host:

.. code-block:: python

   cfg.host = HostConfig(enabled=True, dsps_ssp_fn="tempdata.h5")
   cfg.host = HostConfig(enabled=False)   # quasar-dominated spectra, quick tests

``sfh_model="delayed"`` (default)
   A delayed-:math:`\tau` star-formation history,
   :math:`\mathrm{SFR}(t) \propto t\,\exp(-t/\tau)` from the onset of star formation
   up to the fitted age. It samples a few physical parameters (``sfh_age_gyr``,
   ``sfh_tau_gyr``, stellar mass and metallicity) and is the more stable choice for
   most spectra.

``sfh_model="flexible"``
   Free SSP template weights over the ``age_grid_gyr`` × ``logzsol_grid`` grid. It
   follows more detailed stellar-population structure but has more parameters and
   is easier to under-constrain when the host absorption features or the wavelength
   coverage are weak. It is the closest to PyQSOFit's host decomposition (see
   :doc:`pyqsofit`); the weights are less directly physical than the delayed SFH's
   age, ``tau``, mass and metallicity.

The host includes a line-of-sight velocity distribution, fitted with the stellar
velocity dispersion.

Instrumental resolution
-----------------------

Instrumental-resolution modeling is off by default
(``apply_instrumental_resolution=False``) and needs a positive scalar
``resolving_power`` :math:`R=\lambda/\Delta\lambda_{\rm FWHM}` on
:class:`jaxqsofit.SpectroscopyData`. Turning it on is recommended, particularly for
narrow lines or the host velocity dispersion: the line-spread function then enters
the forward model, so the nonlinear width correction propagates through the
posterior and marginally resolved features are handled correctly.

When it is on, intrinsic and instrumental Gaussian widths are added in quadrature
during the fit. The sampled and reported line widths are the intrinsic,
resolution-corrected widths, as with PyQSOFit's optional correction; the widths used
to generate the model are stored as ``line_sig_effective_per_component`` and
``gal_sigma_effective_kms``.

When it is off, even with a ``resolving_power``, reported line and host widths are
those of the observed, instrument-broadened features. They can be deconvolved
afterwards for well-resolved Gaussian features, but that does not propagate the
correction through the posterior. Broad quasar lines are usually insensitive to the
difference; narrow or unresolved features are not.

BAL absorption
--------------

Built-in broad absorption line (BAL) modeling is enabled with
:class:`jaxqsofit.BALConfig`:

.. code-block:: python

   from jaxqsofit import BALConfig

   cfg.bal = BALConfig(
       enabled=True,
       tau_scale=0.25,
       covering_loc=0.15,
       covering_scale=0.12,
       covering_high=0.70,
   )
   q = JAXQSOFit(cfg)
   result = q.fit()
   bi, bi_err = q.balnicity_index()

This adds multiplicative absorption components for N V, Si IV and C IV that share
the outflow velocity, optical depth and covering fraction across the troughs. They
appear as ``BAL`` in plots. The BAL tutorial shows a complete fit.

Custom components
-----------------

Additional components, such as an alternative iron template, another continuum
component or a non-Gaussian line profile, are added through ``cfg.lines``
(``components``, ``custom_components`` and ``custom_line_components``). Their
evaluators must be JAX-compatible. The custom-components tutorial
(:doc:`tutorials`) walks through each kind, and :doc:`api/custom_components` lists
the helpers. The same component definitions work in ``jaxsedfit`` joint fits.
