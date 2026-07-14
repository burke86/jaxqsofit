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
       AGNConfig,
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
       spectroscopy=SpectroscopyData(
           wave_obs=lam,
           fluxes=flux,
           errors=err,
           resolving_power=2000.0,
           apply_instrumental_resolution=True,
       ),
       continuum=ContinuumConfig(fit_feii=True, fit_balmer_continuum=True),
       host=HostConfig(enabled=True, dsps_ssp_fn='tempdata.h5'),
       inference=InferenceConfig(method='nuts'),
       output=OutputConfig(save_result=False, plot_fig=True),
   )
   q = JAXQSOFit(cfg)
   result = q.fit()
   result.samples
   result.plot_corner(show_plot=False)

Instrumental resolution modeling
---------------------------------

Instrumental resolution modeling is opt-in and defaults to
``apply_instrumental_resolution=False``.  It can be enabled only when a
positive scalar ``resolving_power`` :math:`R=\lambda/\Delta\lambda_{\rm FWHM}`
is provided.  Enabling it is recommended, particularly when fitting narrow
emission lines or host-galaxy velocity dispersion, because including the line
spread function in the forward model propagates the nonlinear width correction
through the posterior and handles marginally resolved features more accurately
than a post-fit correction.

When enabled, ``jaxqsofit`` combines the intrinsic and instrumental Gaussian
widths in quadrature while fitting.  The sampled and normally reported line
widths remain the inferred *intrinsic*, resolution-corrected widths, analogous
to the optional instrumental-resolution correction in PyQSOFit.  The effective
widths actually used to generate the model spectrum are also available as
``line_sig_effective_per_component`` and ``gal_sigma_effective_kms``.

When disabled, even if ``resolving_power`` is present, reported line and host
widths are widths of the observed, instrument-broadened features and are **not**
corrected for instrumental resolution.  They may be deconvolved afterward for
well-resolved Gaussian features, but that does not propagate the nonlinear
correction through the posterior.  Broad quasar lines are usually insensitive
to the difference; narrow or unresolved features are not.

Continuum and host switches
---------------------------

AGN type presets provide a quick way to choose coherent spectral components:

.. code-block:: python

   cfg.agn = AGNConfig(agn_type=1)
   cfg.apply_agn_type_defaults()

   cfg.agn = AGNConfig(agn_type=2)
   cfg.apply_agn_type_defaults()

``agn_type=1`` is the broad-line AGN preset: broad and narrow lines are enabled,
Fe II is enabled, the Balmer continuum is enabled, and the host is enabled.
``agn_type=2`` is the narrow-line/host-dominated preset: broad built-in line
components are removed, narrow lines remain enabled, Fe II and the Balmer
continuum are disabled, and the host is enabled. The preset mutates ordinary
component switches, so call it before any explicit manual overrides that
should win:

.. code-block:: python

   cfg.set_agn_type(2)
   cfg.continuum.fit_feii = True  # explicit override after the preset

For fine-grained control, set the component switches directly.

Use the continuum flags to choose which non-line AGN components are fitted:

.. code-block:: python

   cfg.continuum.fit_feii = True
   cfg.continuum.fit_balmer_continuum = True

Fe II and the Balmer continuum add useful flexibility when the observed
wavelength range covers those features and the spectrum has enough signal to
constrain them. Leave one or both disabled for faster smoke tests, narrow
wavelength coverage, or low signal-to-noise spectra where the extra
components are poorly constrained.

The host-galaxy model is controlled separately:

.. code-block:: python

   cfg.host = HostConfig(enabled=True, dsps_ssp_fn="tempdata.h5")
   cfg.host = HostConfig(enabled=False)

Set ``enabled=False`` for quasar-dominated spectra or quick tests. When the
host is enabled, ``dsps_ssp_fn`` must point to a valid DSPS SSP HDF5 file; a
missing or wrong path is one of the most common setup failures.

The default host assumption is ``sfh_model="delayed"``, a delayed-:math:`\tau`
star-formation history. In this model the host star-formation rate has the
shape

.. math::

   \mathrm{SFR}(t) \propto t\,\exp(-t / \tau),

from the onset of star formation until the fitted population age. The fit
therefore samples a small set of physical host parameters such as
``sfh_age_gyr``, ``sfh_tau_gyr``, stellar mass, and metallicity. This is the
more stable default for most spectra.

Use ``sfh_model="flexible"`` when the spectrum has enough host signal to
support a less restrictive decomposition:

.. code-block:: python

   cfg.host = HostConfig(
       enabled=True,
       sfh_model="flexible",
       dsps_ssp_fn="tempdata.h5",
   )

The flexible model fits free SSP template weights across the configured age
and metallicity grid. It can absorb more detailed stellar-population
structure, but it is higher-dimensional and is easier to underconstrain when
host absorption features or wavelength coverage are weak. This is closest in
spirit to the traditional PyQSOFit-style host decomposition: the host
continuum is represented as a flexible mixture of stellar templates rather
than by a parametric star-formation history. The tradeoff is that the fitted
weights are less directly physical than the delayed-SFH age, tau, mass, and
metallicity parameters.

PSF photometry calibration
--------------------------

Optional PSF-aperture magnitudes add a broadband calibration likelihood on top
of the spectral likelihood. They help constrain gray flux-calibration offsets
and, when a host is enabled, help distinguish compact components from extended
components. The AGN continuum and broad lines are treated as unresolved; the
stellar host and narrow lines are multiplied by an aperture factor
:math:`\eta_{\rm PSF}`.

.. code-block:: python

   from jaxqsofit import PSFPhotometryData

   cfg.psf_photometry = PSFPhotometryData(
       filter_names=["u", "g", "r", "i", "z"],
       magnitudes=[18.9, 18.2, 17.9, 17.7, 17.6],
       magnitude_errors=[0.05, 0.03, 0.03, 0.03, 0.05],
   )

For a gray magnitude offset :math:`\Delta m_{\rm PSF}`, the scale factor is

.. math::

   s_{\rm PSF} = 10^{-0.4\,\Delta m_{\rm PSF}}.

The model spectrum compared to the PSF photometry is

.. math::

   f_{\lambda}^{\rm PSF}
   =
   s_{\rm PSF}
   \left[
   f_{\lambda}^{\rm AGN}
   + f_{\lambda}^{\rm broad}
   + \eta_{\rm PSF}
     \left(f_{\lambda}^{\rm host} + f_{\lambda}^{\rm narrow}\right)
   \right].

For each band :math:`b`, ``jaxqsofit`` computes a synthetic AB magnitude
:math:`m_b^{\rm syn}` from this PSF-space spectrum and applies

.. math::

   m_b^{\rm obs}
   \sim
   \mathcal{N}
   \left(
   m_b^{\rm syn},
   \sqrt{\sigma_{m,b}^2 + \sigma_{\rm phot,extra}^2}
   \right),

where :math:`\sigma_{m,b}` is the catalog magnitude uncertainty and
:math:`\sigma_{\rm phot,extra}` is an inferred extra photometric scatter term.
The PSF bands should overlap the observed spectral wavelength range. This is a
spectral recalibration constraint, not full broadband SED fitting; use
``jaxsedfit`` for full joint SED plus spectroscopy modeling.

Comparison with PyQSOFit
------------------------

``jaxqsofit`` follows the same broad decomposition idea as PyQSOFit: a smooth
AGN continuum, optional Fe II and Balmer-continuum components, a host-galaxy
continuum, and Gaussian emission-line complexes. The line-table fields also
keep the familiar PyQSOFit-style concepts of line names, component names,
velocity ties, width ties, and fixed flux-ratio ties.

The main difference is the modeling and inference backend. ``jaxqsofit`` uses
JAX/NumPyro, so the model can be optimized or sampled with differentiable
probabilistic inference, and posterior predictive draws are available for
component spectra and line measurements. This makes it natural to propagate
uncertainties through quantities such as broad-line FWHM and luminosity.
Because the continuum, host, Fe II, Balmer continuum, and line components are
fit jointly in one probabilistic model, ``jaxqsofit`` also avoids the
redshift-dependent systematics that can arise when the continuum is first fit
in a fixed set of rest-frame windows and then subtracted before line fitting.
In that sense it is a fully Bayesian alternative to a staged
window-continuum workflow.

Like PyQSOFit, instrumental-resolution correction is optional.  In
``jaxqsofit`` it is controlled by ``apply_instrumental_resolution`` and only
operates when ``resolving_power`` is supplied.  It is disabled by default, so
users who require intrinsic line widths should enable it explicitly.

The host model also differs. PyQSOFit-style host decomposition is closest to
``sfh_model="flexible"`` in ``jaxqsofit``: the host is represented as a
flexible mixture of stellar templates. The default ``sfh_model="delayed"``
instead imposes a physical delayed-:math:`\tau` star-formation history, giving
more interpretable parameters at the cost of a stronger assumption.

The PSF-photometry option is another extension beyond a pure spectrum-only
fit. It does not turn ``jaxqsofit`` into a full SED fitter, but it can use
overlapping PSF magnitudes to constrain flux calibration and compact-versus-
extended light. For full broadband SED plus spectroscopy modeling, use
``jaxsedfit``.

Prior configuration
-------------------

Build the default prior bundle from the spectrum with
:meth:`jaxqsofit.PriorConfig.from_spectrum`, then edit semantic prior sections
with ``numpyro.distributions`` objects:

.. code-block:: python

   import numpy as np
   import numpyro.distributions as dist

   from jaxqsofit import PriorConfig

   cfg.prior_config = PriorConfig.from_spectrum(flux=flux, redshift=z)
   cfg.prior_config.powerlaw.slope = dist.TruncatedNormal(
       loc=-1.5,
       scale=0.3,
       low=-3.5,
       high=0.5,
   )
   cfg.prior_config.fe.uv_norm = dist.LogNormal(
       loc=np.log(max(1e-3 * np.nanmedian(np.abs(flux)), 1e-10)),
       scale=0.04,
   )
   cfg.prior_config.host.aperture_scale = dist.Normal(loc=0.0, scale=0.5)

Do not pass flat ``{"dist": ...}`` dictionaries as public prior fields, and
do not use the old ``prior_config.overrides[...]`` style. The low-level model
still serializes priors internally, but user code should stay on the
``PriorConfig`` plus NumPyro-distribution interface.

Line-table customization
------------------------

The built-in tied-line model is seeded from
:data:`jaxqsofit.defaults.DEFAULT_LINE_PRIOR_ROWS`. Each row is a plain
dictionary. See :doc:`api/defaults` for the defaults API reference and the
`rendered defaults source
<https://jaxqsofit.readthedocs.io/en/latest/_modules/jaxqsofit/defaults.html>`__
for the complete line-list table values and helper functions, including
``build_default_bal_components``.

Optional line-list expansions are controlled when constructing the prior
configuration from a spectrum:

.. code-block:: python

   from jaxqsofit import PriorConfig

   cfg.prior_config = PriorConfig.from_spectrum(
       flux=flux,
       redshift=z,
       include_elg_narrow_lines=True,
       include_high_ionization_lines=True,
   )

``include_elg_narrow_lines=True`` appends
:data:`jaxqsofit.defaults.DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS`, a denser
narrow-line list for emission-line-galaxy and host-dominated spectra. It adds
features such as the resolved [O II] doublet, [Ne III], additional Balmer
narrow lines, He I, [O I], [N II], [S II], near-IR Paschen lines, and [S III].
Leave it off for ordinary broad-line quasar fits when the extra weak narrow
features are outside the wavelength range, not scientifically needed, or would
only add unconstrained amplitudes.

``include_high_ionization_lines=True`` appends
:data:`jaxqsofit.defaults.DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS`, a compact
set of high-ionization/coronal narrow lines such as [Ne V], [Fe VII], and
[Fe X]. Enable it for AGN spectra where those features are expected or visible;
otherwise keep it disabled to avoid extra weak-line parameters.

The most commonly edited row fields are:

``linename``
   Output component name used in metadata and plots. Names containing
   ``"_br"`` are treated as broad-line components; other built-in line names
   are treated as narrow components.

``compname``
   The line-complex name used to scope tie indices. Reusing a tie index in
   different complexes does not tie unrelated lines together.

``ngauss``
   Number of Gaussian components expanded from that row. The default table
   already uses multiple broad components for lines such as ``Ha_br``,
   ``Hb_br``, ``MgII_br``, ``CIV_br``, and ``Lya_br``.

Copy the default table before editing it, because the table is a module-level
list of mutable dictionaries:

.. code-block:: python

   from copy import deepcopy

   from jaxqsofit import PriorConfig
   from jaxqsofit.defaults import DEFAULT_LINE_PRIOR_ROWS

   line_table = deepcopy(DEFAULT_LINE_PRIOR_ROWS)
   for row in line_table:
       if row["linename"] == "Hb_br":
           row["ngauss"] = 3

   if cfg.prior_config is None:
       cfg.prior_config = PriorConfig()
   cfg.prior_config.lines.table = line_table

The tie columns follow PyQSOFit-style conventions, scoped by ``compname``:
``vindex`` ties velocity shifts, ``windex`` ties Gaussian widths, and
``findex`` ties amplitudes/flux ratios. Only positive tie indices create a
shared group; zero means the row is independent. Within a positive
``findex`` group, ``fvalue`` sets each component's fixed relative peak
amplitude. For rows with ``findex=0``, ``fvalue`` is only the initial/default
amplitude scale used to seed the independent amplitude prior.

Broad-line measurements
-----------------------

Individual Gaussian summary parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After fitting, ``q.line_result`` contains posterior-median parameters for every
individual Gaussian component, and ``q.line_result_name`` contains the
corresponding field names.  Pair them to make a convenient lookup mapping:

.. code-block:: python

   line = dict(zip(q.line_result_name, q.line_result))

   for component in ["Hb_br_1", "Hb_br_2"]:
       amplitude = line[f"{component}_scale"]
       amplitude_err = line[f"{component}_scale_err"]
       center_lnlam = line[f"{component}_centerwave"]
       center_lnlam_err = line[f"{component}_centerwave_err"]
       sigma_lnlam = line[f"{component}_sigma"]
       sigma_lnlam_err = line[f"{component}_sigma_err"]

       center_angstrom = np.exp(center_lnlam)
       velocity_kms = 299792.458 * (center_lnlam - np.log(4862.68))

       print(
           component,
           amplitude,
           amplitude_err,
           center_angstrom,
           velocity_kms,
           sigma_lnlam,
           sigma_lnlam_err,
       )

Components expanded from a row with ``ngauss > 1`` have names such as
``Hb_br_1`` and ``Hb_br_2`` (with an underscore before the component number).
For each component, the available summary fields are ``_scale``,
``_centerwave``, and ``_sigma``, together with their ``_err`` fields.  The
central values are posterior medians and the ``_err`` values are posterior
standard deviations.

Despite the ``_centerwave`` field name, centers are stored as natural-log
wavelength, ``ln(Angstrom)``, and ``_sigma`` is likewise a width in natural-log
wavelength.  Thus ``np.exp(center_lnlam)`` gives the rest-frame center in
Angstrom.  For a component with laboratory wavelength ``lambda0``, its
velocity offset is ``c * (center_lnlam - np.log(lambda0))``.  The small-width
Gaussian velocity dispersion is approximately ``c * sigma_lnlam``.

The meaning of this reported width depends on the spectroscopy configuration.
With ``apply_instrumental_resolution=True``, ``_sigma`` and
``line_sig_per_component`` are intrinsic, resolution-corrected widths.  With
the default ``False`` setting they describe the instrument-broadened observed
profile and are not corrected.  The effective forward-model widths are stored
separately in ``q.pred_out["line_sig_effective_per_component"]`` when
instrumental modeling is enabled.

These fields are compact median-and-error summaries.  Analyses that require
credible intervals, parameter covariances, or component velocity-separation
distributions should use the posterior draws described below instead.

After fitting, per-component line draws are available in ``q.pred_out``:

.. code-block:: python

   amp = np.asarray(q.pred_out["line_amp_per_component"])
   mu = np.asarray(q.pred_out["line_mu_per_component"])
   sig = np.asarray(q.pred_out["line_sig_per_component"])
   sig_effective = np.asarray(q.pred_out["line_sig_effective_per_component"])

For most workflows, use the helper methods that reconstruct a named broad-line
profile from those draws and then measure FWHM and integrated area:

.. code-block:: python

   from astropy.cosmology import FlatLambdaCDM

   cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

   def flux_to_luminosity(area_1e17, z):
       d_l_cm = cosmo.luminosity_distance(z).to("cm").value
       return area_1e17 * 1.0e-17 * 4.0 * np.pi * d_l_cm**2

   for line_key in ["CIV_br", "MgII_br", "Hb_br", "Ha_br"]:
       fwhm_draws = []
       log_l_draws = []
       for draw_index in range(amp.shape[0]):
           profile = q.line_profile_from_draw(draw_index, line_key)
           fwhm_kms, area_1e17 = q.line_props(profile)
           fwhm_draws.append(fwhm_kms)
           if np.isfinite(area_1e17) and area_1e17 > 0.0:
               log_l_draws.append(np.log10(flux_to_luminosity(area_1e17, q.z)))
           else:
               log_l_draws.append(np.nan)

       f16, f50, f84 = np.nanpercentile(fwhm_draws, [16, 50, 84])
       l16, l50, l84 = np.nanpercentile(log_l_draws, [16, 50, 84])
       print(line_key, f50, f84 - f50, f50 - f16, l50, l84 - l50, l50 - l16)

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
   q.config.inference.dense_mass = True
   q.config.inference.max_tree_depth = 8
   result = q.fit()
   components = result.predict(n_draws=200)

``dense_mass`` and ``max_tree_depth`` are passed directly to NumPyro's NUTS
kernel. For difficult posteriors, lowering ``max_tree_depth`` can cap runtime;
turning ``dense_mass`` off can reduce adaptation cost, but may require more
warmup for strongly correlated parameters.
