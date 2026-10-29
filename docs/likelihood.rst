Likelihood, calibration and priors
==================================

Spectral likelihood
-------------------

The spectrum enters through a Student-t likelihood, which is robust to outliers
and unmodeled absorption. ``cfg.prior_config.student_t_df`` sets its degrees of
freedom (3 by default); lower values are more robust.

The quoted errors are not assumed to be correct. Three terms are added to them in
quadrature:

* ``frac_jitter`` times the model, fitted (prior HalfNormal(0.02));
* ``frac_fe_jitter`` times the Fe II model, fixed at 0.20, for template mismatch;
* ``add_jitter``, a constant, fixed at 0 by default.

Flux calibration
----------------

``cfg.continuum.fit_polynomial_tilt`` (on by default) multiplies the whole model by
a low-order polynomial, to absorb residual spectrophotometric calibration errors.
``cfg.continuum.polynomial_order`` (default 2) is its highest degree. The constant
and linear terms are left out because they duplicate the power-law normalization
and slope, so orders below two add nothing.

PSF photometry
--------------

PSF magnitudes can add a broadband calibration likelihood to the spectral one.
They constrain gray flux-calibration offsets and, with a host, help separate the
compact components from the extended ones. The AGN continuum and broad lines are
treated as unresolved; the host and narrow lines are scaled by an aperture factor
:math:`\eta_{\rm PSF}`.

.. code-block:: python

   from jaxqsofit import PSFPhotometryData

   cfg.psf_photometry = PSFPhotometryData(
       filter_names=["u", "g", "r", "i", "z"],
       magnitudes=[18.9, 18.2, 17.9, 17.7, 17.6],
       magnitude_errors=[0.05, 0.03, 0.03, 0.03, 0.05],
   )

With a gray magnitude offset :math:`\Delta m_{\rm PSF}`, the scale factor is
:math:`s_{\rm PSF} = 10^{-0.4\,\Delta m_{\rm PSF}}`, and the model compared with the
PSF photometry is

.. math::

   f_{\lambda}^{\rm PSF}
   = s_{\rm PSF}
   \left[
   f_{\lambda}^{\rm AGN} + f_{\lambda}^{\rm broad}
   + \eta_{\rm PSF} \left(f_{\lambda}^{\rm host} + f_{\lambda}^{\rm narrow}\right)
   \right].

For each band :math:`b`, ``jaxqsofit`` computes a synthetic AB magnitude
:math:`m_b^{\rm syn}` from this spectrum and uses

.. math::

   m_b^{\rm obs} \sim \mathcal{N}
   \left( m_b^{\rm syn}, \sqrt{\sigma_{m,b}^2 + \sigma_{\rm phot,extra}^2} \right),

where :math:`\sigma_{m,b}` is the catalog error and
:math:`\sigma_{\rm phot,extra}` a fitted extra scatter. The PSF bands should overlap
the spectrum. This is a spectral recalibration, not SED fitting; use ``jaxsedfit``
to fit broadband photometry and spectra together. The PSF recalibration tutorial
(:doc:`tutorials`) shows a complete example. :func:`jaxqsofit.fit_batch` does not
support PSF photometry yet.

Priors
------

If ``cfg.prior_config`` is ``None``, default priors are built from the spectrum's
flux scale. To change them, build the defaults with
:meth:`jaxqsofit.PriorConfig.from_spectrum` and edit its sections with
``numpyro.distributions`` objects:

.. code-block:: python

   import numpy as np
   import numpyro.distributions as dist

   from jaxqsofit import PriorConfig

   cfg.prior_config = PriorConfig.from_spectrum(flux=flux, redshift=z)
   cfg.prior_config.powerlaw.slope = dist.TruncatedNormal(
       loc=-1.5, scale=0.3, low=-3.5, high=0.5
   )
   cfg.prior_config.fe.uv_norm = dist.LogNormal(
       loc=np.log(max(1e-3 * np.nanmedian(np.abs(flux)), 1e-10)), scale=0.04
   )
   cfg.prior_config.fe.op_over_uv = dist.Normal(0.0, 0.4)
   cfg.prior_config.host.aperture_scale = dist.Normal(loc=0.0, scale=0.5)
   cfg.prior_config.lines.amp_scale_mult = 0.20
   cfg.prior_config.student_t_df = 2.5

A custom ``prior_config`` must configure the sections of every enabled component,
and the line model needs a table in ``cfg.prior_config.lines.table`` (see
:doc:`lines`). Do not pass flat ``{"dist": ...}`` dictionaries as prior fields, or
use the old ``prior_config.overrides[...]`` style: the model serializes priors
internally, but user code should use ``PriorConfig`` and NumPyro distributions. The
LaTeX table of all default priors is linked from :doc:`default_line_priors`.
