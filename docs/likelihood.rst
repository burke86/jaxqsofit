Likelihood, calibration and priors
==================================

Besides the physical model, a fit includes terms for flux calibration, apertures and
spectral errors. This table lists them all; the sections below and the linked pages
describe each one.

.. list-table::
   :header-rows: 1
   :widths: 24 32 20 24

   * - Term
     - What it does
     - Default
     - Where
   * - Student-t likelihood
     - Robustness to outlying pixels and unmodeled absorption
     - df = 3
     - below
   * - Spectral jitter
     - Extra error: a fraction of the model and of Fe II, plus a constant
     - fitted fraction
     - below
   * - Calibration polynomial
     - Smooth multiplicative correction to the spectrum's shape
     - on, order 2
     - below
   * - PSF photometry
     - Gray offset and host fraction from PSF magnitudes
     - only with ``psf_photometry``
     - below
   * - Host aperture scale
     - Fraction of the host's light in the fiber or slit
     - fixed at 1
     - below
   * - Instrumental resolution
     - Line-spread function, so widths are intrinsic
     - off
     - :doc:`model`
   * - Milky Way extinction
     - Dereddening with the SFD maps (``Observation.apply_mw_deredden``)
     - on (needs ``ra``, ``dec``)
     - :doc:`installation`

Intrinsic AGN reddening is part of the physical model, not a calibration term; see
:doc:`model`.

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

``cfg.continuum.fit_polynomial_tilt`` (on by default) multiplies the **whole** model,
emission lines included, by a smooth positive curve, to absorb residual
spectrophotometric calibration errors:

.. math::

   P(\lambda) = \exp\Big( \sum_{k=2}^{K} c_k \, \hat{x}^k \Big),
   \qquad x = \frac{\lambda - \lambda_0}{\lambda_0},
   \qquad c_k \sim \mathcal{N}(0, 0.03),

where :math:`\lambda_0` is the center of the fitted (rest-frame) wavelength range,
each power :math:`\hat{x}^k` is normalized to a maximum of one over the spectrum, and
:math:`K` is ``cfg.continuum.polynomial_order`` (default 2, so one coefficient,
``poly_c2``). The constant and linear terms are left out because they duplicate the
power-law normalization and slope, so the polynomial only adds curvature, and orders
below two add nothing. The coefficients' prior allows corrections of a few percent.
``cfg.prior_config.continuum.polynomial_pivot`` sets :math:`\lambda_0` in Å. The
fitted coefficients are ``poly_c2``, ``poly_c3``, ..., and the fitted curve is in the
predictive output ``f_poly_model``. Turn it off
(``fit_polynomial_tilt=False``) for spectra whose shape you trust, or when its
curvature is degenerate with a component you want to measure, such as a weak Balmer
continuum.

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
:math:`\sigma_{\rm phot,extra}` a fitted extra scatter. The priors are fixed:
:math:`\Delta m_{\rm PSF} \sim \mathcal{N}(0, 0.5)` mag,
:math:`\eta_{\rm PSF} \sim {\rm Beta}(2, 2)` (only when the host is fitted; otherwise
:math:`\eta_{\rm PSF} = 1`), and :math:`\sigma_{\rm phot,extra} \sim
{\rm HalfNormal}(0.05)` mag. The fitted values are ``delta_m_psf``, ``eta_psf``,
``scale_psf`` and ``sigma_phot_extra``. These terms enter only the PSF likelihood:
the spectrum itself is not rescaled, but the PSF magnitudes constrain the shared
AGN and host components. The PSF bands should overlap
the spectrum. This is a spectral recalibration, not SED fitting; use ``jaxsedfit``
to fit broadband photometry and spectra together. The PSF recalibration tutorial
(:doc:`tutorials`) shows a complete example. :func:`jaxqsofit.fit_batch` does not
support PSF photometry yet.

Host aperture scale
-------------------

A fiber or slit spectrum usually contains only part of the host galaxy. The physical
host spectrum is multiplied by an aperture scale, which by default is fixed at 1: the
host parameters, such as the stellar mass, then describe the light inside the
aperture. To let them describe the whole galaxy, set a prior on the scale, for example
from an aperture correction:

.. code-block:: python

   import numpyro.distributions as dist

   cfg.prior_config.host.aperture_scale = dist.Normal(loc=-0.5, scale=0.3)

Despite its name, ``host.aperture_scale`` is the prior on the **natural log** of the
scale (output ``log_host_aperture_scale``): ``loc=-0.5`` means about 60% of the
galaxy's light. Without an external constraint the scale is degenerate with the
stellar mass, so give it an informative prior.

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
   cfg.prior_config.lines.amp_scale_mult = 0.20
   cfg.prior_config.student_t_df = 2.5

A custom ``prior_config`` must configure the sections of every enabled component,
and the line model needs a table in ``cfg.prior_config.lines.table`` (see
:doc:`lines`). Do not pass flat ``{"dist": ...}`` dictionaries as prior fields, or
use the old ``prior_config.overrides[...]`` style: the model serializes priors
internally, but user code should use ``PriorConfig`` and NumPyro distributions. The
LaTeX table of all default priors is linked from :doc:`default_line_priors`.
