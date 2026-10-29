Troubleshooting
===============

``SFDQuery`` errors
   The SFD dust maps are missing or ``dustmaps`` cannot find them. Download them
   and set ``config["data_dir"]`` as in :doc:`installation`.

DSPS load errors
   ``HostConfig.dsps_ssp_fn`` must point to a valid DSPS SSP HDF5 file (see
   :doc:`installation`).

Line amplitudes run away
   Tighten the line-table bounds (``maxsca``, ``maxsig``; see :doc:`lines`) and the
   prior scale multipliers such as ``cfg.prior_config.lines.amp_scale_mult``.

Fe II distorts the continuum
   Use stronger shrinkage priors on the Fe II normalizations and narrower priors on
   the Fe II shift and FWHM (see :doc:`likelihood`), or turn Fe II off for spectra
   that do not constrain it.

BAL troughs are overfit
   Reduce ``BALConfig.tau_scale`` or ``BALConfig.covering_high``, or leave
   ``BALConfig.enabled=False`` for spectra without BALs.

NUTS reaches the maximum tree depth
   See the NUTS tuning notes in :doc:`inference`.

Laplace uncertainties look wrong
   Check ``q.approx_result["hessian_min_eigenvalue"]``; values below one flag
   unreliable directions (see :doc:`inference`).
