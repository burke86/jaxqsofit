Fitting many objects
====================

:func:`jaxqsofit.fit_batch` fits many spectra that share a model configuration
with a single compiled program. Fitting spectra one at a time with
:meth:`jaxqsofit.JAXQSOFit.fit` compiles and optimizes each separately.
``fit_batch`` compiles once per group of compatible spectra and fits a whole chunk
of them together with ``jax.vmap``, using ``jaxsedfit.batched``. Use it when you
have a catalog of spectra.

Supported methods are ``"map"`` (``"optax"`` is accepted) and ``"laplace"``, the
batched counterparts of the single-spectrum ``"optax"`` and ``"laplace"`` (see
:doc:`inference`). For NUTS, loop over spectra with :meth:`jaxqsofit.JAXQSOFit.fit`.
PSF photometry is not supported yet.

Example
-------

Configure one :class:`jaxqsofit.JAXQSOFit` per spectrum, usually from catalog rows
with a small helper, then pass the list to ``fit_batch``:

.. code-block:: python

   from jaxqsofit import JAXQSOFit, fit_batch

   def make_fitter(row):
       q = JAXQSOFit.from_arrays(
           lam=row["wave"], flux=row["flux"], err=row["err"], z=row["z"],
           ra=row["ra"], dec=row["dec"], filename=str(row["id"]),
           resolving_power=2000.0, apply_instrumental_resolution=True,
       )
       cfg = q.config
       cfg.host.dsps_ssp_fn = "tempdata.h5"
       cfg.inference.map_steps = 2000
       for key in ("plot_fig", "save_fig", "save_result", "show_plot"):
           setattr(cfg.output, key, False)
       return q

   fitters = [make_fitter(row) for row in catalog]
   results = fit_batch(fitters, method="laplace", chunk_size=4, num_samples=100)

   result = results[0]                       # a FitResult, as from JAXQSOFit.fit()
   result.samples["PL_slope"]                # Laplace draws for the first spectrum
   result.spectrum.line_groups["Hb_br"]      # grouped line measurements per draw
   fitters[0].optax_map_point                # its MAP
   fitters[0].batch_diagnostics              # loss and hessian_min_eigenvalue

``fit_batch`` returns one :class:`jaxqsofit.FitResult` per fitter, in input order,
and leaves each fitter in its fitted state, as after :meth:`jaxqsofit.JAXQSOFit.fit`. The MAP
settings (``map_steps``, ``learning_rate``, ``random_seed``) come from each fitter's
``config.inference``; the MAP is staged as for ``"optax"``.

What can be batched
-------------------

Spectra are grouped automatically. Two spectra share a compiled program when their
configurations agree on everything except per-object data:

* the spectrum itself (wavelengths, fluxes, errors), its redshift and sky position;
* prior values scaled from the data, such as the default amplitude priors.

Everything else defines the group: the preprocessing, continuum, AGN, BAL, host,
line and inference settings, the structure of the prior configuration, the
instrumental-resolution switches and any custom components. A batch can mix
settings, but each distinct combination compiles its own program, so batching only
pays off when many spectra share one.

Two things make the spectral model differ between spectra, and ``fit_batch``
removes both:

* **Pixel counts.** Every spectrum in a group is padded to the group's longest with
  masked pixels (``pixel_mask``) that continue its log-wavelength grid. Masked
  pixels leave the likelihood unchanged.
* **The tied-line table.** It normally contains the lines a spectrum covers, so it
  changes with redshift. Spectra are instead split into redshift windows no wider
  than ``redshift_window`` in ln(1 + z) (default 0.25). Each window shares one table
  spanning its members' rest-frame coverage, and lines outside a spectrum's own
  coverage get zero amplitude (``tied_line_meta["line_mask"]``), so only their
  priors constrain them. Wider windows mean fewer compiled programs but more masked
  lines per spectrum; ``redshift_window=None`` gives each configuration group one
  table.

Compared with :meth:`jaxqsofit.JAXQSOFit.fit`, a batched fit can therefore differ slightly:
a tied group that straddles the coverage edge keeps all its members, with the
uncovered ones masked, and the first (continuum) MAP stage masks the strong-line
windows instead of removing those pixels. The batched model otherwise evaluates the
same log density as the single-spectrum model, to numerical precision.

Results
-------

Each returned :class:`jaxqsofit.FitResult` works like one from
:meth:`jaxqsofit.JAXQSOFit.fit`:

* ``result.samples``: draws of every parameter. For ``"map"`` there is a single
  draw, the MAP; for ``"laplace"`` there are ``num_samples``.
* ``result.method``: ``"optax"`` or ``"laplace"``.
* ``result.spectrum``, ``result.predict(...)``, ``result.save(...)`` and the plots,
  computed from those draws.
* ``fitter.optax_map_point``: the MAP of every parameter (both methods start from
  it).
* ``fitter.batch_diagnostics``: ``loss`` (the negative log posterior at the MAP)
  and, for Laplace, ``hessian_min_eigenvalue``, which is also in
  ``fitter.approx_result`` as after a single ``"laplace"`` fit. Values below one
  mark directions with unreliable Laplace uncertainties (see :doc:`inference`).

Random starting points come from ``seed`` (default: the first fitter's
``inference.random_seed``) and each spectrum's position in ``fitters``.

Performance and memory
----------------------

``chunk_size`` spectra are fitted together (default 16). As a guide, for Chimera
spectra of about 3,300 pixels fitted with the host, Fe II, Balmer continuum and
emission lines (2,000 MAP steps, the Laplace Hessian and 50 draws) on an 8-core
Apple M1 Pro laptop:

* each redshift window compiles two programs once: the fit (40–50 s) and the
  posterior predictive that fills in each fitter's results (3–9 s, longer for
  larger chunks);
* once compiled, the fit takes about 11 s per spectrum in chunks of 4, 8 or 16, and
  about 14 s per spectrum with ``chunk_size=1``. On CPU, chunks larger than 4 do not
  run faster per spectrum;
* filling in the results takes under 1 s per chunk;
* for comparison, :meth:`jaxqsofit.JAXQSOFit.fit` with ``method="laplace"`` takes
  about 90 s per spectrum, most of it compiling, and compiles again for every
  spectrum.

The compiled programs are reused for every chunk of a window, so larger windows and
catalogs amortize the compile time. On CPU they are compiled without two XLA
passes (``constant_folding`` and ``simplify-while-loops``) that otherwise dominate
the compile; see ``jaxsedfit.batched.BATCH_COMPILER_OPTIONS``. On a 4-core GitHub
runner, the 241-spectrum Chimera validation (chunks of 4, 50 draws) takes about
80 minutes, about 20 s per spectrum including compiles, and peaks at 7 GB.

Each fitted :class:`jaxqsofit.JAXQSOFit` keeps its predictive draws, about 0.3 MB per
draw with the default configuration: 17 MB at 50 draws and 130 MB at the default
``num_samples=400``. ``fit_batch`` keeps every fitter until it returns, and clears
jaxsedfit's per-spectrum caches after each chunk. Batched results do not keep the per-draw line profiles (line results
come from the component amplitudes, centers and widths) or the host template grid
(``predict`` rebuilds it). For large catalogs, call ``fit_batch`` one redshift
window at a time and keep only the quantities you need from each call; that adds
no compiles, since each window compiles its own programs anyway.
