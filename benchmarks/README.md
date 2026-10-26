# jaxqsofit Benchmarks

These scripts are manual diagnostics for profiling and pull-request checks. They
are not unit tests, and the exact runtimes depend on the local JAX backend,
hardware, and installed dependency versions.

## Core likelihood profiler

Profile the compiled log-density and value+gradient paths:

```bash
python benchmarks/jaxqsofit_likelihood_profile.py --host-sfh both --n-wave 3690
```

Useful comparison switches:

```bash
python benchmarks/jaxqsofit_likelihood_profile.py --host-sfh delayed --no-cache
python benchmarks/jaxqsofit_likelihood_profile.py --host-sfh flexible --convolution direct
```

The profiler uses a synthetic rest-frame spectrum, excludes JIT compilation from
the reported timings, and prints the active line count, line-group dimensions,
host template matrix shape, parameter count, forward time, and value+gradient
time for several likelihood variants.

## Chimera spectral validation

`chimera_spectra_validation.py` fits the composite Chimera spectra (a COSMOS host
galaxy spectrum from DESI DR1 or LEGA-C plus a weighted SDSS DR7 quasar spectrum)
with `jaxqsofit.fit_batch`, spectrum only, and compares the recovered values with
the truths:

| Quantity | Fitted (median of Laplace draws) | Truth |
|---|---|---|
| Stellar mass | `log_stellar_mass` | COSMOS2015 LePhare mass of the host |
| AGN bolometric luminosity | power-law λLλ × Shen et al. (2011) correction (9.26 at 5100 Å below z = 0.7, 5.15 at 3000 Å below z = 1.9, else 3.81 at 1350 Å) | Shen et al. (2011) `logLbol` + log10(QSO weight) |
| SFR | current SFR of the delayed-τ SFH | COSMOS2015 LePhare SFR (star-forming hosts only) |
| Hα, Hβ, Mg II FWHM and flux | sum of each line's broad and narrow components | Wu & Shen (2022) whole-line values; flux × QSO weight × the quasar's photometric scale |

The spectra and the photometric truths come from jaxsedfit's Chimera validation
sample (`benchmarks/data/chimera_validation_spectra.fits` and
`chimera_validation_sample.csv`); the workflow downloads them from fixed jaxsedfit
commit. The host aperture scale is fixed at one, because the spectra are scaled
to the total photometry. The fits use `fit_batch(method="laplace")`: every quantity
is computed per draw, the point estimate is the median and the interval spans the
2.3rd to 97.7th percentiles. The tables also give the fraction of truths inside
that interval; Laplace errors are statistical only, so expect it below 95%. Spectra are fitted in redshift-ordered slices (`--slice`, default 32) and
each slice's fitters are released before the next, because a fitted `JAXQSOFit`
holds its host template grid and predictive draws (about 80 MB). With
`--isolate-slices`, as in CI, each slice runs in its own process: within one
process the memory floor still rose from slice to slice on the 16 GB runners.

```bash
python benchmarks/chimera_spectra_validation.py run \
  --output-dir /tmp/chimera-spectra-validation \
  --dsps-ssp-fn tempdata.h5 \
  --spectra <jaxsedfit>/benchmarks/data/chimera_validation_spectra.fits \
  --sample <jaxsedfit>/benchmarks/data/chimera_validation_sample.csv   # add --limit 16 for a quick look
```

`data/chimera_line_truth.csv` holds the Wu & Shen (2022) whole-line (broad +
narrow) FWHM and flux of Hα, Hβ and Mg II for the sample quasars, matched within
1″ to the position in their SDSS name (226 of 227). Wu & Shen measured 203 of them
on the same SDSS DR7 spectrum the Chimeras use and 23 on a later SDSS spectrum, so
line fluxes of those 23 can differ through variability. Their broad-line widths and
EWs agree with Shen et al. (2011) to within 0.05 dex in the median. Rebuild it only when the
sample changes:

```bash
python benchmarks/chimera_spectra_validation.py make-truth \
  --catalog dr16q_prop_May01_2024.fits \
  --spectra <jaxsedfit>/benchmarks/data/chimera_validation_spectra.fits
```

`.github/workflows/chimera-spectra-validation.yml` runs this on pull requests that
touch `src/` (and on manual dispatch), uploads the plots and, for pull requests
from branches of this repository, creates or updates a PR comment with the metrics
tables and embedded plots. Nothing is gated yet.
