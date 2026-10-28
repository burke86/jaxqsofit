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
| Broad Hα, Hβ, Mg II FWHM and flux | sum of each line's broad components | Wu & Shen (2022) broad-line values; flux × QSO weight × the quasar's photometric scale |

The spectra and the photometric truths come from jaxsedfit's Chimera validation
sample (`benchmarks/data/chimera_validation_spectra.fits` and
`chimera_validation_sample.csv`); the workflow downloads them from fixed jaxsedfit
commit. The host aperture scale is fixed at one, because the spectra are scaled
to the total photometry. The fits use `fit_batch(method="laplace")`: every quantity
is computed per draw, the point estimate is the median and the interval spans the
2.3rd to 97.7th percentiles. The tables also give the fraction of truths inside
that interval; Laplace errors are statistical only, so expect it below 95%. Spectra are fitted one redshift window at a time, the windows `fit_batch` forms
(at most `--slice` spectra, default 128), and each slice's fitters are released
before the next, because `fit_batch` keeps every fitter until it returns (about
16 MB each at 50 draws). Each window compiles its own programs anyway, so this
adds no compiles. With `--isolate-slices`, as in CI, each slice runs in its own
process: within one process the memory floor still rose from slice to slice on
the 16 GB runners.

```bash
python benchmarks/chimera_spectra_validation.py run \
  --output-dir /tmp/chimera-spectra-validation \
  --dsps-ssp-fn tempdata.h5 \
  --spectra <jaxsedfit>/benchmarks/data/chimera_validation_spectra.fits \
  --sample <jaxsedfit>/benchmarks/data/chimera_validation_sample.csv   # add --limit 16 for a quick look
```

Narrow lines are left out of the line comparison: a strong narrow peak sets the
half maximum of a whole line, and the host's own narrow lines are not in the
quasar truth. jaxqsofit's broad Gaussians have σ ≥ 0.004 in ln λ (FWHM ≥ 2,800
km/s), so narrower Wu & Shen broad lines cannot be matched.

`data/chimera_line_truth.csv` holds the Wu & Shen (2022) broad-line FWHM and
flux of Hα, Hβ and Mg II for the sample quasars, matched within
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

`report` turns a run into one compact Markdown report: a table of RMS · outlier
fraction for each quantity and broad-line measurement in each luminosity bin, with
the change from main's metrics when `--baseline` is given, and the per-bin statistics
and plots behind a fold. It exits non-zero if a bin with at least 10 objects exceeds
its limits in `chimera_spectra_validation_thresholds.json` (set from a reference run
as RMS × 1.15 and outlier fraction + 0.05; update them in the same PR as an intended
change to the recovery). Smaller bins are reported but not gated.

```bash
python benchmarks/chimera_spectra_validation.py report \
  --results /tmp/chimera-spectra-validation --output /tmp/chimera-spectra-validation/report.md
```

`.github/workflows/chimera-spectra-validation.yml` runs this on pull requests that
touch `src/`, on pushes to main and on manual dispatch. Runs on main store their
`metrics.json` and plots under `main/` on the `chimera-spectra-validation-plots`
branch; pull requests from branches of this repository get one comment showing
their change from main, with plots under `pr-<number>/<sha>/`, and the job fails
past a limit.

## PR performance benchmark

`sdss_pr_benchmark.py` fits the SDSS spectrum from the main tutorial with a 200-step
MAP fit (host + lines) and times the jit-compiled value and gradient of that fit's log
density at the MAP, the function every optimizer and sampler step evaluates. `fetch`
downloads the spectrum once; `measure` fits and times the `jaxqsofit` importable in its
process and writes one JSON record; `report` combines base and PR records into a short
comment of value + gradient time, compile time and the final MAP loss (a model-change
flag), and exits non-zero if value + gradient is more than 10% slower beyond 3σ of the
run-to-run scatter, or compile is more than 25% slower.

`.github/workflows/pr-benchmarks.yml` alternates five base and five PR processes on one
runner, each with its side's `src` first on `PYTHONPATH` (both use the PR's jaxsedfit
pin), so drifts of the shared machine cancel; `publish-pr-benchmarks.yml` posts the
comment, also when the benchmark failed.

```bash
python benchmarks/sdss_pr_benchmark.py fetch --output /tmp/bench/spectrum.npz
for i in 1 2 3; do
  PYTHONPATH=../jaxqsofit-main/src python benchmarks/sdss_pr_benchmark.py measure \
    --spectrum /tmp/bench/spectrum.npz --sha main --output /tmp/bench/base-$i.json
  PYTHONPATH=src python benchmarks/sdss_pr_benchmark.py measure \
    --spectrum /tmp/bench/spectrum.npz --sha pr --output /tmp/bench/pr-$i.json
done
python benchmarks/sdss_pr_benchmark.py report --base "/tmp/bench/base-*.json" --pr "/tmp/bench/pr-*.json" \
  --output /tmp/bench/output
```
