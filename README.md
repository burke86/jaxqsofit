# JaxQSOFit

<p align="center">
  <img src="logo.png" alt="JaxQSOFit logo" width="220">
</p>

Bayesian quasar spectral fitting with JAX + NumPyro, including:

- AGN continuum (power law)
- Host galaxy decomposition from DSPS SSP templates
- FeII UV/optical templates
- Balmer continuum
- Tied Gaussian emission-line model
- Student-t likelihood (robust to outliers/absorption)

The main implementation is in [`src/jaxqsofit/core.py`](./src/jaxqsofit/core.py).

## Key Differences vs PyQSOFit

JaxQSOFit is designed around a **joint Bayesian model** of AGN and host components, rather than fitting them in separate stages.

- **Joint host+AGN inference**:
  Host galaxy SPS (DSPS SSP mixture + LOSVD) is fit simultaneously with AGN continuum and lines.
- **More robust for complex spectra**:
  This joint treatment reduces AGN/host degeneracies and yields more stable parameters when spectra have strong blending, absorption, or mixed host contamination.
- **Probabilistic uncertainties**:
  NumPyro/NUTS produces posterior distributions for continuum, host, Fe, Balmer, and line parameters in one consistent model.

## Requirements

- Python 3.10+
- `jax`, `jaxlib`
- `numpyro`
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `astropy`
- `astroquery`
- `extinction`
- `dsps`
- `dustmaps`

## Install

### 1. Create/activate environment

```bash
conda create -n jaxqsofit python=3.12 -y
conda activate jaxqsofit
```

### 2. Install dependencies

CPU example:

```bash
pip install -U pip
pip install numpy scipy matplotlib pandas astropy astroquery extinction dustmaps dsps numpyro
pip install "jax[cpu]"
```

If you want GPU JAX, install `jax`/`jaxlib` following official JAX instructions for your CUDA setup.

### 3. Provide DSPS SSP templates

This code expects an HDF5 SSP file path passed as `dsps_ssp_fn`, e.g. `tempdata.h5`.

Current default in code/notebook examples:

```python
dsps_ssp_fn="tempdata.h5"
```

### 4. Configure dustmaps SFD (one-time)

This repo assumes `dustmaps` is already configured and SFD maps are available.

Typical one-time setup:

```
python setup.py fetch --map-name=sfd
```

## Repo layout

- `src/jaxqsofit/core.py` – fitting code
- `src/jaxqsofit/defaults.py` – default prior/line configs
- `src/jaxqsofit/__init__.py` – package exports
- `tests/` – test directory
- `test.ipynb` – development/example notebook
- `fe_uv.txt`, `fe_optical.txt` – Fe templates
- `tempdata.h5` – DSPS SSP template file (example)
- `data/spec-0332-52367-0639.csv` – example spectrum (`loglam`, `flux`, `ivar`)
- `data/spec-0332-52367-0639-meta.csv` – example metadata (`z`, `ra`, `dec`, etc.)

## Minimal usage

```python
import numpy as np
import jaxqsofit
from astroquery.sdss import SDSS
from astropy import units as u
from astropy.coordinates import SkyCoord

# Example: fetch SDSS spectrum for NGC 5548
coord = SkyCoord.from_name("NGC 5548")
xid = SDSS.query_region(coord, spectro=True, radius=5 * u.arcsec)
sp = SDSS.get_spectra(matches=xid)[0]

tb = sp[1].data
lam = 10 ** tb["loglam"]                 # observed-frame wavelength [A]
flux = tb["flux"]                        # f_lambda
err = 1.0 / np.sqrt(tb["ivar"])          # 1-sigma

# Prefer SDSS pipeline redshift if available, else supply your own z
z = float(sp[2].data["z"][0])

q = jaxqsofit.QSOFit(
    lam, flux, err, z=z,
    ra=float(coord.ra.deg), dec=float(coord.dec.deg),
    plateid=int(sp[0].header.get("plateid", 0)),
    mjd=int(sp[0].header.get("mjd", 0)),
    fiberid=int(sp[0].header.get("fiberid", 0)),
    path="."
)

q.Fit(
    deredden=True,
    fit_lines=True,
    decompose_host=True,
    fit_fe=False,
    fit_bc=True,
    fit_poly=False,
    dsps_ssp_fn="tempdata.h5",
    nuts_warmup=300,
    nuts_samples=600,
    nuts_chains=1,
    plot_fig=True,
    save_fig=False,
)
```

Optional: override any defaults by passing `prior_config`:

```python
prior_config = {
    "student_t_df": 2.5,
    "PL_slope": {"loc": -1.5, "scale": 0.3, "low": -3.5, "high": 0.3},
}
q.Fit(prior_config=prior_config, fit_lines=False, fit_fe=False, fit_bc=True)
```

## Important API notes

- If `prior_config=None`, defaults are auto-built from `src/jaxqsofit/defaults.py` using the input flux scale.
- If you pass a custom `prior_config`, ensure required keys exist for enabled model components.
- `fit_lines=True` requires a line prior table in:
  - `prior_config["line"]["table"]` (preferred), or
  - `prior_config["line_priors"]`, or
  - `prior_config["line_table"]`.
- `fit_fe=False`, `fit_bc=False`, `fit_poly=False`, `decompose_host=False` disable those model blocks.
- Likelihood is Student-t:
  - `prior_config["student_t_df"]` controls tail heaviness.
  - Lower `df` is more robust to outliers.

## Outputs on `QSOFit` object

Common fitted arrays:

- `wave`, `flux`, `err`
- `model_total`
- `f_conti_model`, `f_line_model`
- `f_pl_model`, `f_fe_mgii_model`, `f_fe_balmer_model`, `f_bc_model`
- `host`, `qso`

Posterior artifacts:

- `numpyro_mcmc`, `numpyro_samples`
- `pred_out`
- `_pred_host_draws`, `_pred_bc_draws`, `_pred_cont_draws`

Derived fractions:

- `frac_host_4200`, `frac_host_5100`, `frac_host_2500`
- `frac_bc_2500`

## Troubleshooting

- `SFDQuery` errors:
  - Ensure dustmaps data are downloaded and `config['data_dir']` is set.
- DSPS load errors:
  - Confirm `dsps_ssp_fn` points to a valid SSP HDF5 file.
- Line amplitudes explode:
  - Tighten line prior ranges (`maxsca`, `maxsig`) and scale multipliers.
- Fe fit degrades continuum:
  - Use stronger shrinkage priors on Fe norms and narrower Fe shift/FWHM priors.
