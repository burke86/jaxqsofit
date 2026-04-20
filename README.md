# JAXQSOFit

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
- Low-order multiplicative polynomial basis for spectrophotometric calibration errors
- Uses flexible Gaussian radial basis functions to correct errors near the spectral edges
- Additive and multiplicative intrinsic scatter
- User-specified custom components (e.g., alternative iron template, custom continuum component, custom non-Gaussian line component)
- Optional: fit PSF magnitudes to help constrain host (since large PSF-fiber mag offset implies high host) and slit losses

## Start Here: Tutorial

- GitHub view: [`notebooks/01_jaxqsofit_tutorial.ipynb`](./notebooks/01_jaxqsofit_tutorial.ipynb)

## Documentation

- [Read the Docs](https://jaxqsofit.readthedocs.io/)

> #### `JAXQSOFit` is under active development, breaking API changes are expected.

## Citation

If you use JAXQSOFit in published work, please cite:

- Shen et al. (2019), *ApJS*, 241, 34:
  `https://ui.adsabs.harvard.edu/abs/2019ApJS..241...34S/abstract`
- Hearin et al. (2023), *MNRAS*, 521, 1741 (DSPS):
  `https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.1741H/abstract`
- Green (2018), *JOSS*, 3, 695 (dustmaps):
  `https://ui.adsabs.harvard.edu/abs/2018JOSS....3..695G/abstract`

## Key Differences vs PyQSOFit

JAXQSOFit is designed around a **joint Bayesian model** of AGN and host components, rather than fitting them in separate stages.

- **Joint host+AGN inference**:
  Host galaxy SPS (DSPS SSP mixture + LOSVD) is fit simultaneously with AGN continuum and lines.
- **More robust for complex spectra**:
  This joint treatment reduces AGN/host degeneracies and yields more stable parameters when spectra have strong blending, absorption, or mixed host contamination.
- **Probabilistic uncertainties**:
  NumPyro/NUTS produces posterior distributions for continuum, host, Fe, Balmer, and line parameters in one consistent model.

## Requirements

- Python 3.10+
- `jax`, `jaxlib`
- `jax_cosmo`
- `numpyro`
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `astropy`
- `astroquery`
- `extinction`
- `dsps` ([GitHub](https://github.com/ArgonneCPAC/dsps))
- `dustmaps` ([GitHub](https://github.com/gregreen/dustmaps))

## Install

### 1. Create/activate environment

```bash
conda create -n jaxqsofit python=3.12 -y
conda activate jaxqsofit
```

### 2. Install dependencies

CPU example:

```bash
pip install setuptools numpy scipy matplotlib pandas astropy astroquery extinction dustmaps dsps numpyro jax_cosmo
pip install "jax[cpu]"
```

If you want GPU JAX, install `jax`/`jaxlib` following official JAX instructions for your CUDA setup.

### 3. Provide DSPS SSP templates

This code expects an HDF5 SSP file path passed as `dsps_ssp_fn`, e.g. `tempdata.h5`.

```
curl -L -o tempdata.h5 https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_lgmet_age.h5
```

Always set `dsps_ssp_fn="tempdata.h5"` to the HDF5 SSP template file you want to use.

### 4. Configure dustmaps SFD (one-time)

This repo assumes `dustmaps` is already configured and SFD maps are available.

Typical one-time setup:

```
python setup.py fetch --map-name=sfd
```

After fetching, make sure `dustmaps` is configured to use the directory containing the SFD maps.

## Tutorials

- GitHub view: [`notebooks/01_jaxqsofit_tutorial.ipynb`](./notebooks/01_jaxqsofit_tutorial.ipynb)
- Direct download:
  `https://raw.githubusercontent.com/burke86/jaxqsofit/main/notebooks/01_jaxqsofit_tutorial.ipynb`

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
)

q.fit(
    deredden=True,
    fit_lines=True,
    decompose_host=True,
    fit_fe=False,
    fit_bc=True,
    fit_poly=True,
    dsps_ssp_fn="tempdata.h5",
    nuts_warmup=300,
    nuts_samples=600,
    nuts_chains=1,
    plot_fig=True,
    save_fig=False,
)
```

### Fast fitting option (Optax)

If you want speed over full posterior sampling, use:

```python
q.fit(
    fit_method="optax",
    optax_steps=1500,
    optax_lr=1e-2,
    plot_fig=True,
    save_fig=False,
)
```

This runs a staged MAP optimization (continuum warm start, then full model) and is typically much faster than NUTS.

Optional: override any defaults by passing `prior_config`:

```python
prior_config = {
    "student_t_df": 2.5,
    "PL_slope": {"loc": -1.5, "scale": 0.3, "low": -3.5, "high": 0.3},
}
q.fit(prior_config=prior_config, fit_lines=False, fit_fe=False, fit_bc=True)
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
