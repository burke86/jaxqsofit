import os
import numpy as np
import pytest


def test_sdss_fetch_and_qsofit_init():
    astroquery = pytest.importorskip('astroquery.sdss')
    coordinates = pytest.importorskip('astropy.coordinates')
    units = pytest.importorskip('astropy.units')

    SDSS = astroquery.SDSS
    SkyCoord = coordinates.SkyCoord
    u = units

    ra, dec = 184.0307, -2.2383
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    try:
        xid = SDSS.query_region(coord, radius=5 * u.arcsec, spectro=True)
    except Exception as exc:
        pytest.skip(f'SDSS query unavailable: {exc}')

    if xid is None or len(xid) == 0:
        pytest.skip('No SDSS spectrum found near target coordinates')

    try:
        spectra = SDSS.get_spectra(matches=xid[:1])
    except Exception as exc:
        pytest.skip(f'SDSS spectrum download unavailable: {exc}')

    if not spectra:
        pytest.skip('No SDSS spectra returned')

    from jaxqsofit import JAXQSOFit

    hdu = spectra[0]
    data = hdu[1].data
    lam = np.asarray(10 ** data['loglam'], dtype=float)
    flux = np.asarray(data['flux'], dtype=float)
    ivar = np.asarray(data['ivar'], dtype=float)

    err = np.full_like(flux, np.inf)
    m = ivar > 0
    err[m] = 1.0 / np.sqrt(ivar[m])
    err[~np.isfinite(err)] = 1e-6
    err[err <= 0] = 1e-6

    z = float(xid[0]['z']) if 'z' in xid.colnames else 0.1

    q = JAXQSOFit.from_arrays(lam=lam, flux=flux, err=err, z=z, ra=ra, dec=dec)

    assert q.lam_in.size > 100
    assert np.isfinite(q.flux_in).any()
    assert np.isfinite(q.err_in).any()


def test_sdss_fit_wrms_below_threshold():
    """Run a quick SDSS fit and require normalized residual WRMS < threshold."""
    astroquery = pytest.importorskip('astroquery.sdss')
    coordinates = pytest.importorskip('astropy.coordinates')
    units = pytest.importorskip('astropy.units')

    SDSS = astroquery.SDSS
    SkyCoord = coordinates.SkyCoord
    u = units

    ra, dec = 184.0307, -2.2383
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    try:
        xid = SDSS.query_region(coord, radius=5 * u.arcsec, spectro=True)
    except Exception as exc:
        pytest.skip(f'SDSS query unavailable: {exc}')

    if xid is None or len(xid) == 0:
        pytest.skip('No SDSS spectrum found near target coordinates')

    try:
        spectra = SDSS.get_spectra(matches=xid[:1])
    except Exception as exc:
        pytest.skip(f'SDSS spectrum download unavailable: {exc}')

    if not spectra:
        pytest.skip('No SDSS spectra returned')

    import numpyro.distributions as dist
    from jaxqsofit import (
        ContinuumConfig,
        FitConfig,
        HostConfig,
        InferenceConfig,
        JAXQSOFit,
        LineConfig,
        Observation,
        OutputConfig,
        PreprocessingConfig,
        PriorConfig,
        SpectroscopyData,
    )
    if not os.path.isfile('tempdata.h5'):
        pytest.skip('DSPS SSP template file tempdata.h5 is unavailable')

    hdu = spectra[0]
    data = hdu[1].data
    lam = np.asarray(10 ** data['loglam'], dtype=float)
    flux = np.asarray(data['flux'], dtype=float)
    ivar = np.asarray(data['ivar'], dtype=float)
    wdisp = np.asarray(data['wdisp'], dtype=float) if 'wdisp' in data.names else None

    err = np.full_like(flux, 1e-6, dtype=float)
    good = np.isfinite(ivar) & (ivar > 0)
    err[good] = 1.0 / np.sqrt(ivar[good])
    err[~np.isfinite(err)] = 1e-6
    err[err <= 0] = 1e-6

    try:
        z = float(hdu[2].data['z'][0])
    except Exception:
        z = float(xid[0]['z']) if 'z' in xid.colnames else 0.1

    if wdisp is not None and np.any(np.isfinite(wdisp) & (wdisp > 0)):
        dloglam = float(np.nanmedian(np.diff(np.log10(lam))))
        fwhm_pixels = 2.355 * wdisp[np.isfinite(wdisp) & (wdisp > 0)]
        resolving_power = float(np.nanmedian(1.0 / (np.log(10.0) * dloglam * fwhm_pixels)))
    else:
        resolving_power = 2000.0

    prior_config = PriorConfig.from_spectrum(
        flux=flux,
        redshift=z,
        include_elg_narrow_lines=False,
        include_high_ionization_lines=False,
    )
    prior_config.powerlaw.slope = dist.TruncatedNormal(loc=-1.5, scale=0.3, low=-3.5, high=0.5)
    prior_config.fe.uv_norm = dist.LogNormal(np.log(max(1e-3 * np.median(np.abs(flux)), 1e-10)), 0.04)
    prior_config.fe.op_over_uv = dist.Normal(0.0, 0.4)
    prior_config.lines.dmu_scale_mult = 0.25
    prior_config.lines.sig_scale_mult = 0.25
    prior_config.lines.amp_scale_mult = 0.20
    prior_config.host.sfh_model = "delayed"
    prior_config.host.stellar_mass = dist.TruncatedNormal(loc=10.6, scale=0.4, low=9.5, high=12.0)
    prior_config.host.sfh_age_gyr = dist.Normal(np.log(7.0), 0.3)
    prior_config.host.sfh_tau_over_age = dist.Normal(np.log(0.25), 0.3)
    prior_config.host.metallicity = dist.Normal(0.0, 0.2)
    prior_config.host.metallicity_scatter = dist.Normal(np.log(0.1), 0.3)
    prior_config.host.aperture_scale = dist.Normal(0.0, 0.25)

    q = JAXQSOFit(
        FitConfig(
            observation=Observation(redshift=z, ra=ra, dec=dec, apply_mw_deredden=True),
            spectroscopy=SpectroscopyData(
                wave_obs=lam,
                fluxes=flux,
                errors=err,
                wavelength_dispersion=wdisp,
                resolving_power=resolving_power,
            ),
            preprocessing=PreprocessingConfig(mask_lya_forest=True),
            continuum=ContinuumConfig(
                fit_power_law=True,
                fit_feii=True,
                fit_balmer_continuum=True,
                fit_polynomial_tilt=True,
            ),
            host=HostConfig(
                enabled=True,
                sfh_model="delayed",
                dsps_ssp_fn="tempdata.h5",
                age_grid_gyr=(0.3, 1.0, 3.0, 6.0, 10.0),
                logzsol_grid=(-0.5, 0.0, 0.2),
            ),
            lines=LineConfig(enabled=True),
            inference=InferenceConfig(
                method="optax+nuts",
                map_steps=int(os.getenv('JAXQSOFIT_WRMS_OPTAX_STEPS', '1200')),
                learning_rate=float(os.getenv('JAXQSOFIT_WRMS_OPTAX_LR', '1e-2')),
                num_warmup=int(os.getenv('JAXQSOFIT_WRMS_NUTS_WARMUP', '50')),
                num_samples=int(os.getenv('JAXQSOFIT_WRMS_NUTS_SAMPLES', '50')),
                num_chains=1,
                target_accept_prob=0.9,
            ),
            output=OutputConfig(plot_fig=False, save_fig=False, save_result=False),
            prior_config=prior_config,
        )
    )
    q.fit()

    resid = np.asarray(q.flux) - np.asarray(q.model_total)
    sigma = np.asarray(q.err, dtype=float)

    # Include fitted jitter terms in effective uncertainty when available.
    if getattr(q, 'numpyro_samples', None) is not None:
        s = q.numpyro_samples
        frac_j = float(np.median(np.asarray(s['frac_jitter']))) if 'frac_jitter' in s else 0.0
        add_j = float(np.median(np.asarray(s['add_jitter']))) if 'add_jitter' in s else 0.0
        sigma = np.sqrt(sigma**2 + (frac_j * np.abs(np.asarray(q.model_total)))**2 + add_j**2)

    m = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if np.sum(m) < 10:
        pytest.skip('Not enough valid pixels to evaluate WRMS')

    zres = resid[m] / sigma[m]
    wrms = float(np.sqrt(np.mean(zres**2)))
    threshold = float(os.getenv('JAXQSOFIT_WRMS_THRESHOLD', '1.5'))
    assert wrms < threshold, f'WRMS too high: {wrms:.3f} (threshold={threshold:.3f})'
