"""
Make spectra for first figure
"""
import numpy as np
from astropy import units as u, constants as c
import libpypsg as psg
from libpypsg.cfg.base import Molecule,Profile
from loguru import logger
from matplotlib import pyplot as plt

import paths
import instrument
import core

PMIN = 1e-10 * u.bar
LAYERS_PER_SCALE_HEIGHT = 4
O3_ABN = 1e-4

PL_TEMP = 300*u.K
MEAN_MOLEC_WEIGHT_N2 = 28.97 * u.g /u.mol / c.N_A
MEAN_MOLEC_WEIGHT_H2 = 2.02 * u.g /u.mol / c.N_A
PL_GRAV = 10*u.m/u.s**2
SCALE_HEIGHT_N2 = c.k_B*PL_TEMP/(MEAN_MOLEC_WEIGHT_N2*PL_GRAV)
SCALE_HEIGHT_H2 = c.k_B*PL_TEMP/(MEAN_MOLEC_WEIGHT_H2*PL_GRAV)

def get_n_layers(pmax: u.Quantity):
    return LAYERS_PER_SCALE_HEIGHT * int(
        round(
            np.log(
                (pmax/PMIN).to_value(u.dimensionless_unscaled)
            )
        )
    )
    
def get_star_type(teff: u.Quantity):
    if teff < 4000 * u.K:
        return 'M'
    elif teff < 5000 * u.K:
        return 'K'
    elif teff < 6000 * u.K:
        return 'G'
    elif teff < 7200 * u.K:
        return 'F'
    elif teff < 9700 * u.K:
        return 'A'
    elif teff < 30000 * u.K:
        return 'B'
    else:
        return 'O'

def get_cfg(
    pl_rad: u.Quantity,
    pl_grav: u.Quantity,
    st_rad: u.Quantity,
    st_mass: u.Quantity,
    sy_dist: u.Quantity,
    abn_o3: float,
    mean_molec_weight: u.Quantity,
    pl_temp: u.Quantity,
    pl_press: u.Quantity,
    st_teff: u.Quantity,
    smax: u.Quantity,
    n2:bool
):
    
    pressure: u.Quantity = u.bar * np.logspace(
        np.log10(pl_press.to_value(u.bar)),
        np.log10(PMIN.to_value(u.bar)),
        get_n_layers(pl_press)
    )
    temperature: u.Quantity = pl_temp * np.ones_like(pressure.value)
    stratosphere = pressure < 0.01*u.bar
    
    orb_per = np.sqrt(
        # pylint: disable-next=no-member
        4 * np.pi * (smax**3) / (c.G * st_mass)
    )
    transit_length = orb_per * 2*st_rad / (2*np.pi*smax)
    exp_time = 1*u.s
    n_integrations = int(np.ceil(transit_length/exp_time))
    n_transits = int((instrument.MISSION_LIFE/orb_per).to_value(u.dimensionless_unscaled))
    
    target = psg.cfg.Target(
        object='Exoplanet',
        name='my-planet',
        diameter=2*pl_rad,
        gravity=pl_grav,
        star_distance=smax,
        star_type=get_star_type(st_teff),
        star_temperature=st_teff,
        star_radius=st_rad,
        inclination=90*u.deg,
        season=180*u.deg
    )
    geometry = psg.cfg.Observatory(
        observer_altitude=sy_dist,
        stellar_temperature=st_teff,
    )
    atmosphere = psg.cfg.EquilibriumAtmosphere(
        pressure=pressure[0],
        temperature=temperature[0],
        weight=mean_molec_weight,
        molecules=[
            Molecule('O3','HIT[3]',abn_o3),
            Molecule('N2','HIT[22]',1-abn_o3) if n2 else Molecule('H2','HIT[45]',1-abn_o3)
        ],
        nmax=1,
        lmax=2,
        profile=[
            Profile('PRESSURE',pressure.value,pressure.unit),
            Profile('TEMPERATURE', temperature.value, temperature.unit),
            Profile('O3',np.ones_like(pressure.value)*stratosphere,u.dimensionless_unscaled),
            Profile('N2',np.ones_like(pressure.value),u.dimensionless_unscaled) if n2 else Profile('H2',np.ones_like(pressure.value),u.dimensionless_unscaled)
        ]
    )
    surface = psg.cfg.Surface(
        temperature=temperature[0],
        albedo=0.3,
        emissivity=0.7
    )
    generator = psg.cfg.Generator(
        resolution_kernel=False,
        continuum_model=True,
        gas_model=True,
        apply_telluric_noise=False,
        continuum_stellar=True,
        apply_telluric_obs=False,
        rad_units=u.Unit('W m-2 um-1'),
        log_rad=False
    )
    telescope = psg.cfg.SingleTelescope(
        aperture=instrument.APERTURE_SIZE,
        fov = 1*u.arcsec,
        range1=0.105 *u.um,
        range2=0.8*u.um,
        zodi=0,
        resolution=300*psg.units.resolving_power
    )
    noise = psg.cfg.CCD(
        exp_time=exp_time,
        n_frames=n_integrations*n_transits,
        n_pixels=1,
        read_noise=0.*u.electron,
        dark_current=0.*u.electron/u.s,
        thoughput=1.0,
        emissivity=0.0,
        temperature=50*u.K,
        pixel_depth=1e6*u.electron
    )
    
    cfg = psg.PyConfig(
        target=target,
        geometry=geometry,
        atmosphere=atmosphere,
        surface=surface,
        generator=generator,
        telescope=telescope,
        noise=noise
    )
    return cfg

def make_fig(
    wl: u.Quantity,
    transit_depth: np.ndarray,
    depth_u: float,
    depth_v: float,
    depth_i: float,
    sigma_u: float,
    sigma_v: float,
    sigma_i: float,
    has_o3: bool,
    rad_star: u.Quantity,
    ax: plt.Axes
):    
    ax.plot(wl, transit_depth,color='xkcd:azure')
    ax.errorbar(
        [instrument.FILTER_U.center.to_value(u.um),instrument.FILTER_V.center.to_value(u.um),instrument.FILTER_I.center.to_value(u.um)],
        [depth_u,depth_v,depth_i],
        xerr=[instrument.FILTER_U.width.to_value(u.um),instrument.FILTER_V.width.to_value(u.um),instrument.FILTER_I.width.to_value(u.um)],
        yerr=[sigma_u,sigma_v,sigma_i],
        color='xkcd:golden rod',
        fmt='o'
    )
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('Transit Depth')
    ax.set_title(f'Has O3: {has_o3}')
    
def get_depth(
    in_transit: u.Quantity,
    out_transit: u.Quantity,
    noise: u.Quantity,
    wl: u.Quantity,
    bandpass: core.Bandpass
):
    out_band,out_sigma = bandpass.photometry(out_transit,noise,wl)
    in_band,in_sigma = bandpass.photometry(in_transit,noise,wl)
    depth_band = (out_band-in_band)/out_band
    sigma_depth = np.sqrt(in_sigma**2 + out_sigma**2)/out_band
    return depth_band,sigma_depth

def depth_to_scale_height(
    transit_depth: np.ndarray,
    error: np.ndarray,
    r_star:u.Quantity,
    r_planet:u.Quantity,
    scale_height:u.Quantity
):
    rp_rstar = np.sqrt(transit_depth)
    rp_rstar_error = error/2/transit_depth * rp_rstar
    r_p_wl = rp_rstar*r_star
    r_p_wl_error = rp_rstar_error*r_star
    atm_height = r_p_wl - r_planet
    atm_height_error = r_p_wl_error
    return (atm_height/scale_height).to_value(u.dimensionless_unscaled),(atm_height_error/scale_height).to_value(u.dimensionless_unscaled)


if __name__ == '__main__':
    psg.docker.set_url_and_run()
    plt.style.use('bmh')
    plt.tick_params(axis='both', which='major', labelsize=8,)
    logger.info(f'Scale = {SCALE_HEIGHT_N2.to_value(u.km):.1f} km')
    plt.close('all')
    fig = plt.figure(figsize=(6,6))
    ax_m = fig.add_subplot(2,1,1)
    ax_g = fig.add_subplot(2,1,2)
    fig.subplots_adjust(hspace=0)
    
    cfg_m_has_o3 = get_cfg(
        pl_rad = 1*u.R_earth,
        pl_grav = 10*u.m/u.s**2,
        st_rad=0.2*u.R_sun,
        st_mass=0.2*u.M_sun,
        st_teff=3000*u.K,
        sy_dist=10*u.pc,
        smax = 0.05*u.AU,
        mean_molec_weight=28.97*u.g/u.mol,
        abn_o3=O3_ABN,
        pl_temp=300*u.K,
        pl_press=1*u.bar,
        n2=True
    )
    caller = psg.APICall(cfg_m_has_o3,'all')
    response = caller()
    rad = response.rad
    transit_depth = (rad['Stellar']-rad['Total'])/rad['Stellar']
    depth_scale_height,_ = depth_to_scale_height(transit_depth,np.ones_like(transit_depth),r_star=cfg_m_has_o3.target.star_radius.value, r_planet=1*u.R_earth,scale_height=SCALE_HEIGHT_N2)
    wl = rad.wl
    x = [instrument.FILTER_U.center.to_value(u.um),instrument.FILTER_B.center.to_value(u.um),instrument.FILTER_V.center.to_value(u.um)]
    x_width = [instrument.FILTER_U.width.to_value(u.um),instrument.FILTER_B.width.to_value(u.um),instrument.FILTER_V.width.to_value(u.um)]
    y_and_err = [
        get_depth(
            in_transit=rad['Total'],
            out_transit=rad['Stellar'],
            noise=rad['Noise'],
            wl=wl,
            bandpass=f
        )
        for f in [instrument.FILTER_U, instrument.FILTER_B, instrument.FILTER_V]
    ]
    y = np.array([_y[0] for _y in y_and_err])
    yerr = np.array([_y[1] for _y in y_and_err])
    y, yerr = depth_to_scale_height(y,yerr,r_star=cfg_m_has_o3.target.star_radius.value, r_planet=1*u.R_earth,scale_height=SCALE_HEIGHT_N2)
    
    ax_m.plot(wl,depth_scale_height, color='xkcd:grass green',zorder=-100)
    ax_m.errorbar(x,y,yerr=yerr,xerr=x_width,color='xkcd:grass green',fmt='o',mec='k',mew=1,capsize=3,ecolor='k',label='Has O$_3$')
    ax_m.set_ylabel('Transit Depth ($H$)')
    
    cfg_m_no_o3 = get_cfg(
        pl_rad = 1*u.R_earth,
        pl_grav = 10*u.m/u.s**2,
        st_rad=0.2*u.R_sun,
        st_mass=0.2*u.M_sun,
        st_teff=3000*u.K,
        sy_dist=10*u.pc,
        smax = 0.05*u.AU,
        mean_molec_weight=28.97*u.g/u.mol,
        abn_o3=1e-20,
        pl_temp=300*u.K,
        pl_press=1.0*u.bar,
        n2=True
    )
    caller = psg.APICall(cfg_m_no_o3,'all')
    response = caller()
    rad = response.rad
    transit_depth = (rad['Stellar']-rad['Total'])/rad['Stellar']
    depth_scale_height,_ = depth_to_scale_height(transit_depth,np.ones_like(transit_depth),r_star=cfg_m_no_o3.target.star_radius.value, r_planet=1*u.R_earth,scale_height=SCALE_HEIGHT_N2)
    wl = rad.wl
    x = [instrument.FILTER_U.center.to_value(u.um),instrument.FILTER_B.center.to_value(u.um),instrument.FILTER_V.center.to_value(u.um)]
    x_width = [instrument.FILTER_U.width.to_value(u.um),instrument.FILTER_B.width.to_value(u.um),instrument.FILTER_V.width.to_value(u.um)]
    y_and_err = [
        get_depth(
            in_transit=rad['Total'],
            out_transit=rad['Stellar'],
            noise=rad['Noise'],
            wl=wl,
            bandpass=f
        )
        for f in [instrument.FILTER_U, instrument.FILTER_B, instrument.FILTER_V]
    ]
    y = np.array([_y[0] for _y in y_and_err])
    yerr = np.array([_y[1] for _y in y_and_err])
    y, yerr = depth_to_scale_height(y,yerr,r_star=cfg_m_no_o3.target.star_radius.value, r_planet=1*u.R_earth,scale_height=SCALE_HEIGHT_N2)
    ax_m.plot(wl,depth_scale_height, color='xkcd:azure',zorder=-100)
    ax_m.errorbar(x,y,yerr=yerr,xerr=x_width,color='xkcd:azure',fmt='o',mec='k',mew=1,capsize=3,ecolor='k',label='No O$_3$')
    ax_m.set_ylim(-2,22)
    ax_m.legend()
    
    
    cfg_g_has_o3 = get_cfg(
        pl_rad = 2.5*u.R_earth,
        pl_grav = 10*u.m/u.s**2,
        st_rad=1*u.R_sun,
        st_mass=1*u.M_sun,
        st_teff=5800*u.K,
        sy_dist=200*u.pc,
        smax = 1*u.AU,
        mean_molec_weight=28.97*u.g/u.mol,
        abn_o3=O3_ABN,
        pl_temp=300*u.K,
        pl_press=1.0*u.bar,
        n2=False
    )
    caller = psg.APICall(cfg_g_has_o3,'all')
    response = caller()
    rad = response.rad
    transit_depth = (rad['Stellar']-rad['Total'])/rad['Stellar']
    depth_scale_height,_ = depth_to_scale_height(transit_depth,np.ones_like(transit_depth),r_star=cfg_g_has_o3.target.star_radius.value, r_planet=2*u.R_earth,scale_height=SCALE_HEIGHT_H2)
    wl = rad.wl
    x = [instrument.FILTER_U.center.to_value(u.um),instrument.FILTER_B.center.to_value(u.um),instrument.FILTER_V.center.to_value(u.um)]
    x_width = [instrument.FILTER_U.width.to_value(u.um),instrument.FILTER_B.width.to_value(u.um),instrument.FILTER_V.width.to_value(u.um)]
    y_and_err = [
        get_depth(
            in_transit=rad['Total'],
            out_transit=rad['Stellar'],
            noise=rad['Noise'],
            wl=wl,
            bandpass=f
        )
        for f in [instrument.FILTER_U, instrument.FILTER_B, instrument.FILTER_V]
    ]
    y = np.array([_y[0] for _y in y_and_err])
    yerr = np.array([_y[1] for _y in y_and_err])
    y, yerr = depth_to_scale_height(y,yerr,r_star=cfg_g_has_o3.target.star_radius.value, r_planet=2*u.R_earth,scale_height=SCALE_HEIGHT_H2)
    ax_g.plot(wl,depth_scale_height, color='xkcd:grass green',zorder=-100)
    ax_g.errorbar(x,y,yerr=yerr,xerr=x_width,color='xkcd:grass green',fmt='o',mec='k',mew=1,capsize=3,ecolor='k',label='Has O$_3$')
    ax_g.set_xlabel('$\\lambda ~(\\rm \\mu m)$')
    ax_g.set_ylabel('Transit Depth ($H$)')
    
    cfg_g_no_o3 = get_cfg(
        pl_rad = 2.5*u.R_earth,
        pl_grav = 10*u.m/u.s**2,
        st_rad=1*u.R_sun,
        st_mass=1*u.M_sun,
        st_teff=5800*u.K,
        sy_dist=200*u.pc,
        smax = 1*u.AU,
        mean_molec_weight=28.97*u.g/u.mol,
        abn_o3=1e-20,
        pl_temp=300*u.K,
        pl_press=1.0*u.bar,
        n2=False
    )
    caller = psg.APICall(cfg_g_no_o3,'all')
    response = caller()
    rad = response.rad
    transit_depth = (rad['Stellar']-rad['Total'])/rad['Stellar']
    depth_scale_height,_ = depth_to_scale_height(transit_depth,np.ones_like(transit_depth),r_star=cfg_g_no_o3.target.star_radius.value, r_planet=2*u.R_earth,scale_height=SCALE_HEIGHT_H2)
    wl = rad.wl
    x = [instrument.FILTER_U.center.to_value(u.um),instrument.FILTER_B.center.to_value(u.um),instrument.FILTER_V.center.to_value(u.um)]
    x_width = [instrument.FILTER_U.width.to_value(u.um),instrument.FILTER_B.width.to_value(u.um),instrument.FILTER_V.width.to_value(u.um)]
    y_and_err = [
        get_depth(
            in_transit=rad['Total'],
            out_transit=rad['Stellar'],
            noise=rad['Noise'],
            wl=wl,
            bandpass=f
        )
        for f in [instrument.FILTER_U, instrument.FILTER_B, instrument.FILTER_V]
    ]
    y = np.array([_y[0] for _y in y_and_err])
    yerr = np.array([_y[1] for _y in y_and_err])
    y, yerr = depth_to_scale_height(y,yerr,r_star=cfg_g_no_o3.target.star_radius.value, r_planet=2*u.R_earth,scale_height=SCALE_HEIGHT_H2)
    ax_g.plot(wl,depth_scale_height, color='xkcd:azure',zorder=-100)
    ax_g.errorbar(x,y,yerr=yerr,xerr=x_width,color='xkcd:azure',fmt='o',mec='k',mew=1,capsize=3,ecolor='k',label='No O$_3$')
    
    ax_m.text(0.25, 0.85, 'Earth around M-dwarf @ 10 pc', transform=ax_m.transAxes, fontweight='bold')
    ax_g.text(0.35, 0.85, 'Sub-Neptune around Sun @ 200 pc', transform=ax_g.transAxes, fontweight='bold')
    
    fig.savefig(paths.figures / 'spectrum.pdf', bbox_inches='tight')