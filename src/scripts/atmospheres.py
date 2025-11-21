from functools import partial
from pathlib import Path
import numpy as np
from astropy import units as u, constants as c
import libpypsg as psg
from libpypsg.cfg.base import Molecule,Profile
from loguru import logger
from matplotlib import pyplot as plt
from dataclasses import dataclass

import paths
import instrument
import core

PMIN = 1e-10 * u.bar
LAYERS_PER_SCALE_HEIGHT = 4
O3_ABN = 1e-4

@dataclass
class StellarType:
    name: str
    teff: u.Quantity
    rad: u.Quantity
    mass: u.Quantity
M8 = StellarType(
    name='M8',
    teff=2500 * u.K,
    rad=0.11*u.R_sun,
    mass=0.08*u.M_sun
)
M5 = StellarType(
    name='M5',
    teff=3000 * u.K,
    rad=0.14*u.R_sun,
    mass=0.1221*u.M_sun
)
M0 = StellarType(
    name='M0',
    teff=4000 * u.K,
    rad=0.55*u.R_sun,
    mass=0.55*u.M_sun
)
K0 = StellarType(
    name='K0',
    teff=5000 * u.K,
    rad=0.74*u.R_sun,
    mass=0.88*u.M_sun
)
G2 = StellarType(
    name='G2',
    teff=5800 * u.K,
    rad=1.0*u.R_sun,
    mass=1.0*u.M_sun
)
F2 = StellarType(
    name='F2',
    teff=7000 * u.K,
    rad=1.56*u.R_sun,
    mass=1.68*u.M_sun
)
A0 = StellarType(
    name='A0',
    teff=9500 * u.K,
    rad=1.5*u.R_sun,
    mass=2.3*u.M_sun
)

@dataclass
class Planet:
    name: str
    radius: u.Quantity
    temp: u.Quantity
    psurf: u.Quantity
    grav: u.Quantity
    mean_molec_weight: u.Quantity
    albedo: float
    atm: Path
    
    @property
    def scale_height(self):
        return c.k_B*self.temp/(self.mean_molec_weight/c.N_A*self.grav)

EARTH = Planet(
    name='Earth',
    radius=1*u.R_earth,
    albedo=0.3,
    temp=273*u.K,
    psurf=1*u.bar,
    grav=9.8*u.m/u.s**2,
    mean_molec_weight=28.97 * u.g /u.mol,
    atm=paths.static / 'earth.cfg'
)
VENUS = Planet(
    name='Venus',
    radius=0.95*u.R_earth,
    albedo=0.7,
    temp=450*u.K,
    psurf=90*u.bar,
    grav=8.87*u.m/u.s**2,
    mean_molec_weight=48 * u.g /u.mol,
    atm=paths.static / 'venus.cfg'
)
MARS = Planet(
    name='Mars',
    radius=0.5*u.R_earth,
    albedo=0.25,
    temp=209*u.K,
    psurf=0.006*u.bar,
    grav=3.71*u.m/u.s**2,
    mean_molec_weight=48 * u.g /u.mol,
    atm=paths.static / 'mars.cfg'
)
TITAN = Planet(
    name='Titan',
    radius=0.44*u.R_earth,
    albedo=0.22,
    temp=98*u.K,
    psurf=2*u.bar,
    grav=1.32*u.m/u.s**2,
    mean_molec_weight=28 * u.g /u.mol,
    atm=paths.static / 'titan.cfg'
)

NEP_H20 = Planet(
    name='Neptune',
    radius=4*u.R_earth,
    albedo=0.5,
    temp=500*u.K,
    psurf=1*u.bar,
    grav=11.2*u.m/u.s**2,
    mean_molec_weight=18 * u.g /u.mol,
    atm=paths.static / 'nep_h20.cfg'
)

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
    pl_albedo: float,
    pl_grav: u.Quantity,
    st_rad: u.Quantity,
    st_mass: u.Quantity,
    sy_dist: u.Quantity,
    mean_molec_weight: u.Quantity,
    pl_temp: u.Quantity,
    pl_press: u.Quantity,
    st_teff: u.Quantity,
    has_o3:bool,
    n2:bool
):
    if has_o3:
        abn_o3 = O3_ABN
    else:
        abn_o3 = 1e-20
    
    smax = st_rad/2 * (st_teff/pl_temp)**2 * (1 - pl_albedo)
    
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
        resolution=100*psg.units.resolving_power
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

def make_cfg(
    planet: Planet,
    star: StellarType,
    distance: u.Quantity
):
    cfg = get_cfg(
        pl_rad=planet.radius,
        pl_grav=planet.grav,
        pl_albedo=planet.albedo,
        pl_press=planet.psurf,
        st_teff=star.teff,
        st_rad=star.rad,
        st_mass=star.mass,
        sy_dist=distance,
        mean_molec_weight=planet.mean_molec_weight,
        pl_temp=planet.temp,
        has_o3=False,
        n2=True
    )
    new_cfg = psg.PyConfig.from_file(planet.atm)
    cfg.atmosphere = new_cfg.atmosphere
    return cfg


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

def get_data(
    cfg: psg.PyConfig,
    scale_height: u.Quantity
):
    caller = psg.APICall(cfg,'all')
    caller.reset()
    response = caller()
    rad = response.rad
    transit_depth = (rad['Stellar']-rad['Total'])/rad['Stellar']
    depth_scale_height,_ = depth_to_scale_height(transit_depth,np.ones_like(transit_depth),r_star=cfg.target.star_radius.value, r_planet=cfg.target.diameter.value/2,scale_height=scale_height)
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
    y, yerr = depth_to_scale_height(y,yerr,r_star=cfg.target.star_radius.value, r_planet=cfg.target.diameter.value/2,scale_height=scale_height)
    
    return x,x_width,y,yerr,wl, depth_scale_height

if __name__ == '__main__':
    psg.docker.set_url_and_run()
    plt.style.use('bmh')
    plt.tick_params(axis='both', which='major', labelsize=8,)
    plt.close('all')
    
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    
    OFFSET = 20
    models = [
        (EARTH, M5, 10*u.pc, 'xkcd:azure', 'M-Earth @ 10 pc'),
        (MARS, M5, 10*u.pc, 'xkcd:faded red', 'M-Mars @ 10 pc'),
        # (VENUS, M5, 10*u.pc, 'xkcd:brown orange', 'M-Venus @ 10 pc'),
        (TITAN, M5, 10*u.pc, 'xkcd:orange', 'M-Titan @ 10 pc'),
        (NEP_H20, M5, 10*u.pc, 'xkcd:blue', '500 K Neptune 1% H$_{2}$O$'),
    ]
    for i, model in enumerate(models):
        planet, star, distance, color, label = model
        planet: Planet
        star: StellarType
        distance: u.Quantity
        
        cfg = make_cfg(
            planet=planet,
            star=star,
            distance=distance
        )
        x,x_width,y,yerr,wl, depth_scale_height = get_data(cfg,planet.scale_height)
        ax.plot(wl,i*OFFSET+depth_scale_height,color=color)
        ax.errorbar(x,i*OFFSET+y,xerr=x_width,yerr=yerr,color=color,fmt='o',mec='k',mew=1,capsize=3,ecolor='k',label=label)
    
    plt.tight_layout()
    # plt.savefig(paths.figures/'atmospheres.pdf')
    plt.show()
    
    