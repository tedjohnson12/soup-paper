import numpy as np
from astropy import units as u
from enum import Enum, auto

def spectrum_to_photometry(
    spectrum: u.Quantity,
    uncertainty: u.Quantity,
    wl: u.Quantity
):
    delta_wl = np.diff(wl)
    flux_elements_trap = (spectrum[:-1] + spectrum[1:])/2 * delta_wl
    flux_trap = np.sum(flux_elements_trap)
    
    varience_elements_trap = uncertainty[1:-1]**2 * (delta_wl[1:] + delta_wl[:-1])**2
    varience_trap = np.sum(varience_elements_trap)
    varience = 0.25 * (uncertainty[0]**2 * delta_wl[0]**2 + uncertainty[-1]**2 * delta_wl[-1]**2 + varience_trap)
    
    return flux_trap, np.sqrt(varience)

class Bandpass:
    def __init__(
        self,
        center: u.Quantity,
        width: u.Quantity,
        throughput: float
    ):
        self.center = center
        self.width = width
        self.throughput = throughput
    
    def photometry(self,flux:u.Quantity,sigma:u.Quantity,lam:u.Quantity):
        in_band = (lam >= self.center - self.width) & (lam <= self.center + self.width)
        return spectrum_to_photometry(flux[in_band],sigma[in_band] / np.sqrt(self.throughput),lam[in_band])
    
class PlanetType(Enum):
    ROCKY = auto()
    ICE_GIANT = auto()
    GAS_GIANT = auto()

def get_planet_type(mass: u.Quantity):
    if mass < 10 * u.M_earth:
        return PlanetType.ROCKY
    elif mass < 80 * u.M_earth:
        return PlanetType.ICE_GIANT
    else:
        return PlanetType.GAS_GIANT

class StellarType(Enum):
    M = auto()
    K = auto()
    G = auto()
    F = auto()
    A = auto()
    B = auto()
    O = auto()

def get_stellar_type(teff: u.Quantity):
    if teff < 4000 * u.K:
        return StellarType.M
    elif teff < 5000 * u.K:
        return StellarType.K
    elif teff < 6000 * u.K:
        return StellarType.G
    elif teff < 7200 * u.K:
        return StellarType.F
    elif teff < 9700 * u.K:
        return StellarType.A
    elif teff < 30000 * u.K:
        return StellarType.B
    else:
        return StellarType.O