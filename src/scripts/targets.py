from astropy import units as u, constants as c
import numpy as np

from instrument import FOV
import paths

STELLAR_DENSITY = 0.05* u.solMass/u.pc**3
ALPHA = 2.35
M_MIN = 0.2
M_MAX = 2.0

TOTAL_TARGETS = 7e6
CADENCE = 15*u.min
BITS_PER_INTEGRATION = TOTAL_TARGETS * 64 * u.bit
BITS_PER_SECOND = (BITS_PER_INTEGRATION / CADENCE).to(u.bit *u.s**-1)


GAMMA = (2-ALPHA)/(M_MAX**(2-ALPHA)-M_MIN**(2-ALPHA))

def write_variable(key, val):
    with open(paths.output / f'targets-{key}.txt','w',encoding='utf-8') as f:
        f.write(val)
        
if __name__ == '__main__':
    write_variable('density', f'{STELLAR_DENSITY:latex}')
    write_variable('gamma', f'{GAMMA:.2f}')
    
    omega  = FOV
    # omega = 4*np.pi*u.steradian
    
    n_targets_per_deg2_20pc = GAMMA * omega.to_value(u.steradian) * (20*u.pc)**3 * STELLAR_DENSITY/(1*u.M_sun) * (M_MAX**(1-ALPHA)-M_MIN**(1-ALPHA))/(1-ALPHA)
    n_targets_per_deg2_1000pc = GAMMA * omega.to_value(u.steradian) * (1000*u.pc)**3 * STELLAR_DENSITY/(1*u.M_sun) * (M_MAX**(1-ALPHA)-M_MIN**(1-ALPHA))/(1-ALPHA)
    print(n_targets_per_deg2_20pc)
    print(n_targets_per_deg2_1000pc)
    print(f'kB/s downlink = {BITS_PER_SECOND.to_value(u.kB/u.s):.2f}')