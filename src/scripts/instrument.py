"""
Instrument parameters
"""

from astropy import units as u

from core import Bandpass
import paths

MISSION_LIFE = 10 * u.yr
APERTURE_SIZE = 2 * u.m
FOV = (48 * u.deg)**2

CCD_PIX = 1024 * 2200
N_CCD = 42 * 3

FILTER_U = Bandpass(
    center=0.255 * u.um,
    width=0.05 * u.um,
    throughput=0.5
)

FILTER_B = Bandpass(
    center=0.38 * u.um,
    width=0.05 * u.um,
    throughput=0.5
)

FILTER_V = Bandpass(
    center=0.6 * u.um,
    width=0.05 * u.um,
    throughput=0.5
)

def write_variable(key, val):
    with open(paths.output / f'inst-{key}.txt','w',encoding='utf-8') as f:
        f.write(val)

if __name__ == '__main__':
    write_variable('diam', f'{APERTURE_SIZE:latex}')
    write_variable('lifetime',f'{MISSION_LIFE.to_value(u.yr):.0f}')
    
    write_variable('center_u',f'{FILTER_U.center:latex}')
    write_variable('width_u',f'{FILTER_U.width:latex}')
    write_variable('through_u', f'{FILTER_U.throughput:.2f}')
    
    write_variable('center_b',f'{FILTER_B.center:latex}')
    write_variable('width_b',f'{FILTER_B.width:latex}')
    write_variable('through_b', f'{FILTER_B.throughput:.2f}')
    
    write_variable('center_v',f'{FILTER_V.center:latex}')
    write_variable('width_v',f'{FILTER_V.width:latex}')
    write_variable('through_v', f'{FILTER_V.throughput:.2f}')
    
    write_variable('fov',f'{FOV:latex}')
    
    print(f'Image size in Gb: {CCD_PIX * N_CCD * 8 / 1024 / 1024 / 1024}')
    