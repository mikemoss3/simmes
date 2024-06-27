"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	
Last edited: 2020-05-12

Constants that are used through out the code, so they are made accessible in the Global Constants file.

"""

import numpy as np

# Define constants that are used between methods
bol_lum = [1,100000] # bolumetric luminosity range
c = 3*np.power(10,10) # speed of light, cm/s
omega_m = 0.3 # matter density of the universe
omega_lam = 0.7 # dark energy density of the universe
H0 = 67.4*np.power(10,5) # Hubbles Constant cm/s/Mpc