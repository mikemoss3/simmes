"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines unit tests for the simmes.util_packages.cosmology module

"""

import numpy as np
from scipy.integrate import quad

def testKCorr():
	"""
	Testing time-dilation method
	"""
	from simmes.util_packages.cosmology import k_corr
	from simmes.SPECFUNC import	Band
	
	# Define true k-correction value (see Bloom, Frail, and Sari 2001)
	true_k_corr_val = 0.81

	# Parameters for GRB 970508 taken from Piran, Jimenez, Band 2000.
	alpha = -1.19
	beta = -1.83
	ep = 480.84 * (alpha - beta)  # keV
	spec = Band( alpha = alpha, beta = beta, ep = ep, norm = 1)
	
	z = 0.8350

	# Define comoving energy band to compare (see Bloom, Frail, and Sari 2001)
	emin = 20
	emax = 2000
	Emin = 20
	Emax = 2000

	test_k_corr = k_corr(spec, z, emin=emin, emax=emax, Emin=Emin, Emax=Emax)
	test_k_corr = round(test_k_corr, 2)

	np.testing.assert_equal(test_k_corr, true_k_corr_val)


def testLumDist():
	"""
	Testing luminosity distance method
	"""
	from simmes.util_packages.cosmology import lum_dis

	z = 1

	c = 3*np.power(10,10) # speed of light, cm/s
	omega_m = 0.3 # matter density of the universe
	omega_lam = 0.7 # dark energy density of the universe
	H0 = 67.4*np.power(10,5) # Hubbles Constant cm/s/Mpc

	lum_dis_Mpc = ((1+z)*c/(H0) ) * quad(lambda zi: 1/np.sqrt( ((omega_m*np.power(1+zi,3) )+omega_lam) ),0,z)[0]
	ld = lum_dis_Mpc * 3.086e24 # Mpc -> cm
	
	test_ld = lum_dis(z)

	np.testing.assert_equal(test_ld, ld)

if __name__ == "__main__":

	testKCorr()
	testLumDist()