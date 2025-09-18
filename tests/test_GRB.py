"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines unit tests for the simmes.GRB module

"""

import numpy as np
from scipy.integrate import quad


def testSpectrumShift():
	"""
	Testing whether the spectrum shifting method correctly shifts spectral parameters
	"""
	from simmes.GRB import move_spectrum
	from simmes.SPECFUNC import	CPL
	
	z0 = 0
	z_array = [1, 3, 6, 10]

	alpha = -1.
	ep = 300  # keV
	spec = CPL( alpha = alpha, ep = ep, norm = 1)

	for i in range(len(z_array)):
		z1 = z_array[i]

		test_spec = spec.copy()
		test_spec = move_spectrum(test_spec, z0, z1)

		np.testing.assert_equal(test_spec.params['ep'], ep/(1+z1))

def testFluxChange():
	"""
	Testing whether the spectrum normalization is properly adjusted
	"""
	from simmes.GRB import move_spectrum
	from simmes.util_packages.cosmology import lum_dis, k_corr
	from simmes.SPECFUNC import	CPL
	
	z1 = 1
	z_array = [1, 3, 6, 10]

	emin = 20
	emax = 2000

	alpha = -1.
	ep = 300  # keV
	norm = 1
	spec = CPL( alpha = alpha, ep = ep, norm = norm)
	for i in range(len(z_array)):
		z2 = z_array[i]

		shifted_spec = CPL( alpha = alpha, ep = ep*(1+z1)/(1+z2), norm = norm)

		dl_rat = np.power(lum_dis(z1)/lum_dis(z2), 2.)
		z_rat = np.power((1+z2)/(1+z1) , 2.)
		spec_rat = quad(spec, emin, emax)[0]/quad(shifted_spec, emin, emax)[0]
		kcorr_rat = k_corr(spec, z1, emin=emin, emax=emax)/k_corr(shifted_spec, z2, emin=emin, emax=emax)
		new_norm = dl_rat * z_rat * norm * spec_rat * kcorr_rat
		new_norm = round(new_norm, 6)

		test_spec = CPL( alpha = alpha, ep = ep, norm = 1)
		test_spec = move_spectrum(test_spec, z1, z2, emin=emin, emax=emax)
		test_norm = test_spec.params['norm']
		test_norm = round(test_norm, 6)

		np.testing.assert_equal(test_norm, new_norm)

def testTimeDilation():
	"""
	Testing time-dilation method
	"""
	from simmes.GRB import move_light_curve

	# Create simple square light curve
	light_curve = np.zeros(shape=20, dtype=[("TIME", float), ("RATE",float), ("UNC", float)])
	tb_size = 1
	light_curve['TIME'] = np.arange(-10, 10, step=tb_size)
	light_curve['RATE'][10] = 2 # Peak Flux = 2
	light_curve['RATE'][11] = 1 # Fluence = 3
	
	z0 = 0
	z1 = 1

	# Calculate needed values directly
	non_zero_args = np.argwhere(light_curve['RATE']>0)
	non_zero_start = non_zero_args[0][0]
	non_zero_stop = non_zero_args[-1][0]
	tstart = light_curve['TIME'][non_zero_start] * (1+z1)
	tstop = light_curve['TIME'][non_zero_stop] * (1+z1) + tb_size

	tot_counts = np.sum(light_curve['RATE'])

	# Time dilate light curve
	test_light_curve = move_light_curve(light_curve, z0, z1)
	# Calculate needed values from test 
	test_non_zero_args = np.argwhere(test_light_curve['RATE']>0)
	test_non_zero_start = test_non_zero_args[0][0]
	test_non_zero_stop = test_non_zero_args[-1][0]
	test_tstart = test_light_curve['TIME'][test_non_zero_start]
	test_tstop = test_light_curve['TIME'][test_non_zero_stop]
	test_tot_counts = np.sum(test_light_curve['RATE'])


	# Perform Tests
	# Check signal start and stop times are correct
	np.testing.assert_equal(test_tstart, tstart)
	np.testing.assert_equal(test_tstop, tstop)
	
	# Check total counts are conserved
	np.testing.assert_equal(test_tot_counts, tot_counts)

if __name__ == "__main__":
	testTimeDilation()
	testSpectrumShift()
	testFluxChange()
	