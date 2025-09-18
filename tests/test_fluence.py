"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines unit tests for the simmes.fluence module

"""

import numpy as np

def testFluenceCalc():
	"""
	Testing whether the fluence/peak flux calculation method can accurately measure fluence/peak flux of a square light cure
	"""
	from simmes.fluence import calc_fluence

	# Create simple square light curve
	light_curve = np.zeros(shape=20, dtype=[("TIME", float), ("RATE",float), ("UNC", float)])
	tb_size = 1
	light_curve['TIME'] = np.arange(-10, 10, step=tb_size)
	light_curve['RATE'][10] = 2 # Peak Flux = 2
	light_curve['RATE'][11] = 1 # Fluence = 3

	test_fluence, test_flux = calc_fluence(light_curve, 2, 0)

	np.testing.assert_equal(test_fluence, 3.0)
	np.testing.assert_equal(test_flux, 2.0)

if __name__ == "__main__":
	testFluenceCalc()