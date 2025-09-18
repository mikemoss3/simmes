"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines unit tests for the simmes.bayesian_block module

"""

import numpy as np

def testBayesianBlocks():
	"""
	Testing whether the Bayesian block method can accurately measure duration of a square light cure
	"""
	from simmes.bayesian_block import bayesian_t_blocks

	# Create simple square light curve
	light_curve = np.zeros(shape=20, dtype=[("TIME", float), ("RATE",float), ("UNC", float)])
	tb_size = 1
	light_curve['TIME'] = np.arange(-10, 10, step=tb_size)
	light_curve['RATE'][10] = 2 # Peak Flux = 2
	light_curve['RATE'][11] = 1 # Fluence = 3

	light_curve['UNC'] += 0.01

	test_dur, test_tstart = bayesian_t_blocks(light_curve, dur_per=100)

	np.testing.assert_equal(test_dur, 2.0)
	# True start is at t=0, but Bayesian block algorithm reports bin edges 
	np.testing.assert_equal(test_tstart, -0.5) 

if __name__ == "__main__":
	testBayesianBlocks()