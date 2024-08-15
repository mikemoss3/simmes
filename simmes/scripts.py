"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

This package defines useful scripts which perform calculations based on the simmes simulation package. 
"""

import numpy as np
from simmes.simulations import many_simulations
from scipy.stats import norm, uniform

def find_z_max(grb, z_guess, threshold, num_samples,
	imx, imy, ndets, 
	ndet_max=32768, trials = 20, num_burn_in=50, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False):
	"""
	Method used to estimate the highest redshift a given GRB could be observed
	with a detection rate equal to `threshold`. A Monte-Carlo method is used to sample 
	the redshift to simulate the GRB at. The GRB is simulated at the sampled redshift 
	a number of `trials` times. 

	Attributes:
	------------------------
	grb : GRB 
		GRB class object that holds the template GRB
	z_guess : float
		An initial starting point for the Monte-Carlo algorithm
	threshold : float
		The threshold of successful detections to total trials desired by the user
	resp_mat : RSP
		Response matrix to convolve the template spectrum with. If no response matrix is given, 
		a Swift/BAT response matrix is assumed from the given imx, imy
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	trials : int 
		Number of trials to perform at each sampled redshift 
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not

	Returns:
	------------------------
	z_max : float
		The maximum redshift that the given GRB can be detected above a specified threshold 
		for the specified observing conditions.
	z_samples : ndarray (float)
		Array of redshifts found by the algorithm
	"""

	z_samples = []

	z_curr = z_guess
	# Calculate likelihood for initial redshift 
	lh_curr = _calc_likelihood(grb, z_curr, imx, imy, ndets, trials, threshold)

	for i in range(num_burn_in):
		# Sample new redshift value from distribution
		z_new = norm(loc=z_curr, scale=1).rvs(size=1)[0]
		# Calculate likelihood for new redshift
		lh_new = _calc_likelihood(grb, z_new, imx, imy, ndets, trials, threshold)
		
		# Determine if the new redshift should be accepted or not
		lh_acc = uniform.rvs(size=1)[0]  # Randomly determined acceptance criterion
		if (lh_new / lh_curr) < lh_acc:
			# Accepted: update current redshift and likelihood
			z_curr = z_new
			lh_curr = lh_new

	for i in range(num_burn_in, num_samples):
		# Sample new redshift value from distribution
		z_new = norm(loc=z_curr, scale=1).rvs(size=1)[0]
		# Calculate likelihood for new redshift
		lh_new = _calc_likelihood(grb, z_new, imx, imy, ndets, trials, threshold)
		
		# Determine if the new redshift should be accepted or not
		lh_acc = uniform.rvs(size=1)[0]  # Randomly determined acceptance criterion
		if (lh_new / lh_curr) < lh_acc:
			# Accepted: add new redshift to samples and update current redshift and likelihood
			z_samples.append(z_new)
			z_curr = z_new
			lh_curr = lh_new

	z_max = z_curr

	# Perform simulation `trials` times
	return z_max, z_samples

def _calc_likelihood(grb, z, imx, imy, ndets, trials, threshold):
	param_list = np.array([[z, imx, imy, ndets]])  # Make param list
	sim_results = many_simulations(grb, param_list, trials)  # Perform simulations of burst at this redshift
	det_ratio = len( sim_results[ sim_results['DURATION']>0 ] ) / trials  # Calculate number of successful detections
	lh = np.abs(det_ratio - threshold)  # Calculate difference from threshold

	return lh

