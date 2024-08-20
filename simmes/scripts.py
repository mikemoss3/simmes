"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

This package defines useful scripts which perform calculations based on the simmes simulation package. 
"""

import numpy as np
from simmes.simulations import many_simulations
from scipy.stats import halfnorm

def find_z_max(grb, z_guess, threshold, 
	imx, imy, ndets, 
	ndet_max=32768, trials = 20, tolerance=1, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False):
	"""
	Method used to estimate the highest redshift a given GRB could be observed
	with a detection rate equal to `threshold` (within a given tolerance).
	The GRB is simulated at the sampled redshift a number of `trials` times. 

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
	tolerance : float
		Determines the accuracy range of the method. Accuracy = tolerance * (1/trials), 
		since 1/trials determines the minimum accuracy.
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

	if (threshold > 1) or (threshold < 0):
		print("Threshold must be between [0,1].")
		return 0, None

	tolerance_factor = (1/trials) * tolerance

	z_samples = []  # Keep track of redshift selections 
	z_samples.append(z_guess)
	# Calculate the distance from the threshold value for this redshift 
	det_rat_curr = _calc_dist(grb, z_guess, imx, imy, ndets, trials, threshold)
	# Initialize the difference between the current and previous distance calculations.
	diff_curr = det_rat_curr - threshold  # Should be between -1 and 1

	flag = True
	while flag:
		# Update variables
		det_rat_prev = det_rat_curr
		diff_prev = diff_curr

		# Select new redshift using a half-normal distribution in the direction required to match the threshold
		z_curr = z_samples[-1] + (diff_prev/np.abs(diff_prev))*halfnorm(loc=0, scale=np.abs(diff_prev)).rvs(size=1)[0]
		z_samples.append(z_curr)

		# Calculate distance from threshold for this redshift 
		det_rat_curr = _calc_dist(grb, z_curr, imx, imy, ndets, trials, threshold)
		diff_curr = det_rat_curr - threshold

		if (np.abs(diff_curr) <= tolerance_factor) and (det_rat_curr>0):
			flag = False

	z_max = z_curr

	return z_max, z_samples

def _calc_dist(grb, z, imx, imy, ndets, trials, threshold):
	param_list = np.array([[z, imx, imy, ndets]])  # Make param list
	sim_results = many_simulations(grb, param_list, trials)  # Perform simulations of burst at this redshift
	det_ratio = len( sim_results[ sim_results['DURATION']>0 ] ) / trials  # Calculate number of successful detections

	return det_ratio


