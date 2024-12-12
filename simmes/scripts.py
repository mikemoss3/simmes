"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

This package defines useful scripts which perform calculations based on the simmes simulation package. 
"""

import numpy as np
from scipy.stats import halfnorm
from simmes.simulations import many_simulations
from simmes.RSP import RSP

class Params(object):
	"""
	Object to hold parameters of the search algorithm
	"""
	def __init__(self):
		self.difference = None
		self.z_lo = None
		self.z_hi = None

def find_z_threshold(grb, threshold, 
	imx, imy, ndets, 
	trials = 20, tolerance=1,
	search_method = "Bisection",
	z_min = None, z_max = None, z_guess = None,
	ndet_max=32768, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False, track_z=False):
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
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	trials : int 
		Number of trials to perform at each sampled redshift 
	tolerance : float
		Determines the accuracy range of the method. Accuracy = tolerance * (1/trials), 
		since 1/trials determines the minimum accuracy.
	search_method : string
		Options include "Gaussian" and "Bisection".
		Indicates which search algorithm to use to find the threshold redshift
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
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
		print("Threshold must be between [0, 1].")
		return 0, None

	tolerance_factor = (1/trials) * tolerance  # Calculate tolerance factor

	z_th = z_guess  # Initialize current redshift 
	difference = 0
	if track_z is True:
		z_th_samples = [z_guess]  # Keep track of redshift selections 

	if search_method == "Bisection":
		if (z_min==None) or (z_max==None):
			print("If Bisection search algorithm is selected, initial redshift bounds (z_min and z_max) must be given.")
			return None, None 
		z_lo = z_min
		z_hi = z_max
		z_th = (z_hi + z_lo)/2

		params = Params()
		method = _bisection
	elif search_method == "Guassian":
		if z_guess is None:
			print("If Gaussian search algorithm is selected, initial redshift guess must be given.")
		diff_prev = 0
		z_th = z_guess

		params = Params()
		method = _half_gaussian

	# Calculate the distance from the threshold value for the initial redshift 
	det_rat_curr = _calc_det_rat(grb, z_th, threshold, trials, 
								imx, imy, ndets,  
								ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
								time_resolved=time_resolved, sim_triggers=sim_triggers)
	# Initial difference between the current and desired detection ratio.
	params.difference = det_rat_curr - threshold  # Must be between -1 and 1

	flag = True
	while flag:
		# Update redshift guess (and parameter values)
		z_th, params = method(z_th, params)

		if z_th <= 0: z_th = 1e-3  # Make sure z > 0
		if track_z is True: z_th_samples.append(z_th)  # If indicated, track new redshift guess

		# Calculate detection ratio for the current redshift guess
		det_rat_curr = _calc_det_rat(grb, z_th, threshold, trials, 
									imx, imy, ndets, 
									ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
									time_resolved=time_resolved, sim_triggers=sim_triggers)
		# Calculate difference from threshold for this redshift 
		params.difference = det_rat_curr - threshold

		# If the current difference from the desired detection threshold is within the accepted tolerance (and above zero), then we've found our redshift
		if (np.abs(params.difference) <= tolerance_factor) and (det_rat_curr>0):
			flag = False

	return z_th, z_th_samples

def _bisection(z_th, params):
	"""
	Use a bisection method to determine a new redshift. The difference between the current and the 
	desired detection ratios determines which bisection segment to use and how to update the bounds.

	Attributes:
	------------------------
	z_th : float
		Current threshold redshift guess
	differene : float 
		Difference between the current and desired detection ratios
	z_lo, z_hi : float, float
		Lower and upper bounds of redshift range

	Returns:
	------------------------
	z_th : float
		New threshold redshift guess
	z_lo, z_hi : float, float
		Updated lower and upper bounds of redshift range
	"""

	if params.difference > 0:
		params.z_lo = z_th
		z_th = (params.z_hi + params.z_lo)/2
	if params.difference < 0:
		params.z_hi = z_th
		z_th = (params.z_hi + params.z_lo)/2

	return z_th, params

def _half_gaussian(z_th, params):
	"""
	Use a half-Gaussian distribution to determine the next redshift. The mean of the distribution 
	is the current redshift guess. The standard deviation of the distribution 
	should be the difference between the current and desired detection ratios (sign included).

	Attributes:
	------------------------
	z_th : float
		Mean of the half Gaussian, i.e., the current threshold redshift guess
	difference : float
		Standard deviation of the half Gaussian, i.e., the difference between the current and desired detection ratios

	Returns:
	------------------------
	x : float
		Random parameter value picked from the distribution
	"""

	# Select new redshift using a half-normal distribution in the direction required to match the threshold
	z_th = (params.difference/np.abs(params.difference))*halfnorm(loc=z_th, scale=np.abs(params.difference)).rvs(size=1)[0]
	
	return z_th, params


def _calc_det_rat(grb, z, threshold, trials,
	imx, imy, ndets,  
	ndet_max=32768, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False):
	"""
	Calculates the ratio of successful detections vs the number of simulations performed, i.e., the detection ratio

	Attributes:
	------------------------
	grb : GRB 
		GRB class object that holds the template GRB
	z : float
		An initial starting point for the Monte-Carlo algorithm
	threshold : float
		The threshold of successful detections to total trials desired by the user
	imx, imy : float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	trials : int 
		Number of trials to perform at each sampled redshift 
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	time_resolved : boolean
		Whether or not to use time resolved spectra (held by the GRB object)
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not

	Returns:
	------------------------
	det_ratio : float
		The ratio of successful detections to the number of simulations performed
	"""
	param_list = np.array([[z, imx, imy, ndets]])  # Make param list
	resp_mat = RSP()  # Initialize a response matrix object 
	sim_results = many_simulations(grb, param_list, trials, resp_mat=resp_mat, 
									ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max,
									time_resolved=time_resolved, sim_triggers=sim_triggers)  # Perform simulations of burst at this redshift
	det_ratio = len( sim_results[ sim_results['DURATION']>0 ] ) / trials  # Calculate ratio of successful detections

	return det_ratio


