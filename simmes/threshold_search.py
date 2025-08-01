"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

This package defines useful scripts which perform calculations based on the simmes simulation package. 
"""

import numpy as np
from numpy.random import seed
from scipy.stats import halfnorm
import multiprocessing as mp
from functools import partial

from simmes.simulations import many_simulations
from simmes.RSP import RSP
from simmes.GRB import GRB

class PARAMS(object):
	"""
	Object to hold parameters of the search algorithm
	"""
	def __init__(self, threshold, trials, tolerance, z_tolerance, verbose=False):
		"""
		Method used to estimate the highest redshift a given GRB could be observed
		with a detection rate equal to `threshold` (within a given tolerance).
		The GRB is simulated at the sampled redshift a number of `trials` times. 

		Attributes:
		------------------------
		threshold : float
			Desired detection threshold 
		trials : 
			Number of trial simulated to be performed at each threshold redshift
		tolerance : float
			Acceptance range between the calculated detection ratio and the desired detection threshold
		z_tolerance : float
			Used to prevent the search algorithm from getting stuck
		
		Returns:
		------------------------
		None
		"""

		self.threshold = threshold
		self.trials = trials
		self.tolerance = tolerance
		self.z_tolerance = z_tolerance

		self.z_lo = None  # Initial redshift range lower bound
		self.z_hi = None  # Initial redshift range upper bound

		self.z_th = None  # Threshold redshift to perform simulation
		self.z_th_samples = []  # Use to store z_th values
		self.det_ratio = None  # Current detection ratio found for z_th
		self.det_ratio_samples = [] # Used to store detection ratio values
		self.difference = None  # Difference between detection rate and threshold 
		self.sign = 1  # Sign of `difference` variable

		self.iter = 0  # Iterator

		self.flag = True  # Flag indicating whether the search should continue
		self.verbose = verbose  # Print reason for ending search

	def check_sign(self):
		"""
		Method to check the sign of the difference variable. This is use to determine if the 
		Gaussian search algorithm is stuck (e.g., in a high-z region) and needs a kick.
		"""
		if(np.sign(self.difference) == -1):
			self.iter +=1
			if (self.iter == 10):
				self.z_th/=2
				self.iter = 0
		else:
			self.iter = 0

	def check_zrange(self):
		"""
		Method to check the redshift range. If the range is smaller than the desired z_tolerance, end the search.
		"""
		if( (self.z_hi - self.z_lo) < self.z_tolerance ):
			self.flag = False  # End the search
			if self.verbose is True:
				print("Redshift change is below tolerance.")

	def check_result(self):
		"""
		If the current difference from the desired detection threshold is within the accepted tolerance 
		(and above zero), then we've found our redshift.
		"""
		if (np.abs(self.difference) <= self.tolerance) and (self.det_ratio>self.tolerance):
			self.flag = False
			if self.verbose is True:
				print("Detection ration difference is below tolerance.")

def find_z_threshold(grb, threshold, tolerance, 
	imx, imy, ndets, trials, 
	z_min, z_max, searches=1,
	bgd_size = 20, multiproc=True, workers = mp.cpu_count(),
	search_method = "Bisection", z_tolerance=0.05,
	ndet_max=32768, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False, track_z=False, 
	verbose = False):
	"""
	Method used to estimate the highest redshift a given GRB could be observed
	with a detection rate equal to `threshold` (within a given tolerance).
	The GRB is simulated at the sampled redshift a number of `trials` times. 

	Attributes:
	------------------------
	grb : GRB 
		GRB class object that holds the template GRB
	threshold : float
		The ratio of successful detections to total trials desired by the user
	tolerance : float
		Determines acceptance range around the desired detection threshold 
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	trials : int 
		Number of trials to perform at each sampled redshift 
	z_min, z_max : float, float
		Bounds of the redshift search
	searches : int
		Number of searches for the threshold redshift to be performed
	bgd_size : float
		Background amount to add when adding in a background (in seconds)
	multiproc : boolean
		Indicates if multiprocessing should be used
	workers : int 
		Number of workers to use to use during a multiprocessing run
	search_method : string
		Options include "Gaussian" and "Bisection".
		Indicates which search algorithm to use to find the threshold redshift
	z_tolerance : float
		If the change in redshift between two searches is less than z_tolerance, the simulation will end.
		This is to avoid possible non-covergence issues.
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	track_z : boolean
		Whether or not to track the redshift guesses evaluated by the search algorithm
	verbose : boolean
		Whether or not to print code activity. Only possible when not using multiple cores.

	Returns:
	------------------------
	z_max : float
		The maximum redshift that the given GRB can be detected above a specified threshold 
		for the specified observing conditions.
	z_samples : ndarray(dtype = float) (optional)
		Array of redshifts found by the algorithm. Only returned if track_z = True
	"""

	if (threshold > 1) or (threshold < 0):
		print("Threshold must be between [0, 1].")
		return None

	if (z_min==None) or (z_max==None):
		print("Please supply redshift bounds for the search algorithm.")
		return None
	algorithms = np.array(["Bisection", "Gaussian"])
	if search_method not in algorithms:
		print("Please search methods: Bisection or Gaussian.")
		return None

	if multiproc:

		template_grbs = np.zeros(shape=searches, dtype=GRB)
		for i in range(searches):
			template_grbs[i] = grb.deepcopy() # Create template GRB copies

		# Set up partial function with positional arguments 
		parfunc = partial(_find_z_threshold_work, threshold=threshold, tolerance=tolerance,
								imx=imx, imy=imx, ndets=ndets,
								trials = trials, z_min = z_min, z_max = z_max, bgd_size = bgd_size,
								search_method = search_method, z_tolerance=0.05,
								ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
								time_resolved=time_resolved, sim_triggers=sim_triggers, track_z=track_z, verbose=False)
		# Set up a pool of workers
		pool = mp.Pool(processes=workers, initializer=_init_process_seed)
		# Run threshold searches
		results = pool.map(parfunc, template_grbs)

		# Place all result arrays into single structured array with format [("zth", float), ("ztrack", float)]
		results = np.hstack(results)

	else:
		results = _find_z_threshold_work(grb, threshold=threshold, tolerance=tolerance,
								imx=imx, imy=imx, ndets=ndets,
								trials = trials, z_min = z_min, z_max = z_max, bgd_size = bgd_size,
								search_method = search_method, z_tolerance=0.05,
								ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
								time_resolved=time_resolved, sim_triggers=sim_triggers, track_z=track_z, verbose=verbose)

	return results

def _init_process_seed():
	# Instance a random seed
	seed()

def _find_z_threshold_work(grb, threshold, tolerance,
	imx, imy, ndets, 
	trials, z_min, z_max,
	bgd_size = 20, search_method = "Bisection", z_tolerance=0.05,
	ndet_max=32768, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False, track_z=False, verbose = False):
	"""
	Method used to estimate the highest redshift a given GRB could be observed
	with a detection rate equal to `threshold` (within a given tolerance).
	The GRB is simulated at the sampled redshift a number of `trials` times. 

	Attributes:
	------------------------
	grb : GRB 
		GRB class object that holds the template GRB
	threshold : float
		The ratio of successful detections to total trials desired by the user
	tolerance : float
		Determines acceptance range around the desired detection threshold
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	trials : int 
		Number of trials to perform at each sampled redshift 
	z_min, z_max : float, float
		Bounds of the redshift search
	bgd_size : float
		Background amount to add when adding in a background (in seconds)
	search_method : string
		Options include "Gaussian" and "Bisection".
		Indicates which search algorithm to use to find the threshold redshift
	z_tolerance : float
		If the change in redshift between two searches is less than z_tolerance, the simulation will end.
		This is to avoid possible non-covergence issues.
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	track_z : boolean
		Whether or not to track the redshift guesses evaluated by the search algorithm
	verbose : boolean
		Whether or not to print code activity

	Returns:
	------------------------
	z_max : float
		The maximum redshift that the given GRB can be detected above a specified threshold 
		for the specified observing conditions.
	z_samples : ndarray(dtype = float) (optional)
		Array of redshifts found by the algorithm. Only returned if track_z = True
	"""

	# Set up parameter storage and search method 
	p = PARAMS(threshold=threshold, trials=trials, tolerance=tolerance, z_tolerance=z_tolerance, verbose=verbose)
	p.z_lo = z_min
	p.z_hi = z_max
	if search_method == "Bisection":
		method = _bisection
		p.z_th = (p.z_hi + p.z_lo)/2
	elif search_method == "Gaussian":
		method = _half_gaussian
		p.difference = 0
		p.z_th = np.random.uniform(low=p.z_lo, high=p.z_hi)

	# Calculate the distance from the threshold value for the initial redshift 
	p.det_ratio = _calc_det_rat(grb=grb, z=p.z_th, trials=p.trials, 
								imx=imx, imy=imy, ndets=ndets, bgd_size=bgd_size, 
								ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
								time_resolved=time_resolved, sim_triggers=sim_triggers)

	if track_z is True: 
		# If indicated, track new redshift guess and the new detection ratio
		p.z_th_samples.append(p.z_th) 
		p.det_ratio_samples.append(p.det_ratio)

	# Initial difference between the current and desired detection ratio.
	p.difference = p.det_ratio - p.threshold  # Must be between -1 and 1
	if verbose is True:
		print("At z = {:.2f}, detection ratio = {:.3f}".format(p.z_th, p.det_ratio))
		
	# Check if we've found a redshift with a detection ratio within the tolerance of the desired detection ratio
	p.check_result()

	while p.flag:
		# Update redshift guess (and parameter values)
		method(p)

		if p.z_th <= 0: p.z_th = 1e-3  # Make sure z > 0 
		
		# Calculate detection ratio for the current redshift guess
		p.det_ratio = _calc_det_rat(grb=grb, z=p.z_th, trials=p.trials, 
									imx=imx, imy=imy, ndets=ndets, bgd_size=bgd_size,
									ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
									time_resolved=time_resolved, sim_triggers=sim_triggers)
		if track_z is True: 
			# If indicated, track new redshift guess and the new detection ratio
			p.z_th_samples.append(p.z_th) 
			p.det_ratio_samples.append(p.det_ratio)

		# Calculate difference from threshold for this redshift 
		p.difference = p.det_ratio - p.threshold

		if verbose is True:
			print("At z = {:.2f}, detection ratio = {:.3f}".format(p.z_th, p.det_ratio))

		# Check if we've found a redshift with a detection ratio within the tolerance of the desired detection ratio
		p.check_result()

	if track_z is True:
		return np.array( (p.z_th, p.det_ratio, np.array(p.z_th_samples), np.array(p.det_ratio_samples)), dtype=[("zth",float), ("DetRat", float), ("ztrack", object), ("DetRattrack", object)] )
	else:
		return np.array( (p.z_th, p.det_ratio), dtype=[("zth", float), ("DetRat", float)] )

def _bisection(params):
	"""
	Use a bisection method to determine a new redshift. The difference between the current and the 
	desired detection ratios determines which bisection segment to use and how to update the bounds.

	Attributes:
	------------------------
	params : PARAMS
		PARAMS object that necessary redshift guess infomation.
	"""

	if params.difference > 0:
		params.z_lo = params.z_th
		params.z_th = (params.z_hi + params.z_lo)/2
	if params.difference < 0:
		params.z_hi = params.z_th
		params.z_th = (params.z_hi + params.z_lo)/2

	params.check_zrange()

def _half_gaussian(params):
	"""
	Use a half-Gaussian distribution to determine the next redshift. The mean of the distribution 
	is the current redshift guess. The standard deviation of the distribution 
	should be the difference between the current and desired detection ratios (sign included).

	Attributes:
	------------------------
	params : PARAMS
		PARAMS object that necessary redshift guess infomation.
	"""

	# Select new redshift using a half-normal distribution in the direction required to match the threshold
	params.z_th = params.z_th + np.sign(params.difference)*halfnorm(loc=0, scale=np.abs(params.difference)/2).rvs(size=1)[0]

	# The below check is used to make sure the search doesn't get stuck searching in a region of no detections (i.e., too-high redshifts)
	params.check_sign()

def _calc_det_rat(grb, z, trials,
	imx, imy, ndets, bgd_size = 20, 
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
	imx, imy : float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	bgd_size : float
		Background amount to add when adding in a background (in seconds)
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
	resp_mat.load_SwiftBAT_resp(imx, imy)  # Load Swift/BAT response according to given imx, imy
	sim_results = many_simulations(grb, param_list, trials, resp_mat=resp_mat, 
									ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max,
									bgd_size=bgd_size,
									time_resolved=time_resolved, sim_triggers=sim_triggers)  # Perform simulations of burst at this redshift

	det_ratio = len( sim_results[ sim_results['DURATION']>0 ] ) / trials  # Calculate ratio of successful detections

	return det_ratio
