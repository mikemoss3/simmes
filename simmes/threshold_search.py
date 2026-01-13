"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

This package defines useful scripts to determine the redshift at which a simulated GRB reaches a desired detection rate.
"""

import numpy as np
from numpy.random import seed
import multiprocessing as mp
from functools import partial
from scipy.optimize import curve_fit

from simmes.simulations import many_simulations
from simmes.RSP import RSP
from simmes.GRB import GRB

from dataclasses import dataclass

@dataclass
class PARAMS:
	"""
	Object to hold parameters values for the sigmoid function used to fit detection fraction curves

	See the Wikipedia article on Generalized Logistics Function for parameter details. 
	https://en.wikipedia.org/wiki/Generalised_logistic_function
	"""

	# Set initial parameter values
	A : float = 1.
	K : float = 0.
	B : float = 1.
	C : float = 1.
	Q : float = 1.
	M : float = 1.
	nu : float = 1.

	# Set parameter bounds
	B_lo : float = 0.
	B_hi : float = 10.

	C_lo : float = 0.
	C_hi : float = 2.

	Q_lo : float = 1.
	Q_hi : float = 20.

	M_lo : float = -10.
	M_hi : float = 40.

	nu_lo : float = 0.
	nu_hi : float = 3.

	def __init__(self, B=None, C=None, Q=None, M=None, nu=None):
		"""
		Initialize the PARAMS object with initial parameter values. 

		Attributes:
		------------------------
		B : float
		C : float
		Q : float
		M : float 
		nu : float

		Returns:
		------------------------
		None
		"""
		self.B = B
		self.C = C
		self.Q = Q
		self.M = M
		self.nu = nu


def find_detectoin_rate_curve(grb, detection_rate, trials, 
	imx, imy, ndets, 
	bgd_size = 20, ndet_max=32768, band_rate_min=14, band_rate_max=350, 
	multiproc=True, searches=1, workers = mp.cpu_count(),
	time_resolved=False, sim_triggers=False, verbose = False):
	"""
	Method used to estimate the highest redshift a given GRB could be observed
	with a detection rate equal to `threshold` (within a given tolerance).
	The GRB is simulated at the sampled redshift a number of `trials` times. 

	Attributes:
	------------------------
	grb : GRB 
		GRB class object that holds the template GRB
	detection_rate : float
		The ratio of successful detections to total trials desired by the user
	trials : int 
		Number of trials to perform at each sampled redshift 	
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	bgd_size : float
		Background amount to add when adding in a background (in seconds)
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	multiproc : boolean
		Indicates if multiprocessing should be used
	searches : int
		Number of searches for the threshold redshift to be performed
	workers : int 
		Number of workers to use to use during a multiprocessing run
	time_resolved : boolean
		Whether or not time-resolved spectra (held by the GRB object) will be used for the simulations 
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	verbose : boolean
		Whether or not to print code activity. Only possible when not using multiple cores.

	Returns:
	------------------------

	"""

	# Redshift values to evaluate detection rates at
	z_vals = np.linspace(0.1, 15, num=15)

	# Array to store detections rates.
	det_rates_arr = np.zeros(shape=len(z_vals))

	# if multiproc:

	# 	template_grbs = np.zeros(shape=searches, dtype=GRB)
	# 	for i in range(searches):
	# 		template_grbs[i] = grb.deepcopy() # Create template GRB copies

	# 	# Set up partial function with positional arguments 
	# 	parfunc = partial(_calc_det_rat, trials = trials,
	# 							imx=imx, imy=imx, ndets=ndets,
	# 							bgd_size = bgd_size, ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
	# 							time_resolved=time_resolved, sim_triggers=sim_triggers, verbose=False)

	# 	# Set up a pool of workers
	# 	pool = mp.Pool(processes=workers, initializer=_init_process_seed)
	# 	# Run threshold searches
	# 	results = pool.map(parfunc, template_grbs)

	# 	# Place all result arrays into single structured array with format [("zth", float), ("ztrack", float)]
	# 	results = np.hstack(results)

	for i in range(len(z_vals)):
		det_rates_arr[i] = _calc_det_rat(grb, z=z_vals[i], trials = trials,
								imx=imx, imy=imx, ndets=ndets,
								bgd_size = bgd_size, ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
								time_resolved=time_resolved, sim_triggers=sim_triggers, verbose=verbose)

	params = PARAMS()

	# Fit detection rates with sigmoid to obtain empirical detection rates curve
	popt, pcov = curve_fit(detection_rate_sigmoid_fitter, xdata=z_vals, ydata=det_rates_arr, 
							p0=[params.B, params.C, params.Q, params.M, params.nu], 
							bounds=[(params.B_lo, params.C_lo, params.Q_lo, params.M_lo, params.nu_lo), 
									(params.B_hi, params.C_hi, params.Q_hi, params.M_hi, params.nu_hi)] )

	# Store best-fit values in a parameter class
	params.B = popt[0]
	params.C = popt[1]
	params.Q = popt[2]
	params.M = popt[3]
	params.nu = popt[4]

	return params

def _init_process_seed():
	# Instance a random seed
	seed()

def _calc_det_rat(grb, z, trials,
	imx, imy, ndets, 
	bgd_size = 20, ndet_max=32768, band_rate_min=14, band_rate_max=350, 
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
		Whether or not time-resolved spectra (held by the GRB object) will be used for the simulations 
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not

	Returns:
	------------------------
	detection_ratio : float
		The ratio of successful detections to the number of simulations performed
	"""
	param_list = np.array([[z, imx, imy, ndets]])  # Make param list

	resp_mat = RSP()  # Initialize a response matrix object 
	resp_mat.load_SwiftBAT_resp(imx, imy)  # Load Swift/BAT response according to given imx, imy
	
	sim_results = many_simulations(grb, param_list, trials, resp_mat=resp_mat, 
									ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max,
									bgd_size=bgd_size,
									time_resolved=time_resolved, sim_triggers=sim_triggers)  # Perform simulations of burst at this redshift

	detection_rate = len( sim_results[ sim_results['DURATION']>0 ] ) / trials  # Calculate ratio of successful detections

	return detection_rate


def detection_rate_sigmoid_fitter(z, B, C, Q, M, nu):
	"""
	Generalized logistics function used to fit the detection rate curve

	Attributes:
	------------------------
	z : float
		Redshift where the sigmoid will be evaluated

	Returns:
	------------------------
	val : float
		Value of the function at location z

	"""
	A = 1 # Left horizontal asymptote 
	# K = 0 # Right horizontal asymptote 
	numerator = -1 # = -(K - A)
	denominator = np.power(C + Q*np.exp(-B * (z-M)), 1/nu)

	val = A + (numerator/denominator)

	return val

def inverse_sigmoid(detection_rate, B, C, Q, M, nu):
	"""
	Method to calculate the redshift at which a burst has the desired detection rate. 
	This calculation uses the Inverse of the generalized logistics functions given the input parameter values.

	Attributes:
	------------------------
	detection_rate : float (0, 1)
		Desired detection rate along the best-fit sigmoid 

	Returns:
	------------------------
	z : float
		redshift at which the burst has a detection rate equal to detection_rate

	"""

	A = 1
	K = 0
	term1= np.power( (K-A)/(detection_rate-A), nu)
	term2= (term1 - C)/Q
	term3 = -np.log(term2)/B
	z = term3 + M

	return z