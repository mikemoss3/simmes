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
import warnings 

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
	A : float = 1. # Fixed value
	K : float = 0. # Fixed value

	B : float = 1.
	C : float = 1.
	Q : float = 1.
	M : float = 1.
	nu : float = 1.

	# Parameter uncertainties
	uncB : float = 0.
	uncC : float = 0.
	uncQ : float = 0.
	uncM : float = 0
	uncnu : float = 0.

	# Parameter bounds
	B_lo : float = 0.
	B_hi : float = 100.

	C_lo : float = 0.
	C_hi : float = 20.

	Q_lo : float = 0.
	Q_hi : float = 20.

	M_lo : float = -20.
	M_hi : float = 20.

	nu_lo : float = 0.
	nu_hi : float = 10.

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
		if B is not None:
			self.B = B
		if C is not None:
			self.C = C
		if Q is not None:
			self.Q = Q
		if M is not None:
			self.M = M
		if nu is not None:
			self.nu = nu

	def __str__(self):
		names = ["A", "K", "B", "C", "Q", "M", "nu"]
		fields = [self.A, self.K, self.B, self.C, self.Q, self.M, self.nu]

		return '\n'.join("{} = {:.3e}".format(names[i], fields[i]) for i in range(len(names)))

	def get(self, var):
		"""
		Method that returns the current value and uncertainty of the specified variable

		Attributes:
		------------------------
		var = str
			Variable field name

		Returns:
		------------------------
		ret : np.array([float, float])
			Array storing the current parameter value and uncertainty
		"""
		if var == "A":
			ret = np.array([self.A, 0.])
		if var == "K":
			ret = np.array([self.K, 0.])
		if var == "B":
			ret = np.array([self.B, self.uncB])
		if var == "C":
			ret = np.array([self.C, self.uncC])
		if var == "Q":
			ret = np.array([self.Q, self.uncQ])
		if var == "M":
			ret = np.array([self.M, self.uncM])
		if var == "nu":
			ret = np.array([self.nu, self.uncnu])

		return ret

	def save_params(self, fn):
		"""
		Method to save the parameter values to a file with the defined file path

		Attributes:
		------------------------
		fn = str
			File name and path
		"""

		out_arr = np.array([
			
			], dtype=float)
		with open(fn, "w") as f:
			f.write("# A\tUncA\tK\tUncK\tB\tUncB\tC\tUncC\tQ\tUncQ\tM\tUncM\tnu\tUncnu\n")
			f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(self.get("B")[0], self.get("B")[1], 
																																			self.get("C")[0], self.get("C")[1], 
																																			self.get("Q")[0], self.get("Q")[1], 
																																			self.get("M")[0], self.get("M")[1], 
																																			self.get("nu")[0], self.get("nu")[1]))

	def load_params_from_file(self, fn):
		"""
		Method to load parameter values from a file with the defined file path

		Attributes:
		------------------------
		fn = str
			File name and path
		"""

		params = np.genfromtxt(fname=fn)

		params.B = params[0]
		params.uncB = params[1]
		params.C = params[2]
		params.uncC = params[3]
		params.Q = params[4]
		params.uncQ = params[5]
		params.M = params[6]
		params.uncM = params[7]
		params.nu = params[8]
		params.uncnu = params[9]

def _save_det_curve_samples(fn_prefix, z_vals, detection_rates):
	"""
	Method to save the parameter values to a file with the defined file path

	Attributes:
	------------------------
	fn_prefix = str
		File name prefix (and path) to save file to
	z_vals : np.ndarray([floats])
		Array of sampled redshift values 
	detection_rates : np.ndarray([floats])
		Array of sampled detection rates at the given redshift values
	"""

	combined_data = list(zip(z_vals, detection_rates))
	np.savetxt(fname=fn_prefix+"_det_curves_samples.txt", X=combined_data, fmt="%.3f %.3f", header="z\tDet Frac.")

def sample_detectoin_rate_curve(grb, trials,
	imx, imy, ndets, 
	z_vals = None, z_max = 15, num_samples = 15,
	bgd_size = 20, ndet_max=32768, band_rate_min=15, band_rate_max=150, 
	multiproc=True, workers = mp.cpu_count(),
	time_resolved=False, measure_durs=False, sim_triggers=True, verbose = False, fn_prefix=None):
	"""
	Method used to estimate the highest redshift a given GRB could be observed
	with a detection rate equal to `threshold` (within a given tolerance).
	The GRB is simulated at the sampled redshift a number of `trials` times. 

	Attributes:
	------------------------
	grb : GRB 
		GRB class object that holds the template GRB
	trials : int 
		Number of trials to perform at each sampled redshift 	
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	z_vals : np.1darray of floats
		Array of redshift values to calculate the detection fraction. If None is given
		z_max and num_samples will be used to generate a z_vals array.
	z_max : float 
		Maximum redshift to test out to 
	num_samples : int
		Number of redshifts between z and z_max to test the detectio rate at 
	bgd_size : float
		Background amount to add when adding in a background (in seconds)
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	multiproc : boolean
		Indicates if multiprocessing should be used
	workers : int 
		Number of workers to use to use during a multiprocessing run
	time_resolved : boolean
		Whether or not time-resolved spectra (held by the GRB object) will be used for the simulations 
	measure_durs : boolean
		Indicates whether to use Bayesian blocks to measure the duration of the simulated light curve or not
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	verbose : boolean
		Whether or not to print code activity. Only possible when not using multiple cores.

	Returns:
	------------------------
	z_vals : np.ndarray([floats])
		Array of sampled redshift values 
	detection_rates : np.ndarray([floats])
		Array of sampled detection rates at the given redshift values

	"""

	# Redshift values to evaluate detection rates at
	if z_vals is None:
		z_vals = np.linspace(grb.z, z_max, num=num_samples)
	else:
		num_samples = len(z_vals)

	# Array to store detections rates.
	detection_rates = np.zeros(shape=num_samples)

	if multiproc:

		template_grbs = np.zeros(shape=num_samples, dtype=GRB)
		for i in range(num_samples):
			template_grbs[i] = grb.deepcopy() # Create template GRB copies

		# Set up partial function with positional arguments 
		parfunc = partial(_calc_det_rat, trials = trials,
								imx=imx, imy=imy, ndets=ndets,
								bgd_size = bgd_size, ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
								time_resolved=time_resolved, measure_durs=measure_durs, sim_triggers=sim_triggers, verbose=False)

		# # Set up a pool of workers
		# pool = mp.Pool(processes=workers, initializer=_init_process_seed)
		# # Run threshold searches
		# results = pool.starmap(parfunc, list(zip(template_grbs, z_vals)))

		warnings.filterwarnings("ignore")
		with mp.Pool(processes=workers, initializer=_init_process_seed) as pool:
			results = pool.starmap(parfunc, list(zip(template_grbs, z_vals)))

		# Place all result arrays into single structured array with format [("zth", float), ("ztrack", float)]
		detection_rates = np.hstack(results)
	else:
		for i in range(len(z_vals)):
			detection_rates[i] = _calc_det_rat(grb, z=z_vals[i], trials = trials,
									imx=imx, imy=imy, ndets=ndets,
									bgd_size = bgd_size, ndet_max=ndet_max, band_rate_min=band_rate_min, band_rate_max=band_rate_max, 
									time_resolved=time_resolved, measure_durs=measure_durs, sim_triggers=sim_triggers, verbose=verbose)

	if fn_prefix is not None:
		_save_det_curve_samples(fn_prefix=fn_prefix, z_vals=z_vals, detection_rates=detection_rates)

	return z_vals, detection_rates

def _init_process_seed():
	# Instance a random seed
	seed()


def _calc_det_rat(grb, z, trials,
	imx, imy, ndets, 
	bgd_size = 20, ndet_max=32768, band_rate_min=15, band_rate_max=150, 
	time_resolved=False, measure_durs=False, sim_triggers=True, verbose=False):
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
	measure_durs : boolean
		Indicates whether to use Bayesian blocks to measure the duration of the simulated light curve or not
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	verbose : boolean
		Whether or not to print code activity.

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
									bgd_size=bgd_size, quick=True,
									time_resolved=time_resolved, measure_durs=measure_durs, sim_triggers=sim_triggers, verbose=verbose)  # Perform simulations of burst at this redshift

	detection_rate = len( sim_results[ sim_results['Triggered']>0 ] ) / trials  # Calculate ratio of successful detections

	return detection_rate


def fit_detection_rate_curve(z_vals, detection_rates, params=None, **kwargs):
	"""
	Method to fit the given sampled points from a detection rate curve using the scipy.curve_fit package

	Attributes:
	------------------------
	z_vals : np.ndarray([floats])
		Array of sampled redshift values 
	detection_rates : np.ndarray([floats])
		Array of sampled detection rates at the given redshift values
	params : PARAMS, optional
		Object to provide user-defined initial parameter values and parameter bounds

	Returns:
	------------------------
	params : PARAMS
		Object storing the best fit parameter values

	"""

	if params is None:
		params = PARAMS()

	# Fit detection rates with sigmoid to obtain empirical detection rates curve
	# popt, pcov = curve_fit(detection_rate_sigmoid_fitter, xdata=z_vals, ydata=detection_rates, 
	popt, pcov = curve_fit(lambda z, B, C, Q, M, nu: detection_rate_sigmoid_fitter(z, B, C, Q, M, nu, A=1, K=0), 
							xdata=z_vals, ydata=detection_rates, 
							p0=[params.B, params.C, params.Q, params.M, params.nu], 
							bounds=[(params.B_lo, params.C_lo, params.Q_lo, params.M_lo, params.nu_lo), 
									(params.B_hi, params.C_hi, params.Q_hi, params.M_hi, params.nu_hi)], **kwargs)

	# Store best-fit values in a parameter class
	params.B = popt[0]
	params.C = popt[1]
	params.Q = popt[2]
	params.M = popt[3]
	params.nu = popt[4]

	perr = np.sqrt(np.diag(pcov))

	params.uncB = perr[0]
	params.uncC = perr[1]
	params.uncQ = perr[2]
	params.uncM = perr[3]
	params.uncnu = perr[4]

	return params

def detection_rate_sigmoid_fitter(z, B, C, Q, M, nu, A=1, K=0):
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
	# A = 1 # Left horizontal asymptote 
	# K = 0 # Right horizontal asymptote 
	numerator = (K - A)
	denominator = np.power(C + Q*np.exp(-B * (z-M)), 1/nu)

	val = A + (numerator/denominator)

	return val

def calc_z_threshold(detection_rate, B, C, Q, M, nu, A=1, K=0):
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

	term1 = np.power( (K-A)/(detection_rate-A), nu)
	term2 = (term1 - C)/Q
	term3 = -np.log(term2)/B
	z = term3 + M

	return z