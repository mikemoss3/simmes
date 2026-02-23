"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	
Last edited: 2025-07-15

This file defines functions that deal with adding fluctuations/variations to synthetic light curves and spectra

"""

import numpy as np
from pathlib import Path
import pickle
from scipy.stats import rv_discrete
path_here = Path(__file__).parent

################################################
# Random Light Curve Variance Methods
################################################

def rand_draw(PCODE, NDETS, dr_max=0.05, pcode_max = 1.05, ndet_max = 32768):
	"""
	Randomly selects a PCODE and NDETS from a circular region around the center 
	given by (x, y) = (PCODE, NDETS).
	
	Attributes:
	--------------
	PCODE : float 
		Partial coding fraction of the observation. 
	NDETS : int 
		Number of detector on the detector plane enabled during observations
	dr_max : float
		Percentage max away from highest possible evaluation 
	pcode_max : float 
		Maximum possible partial coding
	ndets_max : int 
		Maximum possible number of enabled detectors

	Returns:
	--------------
	variance : float
		Randomly selected background variance. Units counts / sec / cm^2.
	"""

	theta_i = np.random.randint(low=0, high=360)
	dr_i = np.random.uniform(low=0, high=dr_max)

	dpcode = np.cos(theta_i)*dr_i
	dndet = np.sin(theta_i)*dr_i

	pcode_i = PCODE + pcode_max*dpcode
	pcode_i = np.min([ pcode_i, pcode_max ])
	pcode_i = np.max([ pcode_i, 0. ])

	ndets_i = NDETS + ndet_max*dndet
	ndets_i = np.min([ ndets_i, ndet_max])
	ndets_i = np.max([ ndets_i, 0. ])

	return pcode_i, ndets_i 

def rand_lc_variance(PCODE, NDETS, size=1, dr_max=0.05, pcode_max = 1.05, ndet_max = 32768):
	"""
	Randomly selects a standard deviation value from a 2D PDF created from Swift/BAT observations.
	
	Attributes:
	--------------
	PCODE : float 
		Partial coding fraction of the observation. 
	NDETS : int 
		Number of detector on the detector plane enabled during observations 
	size : int
		Number of backgrounds to sample
	dr_max : float
		Percentage max away from highest possible evaluation 
	pcode_max : float 
		Maximum possible partial coding
	ndets_max : int 
		Maximum possible number of enabled detectors

	Returns:
	--------------
	variance : float
		Randomly selected background variance. Units counts / sec / cm^2.
	"""

	variance = np.zeros(shape=size)

	# Load interpolated background variances
	with open(path_here.joinpath("files-det-ang-dependence/pickled_interpolator.pck"), 'rb') as file_handle:
		f_inter = pickle.load(file_handle)

	for i in range(size):
		rand_pcode, rand_ndets = rand_draw(PCODE=PCODE, NDETS=NDETS, dr_max=dr_max, pcode_max=pcode_max, ndet_max=ndet_max)
		variance[i] = f_inter(rand_pcode, rand_ndets)

	return variance

def add_light_curve_flucations(light_curve, t_bin_size, PCODE, NDETS, dr_max=0.05, pcode_max = 1.05, ndet_max = 32768):
	"""
	Method to add a randomly variance to a Swift/BAT mask-weighted light curve. 
	The variance is take from a 2D distribution created from the 
	measured light curve background variances observed by Swift/BAT.

	Attributes:
	--------------
	light_curve : np.ndarray with [("RATE", float), ("UNC", float)]
		Light curve array
	t_bin_size : float
		Time bin size (in seconds)
	PCODE : float
		Partial coding fraction of the observation
	NDETS : int
		Number of detectors enabled on the detector plane
	dr_max : float
		Percentage max away from highest possible evaluation 
	pcode_max : float 
		Maximum possible partial coding
	ndets_max : int 
		Maximum possible number of enabled detectors

	Returns:
	--------------
	light_curve : np.ndarray with [("RATE", float), ("UNC", float)]
		Light curve array
	"""

	variance = rand_lc_variance(PCODE, NDETS, size=1, dr_max=dr_max, pcode_max=pcode_max, ndet_max=ndet_max)  # counts / sec / cm^2
	
	variance *= 0.16  # counts / sec / det
	variance /= np.sqrt(t_bin_size) # scale for time-bin size

	# Fluctuate the background according to a Normal distribution around 0 with a standard variation equal to the background variance
	light_curve['RATE'] += np.random.normal( loc=np.zeros(shape=len(light_curve)), scale=variance)
	# Set the uncertainty of the count rate to the variance. 
	light_curve['UNC'] = np.ones(shape=len(light_curve))*variance

	return light_curve


################################################
# Random Spectrum Variance Methods
################################################

def fred_function(t, fm, tm, r, d):
	"""
	FRED shaped curve based on Equation 22 from Kocevski et al. 2003; power law rise and exponential decay.

	Attributes:
	--------------
	t = time since trigger
	fm = flux at peak of the pulse (fm = F(tm))
	tm = t_max or the peak time of the pulse 
	r = rise constant
	d = decay constant 

	Returns:
	--------------
	flux : float
		amplitude of the FRED function at time t
	"""

	flux = fm*np.power(t/tm,r)*np.power( (d/(d+r)) + ((r/(d+r))*np.power(t/tm,r+1)) ,-(r+d)/(r+1))

	return flux


def rand_spec_variance(fm, tm, r, d, cut_min, cut_max, size=1):
	"""
	Randomly selects a value a PDF based on a FRED function described in Kocevski et al 2003. 
	
	Attributes:
	--------------
	size : int
		Number of backgrounds to sample
	fm : float
		flux at peak of the pulse (fm = F(tm))
	tm : float
		t_max or the peak time of the pulse 
	r : float
		rise constant
	d : float
		decay constant
	cut_min, cut_max : float, float
		Bounds of the PDF. Generally should be used to omit any 
		anomolous variances found outside of these cuts.

	Returns:
	--------------
	variance : float
		Randomly selected background variance. Units counts / sec / cm^2.
	"""

	x_range = np.linspace(cut_min, cut_max, num=100)

	# Create distribution
	distrib = rv_discrete(a=cut_min, b=cut_max, values=(x_range, fred_function(x_range, fm, tm, r, d)) ) 
	# Select random value
	variance = distrib.rvs(size=size)

	return variance


def add_spec_fluctuations(spectrum):
	"""
	Method to add fluctuations to a Swift/BAT mask-weighted spectrum.
	The variances are take from statistical error distributions created from the 
	measured Swift/BAT spectra. Systematic errors fluctuations are taken from the
	known Swift/BAT systematic error table.

	Attributes:
	--------------
	spectrum : np.ndarray with [("ENERGY", float), ("RATE", float)]
		Spectrum array

	Returns:
	--------------
	spectrum : np.ndarray with [("ENERGY", float), ("RATE", float)]
		Spectrum array

	"""

	# Load best-fit parameters of per-energy-channel statistical error distributions
	stat_errors = np.loadtxt(path_here.joinpath("files-spec-unc/spec-unc-fit-params.txt"), 
								dtype=[("CHAN", int), ("FM", float), ("TM", float), ("R", float), ("D", float)])
	# Load per-energy-channel fractional systematic errors
	sys_err_arr = np.loadtxt(path_here.joinpath("files-spec-unc/bat-chan-sys-err-fracs.txt"), 
								dtype=[("CHAN", int), ("SYS_ERR", float)])

	# For each channel, grab statistical variances.
	variances = np.zeros(shape=len(spectrum))
	for i in range(len(spectrum)):
		# Grab statistical error PDF for this channel
		variances[i] = rand_spec_variance(fm = stat_errors["FM"][i], 
								tm = stat_errors["TM"][i], 
								r = stat_errors["R"][i], 
								d = stat_errors["D"][i],
								cut_min = 0, cut_max = 0.02)

	# Add statistical fluctuations
	spectrum['RATE'] += np.random.normal( loc=np.zeros(shape=len(spectrum)), scale=variances)

	# Add systematic fluctuations
	spectrum['RATE'] += np.random.normal( loc=np.zeros(shape=len(spectrum)), scale=np.abs(spectrum['RATE']*sys_err_arr["SYS_ERR"]))

	return spectrum