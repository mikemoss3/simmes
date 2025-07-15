"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	
Last edited: 2025-07-15

This file defines functions that deal with adding fluctuations/variations to synthetic light curves and spectra

"""

import numpy as np
from scipy.stats import rv_discrete
from pathlib import Path
path_here = Path(__file__).parent

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


def rand_variance(fm, tm, r, d, cut_min, cut_max, size=1):
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

def add_light_curve_flucations(light_curve, t_bin_size):
	"""
	Method to add a randomly variance to a Swift/BAT mask-weighted light curve. 
	The variance is take from a distribution created from the 
	measured light curve background variances observed by Swift/BAT.

	Attributes:
	--------------
	light_curve : np.ndarray with [("RATE", float), ("UNC", float)]
		Light curve array
	t_bin_size : float
		Time bin size (in seconds)

	Returns:
	--------------
	light_curve : np.ndarray with [("RATE", float), ("UNC", float)]
		Light curve array
	"""

	# There are only anomolous variances found outside of these cuts
	cut_min = 0.02
	cut_max = 0.25

	parameters = [0.04305558, 0.06859141, 9.92930295, 3.06620674]
	"""
	These parameter values were found in a separate fit.
	
	Parameters:
	fm = flux at peak of the pulse (fm = F(tm))
	tm = t_max or the peak time of the pulse 
	r = rise constant
	d = decay constant
	"""

	# Pull a random background variance from the distribution created from observed values
	variance = rand_variance(fm = parameters[0], tm = parameters[1] , r = parameters[2], d = parameters[3],
											cut_min = cut_min, cut_max = cut_max)  # counts / sec / cm^2
	
	variance *= 0.16  # counts / sec / det
	variance /= np.sqrt(t_bin_size) # scale for time-bin size

	# Fluctuate the background according to a Normal distribution around 0 with a standard variation equal to the background variance
	light_curve['RATE'] += np.random.normal( loc=np.zeros(shape=len(light_curve)), scale=variance)
	# Set the uncertainty of the count rate to the variance. 
	light_curve['UNC'] = np.ones(shape=len(light_curve))*variance

	return light_curve

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
		variances[i] = rand_variance(fm = stat_errors["FM"][i], 
								tm = stat_errors["TM"][i], 
								r = stat_errors["R"][i], 
								d = stat_errors["D"][i],
								cut_min = 0, cut_max = 0.02)
		
	# Add statistical fluctuations
	spectrum['RATE'] += np.random.normal( loc=np.zeros(shape=len(spectrum)), scale=variances)

	# Add systematic fluctuations
	spectrum['RATE'] += np.random.normal( loc=np.zeros(shape=len(spectrum)), scale=np.abs(spectrum['RATE']*sys_err_arr["SYS_ERR"]))

	return spectrum