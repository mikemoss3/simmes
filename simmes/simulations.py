"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

This package defines functions used to obtain the duration and fluence measurements for simulated GRBs
"""

import numpy as np
from scipy.stats import rv_discrete

from simmes.GRB import GRB
from simmes.RSP import RSP
from simmes.bayesian_block import bayesian_t_blocks
from simmes.fluence import calc_fluence
from simmes.util_packages.datatypes import dt_sim_res
from simmes.util_packages.det_ang_dependence import find_pcode, find_inc_ang, fraction_correction


import matplotlib.pyplot as plt
from simmes.PLOTS import PLOTGRB

def simulate_observation(synth_grb, template_grb, resp_mat, 
	imx, imy, ndets, 
	z_p=0, ndet_max=32768, band_rate_min=14, band_rate_max=350, 
	time_resolved=False, sim_triggers=False, sim_bgd=True):
	"""
	Method to complete a simulation of a synthetic observation based on the input source frame GRB template and the desired observing conditions

	Attributes:
	--------------
	template_grb : GRB 
		GRB class object that holds the source frame information of the template GRB
	synth_grb : GRB 
		GRB class object that will hold the simulated light curve
	resp_mat : RSP
		Response matrix to convolve the template spectrum with. If no response matrix is given, a Swift/BAT response matrix is assumed from the given imx, imy
	imx, imy : 	float, float 
		The x and y position of the GRB on the detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	z_p : float 
		Redshift of synthetic GRB
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	band_rate_min, band_rate_max : float, float
		Minimum and maximum of the energy band over which to calculate source photon flux
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	"""

	# Initialize synthetic GRB
	synth_grb.imx, synth_grb.imy = imx, imy
	synth_grb.z = z_p

	# Apply distance corrections to GRB light curve and spectrum
	synth_grb.move_to_new_frame(z_o=template_grb.z, z_p=z_p)

	# Calculate the fraction of the detectors currently enabled 
	det_frac = ndets / ndet_max # Current number of enabled detectors divided by the maximum number of possible detectors

	if time_resolved == False:
		# Fold spectrum through instrument response and calculate the count rate in the observation band
		folded_spec = resp_mat.fold_spec(synth_grb.specfunc)  # counts / sec / keV
		rate_in_band = band_rate(folded_spec, band_rate_min, band_rate_max) * det_frac # counts / sec

		# Using the total count rate from the spectrum and the relative flux level of the light curve, make a new light curve
		# The synthetic GRB light curve technically has units of counts / sec / cm^2, but we are only using it as a template for relative flux values. 
		# The actual flux is set by the band rate, which is in units of counts / sec 
		synth_grb.light_curve['RATE'] *= rate_in_band # counts / sec
	else:
		# Time-resolved analysis is True
		# If there is any interval of the light curve that is not covered by the time resolved spectra, use time-integrated spectrum
		folded_spec = resp_mat.fold_spec(synth_grb.specfunc)  # counts / sec / keV
		rate_in_band = band_rate(folded_spec, band_rate_min, band_rate_max) * det_frac # counts / sec
		
		arg_t_start = np.argmax(synth_grb.light_curve['TIME']>=synth_grb.spectrafuncs[0]['TSTART'])
		if arg_t_start > 0: 
			synth_grb.light_curve[:arg_t_start]['RATE'] *= rate_in_band # counts / sec
		
		arg_t_end = np.argmax(synth_grb.light_curve['TIME']>=synth_grb.spectrafuncs[-1]['TEND'])
		if arg_t_end > 0:
			synth_grb.light_curve[arg_t_end:]['RATE'] *= rate_in_band # counts / sec

		# Fold time-resolved spectrum
		for i in range(len(synth_grb.spectrafuncs)):
			folded_spec = resp_mat.fold_spec(synth_grb.spectrafuncs[i]['SPECFUNC'])  # counts / sec / keV
			rate_in_band = band_rate(folded_spec, band_rate_min, band_rate_max) * det_frac # counts / sec

			arg_t_start = np.argmax(synth_grb.light_curve['TIME']>=synth_grb.spectrafuncs[i]['TSTART'])
			arg_t_end = np.argmax(synth_grb.light_curve['TIME']>=synth_grb.spectrafuncs[i]['TEND'])
			synth_grb.light_curve[arg_t_start:arg_t_end]['RATE'] *= rate_in_band # counts / sec

	# If we are testing the trigger algorithm:
		# Modulate the light curve by the folded spectrum normalization for each energy band 
		# Calculate the fraction of the quadrant exposure 


	# Apply mask-weighting approximation to source rate signal 
	synth_grb.light_curve = apply_mask_weighting(synth_grb.light_curve, imx, imy, ndets) # counts / sec / det

	if sim_bgd == True:
		# Add mask-weighted background rate to either side of mask-weighted source signal
		synth_grb.light_curve = add_background(synth_grb.light_curve, t_buffer=template_grb.t_buffer, dt = template_grb.dt) # counts / sec / det


	return synth_grb

def band_rate(spectrum, emin, emax):
	"""
	Method to calculate the rate by taking the sum of the spectrum across a specified energy band

	Attributes:
	--------------
	spectrum : np.ndarray with [("RATE",float), ("ENERGY", float)]
		Array storing the spectrum 
	emin, emax : float, float
		Minimum and maximum energy to sum the spectrum over
	"""

	return np.sum(spectrum['RATE'][np.argmax(spectrum['ENERGY']>=emin):np.argmax(spectrum['ENERGY']>=emax)])

def apply_mask_weighting(light_curve, imx, imy, ndets):
	"""
	Method to apply mask weighting to a light curve assuming a flat background.
	
	Mask-weighted means:
		1. Background subtraction
		2. Per detector
		3. Per illuminated detector (partial coding fraction)
		4. Fraction of detector illuminated (mask correction)
		5. On axis equivalent (effective area correction for off-axis bursts)

	Attributes:
	--------------
	light_curve : np.ndarray with [("RATE", float), ("UNC", float)]
		light curve array
	imx, imy : float, float
		
	"""

	# From imx and imy, find pcode and the angle of incidence
	pcode = find_pcode(imx, imy)
	angle_inc = find_inc_ang(imx,imy) # rad

	# Total mask-weighting correction
	correction = np.cos(angle_inc)*pcode*ndets*fraction_correction(imx, imy) # total correction factor

	if pcode == 0:
		# Source was not in the field of view
		light_curve['UNC'] *= 0
		light_curve['RATE'] *=0 # counts / sec / dets
		return light_curve

	# Calculate the mask-weighted RATE column
	light_curve['RATE'] = light_curve["RATE"]/correction # background-subtracted counts / sec / dets

	return light_curve

def add_background(light_curve, t_buffer, dt):
	"""
	Method that adds a background interval before and after the source signal 

	Attributes:
	------------------------
	light_curve : np.ndarray([("TIME",float), ("RATE",float), ("UNC",float)])
		Light curve array
	buffer : float
		Duration (sec) of the background interval to be added to either side of the existing light curve
	dt : float
		time bin size
	"""

	sim_lc_length = int( (2*t_buffer/dt) + len(light_curve) ) # Length of the new light curve

	# Initialize an empty background light curve 
	bgd_lc = np.zeros(shape=sim_lc_length, dtype=[('TIME', float), ('RATE',float), ('UNC',float)])

	# Fill the time axis from synth_grb-buffer to synth_grb+buffer with correct time bin sizes 
	bgd_lc['TIME'] = np.arange(
		start=light_curve['TIME'][0]- t_buffer, 
		stop= light_curve['TIME'][-1]+t_buffer+dt, 
		step= dt
		)[:len(bgd_lc)]

	# Pull a random background variance from the distribution created from observed values
	variance = rand_background_variance()
	# Correct for time-bin size
	variance /= np.sqrt(dt)

	# Fluctuate the background according to a Normal distribution around 0 with a standard variation equal to the background variance
	bgd_lc['RATE'] = np.random.normal( loc=np.zeros(shape=len(bgd_lc)), scale=variance)
	# Set the uncertainty of the count rate to the variance. 
	bgd_lc['UNC'] = np.ones(shape=len(bgd_lc))*variance

	len_sim = len(light_curve['RATE']) # Length of the source signal
	argstart = np.argmax(bgd_lc['TIME']>=light_curve['TIME'][0]) # Start index of the signal in the new light curve	
	bgd_lc[argstart: argstart+len_sim]['RATE'] += light_curve['RATE'] # Add signal to background light curve

	return bgd_lc

def fit_function(t,fm,tm,r,d):
	"""
	FRED shaped light curve based on Equation 22 from Kocevski et al. 2003, basd on power law rise and exponential decay

	Attributes:
	t = time since trigger
	fm = flux at peak of the pulse (fm = F(tm))
	tm = t_max or the peak time of the pulse 
	r = rise constant
	d = decay constant 

	"""

	flux = fm*np.power(t/tm,r)*np.power( (d/(d+r)) + ((r/(d+r))*np.power(t/tm,r+1)) ,-(r+d)/(r+1))

	return flux

def rand_background_variance(size=1):
	"""
	Method that return a randomly selected from a distribution created from the measured background variances observed by Swift/BAT.

	The distribution function is created from the FRED function described in Kocevski 2003. 
	The parameter values were found in a separate fit.  
	"""

	# There are only anomolous variances found outside of these cuts
	cut_min = 0.02
	cut_max = 0.25

	x_range = np.linspace(cut_min, cut_max)

	# parameters = 0.0869, 0.0686, 9.929, 3.066 
	parameters = [0.08694067, 0.06859141, 9.92930295, 3.06620674]
	"""
	Parameters:
	fm = flux at peak of the pulse (fm = F(tm))
	tm = t_max or the peak time of the pulse 
	r = rise constant
	d = decay constant
	"""

	# Create distribution
	distrib = rv_discrete(a=cut_min, b=cut_max, values=(x_range, fit_function(x_range, *parameters)) ) 
	# Select random value
	variance = distrib.rvs(size=size)

	return variance

def many_simulations(template_grb, param_list, trials, 
	dur_per = 90, ndet_max=32768, 
	time_resolved=False, sim_triggers=False, out_file_name = None, ret_ave = False, keep_synth_grbs=False, verbose=False):
	"""
	Method to perform multiple simulations for each combination of input parameters 

	Attributes:
	----------
	template_grb : GRB
		GRB object used as a template light curve and spectrum
	param_list : np.ndarray
		List with all combinations of redshifts, imx, imy, and ndets to simulate the template GRB with
	trials : int
		Number of trials for each parameter combination
	dur_per : float
		Duration percentage to find using Bayesian blocks, i.e., dur_pur = 90 returns the T_90 of the burst
	ndet_max : int
		Maximum number of detectors on the detector plane (for Swift/BAT ndet_max = 32,768)
	sim_triggers : boolean
		Whether or not to simulate the Swift/BAT trigger algorithms or not
	out_file_name : string
		File name of .txt file to write the simulation result out to. 
	"""

	# Make a list to hold the simulation results
	sim_results = np.zeros(shape=int(len(param_list)*trials),dtype=dt_sim_res)
	sim_result_ind = 0

	if keep_synth_grbs is True:
		synth_grb_arr = np.zeros(shape=len(param_list),dtype=GRB)

	if verbose is True:
		print("Tot number of param combinations for GRB {} = {} ".format( template_grb.grbname ,len(param_list)) )

	# Simulate an observation for each parameter combination
	# Make a Response Matrix object
	resp_mat = RSP()

	for i in range(len(param_list)):
		if verbose is True:
			print("Param combination {}/{}:\n\tz = {}\n\timx, imy = {},{}\n\tndets={}".format(i+1, len(param_list), param_list[i][0], param_list[i][1], param_list[i][2], param_list[i][3]))
	
		# Load Swift BAT response based on the IMX, IMY position on the detector plane 
		resp_mat.load_SwiftBAT_resp(param_list[i][1], param_list[i][2])
		
		for j in range(trials):
			# if verbose is True:
				# print("\t\tTrial ",j)

			synth_grb = template_grb.copy()

			sim_results[["z", "imx", "imy", "ndets"]][sim_result_ind] = (param_list[i][0], param_list[i][1], param_list[i][2], param_list[i][3])

			simulate_observation(synth_grb = synth_grb, template_grb=template_grb,resp_mat=resp_mat, z_p=param_list[i][0], imx=param_list[i][1], imy=param_list[i][2], ndets=param_list[i][3], ndet_max=ndet_max, time_resolved = time_resolved, sim_triggers=sim_triggers)
			sim_results[["DURATION", "TSTART"]][sim_result_ind] = bayesian_t_blocks(synth_grb.light_curve, dur_per=dur_per) # Find the Duration and the fluence 
			if sim_results['DURATION'][sim_result_ind] > 0:	
				sim_results[["T100DURATION", "T100START"]][sim_result_ind] = bayesian_t_blocks(synth_grb.light_curve, dur_per=99) # Find the Duration and the fluence 
				sim_results[["FLUENCE","1sPeakFlux"]][sim_result_ind] = calc_fluence(synth_grb.light_curve, sim_results["DURATION"][sim_result_ind], sim_results['TSTART'][sim_result_ind])
				sim_results[["T100FLUENCE","1sPeakFlux"]][sim_result_ind] = calc_fluence(synth_grb.light_curve, sim_results["T100DURATION"][sim_result_ind], sim_results['T100START'][sim_result_ind])

			# Increase simulation index
			sim_result_ind +=1
		if keep_synth_grbs is True:
			synth_grb_arr[i] = synth_grb.copy()

	if out_file_name is not None:
		np.save(out_file_name, sim_results)

	if ret_ave is True:
		sim_results = make_ave_sim_res(sim_results)

	if keep_synth_grbs is True:
		return sim_results, synth_grb_arr
	else:
		return sim_results

def make_param_list(z_arr, imx_arr, imy_arr, ndets_arr):
	"""
	Method to make a list of all parameter combinations from the given parameter values.

	Attributes:
	----------
	z_arr : np.ndarray
		Array of redshifts to simulate the GRB at
	imx_arr, imy_arr : np.ndarry, np.ndarray
		Array of (imx,imy) points i.e., where the simualted sources will be located on the detector 
	ndets_arr : np.ndarray
		Array of values to use for the number of enabled detectors during the observation simulations
	"""

	# Make a list of all parameter combinations	
	param_list = np.array(np.meshgrid(z_arr, imx_arr, imy_arr,ndets_arr)).T.reshape(-1,4)

	return param_list


def make_ave_sim_res(sim_results, omit_nondetections=True):
	"""
	Method to make an average sim_results array for each unique parameter combination

	Attributes:
	----------
	sim_results : np.ndarray
		sim_results array 
	"""
	if omit_nondetections is True:
		sim_results = sim_results[sim_results['DURATION']>0]

	unique_rows = np.unique(sim_results[["z","imx","imy","ndets"]])

	ave_sim_results = np.zeros(shape=len(unique_rows),dtype=dt_sim_res)

	ave_sim_results[["z","imx","imy","ndets"]] = unique_rows

	for i in range(len(unique_rows)):
		ave_sim_results["DURATION"][i] = np.sum(sim_results["DURATION"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])
		ave_sim_results["T100DURATION"][i] = np.sum(sim_results["T100DURATION"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])
		ave_sim_results["TSTART"][i] = np.sum(sim_results["TSTART"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])
		ave_sim_results["T100START"][i] = np.sum(sim_results["T100START"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])
		ave_sim_results["FLUENCE"][i] = np.sum(sim_results["FLUENCE"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])
		ave_sim_results["T100FLUENCE"][i] = np.sum(sim_results["FLUENCE"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])
		ave_sim_results["1sPeakFlux"][i] = np.sum(sim_results["1sPeakFlux"][ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])/len(sim_results[ sim_results[["z","imx","imy","ndets"]] == unique_rows[i]])

	return ave_sim_results

def save_sim_results(fname, sim_results):
	"""
	Method to make an average sim_results array for each unique parameter combination

	Attributes:
	----------
	fname : string
		file name to save array to
	sim_results : np.ndarray
		sim_results array 
	"""

	np.save(fname, sim_results)