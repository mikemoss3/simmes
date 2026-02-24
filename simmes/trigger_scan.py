import numpy as np
import heasoftpy as hsp
import subprocess
from subprocess import STDOUT
from astropy.io import fits

from pathlib import Path
path_here = Path(__file__).parent

from simmes.simulations import band_rate
from simmes.util_packages.fluctuations import rand_lc_variance
from simmes.util_packages.det_ang_dependence import find_pcode
import simmes.util_packages.datatypes as datatypes

# Load trigger algorithms from .dat files
all_trigalgs = np.genfromtxt('util_packages/files-swift-trigger-algs/trigger_info_list.dat', dtype = datatypes.trigalg_dtype)

def get_trig_alg_params(trigalg_crit):
	"""
	Method to grab trigger algorithm parameters

	Attributes:
	--------------
	trigalg_crit : int
		The trigger algorithm criterion number

	Returns:
	--------------
	trig_alg_params : np.dtype([('criterion', int), ('bg1dur', float), ('fgdur', float), ('bg2dur', float), 
						('elapsedur', float), 
						('q0', bool), ('q1', bool), ('q2', bool), ('q3', bool), 
						('enband', int), ("sigmasquare", float), ("tskip", float), ("flag", bool)])

	"""

	trig_alg_params = all_trigalgs[all_trigalgs["criterion"] == trigalg_crit]
	return trig_alg_params

def scan_BAT_trigalgs(quad_band_light_curve):
	"""
	Method that takes in a BAT light curve and tests if any BAT trigger algorithms successfully trigger 

	Attributes:
	--------------
	quad_band_lc : np.ndarray(
						[("TIME", float), 
						("RATE", float),
						("1525", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
						("1550", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
						("25100", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
						("50350", [("q0", float), ("q1", float), ("q2", float), ("q3", float)])]
						)
		Quad band light curve that stores the four-quadrant, four-energy channel light curve required to test 
		Swift/BAT trigger algorithms

	Returns:
	--------------
	trigger : boolean
		Indicates if the one of the trigger algorithms realized a successful trigger on the source light curve
	snr_max : float
		Max signal-to-noise (SNR) ratio obtained during scan (defaults to -1e20 if no trigger)
	trig_time_start : float
		Time at which max SNR trigger occurs (defaults to -1e20 if no trigger)
	trigalg : int 
		Criterion number of successful trigger
	"""

	trigger = False
	trigalg = 000
	SNR_max = -1e10
	trig_time_start = -1e10

	# Time bin size
	tbin_size = (quad_band_light_curve['TIME'][1] - quad_band_light_curve['TIME'][0])
	# Only test trigger algorithms that have foregrounds shorter than the time bin size
	trigalg_list = all_trigalgs[all_trigalgs["fgdur"] >= tbin_size]

	# Apply each trigger algorithm to light curve
	for i in range(len(trigalg_list)):
		# print("Testing algorithm # {}".format(trigalg_list[i]['criterion']))
		tmp_trigger, tmp_snr, tmp_trig_time_start = test_trigger_alg(quad_band_light_curve, *[*trigalg_list[i]][1:-1])

		if tmp_snr > SNR_max:
			trigger = tmp_trigger
			SNR_max = tmp_snr
			trig_time_start = tmp_trig_time_start
			trigalg = trigalg_list[i]['criterion']

	return trigger, SNR_max, trig_time_start, trigalg

def test_trigger_alg(quad_band_light_curve, bg1dur, fgdur, bg2dur, elapsedur, q0, q1, q2, q3, enband, sigmasquare, tskip, dt=None, verbose=True):
	"""

	Attributes:
	--------------
	quad_band_light_curve : 

	bg1dur : float
		First background interval (in units of seconds)
	fgdur : float
		Foreground interval (in units of seconds)
	bg2dur : float
		Second background interval (in units of seconds)
	elapsedur : float
		Interval between the background intervals and 
		the foreground interval (in units of seconds)
	q0, q1, q2, q3 : 0 or 1
		Indicates which of the quadrants are relevant to this trigger algorithm
	en_band : 0, 1, 2, 3
		Indicates which energy band to consider for this trigger algorithm
	dt : float
		Time bin size (if left as None, dt will be calculated from the light curve)
	verbose : bool
		Whether to print error messages handled in this function or not

	Returns:
	--------------
	image_threshold_flag : bool
		Indicates whether there was a successful trigger or not
	snr_max : float
		Max signal-to-noise (SNR) ratio obtained during scan (defaults to -1e20 if no trigger)
	trig_time_start : float
		Time at which max SNR trigger occurs (defaults to -1e20 if no trigger)
	"""

	# For rate trigger testing, combine the counts in the energy band and quadrant relevant to this trigger
	enband_str_list = ["1525", "1550", "25100", "50350"]
	enband_str = enband_str_list[enband]
	comb_quad_light_curve = quad_band_light_curve[enband_str]["q0"] * q0 + quad_band_light_curve[enband_str]["q1"] * q1 + quad_band_light_curve[enband_str]["q2"] * q2 + quad_band_light_curve[enband_str]["q3"] * q3

	snr_max = -1e20
	trig_time_start = -1e20

	rate_threshold_flag = False
	image_threshold_flag = False
	tot_interval = bg1dur + elapsedur + fgdur + elapsedur + bg2dur
	if dt is None:
		dt = quad_band_light_curve['TIME'][2] - quad_band_light_curve['TIME'][1]

	bg1size = int(np.floor(bg1dur / dt))
	fgsize = int(np.floor(fgdur / dt))
	bg2size = int(np.floor(bg2dur / dt))
	elapsesize = int(np.floor(elapsedur / dt))

	lc_duration = quad_band_light_curve['TIME'][-1] - quad_band_light_curve['TIME'][0]
	if (lc_duration < tot_interval) and (verbose==True):
		print("Wrong! Light curve is shorter than the trigger time bracket! Algorithm skipped.")
		return False, 1e-20, 1e-20
	if (fgsize == 0) and (verbose==True):
		print("Wrong! Foreground size cannot be zero. Algorithm skipped.")
		return False, 1e-20, 1e-20

	# Step through light curve to test for trigger
	for i in range(len(quad_band_light_curve)-(bg1size+elapsesize+fgsize+elapsesize+bg2size)):
		# Using the quad band light curves relevant to this trigger, 
		# calculate the counts in first background, foreground, and second background 
		bg1cnts = np.sum(comb_quad_light_curve[ i:i+bg1size])
		fgcnts = np.sum(comb_quad_light_curve[ i+bg1size+elapsesize : i+bg1size+elapsesize+fgsize ])
		bg2cnts = np.sum(comb_quad_light_curve[ i+bg1size+elapsesize+fgsize+elapsesize : i+bg1size+elapsesize+fgsize+elapsesize+bg2size ])
		
		# Which method to use to calculate the signal-to-noise ratio (SNR) depends on the type of trigger being used
		snr_rate = calc_SNR(Nbk1=bg1cnts, tbk1=bg1dur, Nfg=fgcnts, tfg=fgdur, Nbk2=bg2cnts, tbk2=bg2dur, verbose=verbose)

		# Check for trigger
		if snr_rate > np.sqrt(sigmasquare):
			rate_threshold_flag = True
			# Rate trigger was passed, now check for image trigger.
			# Total Counts in first background, foreground, and second background 
			bg1cnts = np.sum(quad_band_light_curve['RATE'][ i:i+bg1size])
			fgcnts = np.sum(quad_band_light_curve['RATE'][ i+bg1size+elapsesize : i+bg1size+elapsesize+fgsize ])
			bg2cnts = np.sum(quad_band_light_curve['RATE'][ i+bg1size+elapsesize+fgsize+elapsesize : i+bg1size+elapsesize+fgsize+elapsesize+bg2size ])

			snr_img = calc_SNR(Nbk1=bg1cnts, tbk1=bg1dur, Nfg=fgcnts, tfg=fgdur, Nbk2=bg2cnts, tbk2=bg2dur, verbose=verbose)

			if snr_img > 7:
				image_threshold_flag = True

				if snr_rate > snr_max:
					snr_max = snr_rate
					trig_time_start = quad_band_light_curve['TIME'][i+bg1size+elapsesize]

	if image_threshold_flag:
		return image_threshold_flag, snr_max, trig_time_start
	else: 
		return image_threshold_flag, -1e20, -1e20 

def calc_SNR(Nbk1, tbk1, Nfg, tfg, Nbk2, tbk2, verbose=True):
	"""
	Method to calculate signal score for input trigger algorithm parameters. This 
	method reduces to Fenimore et al. 2003 (Eq. 1) when a single background interval is used.

	Attributes:
	--------------
	Nbk1 : float
		Counts in the first background interval
	tbk1 : float 
		Duration of the first background interval
	Nfg : float
		Counts in the foreground interval
	tfg : float
		Duration of the foreground interval
	Nbk2 : float
		Counts in the Second background interval
	tbk2 : float
		Duration of the Second background interval
	verbose : bool
		Whether to print error messages handled in this function or not

	Returns:
	--------------
	snr : float
		Calculated signal score 

	"""

	if tbk1 != 0 and tbk2 != 0:
		beta = 0.5 * ((Nbk1*tfg/tbk1) + (Nbk2*tfg/tbk2) )
	elif tbk1 == 0 and tbk2 != 0:
		beta = Nbk2*tfg/tbk2
	elif tbk1 != 0 and tbk2 == 0:
		beta = Nbk1*tfg/tbk1
	else:
		print("Wrong! Both background intervals have zero durations.")
		return -1e20

	snr = np.sqrt((Nfg - beta)**2 / np.abs(beta))

	return snr

def make_BAT_quad_band_light_curves(light_curve, folded_spec, imx, imy, sim_var=True, variance=None):
	"""
	Method that takes in a BAT light curve and splits it into four 
	quadrant light curve components based on the location of the source on the detector plane. 

	Attributes:
	--------------
	light_curve : np.ndarray([ ("TIME", float), ("RATE", float) ])
		Array that stores the source light curve
	folded_spec : SPEC 
		Observed spectrum (i.e., source spectrum already folded through instrument response matrix)
	imx, imy : float, float
		Source position on the BAT detector plane
	ndets : int
		Number of detectors enabled during the synthetic observation 
	sim_var : boolean
		Whether or not to include noise fluctuations (e.g., for when you want to test things without variations)
	variance : float 
		Variance level (in counts / sec / on-axis fully-illuminated detector)


	Returns:
	--------------
	quad_band_lc : np.ndarray(
						[("TIME", float), 
						("RATE", float),
						("1525", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
						("1550", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
						("25100", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
						("50350", [("q0", float), ("q1", float), ("q2", float), ("q3", float)])]
						)
		Array storing the four quadrant four energy band light curves
	"""
	if (sim_var is True) and (variance is None):
		print("Wrong! If sim_var is True, a variance level must be given.")
		return 1


	quad_lc = np.zeros(shape=len(light_curve), dtype=datatypes.quad_lc_dtype)
	
	quad_lc['TIME'] = light_curve['TIME']
	quad_lc['RATE'] = light_curve['RATE'] * band_rate(folded_spec, 15., 350.) * 2.


	# Create source mask from sample DPI 
	maskwt_res = hsp.batmaskwtimg(outfile='src.mask', attitude="NONE", 
									ra=imx, dec=imy, coord_type='tanxy', clobber='yes', 
									infile=path_here.joinpath("util_packages/files-swift-trigger-algs/sample.dpi"))
	if maskwt_res.returncode != 0:
		print("Wrong! Failed to created src.mask, return code: {}\nPerhaps initialize Heasoft and CALDB?\n".format(maskwt_res.returncode))
		return maskwt_res.returncode

	mask = fits.getdata("src.mask")
	mask = np.flip(mask, axis=0)
	# For Swift BAT: mask.shape = (173, 286)

	tot_num_elems = np.sum(mask!=0)
	elem_fracs = np.zeros(shape=4)
	elem_fracs[0] = np.sum(mask[ 87:, :144] !=0) / tot_num_elems #  q0
	elem_fracs[1] = np.sum(mask[ 87:, 144:] !=0) / tot_num_elems #  q1
	elem_fracs[2] = np.sum(mask[ :87, :144] !=0) / tot_num_elems #  q2
	elem_fracs[3] = np.sum(mask[ :87, 144:] !=0) / tot_num_elems #  q3

	rates_in_bands = np.zeros(shape=4)
	rates_in_bands[0] = band_rate(folded_spec, 15, 25) * 2.
	rates_in_bands[1] = band_rate(folded_spec, 15, 50) * 2.
	rates_in_bands[2] = band_rate(folded_spec, 25, 100) * 2.
	rates_in_bands[3] = band_rate(folded_spec, 50, 350) * 2.

	en_ranges = ["1525", "1550", "25100", "50350"]
	quads = ["q0", "q1", "q2", "q3"]
	for i in range(4):
		for j in range(4):
			quad_lc[en_ranges[i]][quads[j]] = light_curve['RATE'] * rates_in_bands[i] * elem_fracs[j]
			
			if sim_var is True:
				quad_lc[en_ranges[i]][quads[j]] += np.random.normal( loc=np.zeros(shape=len(light_curve)), scale=variance) * rates_in_bands[i] * elem_fracs[j]

	subprocess.run(["rm src.mask"], shell=True, stderr=STDOUT)

	return quad_lc
