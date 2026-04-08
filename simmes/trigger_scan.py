import numpy as np
import heasoftpy as hsp
import subprocess
from subprocess import STDOUT
from astropy.io import fits

from pathlib import Path
path_here = Path(__file__).parent

from simmes.simulations import band_rate
import simmes.util_packages.datatypes as datatypes

# Load trigger algorithms from .dat files
all_trigalgs = np.genfromtxt(path_here.joinpath('util_packages/files-swift-trigger-algs/trigger_info_list.dat'), dtype = datatypes.trigalg_dtype)

class Trigger:
	flag = False
	SNR_max = -1e20
	start_time = -1e20
	criterion = None


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

def scan_BAT_trigalgs(quad_band_light_curve, quick=False):
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

	quick : boolean 
		If true, the scan will cease as soon as a successful trigger is made. If false, all trigger
		algorithms will be scanned.

	Returns:
	--------------
	curr_trig : Trigger
		Object storing the trigger information
	"""

	curr_trig = Trigger()

	# Time bin size
	tbin_size = (quad_band_light_curve['TIME'][1] - quad_band_light_curve['TIME'][0])
	# Only test trigger algorithms that have foregrounds greater than the time bin size
	trigalg_list = all_trigalgs[all_trigalgs["fgdur"] > tbin_size]

	# Only test trigger criteria that are flagged true (some triggers are ignored)
	trigalg_list = trigalg_list[trigalg_list["flag"] >= 0]

	# Apply each trigger algorithm to light curve
	for i in range(len(trigalg_list)):
		# print("Testing algorithm # {}".format(trigalg_list[i]['criterion']))
		tmp_flag, tmp_snr, tmp_trig_time_start = test_trigger_alg(quad_band_light_curve, *[*trigalg_list[i]][1:-1], verbose=False)

		if tmp_snr > curr_trig.SNR_max:
			curr_trig.flag = tmp_flag
			curr_trig.SNR_max = tmp_snr
			curr_trig.start_time = tmp_trig_time_start
			curr_trig.criterion = trigalg_list[i]['criterion']

			if (quick is True) and (curr_trig.flag == True):
				return curr_trig

	return curr_trig

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

	bg1size = int(np.ceil(bg1dur / dt))
	fgsize = int(np.ceil(fgdur / dt))
	bg2size = int(np.ceil(bg2dur / dt))
	elapsesize = int(np.ceil(elapsedur / dt))

	lc_duration = quad_band_light_curve['TIME'][-1] - quad_band_light_curve['TIME'][0]
	if (lc_duration < tot_interval) and (verbose==True):
		print("Wrong! Light curve is shorter than the trigger time bracket! Algorithm skipped.")
		return False, 1e-20, 1e-20
	if (fgsize == 0) and (verbose==True):
		print("Wrong! Foreground size cannot be zero. Algorithm skipped.")
		return False, 1e-20, 1e-20

	# Use the prefix sum algorithm (a.k.a. use the cumulative sum)
	summed_lc = np.cumsum(comb_quad_light_curve)
	summed_lc = np.concatenate([[0.0], summed_lc]) # Necessary for correct indexing

	# Calculate summations for each interval by taking the difference of the cumulative sum curve at different offsets. 
	fgcnts_list = summed_lc[bg1size+elapsesize+fgsize : -(elapsesize+bg2size)] - summed_lc[bg1size+elapsesize: -fgsize - (elapsesize+bg2size)]

	# Number of bytes we have to skip in memory to move to the next position along a certain axis
	n = comb_quad_light_curve.strides[0]
	# Number of sliding windows to calculate standard deviation over
	nrows = comb_quad_light_curve.size - (bg1size +elapsesize+fgsize+elapsesize+bg2size) + 1

	if bg1size != 0:
		bg1cnts_list = summed_lc[bg1size : -(elapsesize+fgsize+elapsesize+bg2size)] - summed_lc[: -bg1size - (elapsesize+fgsize+elapsesize+bg2size)]

		# Window size
		W = bg1size
		# Make sliding windows a strides
		a2D = np.lib.stride_tricks.as_strided(comb_quad_light_curve[:-(elapsesize+fgsize+elapsesize+bg2size)], shape=(nrows, W), strides=(n,n))
		# Calculate standard deviation for all windows
		bg1std_list = np.std(a2D, axis=1)
	else: 
		bg1cnts_list = np.zeros(shape=len(fgcnts_list))
		bg1std_list = np.zeros(shape=len(fgcnts_list))

	if bg2size != 0:
		bg2cnts_list = summed_lc[bg1size+elapsesize+fgsize+elapsesize+bg2size:] - summed_lc[bg1size+elapsesize+fgsize+elapsesize: -bg2size] 

		W = bg2size
		a2D = np.lib.stride_tricks.as_strided(comb_quad_light_curve[bg1size+elapsesize+fgsize+elapsesize:], shape=(nrows, W), strides=(n,n))
		bg2std_list = np.std(a2D, axis=1)
	else:
		bg2cnts_list = np.zeros(shape=len(fgcnts_list))
		bg2std_list = np.zeros(shape=len(fgcnts_list))

	# Calculate SNR for all summation combinations.
	snr_vals = calc_SNR(Nbk1=bg1cnts_list, tbk1=bg1dur, stdbk1= bg1std_list,
						Nfg=fgcnts_list, tfg=fgdur, 
						Nbk2=bg2cnts_list, tbk2=bg2dur, stdbk2=bg2std_list)

	# Is any snr value > np.sqrt(sigmasquare)? 
	snr_thresh = np.sqrt(sigmasquare)
	if any(x > snr_thresh for x in snr_vals):
		# If yes, calculate SNR for image triggers (i.e., using full quad rates)

		# Use same algorithm
		quad_summed_lc = np.cumsum(quad_band_light_curve['RATE'])
		quad_summed_lc = np.concatenate([[0.0], quad_summed_lc])

		quad_fgcnts_list = quad_summed_lc[bg1size+elapsesize+fgsize : -(elapsesize+bg2size)] - quad_summed_lc[bg1size+elapsesize: -fgsize - (elapsesize+bg2size)]

		n = quad_summed_lc.strides[0] 

		if bg1size != 0:
			quad_bg1cnts_list = quad_summed_lc[bg1size : -(elapsesize+fgsize+elapsesize+bg2size)] - quad_summed_lc[: -bg1size - (elapsesize+fgsize+elapsesize+bg2size)]

			W = bg1size 
			slides = np.lib.stride_tricks.as_strided(quad_summed_lc[:-(elapsesize+fgsize+elapsesize+bg2size)], shape=(nrows, W), strides=(n,n))
			quad_bg1std_list = np.std(slides, axis=1)
		else: 
			quad_bg1cnts_list = np.zeros(shape=len(quad_fgcnts_list))
			quad_bg1std_list = np.zeros(shape=len(quad_fgcnts_list))

		if bg2size != 0:
			quad_bg2cnts_list = quad_summed_lc[bg1size+elapsesize+fgsize+elapsesize+bg2size:] - quad_summed_lc[bg1size+elapsesize+fgsize+elapsesize: -bg2size] 

			W = bg2size
			slides = np.lib.stride_tricks.as_strided(quad_summed_lc[bg1size+elapsesize+fgsize+elapsesize:], shape=(nrows, W), strides=(n,n))
			quad_bg2std_list = np.std(slides, axis=1)
		else:
			quad_bg2cnts_list = np.zeros(shape=len(quad_fgcnts_list))
			quad_bg2std_list = np.zeros(shape=len(quad_fgcnts_list))


		snr_vals_img = calc_SNR(Nbk1=quad_bg1cnts_list, tbk1=bg1dur, stdbk1=quad_bg1std_list, 
								Nfg=quad_fgcnts_list, tfg=fgdur, 
								Nbk2=quad_bg2cnts_list, tbk2=bg2dur, stdbk2=quad_bg2std_list)

		# If any snr_vals_img are > 7, 
		if any(x > 7 for x in snr_vals_img):
			# Then we have a successful trigger 
			image_threshold_flag = True

			snr_max = np.max(snr_vals)
			index = np.where(snr_vals==snr_max)[0][0]
			trig_time_start = quad_band_light_curve['TIME'][index] + bg1dur + elapsedur


	if image_threshold_flag:
		return image_threshold_flag, snr_max, trig_time_start
	else: 
		return image_threshold_flag, -1e20, -1e20 


def calc_SNR(Nbk1, tbk1, stdbk1, Nfg, tfg, Nbk2, tbk2, stdbk2):
	"""
	Method to calculate signal score for input trigger algorithm parameters. This 
	method reduces to Fenimore et al. 2003 (Eq. 1) when a single background interval is used.

	Attributes:
	--------------
	Nbk1 : float or np.ndarray of floats
		Counts in the first background interval
	tbk1 : float or np.ndarray of floats
		Duration of the first background interval
	stdbk1 : float or np.ndarray of floats
		Standard deviation of the first background interval
	Nfg : float or np.ndarray of floats
		Counts in the foreground interval
	tfg : float or np.ndarray of floats
		Duration of the foreground interval
	Nbk2 : float or np.ndarray of floats
		Counts in the Second background interval
	tbk2 : float or np.ndarray of floats
		Duration of the Second background interval
	stdbk2 : float or np.ndarray of floats
		Standard deviation of the second background interval

	Returns:
	--------------
	snr : float
		Calculated signal score 

	"""

	if (tbk1 != 0) and (tbk2 != 0):
		beta = 0.5 * ((Nbk1*tfg/tbk1) + (Nbk2*tfg/tbk2) )
	elif (tbk1 == 0) and (tbk2 != 0):
		beta = Nbk2*tfg/tbk2
	elif (tbk1 != 0) and (tbk2 == 0):
		beta = Nbk1*tfg/tbk1
	else:
		print("Wrong! Both background intervals have zero durations.")
		return -1e20

	# Avoid divide by zero results.
	beta = np.asarray(beta)
	beta += (np.sign(beta) * 1e-3)
	beta[beta==0] = 1e-3

	snr = (Nfg - beta) / (stdbk1 + stdbk2)/2.

	return snr

def make_BAT_quad_band_light_curves(light_curve, folded_spec, imx, imy):
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

	quad_lc = np.zeros(shape=len(light_curve), dtype=datatypes.quad_lc_dtype)
	
	quad_lc['TIME'] = light_curve['TIME']
	quad_lc['RATE'] = light_curve['RATE']

	rand_int = np.random.randint(low=0, high=1e3) # use random interger for file creation, mostly to avoid multiprocessing error

	# Create source mask from sample DPI 
	with hsp.utils.local_pfiles_context():
		maskwt_res = hsp.batmaskwtimg(outfile="src.mask.{}".format(rand_int), attitude="NONE", 
										ra=imx, dec=imy, coord_type='tanxy', clobber='yes', 
										infile=path_here.joinpath("util_packages/files-swift-trigger-algs/sample.dpi"))
	if maskwt_res.returncode != 0:
		print("Wrong! Failed to created src.mask.{}, return code: {}\nPerhaps initialize Heasoft and CALDB?\n".format(rand_int, maskwt_res.returncode))
		return maskwt_res.returncode

	try:
		mask = fits.getdata("src.mask.{}".format(rand_int))
	except: 
		print("Could not load src.mask.{}".format(rand_int))
		return 1

	mask = np.flip(mask, axis=0)
	# For Swift BAT: mask.shape = (173, 286)

	tot_num_elems = np.sum(mask!=0)
	elem_fracs = np.zeros(shape=4)
	elem_fracs[0] = np.sum(mask[ 87:, :144] !=0) / tot_num_elems #  q0
	elem_fracs[1] = np.sum(mask[ 87:, 144:] !=0) / tot_num_elems #  q1
	elem_fracs[2] = np.sum(mask[ :87, :144] !=0) / tot_num_elems #  q2
	elem_fracs[3] = np.sum(mask[ :87, 144:] !=0) / tot_num_elems #  q3

	total_band_rate = band_rate(folded_spec, 15., 350.)
	rates_in_bands = np.zeros(shape=4)
	rates_in_bands[0] = band_rate(folded_spec, 15., 25.) / total_band_rate
	rates_in_bands[1] = band_rate(folded_spec, 15., 50.) / total_band_rate
	rates_in_bands[2] = band_rate(folded_spec, 25., 100.) / total_band_rate
	rates_in_bands[3] = band_rate(folded_spec, 50., 350.) / total_band_rate

	en_ranges = ["1525", "1550", "25100", "50350"]
	quads = ["q0", "q1", "q2", "q3"]
	for i in range(4):
		for j in range(4):
			quad_lc[en_ranges[i]][quads[j]] = light_curve['RATE'] * rates_in_bands[i] * elem_fracs[j]
			
	subprocess.run(["rm src.mask.{}".format(rand_int)], shell=True, stderr=STDOUT)

	return quad_lc
