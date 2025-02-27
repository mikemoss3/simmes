"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com   

Defines the Bayesian block method to calculate the duration of a GRB from a supplied light curve

"""

import numpy as np
from astropy.stats import bayesian_blocks

def bayesian_t_blocks(light_curve, dur_per=90, ncp_prior=6):
	"""
	Method to extract the duration and photon fluence of a GRB from a supplied light curve using a Bayesian block method. 

	Attributes:
	--------------
	light_curve : nd.array(dtype=[("TIME", float), ("RATE", float), ("UNC", float)])
		Array that stores the light curve 
	dur_per : float
		Indicates the percentage of the total fluence to be enclosed within the reported duration (i.e., T90 corresponds to dur_per = 90)
	ncp_prior : int
		Initial guess at the number of change points used by the Bayesian block algorithm

	Returns:
	--------------
	duration : float
		Total duration found for the event
	t_start : float
		Start time found for the event
	"""

	# custom_bb is a condensed bayesian block algorithm adapted from the astropy function. It's faster but more unsafe!

	# Astropy bayesian block algorithm -- it is much safer to use since it handles exceptions better, however its an order of magnitude
	try:
		bin_edges = bayesian_blocks(t=light_curve['TIME'], x=light_curve['RATE'], sigma=light_curve['UNC'], fitness="measures", ncp_prior=ncp_prior) # Find the T90 and the fluence 
		# bin_edges = custom_bb(light_curve=light_curve, ncp_prior=ncp_prior)
	except:
		bin_edges = bayesian_blocks(t=light_curve['TIME'], x=light_curve['RATE'], sigma=light_curve['ERROR'], fitness="measures", ncp_prior=ncp_prior) # Find the T90 and the fluence 
		# bin_edges = custom_bb(light_curve=light_curve, ncp_prior=ncp_prior)

	# Check if any GTI (good time intervals) were found
	if len(bin_edges) <= 3:
		# If true, then no GTI's were found     
		return 0., 0.
	else:
		# Calculate total duration and start time 
		t_dur_tot = bin_edges[-2] - bin_edges[1]
		t_start_tot = bin_edges[1]

		if dur_per == 100:
			return t_dur_tot, t_start_tot
		else:
			## Find TXX
			emission_interval = light_curve[np.argmax(t_start_tot<=light_curve['TIME']):np.argmax((t_start_tot+t_dur_tot)<=light_curve['TIME'])]
			if len(emission_interval) == 0:
				# Then no Bayesian blocks were found.
				return 0, 0
			# Find the total fluence 
			tot_fluence = np.sum(emission_interval['RATE'])
			# Find the normalized cumulative sum between the total duration 
			cum_sum_fluence = np.cumsum(emission_interval['RATE'])/tot_fluence
			# Find the time interval that encompasses dur_per of the burst fluence
			per_start = ((100 - dur_per)/2)/100
			per_end = 1 - per_start
			t_start =  emission_interval['TIME'][np.argmax(per_start <= cum_sum_fluence)]
			t_end = emission_interval['TIME'][np.argmax(per_end <= cum_sum_fluence)]

			duration = t_end - t_start

	return duration, t_start

def custom_bb(light_curve, ncp_prior):
	"""
	A Bayesian block algorithm adapted from the astropy.bayesian_blocks function.
	This version is faster than the astropy version because it removes some of the error safety and handling as well as only uses
	the ncp_prior parameter for the algorithm

	Attributes:
	--------------
	light_curve : nd.array(dtype=[("TIME", float), ("RATE", float)])
		Initial guess at the number of change points used by the Bayesian block algorithm
	ncp_prior : int
		Initial guess at the number of change points used by the Bayesian block algorithm

	Returns:
	--------------
	bin_edges : array of floats 
		Array of bin edges for bracketing the significant time bins of the event
	"""

	t=light_curve['TIME']
	x=light_curve['RATE']
	sigma=light_curve['UNC']

	# compute values needed for computation
	ak_raw = np.ones_like(x) / sigma**2
	bk_raw = x / sigma**2

	# create length-(N + 1) array of cell edges
	edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])

	# arrays to store the best configuration
	N = len(t)
	best = np.zeros(N, dtype=float)
	last = np.zeros(N, dtype=int)

	# ----------------------------------------------------------------
	# Start with first data cell; add one cell at each iteration
	# ----------------------------------------------------------------
	### Stripped down astropy implementation
	for R in range(N):
		# a_k: eq. 31
		a_k = 0.5 * np.cumsum(ak_raw[: (R + 1)][::-1])[::-1]

		# b_k: eq. 32
		b_k = -np.cumsum(bk_raw[: (R + 1)][::-1])[::-1]

		# evaluate fitness function
		fit_vec = fitness(a_k, b_k)

		A_R = fit_vec - ncp_prior
		A_R[1:] += best[:R]

		i_max = np.argmax(A_R)
		last[R] = i_max
		best[R] = A_R[i_max]

	### Even more stripped down astropy implementations
	# a_k = 0.5 * np.cumsum(ak_raw)  # a_k: eq. 31
	# b_k = np.cumsum(bk_raw)  # b_k: eq. 32
	# fit_vec = fitness(a_k, b_k)  # Evaluate fitness function, log(L^k_max) eq. 41
	# for R in range(N):
	# 	A_R = fit_vec[:R+1] - ncp_prior
	# 	A_R[1:] += best[:R]

	# 	i_max = np.argmax(A_R)  # Find max likelihood
	# 	last[R] = i_max  # Record max likelihood index
	# 	best[R] = A_R[i_max]  # Record max likelihood

	# ----------------------------------------------------------------
	# Now find changepoints by iteratively peeling off the last block
	# ----------------------------------------------------------------
	change_points = np.zeros(N, dtype=int)
	i_cp = N
	ind = N
	while i_cp > 0:
		i_cp -= 1
		change_points[i_cp] = ind
		if ind == 0:
			break
		ind = last[ind - 1]
	if i_cp == 0:
		change_points[i_cp] = 0
	change_points = change_points[i_cp:]

	return edges[change_points]


def fitness(a_k, b_k):
	"""
	The fitness function of the Bayesian block algorithm, i.e., Eq. 41 from Scargle 2013

	Attributes:
	--------------
	a_k : float
	b_k : float

	Returns:
	--------------
	value : float
	"""
	return (b_k * b_k) / (4 * a_k)