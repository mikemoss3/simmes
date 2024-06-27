import numpy as np

def calc_fluence(light_curve, duration, tstart):
	"""
	Method to calculate photon fluence of a light curve

	Attributes:
	---------
	light_curve : nd.array(dtype=[("TIME", float), ("RATE", float), ("UNC", float)])
		Array that stores the light curve 
	duration : float
		Length of the event 
	tstart : float
		What time the event begins
	"""

	# Calculate photon fluence of the light curve within the specified time interval
	fluence = np.sum(light_curve['RATE'][(light_curve['TIME'] > tstart ) & (light_curve['TIME'] < tstart+duration)])

	# Calculate the 1 second peak flux 
	dt = light_curve['TIME'][1] - light_curve['TIME'][0]
	flux_peak_1s = np.max(light_curve['RATE'])

	return fluence, flux_peak_1s