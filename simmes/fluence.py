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
	# Light curve time step
	dt = light_curve['TIME'][1] - light_curve['TIME'][0]

	# Calculate photon fluence of the light curve within the specified time interval
	fluence = np.sum(light_curve['RATE'][(light_curve['TIME'] >= tstart ) & (light_curve['TIME'] <= tstart+duration)]) * dt

	# Calculate the average 1 second peak flux 
	if dt < 1:
		num_bins = int(np.ceil(1 / dt)) # number of time bins that make up one second 

		arg_max = np.argwhere(light_curve['RATE'] == np.max(light_curve['RATE']) )[0][0]
		flux_peak_1s = np.mean( light_curve['RATE'][arg_max: arg_max+num_bins] )

	else:
		flux_peak_1s = np.max(light_curve['RATE'])

	return fluence, flux_peak_1s