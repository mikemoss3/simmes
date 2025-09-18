"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines unit tests for the simmes.simulations module

"""

import numpy as np
from scipy.integrate import quad

def testBandRate():
	"""
	Testing whether the band rate calculation method is correct
	"""
	from simmes.simulations import band_rate
	from simmes.SPECFUNC import PL

	spec = PL( alpha = 0, norm = 1)

	# Swift/BAT energy bins
	energy_bins = np.array([(0.00000, 10.0000), (10.0000, 12.0000), (12.0000, 14.0000), (14.0000, 16.0000),
				(16.0000, 18.0000), (18.0000, 20.0000), (20.0000, 22.0000), (22.0000, 24.0000), (24.0000, 26.0000),
				(26.0000, 28.0000), (28.0000, 30.1000), (30.1000, 32.1000), (32.1000, 34.2000), (34.2000, 36.3000),
				(36.3000, 38.3000), (38.3000, 40.4000), (40.4000, 42.5000), (42.5000, 44.6000), (44.6000, 46.8000), 
				(46.8000, 48.9000), (48.9000, 51.1000), (51.1000, 53.2000), (53.2000, 55.4000), (55.4000, 57.6000), 
				(57.6000, 59.8000), (59.8000, 62.0000), (62.0000, 64.2000), (64.2000, 66.4000), (66.4000, 68.7000),
				(68.7000, 70.9000), (70.9000, 73.2000), (73.2000, 75.4000), (75.4000, 77.7000), (77.7000, 80.0000),
				(80.0000, 82.3000), (82.3000, 84.6000), (84.6000, 87.0000), (87.0000, 89.3000), (89.3000, 91.7000),
				(91.7000, 94.0000), (94.0000, 96.4000), (96.4000, 98.8000), (98.8000, 101.200), (101.200, 103.600),
				(103.600, 106.000), (106.000, 108.400), (108.400, 110.900), (110.900, 113.300), (113.300, 115.800),
				(115.800, 118.200), (118.200, 120.700), (120.700, 123.200), (123.200, 125.700), (125.700, 128.300),
				(128.300, 130.800), (130.800, 133.300), (133.300, 135.900), (135.900, 138.400), (138.400, 141.000),
				(141.000, 143.600), (143.600, 146.200), (146.200, 148.800), (148.800, 151.400), (151.400, 154.100),
				(154.100, 156.700), (156.700, 159.400), (159.400, 162.000), (162.000, 164.700), (164.700, 167.400),
				(167.400, 170.100), (170.100, 172.800), (172.800, 175.500), (175.500, 178.200), (178.200, 181.000),
				(181.000, 183.700), (183.700, 186.500), (186.500, 189.300), (189.300, 192.100), (192.100, 194.900), 
				(194.900, 6553.60)], dtype=[("E_MIN", float), ("E_MAX", float)] )

	# emin = (14.0000 + 16.0000)/2.
	# emax = (192.100 + 194.900)/2.
	emin = 16.
	emax = 194.9

	# Calculate rate in band
	rate_in_band = quad(spec, emin, emax)[0]

	# Calculate rate in band from a binned spectrum
	test_binned_spec = np.zeros(shape=len(energy_bins), dtype=[("ENERGY", float), ("RATE", float)])  # photons / s / cm^2
	for i in range(len(energy_bins)):
		test_binned_spec['ENERGY'][i] = (energy_bins[i]["E_MAX"]+energy_bins[i]["E_MIN"])/2
		test_binned_spec['RATE'][i] = quad(spec, energy_bins[i]["E_MIN"], energy_bins[i]["E_MAX"])[0]

	test_rate_in_band = band_rate(test_binned_spec, emin, emax)

	np.testing.assert_equal(test_rate_in_band, rate_in_band)


if __name__ == "__main__":
	testBandRate()
