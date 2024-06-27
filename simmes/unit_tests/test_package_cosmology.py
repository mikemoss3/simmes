"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the unit tests for the cosmological function package

"""

import unittest
import numpy as np

from util_packages.package_cosmology import lum_dis, k_corr
from packages.class_SPECFUNC import PL, CPL

class TestCosmology(unittest.TestCase):
	def test_lum_dis(self):
		z = 1  # input redshift
		known_dis = 2.1192497866435235e28  # cm, known luminosity distance
		
		self.assertEqual(lum_dis(z), known_dis, "Luminosity distance is incorrect.")

	def test_k_corr(self):
		""" Test k-correction calculation method """
		z = 1  # Redsihft
		emin = 5  # Energy band minimum
		emax = 150  # Energy band maximum 

		power_law = PL(alpha=-1,norm=1)  # Instance of a power law spectral function 
		known_k_corr_PL = 344.824137  # Known k-correction value
		diff_PL = np.abs( known_k_corr_PL - k_corr( power_law, z, emin, emax) )

		allowed_err = 1e-5

		self.assertTrue( diff_PL < allowed_err, "k-correction calculation using a power law is outside allowed tolerance.")


def suite():
	suite = unittest.TestSuite()
	suite.addTest(TestCosmology('test_lum_dis'))
	suite.addTest(TestCosmology('test_k_corr'))
	return suite