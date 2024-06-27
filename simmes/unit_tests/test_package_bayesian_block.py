"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the unit tests for the Bayesian block package 

"""

import unittest

from packages.package_bayesian_block import bayesian_t_blocks
from packages.class_GRB import GRB

class TestBayesianBlock(unittest.TestCase):
	def setUp(self):
		# Real GRB Light curve file 
		light_curve_fn = "./unit_tests/test_files/grb_130831A_1chan_64ms.txt"
		# Make GRB object to store light curve
		self.real_grb = GRB(light_curve_fn=light_curve_fn)

	def test_T90_real_grb(self):
		# Known T90 duration
		known_T90 = 32.51199996471405 # sec

		# Find duration using Bayesian block routine
		duration, timestart, fluence = bayesian_t_blocks(self.real_grb,dur_per=90)
		
		self.assertEqual(duration, known_T90, "Bayesian Block did not find the correct duration for the real GRB light curve.")

	def test_T90_start_real_grb(self):
		# Known T90 duration
		known_T90_start = 0.6080000400543213 # sec

		# Find duration using Bayesian block routine
		duration, timestart, fluence = bayesian_t_blocks(self.real_grb,dur_per=90)
		
		self.assertEqual(timestart, known_T90_start, "Bayesian Block did not find the correct duration for the real GRB light curve.")

	def test_T90_fluence_real_grb(self):
		# Known T90 duration
		known_T90_fluence = 151.05826053713116 # erg

		# Find duration using Bayesian block routine
		duration, timestart, fluence = bayesian_t_blocks(self.real_grb,dur_per=90)
		
		self.assertEqual(fluence, known_T90_fluence, "Bayesian Block did not find the correct duration for the real GRB light curve.")

def suite():
	suite = unittest.TestSuite()
	suite.addTest(TestBayesianBlock('test_T90_real_grb'))
	suite.addTest(TestBayesianBlock('test_T90_start_real_grb'))
	suite.addTest(TestBayesianBlock('test_T90_fluence_real_grb'))
	return suite