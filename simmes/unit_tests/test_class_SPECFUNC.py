"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the unit tests for the SPECFUNC class

"""

import unittest
import numpy as np

from packages.class_SPECFUNC import PL, CPL, Blackbody, Band


class TestSpectralFunctions(unittest.TestCase):
	def setUp(self):
		self.power_law = PL()
		self.cutoff_power_law = CPL()
		self.blackbody = Blackbody()
		self.band = Band()

	def test_PL_eval(self):
		""" Test if the evaluation function for the power law class is correct """
		energy = 10 # energy to evaluate the power law at
		ans = 0.1 # expected flux photons / keV
		self.assertEqual(self.power_law(energy), ans, "Power law did not return correct flux.")

	def test_CPL_eval(self):
		""" Test if the evaluation function for the cut-off power law class is correct """
		energy = 10 # energy to evaluate the power law at
		ans = 0.09048374180359596 # expected flux photons / keV
		diff = np.abs(ans - self.cutoff_power_law(energy)) # Difference between known and test
		allowed_err = 1e-6 # Allowed error tolerance
		self.assertTrue(diff<allowed_err, "The Cut-off Power Law function did not return a flux value within the error tolerance.")

	def test_Blackbody_eval(self):
		""" Test if the evaluation function for the Blackbody class is correct """
		energy = 10 # energy to evaluate the power law at
		ans = 0.584117029519693 # expected flux photons / keV
		diff = np.abs(ans - self.blackbody(energy)) # Difference between known and test
		allowed_err = 1e-6 # Allowed error tolerance
		self.assertTrue(diff<allowed_err, "Blackbody function did not return a flux within the error tolerance.")

	def test_Band_eval(self):
		""" Test if the evaluation function for the Band function class is correct """
		
		# Test below the break energy
		energy = 10 # energy to evaluate the power law at
		ans = 9.753099120283327 # expected flux photons / keV
		diff = np.abs(ans - self.band(energy)) # Difference between known and test
		allowed_err = 1e-6 # Allowed error tolerance
		self.assertTrue(diff<allowed_err, "The Band function did not return a flux value within the error tolerance when evaluated below the peak energy.")

		# Test above the break energy
		energy = 800 # energy to evaluate the power law at
		ans = 0.022992465073215146 # expected flux photons / keV
		diff = np.abs(ans - self.band(energy)) # Difference between known and test
		allowed_err = 1e-6 # Allowed error tolerance
		self.assertTrue(diff<allowed_err, "The Band function did not return a flux value within the error tolerance when evaluated above the peak energy.")

	def test_get_param_names(self):
		""" Test if the get_param_names() method of SPECFUNC return the correct names """
		pl_ans = ['alpha', 'norm', 'enorm']
		cpl_ans = ['ep', 'alpha', 'norm', 'enorm']
		band_ans = ['ep', 'alpha', 'beta', 'norm', 'enorm']
		self.assertEqual(self.power_law.get_param_names(), pl_ans, "Incorrect parameter names were returned for the PL class.") 
		self.assertEqual(self.cutoff_power_law.get_param_names(), cpl_ans, "Incorrect parameter names were returned for the CPL class.") 
		self.assertEqual(self.band.get_param_names(), band_ans, "Incorrect parameter names were returned for the Band class.") 

	def test_get_param_vals(self):
		""" Test if the get_param_vals() method of SPECFUNC return the correct values """
		pl_ans = [-1.0, 1.0, 1.0]
		cpl_ans = [100.0, -1.0, 1.0, 1.0]
		band_ans = [400.0, -1.0, -2.0, 1.0, 100.0]
		self.assertEqual(self.power_law.get_param_vals(), pl_ans, "Incorrect parameter values were returned for the PL class.") 
		self.assertEqual(self.cutoff_power_law.get_param_vals(), cpl_ans, "Incorrect parameter values were returned for the CPL class.") 
		self.assertEqual(self.band.get_param_vals(), band_ans, "Incorrect parameter values were returned for the Band class.") 

	def test_set_params(self):
		""" Test if the set_param() method of SPECFUNC is abel to set a parameter value correctly """
		self.power_law.set_params(alpha=-1.5) # Set value of power law index to -1.5
		self.cutoff_power_law.set_params(alpha=-1.5) # Set value of low-energy power law index to -1.5
		self.band.set_params(alpha=-1.5) # Set value of power law index to -1.5
		self.assertEqual(self.power_law.params['alpha'], -1.5, "Incorrectly set parameter to new value for PL class.")
		self.assertEqual(self.cutoff_power_law.params['alpha'], -1.5, "Incorrectly set parameter to new value for CPL class.")
		self.assertEqual(self.band.params['alpha'], -1.5, "Incorrectly set parameter to new value for Band class.")

		# Reset parameter value for other tests
		self.power_law.set_params(alpha=-1.)
		self.cutoff_power_law.set_params(alpha=-1.)
		self.band.set_params(alpha=-1.)

	def test_make_spectrum(self):
		""" Test if the a spectrum is made properly using the SPECFUNC::make_spectrum() method """

		# Call make_spectrum method
		test_spec = self.power_law.make_spectrum(emin=1,emax=10,num_bins=5)


		ans_spec = np.array([( 1., 1.),
						( 1.77827941, 0.56234133),
						( 3.16227766, 0.31622777),
						( 5.62341325, 0.17782794),
						(10., 0.1)],dtype=[("ENERGY",float),("RATE",float)])

		# Confirm that the first and last values are the same (no rounding error in this example)
		self.assertEqual(test_spec['ENERGY'][0], ans_spec['ENERGY'][0], "First energy value was not correct.")
		self.assertEqual(test_spec['RATE'][0], ans_spec['RATE'][0], "First rate was not correct.")
		self.assertEqual(test_spec['ENERGY'][-1], ans_spec['ENERGY'][-1], "Final energy value was not correct.")
		self.assertEqual(test_spec['RATE'][-1], ans_spec['RATE'][-1], "Final rate was not correct.")

		# Test if energy value array is the same
		self.assertTrue(np.allclose(test_spec['ENERGY'],ans_spec['ENERGY']), "Spectrum energy axis was not made correctly.")
		# Test if the rate at each energy is the same
		self.assertTrue(np.allclose(test_spec['RATE'],ans_spec['RATE']), "Spectrum rates were not calculated correctly.")

	def test_energy_flux(self):
		""" Test if the energy flux is calculated correctly."""

		en_lo = 1. # Lower energy bound
		en_hi = 10. # Upper energy bound
		kev2erg = 1000*1.60217657e-12 # Conversion between keV to erg 

		# Known answer for power law
		ans = np.log(en_hi/en_lo)*kev2erg
		# Absolute difference 
		difference = np.abs(ans - self.power_law._energy_flux(en_lo, en_hi))
		# Allowed error
		allowed_err = 1e-8

		self.assertTrue((difference<allowed_err), "Energy flux was not within error tolerance.")

	def test_energy_flux(self):
		""" Test if the spectrum normalization is calculated correctly."""

		flux = 1. # 
		en_lo = 1. # Lower energy bound
		en_hi = 10. # Upper energy bound
		kev2erg = 1000*1.60217657e-12 # Conversion between keV to erg 

		# Known answer for power law
		ans = 1. / ( np.log(en_hi/en_lo)*kev2erg )
		# Absolute difference 
		difference = np.abs(ans - self.power_law._find_norm(flux, en_lo, en_hi))
		# Allowed error
		allowed_err = 1e-2

		self.assertTrue((difference<allowed_err), "Energy flux was not within error tolerance.")

def suite():
	suite = unittest.TestSuite()
	suite.addTest(TestSpectralFunctions('test_PL_eval'))
	suite.addTest(TestSpectralFunctions('test_CPL_eval'))
	suite.addTest(TestSpectralFunctions('test_Blackbody_eval'))
	suite.addTest(TestSpectralFunctions('test_Band_eval'))
	suite.addTest(TestSpectralFunctions('test_get_param_names'))
	suite.addTest(TestSpectralFunctions('test_get_param_vals'))
	suite.addTest(TestSpectralFunctions('test_set_params'))
	suite.addTest(TestSpectralFunctions('test_make_spectrum'))
	suite.addTest(TestSpectralFunctions('test_energy_flux'))
	suite.addTest(TestSpectralFunctions('test_energy_flux'))

	return suite