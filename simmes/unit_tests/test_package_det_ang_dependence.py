"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the unit tests for the detector angle dependence package

"""

import unittest
import numpy as np

from util_packages.package_det_ang_dependence import find_grid_id, find_inc_ang, find_pcode


class TestDetAngDependence(unittest.TestCase):
	def setUp(self):
		self.imx0, self.imy0 = 0., 0.
		self.imx, self.imy = 0.5, 0.5

	def test_find_grid_id(self):
		""" Test the find grid ID method """
		test_on_axis = find_grid_id(self.imx0,self.imy0)
		test_off_axis = find_grid_id(self.imx,self.imy)

		self.assertEqual(test_on_axis, '17', "The wrong Grid ID was found for the on-axis case")
		self.assertEqual(test_off_axis, '25', "The wrong Grid ID was found for the off-axis case")

	def test_find_inc_ang(self):
		""" Test the find incident angle method """
		test_on_axis = find_inc_ang(self.imx0,self.imy0)
		test_off_axis = find_inc_ang(self.imx,self.imy)

		self.assertEqual(test_on_axis, 0, "The wrong incident angle was found for the on-axis case")
		self.assertEqual(test_off_axis, 0.6154797086703874, "The wrong incident angle was found for the off-axis case")

	def test_find_pcode(self):
		""" Test the find PCODE method """
		test_on_axis = find_pcode(self.imx0,self.imy0)
		test_off_axis = find_pcode(self.imx,self.imy)

		self.assertEqual(test_on_axis, 1.0, "The wrong partial coding fraction was found for the on-axis case")
		self.assertEqual(test_off_axis, 0.5, "The wrong partial coding fraction was found for the off-axis case")

def suite():
	suite = unittest.TestSuite()
	suite.addTest(TestDetAngDependence('test_find_grid_id'))
	suite.addTest(TestDetAngDependence('test_find_inc_ang'))
	suite.addTest(TestDetAngDependence('test_find_pcode'))

	return suite