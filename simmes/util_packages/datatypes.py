"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines data types commonly used in this code.
This makes it simple to change the shape of a structured array by only having it defined in one place.

"""

import numpy as np

# Data type for simulation results 
dt_sim_res = np.dtype([("DURATION",float),
	("TSTART",float),("T100DURATION",float),("T100START",float),
	("FLUENCE",float),("T100FLUENCE",float),("1sPeakFlux",float),
	("z",float),("imx",float),("imy",float),("ndets",float)])

# Light curve data type
lc_type = [('TIME', float), ('RATE',float), ('UNC',float)]

# Quad band light curve data type
quad_lc_dtype = np.dtype(
					[("TIME", float),
					("RATE", float),
					("1525", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
					("1550", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
					("25100", [("q0", float), ("q1", float), ("q2", float), ("q3", float)]), 
					("50350", [("q0", float), ("q1", float), ("q2", float), ("q3", float)])]
					)