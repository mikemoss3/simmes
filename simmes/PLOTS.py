"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the base class and methods used for plotting within the simmes package.

"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simmes.util_packages.cosmology import lum_dis, k_corr
from simmes.SPECFUNC import SPECFUNC

class PLOTS(mpl.axes.Axes):
	"""
	Base class that defines methods used by other plot super classes  

	Attributes:
	----------
	fontsize : int
		Size of plot axis label
	fontweight : str
		Boldness level of the text [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
	"""

	def __init__(self, fontsize = 20, titlesize=20):

		self.fontsize = fontsize
		self.titlesize = titlesize




