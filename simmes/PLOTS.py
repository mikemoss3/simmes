"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the base class and methods used for plotting within the simmes package.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt

class PLOTS(object):
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

	def show(self):
		plt.show()

	def close(self):
		plt.close()

	def savefig(self, fname, dpi=400):
		plt.savefig(fname, dpi = dpi)




