"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the base class and methods used for plotting response matrices.

"""

import numpy as np
import matplotlib.pyplot as plt

from simmes.PLOTS import PLOTS


class PLOTRSP(PLOTS):
	"""
	Class to plot response matrice from a RSP class object

	Attributes:
	----------
	RSP : RSP
		Response function object 
	"""
	def __init__(self):
		PLOTS.__init__(self)

	def plot_heatmap(self, RSP, ax=None, E_phot_bounds=None, E_chan_bounds=None, **kwargs):
		""" 
		Plot heat map of the response matrix 

		Attributes:
		----------
		RSP : RSP
			Response function object 
		ax : matplotlib.axes
			Axis on which to create the figure
		E_phot_bounds : np.ndarray
			Photon energy bin edges
		E_chan_bounds : np.ndarray
			Channel energy bin edges
		"""

		if ax is None:
			ax = plt.figure().gca()
		fig = plt.gcf()

		im = ax.pcolormesh(RSP.ECHAN_MID, RSP.ENERG_MID, RSP.MATRIX, shading='auto', **kwargs)

		if E_chan_bounds is None:
			ax.set_xlim(RSP.ECHAN_HI[0], RSP.ECHAN_LO[-1])
		else:
			ax.set_xlim(E_chan_bounds[0], E_chan_bounds[1])
		if E_phot_bounds is None:
			ax.set_ylim(RSP.ENERG_HI[0], RSP.ENERG_LO[-5])
		else:
			ax.set_xlim(E_phot_bounds[0], E_phot_bounds[1])

		ax.set_xlabel('Instrument Channel Energy (keV)')
		ax.set_ylabel('Photon Energy (keV)')

		cbar = fig.colorbar(im)
		cbar.ax.set_ylabel('Probability', rotation=270, labelpad=15)



	def plot_effarea(self, RSP, ax=None, det_area=1, E_phot_bounds=None, norm=1, **kwargs):
		"""
		Plot heat map of the response matrix 

		Attributes:
		----------
		RSP : RSP
			Response function object 
		ax : matplotlib.axes
			Axis on which to create the figure
		det_area : float
			Surface area of individual detector element
		E_phot_bounds : np.ndarray
			Photon energy bin edges
		E_chan_bounds : np.ndarray
			Channel energy bin edges
		norm : float
			Optional normalization factor
		"""
		
		if ax is None:
			ax = plt.figure().gca()

		# eff_area = np.sum(self.MATRIX,axis=1)/(self.ENERG_HI-self.ENERG_LO)
		eff_area = np.zeros( shape=RSP.num_phot_bins )
		for i in range( RSP.num_phot_bins ):
			for j in range( RSP.num_chans ):
				eff_area[i] += RSP.MATRIX[j][i]
		
		eff_area*=det_area

		ax.step(RSP.ENERG_MID, eff_area*norm, **kwargs)

		if E_phot_bounds is None:
			ax.set_xlim(RSP.ENERG_MID[0], RSP.ENERG_MID[-1])
		else:
			ax.set_xlim(E_phot_bounds[0], E_phot_bounds[1])

		ax.set_xscale('log')
		# ax.set_yscale('log')

		ax.set_xlabel('Incident Photon Energy (keV)')
		ax.set_ylabel(r'Effective Area (cm$^2$)')
