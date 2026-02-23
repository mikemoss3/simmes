"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the class and methods used for plotting GRB data (both observed and simulated)

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from simmes.PLOTS import PLOTS


class PLOTGRB(PLOTS):
	def __init__(self):
		PLOTS.__init__(self)

	def plot_light_curves(self, grbs, t_window=None, labels=None, ax=None, alpha=0.7, norm=1, **kwargs):
		"""
		Method to plot the average duration percentage as a function of the position on the detector plane

		Attributes:
		----------
		grbs : GRB, array of GRB
			Either a single instance of a GRB or an array of GRBs for which the light curves will be plotted on the same axis
		t_window : tuple 
			Contains the minimum and maximum times over which to plot the light curve. If t_window = None, the entire light curve will be plotted.
		ax : matplotlib.axes
			Axis on which to create the figure
		norm : float
			An arbitrary factor to multiply the y-axis values by 
		labels : str, list
			Label for the grb(s) to be plotted
		"""

		if ax is None:
			ax = plt.figure().gca()

		# For an array of GRBs
		if hasattr(grbs,'__len__'):
			for i in range(len(grbs)):
				if labels is None:
					ax.errorbar(x=grbs[i].light_curve['TIME'],
						y=grbs[i].light_curve['RATE']*norm, yerr=grbs[i].light_curve['UNC']*norm,
						fmt="",drawstyle="steps-mid", alpha=alpha, **kwargs)
				else:
					ax.errorbar(x=grbs[i].light_curve['TIME'], 
						y=grbs[i].light_curve['RATE']*norm, yerr=grbs[i].light_curve['UNC']*norm,
						fmt="", drawstyle="steps-mid", alpha=alpha, 
						label="{}".format(labels[i]), **kwargs)
		# For a single GRB
		else:
			ax.errorbar(x=grbs.light_curve['TIME'], 
				y=grbs.light_curve['RATE']*norm, yerr=grbs.light_curve['UNC']*norm, 
				fmt="", drawstyle="steps-mid", alpha=alpha, 
				label=labels, **kwargs)

		ax.set_xlabel("Time (sec)")
		ax.set_ylabel("Rate (counts/sec)")

		if t_window is not None:
			ax.set_xlim(t_window)

		if labels is not None:
			ax.legend(fontsize=self.fontsize-2)

	def plot_spectrum(self, grb, e_window, time_resolved = False, bins = None, folded=False, rsp = None, ax=None, alpha=0.8, norm=1, cmap='Blues', **kwargs):
		"""
		Method to plot the average duration percentage as a function of the position on the detector plane

		Attributes:
		----------
		grbs : GRB
			A GRB object containing the light curves to be plotted 
		e_window : tuple 
			Contains the minimum and maximum energies over which to plot the spectral functions
		folded : boolean
			Indicates whether the plotted spectrum should first be folded through the provided instrument response matrix
		rsp : RSP
			Response matrix object that holds the instrument response matrix that the spectrum may be folded through (if indicated)
		ax : matplotlib.axes
			Axis on which to create the figure
		norm : float
			An arbitrary factor to multiply the y-axis values by 
		"""

		if ax is None:
			ax = plt.figure().gca()

		if (folded == True) and (rsp == None):
			print("Must provide a response matrix to fold spectrum through.")
			return;

		if time_resolved is False:
			spectrum = grb.make_spectrum(emin = e_window[0], emax = e_window[1], num_bins=bins)

			ax.step(x=spectrum['ENERGY'],y=spectrum['ENERGY']**2 * spectrum['RATE']*norm, where="mid", alpha=alpha,**kwargs)
			# ax.errorbar(x=spectrum['ENERGY'],y=spectrum['RATE']*norm,yerr=spectrum['UNC']*norm,fmt="",drawstyle="steps-mid",alpha=alpha,**kwargs)
		else:
			_col_map = mpl.colormaps[cmap]
			colors = _col_map(np.linspace(0, 1, len(grb.spectrafuncs)))
			for i in range(len(grb.spectrafuncs)):
				spectrum = grb.make_spectrum(emin = e_window[0], emax = e_window[1], num_bins=bins, spec_num=i)
				ax.step(x=spectrum['ENERGY'],y=spectrum['ENERGY']**2 * spectrum['RATE']*norm, where="mid", alpha=alpha, color=colors[i], **kwargs)

		ax.set_xscale('log')
		ax.set_yscale('log')

		ax.set_xlabel("Time (sec)")
		ax.set_ylabel("Rate (counts/sec)")

