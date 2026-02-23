"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the class and methods used for plotting the results from a sample of simulated bursts.

"""

import numpy as np
import matplotlib.pyplot as plt

from simmes.PLOTS import PLOTS

class PLOTSAMPLE(PLOTS):
	def __init__(self):
		PLOTS.__init__(self)

	def cumulative_durations(self, data, ax = None, bins=None, bin_min=None, bin_max=None, normed=False, **kwargs):
		"""
		Method to plot the cumulative distribution of durations for a sample of simulation results  

		Attributes:
		----------
		data : dt_sim_res, np.ndarray of with dtype [("DURATION", float)]
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		bins : array
			Array of bin edges
		bin_min, bin_max : float, float
			If `bins` is not provided, then bins will be created between bin_min and bin_max
		normed : bool
			Indicates if the data should be normalized by the maximum of the duration
		"""

		if ax is None:
			ax = plt.figure().gca()
		# fig = plt.gcf()

		norm = 1
		if normed == True:
			# norm = np.min(data['DURATION'][data['DURATION']>0])
			norm = np.max(data['DURATION'])

		if bins is None:
			if bin_min is None:
				bin_min	= 0.1
			if bin_max is None:
				bin_max = np.max(data['DURATION'])

			bins = np.logspace(start=np.log10(bin_min), stop = np.log10(bin_max), num=100)

		self._make_cumu_plot(data['DURATION']/norm, bins=bins, ax=ax, **kwargs)

		ax.set_xscale("log")
		# ax.set_yscale("log")

		# ax.set_xlim(1)
		# ax.set_ylim(0,1.05)

		if "label" in kwargs:
			ax.legend()

		ax.set_xlabel(r"T$_{90}$ (sec)")
		ax.set_ylabel("Cumulative Probability")
		# ax.set_title("T90 Distrubtions")


	def cumulative_fluence(self, data, ax = None, bins = None, bin_min=None, bin_max=None, **kwargs):
		"""
		Method to plot the cumulative distribution of durations for a sample of simulation results  

		Attributes:
		----------
		data : dt_sim_res, np.ndarray of with dtype [("FLUENCE", float)]
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		bins : array
			Array of bin edges
		bin_min, bin_max : float, float
			If `bins` is not provided, then bins will be created between bin_min and bin_max
		normed : bool
			Indicates if the data should be normalized by the maximum of the fluence
		"""

		if ax is None:
			ax = plt.figure().gca()
		# fig = plt.gcf()

		if bins is None:
			if bin_min is None:
				bin_min	= 0.01
			if bin_max is None:
				bin_max = np.max(data['FLUENCE']) 

			bins = np.logspace(start=np.log10(bin_min), stop = np.log10(bin_max), num=100)
		

		self._make_cumu_plot(data["FLUENCE"], bins=bins, ax=ax, **kwargs)

		ax.set_xscale("log")
		# ax.set_yscale("log")

		# ax.set_xlim(0.5)

		if "label" in kwargs:
			ax.legend()

		ax.set_xlabel("Fluence (cnts/det)")
		ax.set_ylabel("Cumulative Probability")
		# ax.set_title("Fluence Distrubtion")


	def cumulative_peak_flux(self, data, ax = None, bins = None, bin_min=None, bin_max=None, **kwargs):
		"""
		Method to plot the cumulative distribution of durations for a sample of simulation results  

		Attributes:
		----------
		data : dt_sim_res, np.ndarray of with dtype [("1sPeakFlux", float)]
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		bins : array
			Array of bin edges
		bin_min, bin_max : float, float
			If `bins` is not provided, then bins will be created between bin_min and bin_max
		normed : bool
			Indicates if the data should be normalized by the maximum of the peak flux
		"""

		if ax is None:
			ax = plt.figure().gca()
		# fig = plt.gcf()

		if bins is None:
			if bin_min is None:
				bin_min	= 0.01
			if bin_max is None:
				bin_max = np.max(data['1sPeakFlux']) 

			bins = np.logspace(start=np.log10(bin_min), stop = np.log10(bin_max), num=100)
		

		self._make_cumu_plot(data["1sPeakFlux"], bins=bins, ax=ax, **kwargs)

		ax.set_xscale("log")
		# ax.set_yscale("log")

		if "label" in kwargs:
			ax.legend()

		ax.set_xlabel("1s Peak Flux (counts/sec/det)")
		ax.set_ylabel("Cumulative Probability")
		# ax.set_title("1s Peak Flux Distrubtion")

	def _make_cumu_plot(self, values, bins, ax, **kwargs):

		# Make histogram
		count, edges = np.histogram(values, bins=bins)
		# Make cumulative distribution 
		cum_count = np.cumsum(count)
		# Plot cumulative distribution 
		ax.stairs(cum_count/np.max(cum_count), edges, **kwargs)
