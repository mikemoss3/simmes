"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the class and methods used for plotting simulation results.

"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simmes.util_packages.cosmology import lum_dis, k_corr
from simmes.RSP import RSP
from simmes.SPECFUNC import SPECFUNC

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

	def __init__(self, fontsize = 13, fontweight = "normal"):

		self.fontsize = fontsize
		self.fontweight = fontweight

	def plot_aesthetics(self, ax, xax=True, yax=True):
		"""
		This function is used to make bold and increase the font size of all plot tick markers

		Attributes:
		----------
		ax : matplotlib.axes
			Axis object to apply aesthetic changes to
		xax : boolean
			Indicates whether to apply to changes to x-axis
		yax : boolean
			Indicates whether to apply to changes to y-axis
		"""

		fontsize = self.fontsize
		fontweight = self.fontweight

		if xax is True:
			for tick in ax.xaxis.get_major_ticks():
				tick.label1.set_fontsize(fontsize=fontsize)
				tick.label1.set_fontweight(fontweight)

				tick.label2.set_fontsize(fontsize=fontsize)
				tick.label2.set_fontweight(fontweight)

			for tick in ax.xaxis.get_minor_ticks():
				tick.label1.set_fontweight(fontweight)

				tick.label2.set_fontweight(fontweight)
		else:
			ax.set_xticklabels([])
			# ax.set_xlabel()
			ax.xaxis.set_visible(False)

		if yax is True:
			for tick in ax.yaxis.get_major_ticks():
				tick.label1.set_fontsize(fontsize=fontsize)
				tick.label1.set_fontweight(fontweight)

				tick.label2.set_fontsize(fontsize=fontsize)
				tick.label2.set_fontweight(fontweight)

			for tick in ax.yaxis.get_minor_ticks():
				tick.label1.set_fontweight(fontweight)

				tick.label2.set_fontweight(fontweight)
		else:
			ax.set_yticklabels([])
			# ax.set_ylabel()
			ax.yaxis.set_visible(False)
			
		ax.tick_params(direction="in",which="both")
		ax.margins(x=0,y=0)

		# Make the cursor x, y coordinates display in matplotlib window
		ax.format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)

	def show(self):
		plt.show()

	def tight_layout(self):
		plt.tight_layout()

	def close(self):
		plt.close()

	def savefig(self, fname, dpi=400):
		plt.savefig(fname, dpi = dpi)

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

		ax.set_xlabel("Time (sec)", fontsize=self.fontsize, fontweight=self.fontweight)
		ax.set_ylabel("Rate (counts/sec)", fontsize=self.fontsize, fontweight=self.fontweight)

		if t_window is not None:
			ax.set_xlim(t_window)

		if labels is not None:
			ax.legend(fontsize=self.fontsize-2)

		self.plot_aesthetics(ax)

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

		ax.set_xlabel("Time (sec)", fontsize=self.fontsize, fontweight=self.fontweight)
		ax.set_ylabel("Rate (counts/sec)", fontsize=self.fontsize, fontweight=self.fontweight)

		self.plot_aesthetics(ax)
		ax.margins(y=0.1)


class PLOTSIMRES(PLOTS):
	def __init__(self):
		PLOTS.__init__(self)

	def duration_overlay(self, sim_results, light_curve, order_type=2, ax=None, **kwargs):
		"""
		Method to plot simulated duration measures overlayed on template light curve

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		light_curve : np.ndarray
			Light curve to plot under the simulated duration measurements
		ax : matplotlib.axes
			Axis on which to create the figure
		order_type : float
			Order of the sim_results lines. [ 0 = No order | 1 = Time Start | 2 = Time Duration ]
		"""

		if ax is None:
			ax = plt.figure().gca()

		# Plot light curve
		ax.step(light_curve['TIME'], light_curve['RATE'], color="k", alpha=0.5, **kwargs)

		# Order sim_results
		# Take lines and order them
		# 0 == No order
		# 1 == Time Start
		# 2 == Time Duration
		if order_type == 0:
			sorted_sim_results = sim_results
		elif order_type == 1:
			sorted_sim_results = np.sort(sim_results, order='TSTART')
		elif order_type == 2:
			sorted_sim_results = np.flip(np.sort(sim_results, order="DURATION"))

		# Plot simulated duration measurements
		y_pos = np.linspace(np.max(light_curve['RATE'])*0.05, np.max(light_curve['RATE'])*0.95, len(sorted_sim_results))
		for i in range(len(sorted_sim_results)):
			ax.hlines(y=y_pos[i], 
				xmin=sorted_sim_results[i]['TSTART'], xmax=(sorted_sim_results[i]['TSTART']+sorted_sim_results[i]['DURATION']), 
				color="C1", alpha=0.7)

		ax.set_xlabel("Time (sec)", fontsize=self.fontsize, fontweight=self.fontweight)
		ax.set_ylabel("Rate (counts/sec)", fontsize=self.fontsize, fontweight=self.fontweight)

		self.plot_aesthetics(ax)

	def dur_vs_param(self, sim_results, obs_param, dur_frac=False, t_true=None, ax=None, marker=".", joined=False, **kwargs):
		"""
		Method to plot duration vs observing parameter (e.g., redshift, pcode, ndets)

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		obs_param : str
			The sim_result column field name of the parameter to be plotted against the duration (e.g., "z", "imx", or "ndets")
		t_true : float
			Value of the true duration of the burst. If given, a horizontal line will be marked at t_true.
		dur_frac : boolean
			Indicates whether the y-axis will be simply the duration measure (dur_frac = False) or the fraction of the true duration (dur_frac = True). If True, t_true must be supplied
		ax : matplotlib.axes
			Axis on which to create the figure
		"""

		if dur_frac is True:
			if t_true is None:
				print("A true duration must be given to create duration fraction axis.")
				return;
			sim_results['DURATION'] /= t_true

		if ax is None:
			ax = plt.figure().gca()

		if joined is True:
			line, = ax.step(sim_results[obs_param], sim_results['DURATION'], where="mid", **kwargs)
		else:
			line, = ax.scatter(sim_results[obs_param], sim_results['DURATION'],marker=marker, **kwargs)

		ax.set_xlabel("{}".format(obs_param), fontsize=self.fontsize, fontweight=self.fontweight)
		ax.set_ylabel("Duration (sec)", fontsize=self.fontsize, fontweight=self.fontweight)

		if "label" in kwargs:
			ax.legend()

		self.plot_aesthetics(ax)
		ax.margins(x=0.1,y=0.05)

		return line

	def det_plane_map(self, sim_results, ax=None, imx_max=1.75*1.1, imy_max=0.875*1.1, inc_grids=False, **kwargs):
		"""
		Method to plot the average duration percentage as a function of the position on the detector plane

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		imx_max, imy_max : float, float
			Defines the maximum (and minimum) values of the x and y plane on the detector
		inc_grids : bool
			Whether to include where the grid separations lie
		"""

		if ax is None:
			ax = plt.figure(figsize=(3*3.5, 3*1.75)).gca()
		fig = plt.gcf()


		if inc_grids is True:
			x_divs = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
			y_divs = [-0.875, -0.525, -0.175, 0.175, 0.525, 0.875]

			ax.plot([x_divs[0], x_divs[0]], [y_divs[1], y_divs[-2]], color="k", alpha=0.2)
			ax.plot([x_divs[-1], x_divs[-1]], [y_divs[1], y_divs[-2]], color="k", alpha=0.2)

			ax.plot([x_divs[1], x_divs[-2]], [y_divs[0], y_divs[0]], color="k", alpha=0.2)
			ax.plot([x_divs[1], x_divs[-2]], [y_divs[-1], y_divs[-1]], color="k", alpha=0.2)

			for i in range(1,len(x_divs)-1):
				ax.plot([x_divs[i], x_divs[i]], [y_divs[0], y_divs[-1]], color="k", alpha=0.2)
			for i in range(1,len(y_divs)-1):
				ax.plot([x_divs[0], x_divs[-1]], [y_divs[i], y_divs[i]], color="k", alpha=0.2)


		cmap = mpl.colormaps["viridis"].copy()
		cmap.set_bad(color="gray")
		cmap.set_under(color="gray")
		cmin = np.min(1e-9)

		im = ax.scatter(sim_results['imx'],sim_results['imy'],c=sim_results['DURATION'],cmap=cmap, vmin=cmin, marker=',', **kwargs)
		cbar = fig.colorbar(im)


		ax.set_xlim(-imx_max,imx_max)
		ax.set_ylim(-imy_max,imy_max)

		ax.set_xlabel("IMX",fontsize=self.fontsize,fontweight=self.fontweight)
		ax.set_ylabel("IMY",fontsize=self.fontsize,fontweight=self.fontweight)

		cbar.set_label("Duration (sec)",fontsize=self.fontsize,fontweight=self.fontweight)
		cbar.ax.axhline(2, c='w')
			

		self.tight_layout()
		self.plot_aesthetics(ax)

	def redshift_duration_evo(self, sim_results, ax=None, 
		t_true=1, t_max=None, bins=None, dur_frac=False, 
		log=False, norm=mcolors.LogNorm, inc_cbar=False, 
		cmin = 1, dt = 1,
		inc_cosmo_line=True, **kwargs):
		"""
		Method to plot the measured duration of each synthetic light curve as a function redshift

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		t_true : float
			True duration of the emission
		t_max : float
			y-axis maximum
		bins : None or int or [int, int] or array-like or [array, array]
			The bin specification:
				If `int`, the number of bins for the two dimensions `(nx = ny = bins)`.
				If `[int, int]`, the number of bins in each dimension `(nx, ny = bins)`.
				If array-like, the bin edges for the two dimensions `(x_edges = y_edges = bins)`.
				If `[array, array]`, the bin edges in each dimension `(x_edges, y_edges = bins)`.
		dur_frac : bool
			Indicates whether the y-axis should be actual duration measurement of the ratio of the duration 
			measurement to t_true
		log : bool
			Indicates if the y-axis should be taken to be the log of the data or not
		norm : str or matplotlib.colors.Normalize
			Sets the normalization scale for the density plot according to the colormap 
		inc_cbar : bool
			Indicates whether to include a colorbar or not
		cmin : float
			Sets the minimum cut-off value for the density plot. Any bin with fewer counts than cmin are omitted. 
		dt : float
			The time bin size of the simulated light curve. This is used to determine histogram bins.
		inc_cosmo_line : bool
			Indicates whether to include the t_true*(1+z) line or not
		"""

		if ax is None:
			ax = plt.figure().gca()
		fig = plt.gcf()

		results = sim_results[sim_results['DURATION'] > 0]

		z_min, z_max = np.min(sim_results['z']), np.max(sim_results['z'])
		num_trials = len(results["DURATION"][results['z']==z_min])

		if t_max is None:
			# Determine bins before any cuts are applied
			z_bins = np.unique(sim_results['z'])
			t_bins = np.arange(start=0, stop=np.max(sim_results['DURATION']), step=dt)
			# Make a histogram
			hist, xedges, yedges = np.histogram2d(sim_results['z'], sim_results['DURATION'], bins=[z_bins, t_bins])
			# Remove bins that fall under the cut-off limit
			hist[np.where(hist <= cmin)] = 0
			# Find T max of this histogram
			t_max = yedges[np.max(np.where(hist>0)[1])]

		z_arr = np.linspace(0, z_max*1.1)
		def dilation_line(z):
			return t_true*(1+z)/(1+z_min)

		cmap = mpl.colormaps["viridis"].copy()
		cmap.set_bad(color="w")
		cmap.set_under(color="w")

		dur_arr = results["DURATION"]
		if dur_frac is True:
			if t_true is None:
				print("A true duration must be given to plot a duration fraction.")
				return;
			dur_arr /= t_true
		t_min = 0
		if log is True:
			dur_arr = np.log10(dur_arr)
			t_max = np.log10(t_max)
			t_min = -1

		if bins == None:
			num_z_bins = len(np.unique(sim_results['z']))
			dz = (z_max - z_min)/num_z_bins
			z_bins = np.arange(start=z_min-(dz/2), stop=z_max+(dz/2), step=dz)
			t_bins = np.arange(start=t_min, stop=t_max, step=dt)

		im = ax.hist2d(results['z'], dur_arr, bins=[z_bins, t_bins], cmin=cmin, cmap=cmap, norm=norm(vmin=cmin, vmax= num_trials), **kwargs) # output = counts, xbins, ybins, image

		# h, xedges, yedges = np.histogram2d(results['z'], dur_arr, range= [[z_min, z_max], [t_min, t_max]])
		# xbins = xedges[:-1] + (xedges[1] - xedges[0]) / 2
		# ybins = yedges[:-1] + (yedges[1] - yedges[0]) / 2
		# h = h.T
		# Note: The +1 is to allow for Log scale without requiring mask
		# CS = ax.contour(xbins, ybins, h+1, colors="gray", norm=mcolors.LogNorm(), levels=np.logspace(1, 3, 10))

		if inc_cbar == True:
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			cbar = fig.colorbar(im[3], cax=cax, orientation='vertical', ticks=[0.01*num_trials, 0.1*num_trials, num_trials])
			cbar.ax.set_yticklabels(["1%", "10%", "100%"])

		# if (t_true is not None):
		# 	if (dur_frac is False):
		# 		ax.axhline(y=t_true,color="C1",linestyle="dashed", alpha=0.5, label="True Duration")
		# 	else:
		# 		ax.axhline(y=1,color="C1",linestyle="dashed",alpha=0.5,label="True Duration")

		ax.axvline(x=z_min,color="k", linewidth=2, linestyle="dotted", label="Measured Redshift")
		# ax.axvline(x=z_max, color="C1", linewidth=2, label="Max. Simulated Redshift")
		# ax.axhline(y=2,color="w",linestyle="dashed",alpha=0.5)

		ax.set_xlabel("Redshift",fontsize=self.fontsize,fontweight=self.fontweight)
		if log is False:
			if inc_cosmo_line is True:
				ax.plot(z_arr, dilation_line(z_arr), color="C1", alpha=1, linewidth=3)
			ax.set_ylabel("Duration (sec)",fontsize=self.fontsize,fontweight=self.fontweight)
			ax.set_ylim(0)

		else:
			if inc_cosmo_line is True:
				ax.plot(z_arr, np.log10(dilation_line(z_arr)), color="C1", alpha=1, linewidth=3)
			ax.set_ylabel("log(Duration)",fontsize=self.fontsize,fontweight=self.fontweight)
			ax.set_ylim(-1)

		ax.set_xlim(0, z_max)

		self.tight_layout()
		self.plot_aesthetics(ax)

	def redshift_fluence_evo(self, sim_results, ax=None, 
		F_true=None, F_max=None, F_min=None, bins=None, 
		fluence_frac=False, norm=mcolors.LogNorm, inc_cbar=False, 
		cmin = 1, inc_sensitivity_line = True,
		inc_cosmo_line=False, specfunc=None, e_min=15, e_max=350, **kwargs):
		"""
		Method to plot the measured duration of each synthetic light curve as a function redshift

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		F_true : float
			Observed fluence of the burst at the lowest redshift 
		F_max : float
			y-axis maximum
		F_min : float
			y-axis minimum
		bins : None or int or [int, int] or array-like or [array, array]
			The bin specification:
				If `int`, the number of bins for the two dimensions `(nx = ny = bins)`.
				If `[int, int]`, the number of bins in each dimension `(nx, ny = bins)`.
				If array-like, the bin edges for the two dimensions `(x_edges = y_edges = bins)`.
				If `[array, array]`, the bin edges in each dimension `(x_edges, y_edges = bins)`.
		fluence_frac : bool
			Indicates whether the y-axis should be a fraction of the true fluence
		norm : str or matplotlib.colors.Normalize
			Sets the normalization scale for the density plot according to the colormap 
		inc_cbar : bool
			Indicates whether to include a colorbar or not
		cmin : float
			Sets the minimum cut-off value for the density plot. Any bin with fewer counts than cmin are omitted. 
		inc_cosmo_line : bool
			Indicates whether to include the 5-sigma sensitivty line (for Swift/BAT)
		inc_cosmo_line : bool
			Indicates whether to include the F_true/(1+z) line or not
		specfunc : SPECFUNC
			The observed spectral function of the burst at the lowest redshift 
		e_min, e_max : float, float
			Minimum and maximum energy values of the observed band, used for k-correction calculations. 
			Defaults are 15 and 350 keV, respectively, to represent Swift/BAT energy band.
		"""

		if ax is None:
			ax = plt.figure().gca()
		fig = plt.gcf()

		if (inc_cosmo_line == True) and not isinstance(specfunc, SPECFUNC):
			print("If a cosmological fluence line is to be included, a spectral function of type SPECFUNC must also be given.")
			return;

		results = sim_results[sim_results['FLUENCE'] > 0]

		z_min, z_max = np.min(sim_results['z']), np.max(sim_results['z'])
		num_trials = len(results["DURATION"][results['z']==z_min])

		if F_max is None:
			F_max = np.log10(np.max(sim_results['FLUENCE']))

		if F_true is None:
			F_true = np.mean(sim_results['FLUENCE'][sim_results['z']==z_min])

		cmap = mpl.colormaps["viridis"].copy()
		cmap.set_bad(color="w")
		cmap.set_under(color="w")

		fluence_arr = np.log10(results["FLUENCE"])
		if fluence_frac is True:
			if F_true is None:
				print("A true fluence must be given to plot a duration fraction.")
				return;
			fluence_arr /= F_true

		if F_min is None:
			# Determine bins before any cuts are applied
			num_z_bins = len(np.unique(sim_results['z']))
			dz = (z_max - z_min)/num_z_bins
			z_bins = np.arange(start=z_min-(dz/2), stop=z_max+(dz/2), step=dz)

			f_bins = np.linspace(start=np.min(fluence_arr), stop=F_max, num=50)
			# Make a histogram
			hist, xedges, yedges = np.histogram2d(results['z'], fluence_arr, bins=[z_bins, f_bins])
			# Remove bins that fall under the cut-off limit
			hist[np.where(hist <= cmin)] = 0
			# Find F min of this histogram
			F_min = np.min([-1, yedges[np.min(np.where(hist>0)[1])] ])

		if bins == None:
			num_z_bins = len(np.unique(sim_results['z']))
			dz = (z_max - z_min)/num_z_bins
			z_bins = np.arange(start=z_min-(dz/2), stop=z_max+(dz/2), step=dz)

			f_bins = np.linspace(start=F_min, stop=F_max, num=int( (F_max - F_min)*30) )

		im = ax.hist2d(results['z'], fluence_arr, 
						bins=[z_bins, f_bins], cmin=cmin, cmap=cmap, 
						norm=norm(vmin=cmin, vmax= num_trials), **kwargs)

		if inc_cbar == True:
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			cbar = fig.colorbar(im[3], cax=cax, orientation='vertical', ticks=[0.01*num_trials, 0.1*num_trials, num_trials])
			cbar.ax.set_yticklabels(["1%", "10%", "100%"])

		ax.axvline(x=z_min,color="k",linewidth=2, linestyle="dotted", label="Measured Redshift")
		# ax.axvline(x=z_max,color="C1",linewidth=2, label="Max. Simulated Redshift")

		ax.set_xlabel("Redshift",fontsize=self.fontsize,fontweight=self.fontweight)

		if inc_cosmo_line is True:
			z_arr = np.linspace(z_min, z_max*1.1)
			
			ax.plot(z_arr, 
				np.log10(self._luminosity_distance(z_arr, specfunc, F_true, z_min, e_min, e_max)), 
				color="C1", alpha=1, linewidth=3)

		if inc_sensitivity_line is True:
			z_vals = np.unique(sim_results['z'])
			t_vals = np.zeros(shape=len(z_vals))
			for i in range(len(z_vals)):
				t_vals[i] = np.mean(results['DURATION'][results['z']==z_vals[i]])

			ax.plot(z_vals, np.log10(self._fluence_sens(t_vals)), color="magenta", linewidth=2) # 5-sigma fluence limit 

		ax.set_ylabel(r"log(Photon Fluence) log(cnts cm$^{-2}$)",fontsize=self.fontsize,fontweight=self.fontweight)
		ax.set_ylim(F_min)

		ax.set_xlim(0, z_max)

		# cbar.set_label("Frequency",fontsize=self.fontsize,fontweight=self.fontweight)

		self.tight_layout()
		self.plot_aesthetics(ax)


	def _fluence_sens(self, time):
		# Swift/BAT 5-sigma Fluence sensitivity line (see Baumgartner 2013)
		return 1.18 * 2.4*10**(-2) * time**(1./2.)  # Units of counts / s / det

	def _luminosity_distance(self, z, specfunc, F_true, z_min, e_min, e_max):
		# Analytically calculate fluence evolution across cosmological distances

		arr = np.zeros(shape=len(z))
		for i in range(len(z)):
			new_spec = specfunc.deepcopy()
			# Move spectral function to z_p frame by correcting E_peak or temperature by the redshift 
			# (if spectral function has a peak energy or temperature)
			for j, (key, val) in enumerate(new_spec.params.items()):
				if key == "ep":
					new_spec.params[key] *= (1+z_min)/(1+z[i])
				if key == "temp":
					new_spec.params[key] *= (1+z_min)/(1+z[i])

			k_corr_rat = k_corr(specfunc, z_min, e_min, e_max) / k_corr(new_spec, z[i], e_min, e_max)

			arr[i] = F_true * k_corr_rat * ((1+z[i])/(1+z_min)) * (lum_dis(z_min) / lum_dis(z[i]) )**2
		return arr

	def det_frac(self, sim_results, ax=None, alpha=0.4, step="mid", **kwargs):
		"""
		Method to plot the detection fraction of a GRB as a function of the redshift it was simulated at. 
		The detection fraction is defined as ratio between the number of simulations (at a specific redshift) that 
		measurements were obtained for and the the total number of simulations.

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		"""

		if ax is None:
			ax = plt.figure().gca()
		# fig = plt.gcf()

		zs = np.unique(sim_results['z'])
		trials = len(sim_results[sim_results['z']==zs[0]])

		perc = []
		num_det = []
		for i in range(len(zs)):
			tmp = sim_results[sim_results['z']==zs[i]]
			num_det.append(len(tmp[tmp['DURATION']>0]))
			perc.append( num_det[i] / trials)

		ax.fill_between(x=zs, y1=(num_det+np.sqrt(trials))/trials, y2 = (num_det-np.sqrt(trials))/trials, alpha=alpha, step=step, **kwargs)
		ax.step(zs, perc, where=step, **kwargs)

		ax.set_xlabel("Redshift",fontsize=self.fontsize,fontweight=self.fontweight)
		ax.set_ylabel("Detection Fraction",fontsize=self.fontsize,fontweight=self.fontweight)

		self.tight_layout()
		self.plot_aesthetics(ax)

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

		ax.set_xlim(1)
		ax.set_ylim(0,1.05)

		if "label" in kwargs:
			ax.legend()

		ax.set_xlabel(r"T$_{90}$ (sec)", fontsize=14)
		ax.set_ylabel("Cumulative Histogram", fontsize=14)
		ax.set_title("T90 Distrubtions", fontsize=14)

		self.plot_aesthetics(ax)

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

		ax.set_xlim(0.5)

		if "label" in kwargs:
			ax.legend()

		ax.set_xlabel("Fluence (counts/det)", fontsize=self.fontsize)
		ax.set_ylabel("Cumulative Histogram", fontsize=self.fontsize)
		ax.set_title("Fluence Distrubtion", fontsize=self.fontsize)

		self.plot_aesthetics(ax)

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

		ax.set_xlabel("1s Peak Flux (counts/sec/det)", fontsize=self.fontsize)
		ax.set_ylabel("Normalied Histogram (arb units)", fontsize=self.fontsize)
		ax.set_title("1s Peak Flux Distrubtion", fontsize=self.fontsize)

		self.plot_aesthetics(ax)

	def _make_cumu_plot(self, values, bins, ax, **kwargs):

		# Make histogram
		count, edges = np.histogram(values, bins=bins)
		# Make cumulative distribution 
		cum_count = np.cumsum(count)
		# Plot cumulative distribution 
		ax.stairs(cum_count/np.max(cum_count), edges, **kwargs)


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

		self.plot_aesthetics(ax)


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

		self.plot_aesthetics(ax)
