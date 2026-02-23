"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the class and methods used for plotting simulation results.

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from simmes.PLOTS import PLOTS


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

		ax.set_xlabel("Time (sec)")
		ax.set_ylabel("Rate (counts/sec)")

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

		ax.set_xlabel("{}".format(obs_param))
		ax.set_ylabel("Duration (sec)")

		if "label" in kwargs:
			ax.legend()

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

		ax.set_xlabel("IMX")
		ax.set_ylabel("IMY")

		cbar.set_label("Duration (sec)")
		cbar.ax.axhline(2, c='w')
			
	def redshift_duration_evo(self, sim_results, ax=None, 
		t_true=1, t_max=None, dt = None, t_bins=None, z_bins=None, z_true=None,
		dur_frac=False, log=False, norm=mcolors.LogNorm, inc_cbar=False, 
		cmin = 1, inc_cosmo_line=True, **kwargs):
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
		dt : float
			The time bin size of the simulated light curve. This is used to determine histogram bins.
		t_bins : None or int or array-like
			The bin specification:
				If `int`, the number of duration bins `(ny = t_bins)`.
				If array-like, the bin edges for the duration bins `(y_edges = t_bins)`.
			Takes precedence over dt parameter.
		z_bins : int or array-like
			The bin specification:
				If `int`, the number of redshift bins `(nx = z_bins)`.
				If array-like, the bin edges for the redshift bins `(x_edges = z_bins)`.
		z_true : float
			Observed redshift of the source. This should be used in cases where the minimum redshift of the 
			simulations is greater than the observed redshift 
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
		inc_cosmo_line : bool
			Indicates whether to include the t_true*(1+z) line or not
		"""

		if ax is None:
			ax = plt.figure().gca()
		fig = plt.gcf()

		results = sim_results[sim_results['DURATION'] > 0]


		z_min, z_max = np.min(sim_results['z']), np.max(sim_results['z'])
		num_trials = len(results["DURATION"][results['z']==z_min])
		if z_true is None:
			z_true = z_min

		if t_max is None:
			t_max = np.max(sim_results['DURATION'])

		z_arr = np.linspace(0, z_max*1.1)
		def dilation_line(z):
			return t_true*(1+z)/(1+z_true)

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

		# Determine redshift bins 
		if isinstance(z_bins, int):
			num_z_bins = z_bins
		else:
			num_z_bins = len(np.unique(sim_results['z']))
		dz = (z_max - z_min)/num_z_bins
		if not isinstance(z_bins, np.ndarray):
			z_bins = np.linspace(start=z_min-(dz/2), stop=z_max+(dz/2), num=num_z_bins+1)

		# Determine duration bins 
		if isinstance(t_bins, int):
			num_t_bins = t_bins
		else:
			num_t_bins = int( (t_max - t_min)*50)
		
		if dt is None:
			dt = (t_max - t_min)/num_t_bins
		else:
			# dt value was given by user, and num_t_bins must be defined.
			num_t_bins = int( (t_max - t_min) / dt )
		
		if not isinstance(t_bins, np.ndarray):
			t_bins = np.linspace(start=t_min-(dt/2), stop=t_max+(dt/2), num = num_t_bins)

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

		ax.set_xlabel("Redshift")
		if log is False:
			if inc_cosmo_line is True:
				ax.plot(z_arr, dilation_line(z_arr), color="C1", alpha=1, linewidth=3)
			ax.set_ylabel("Duration (sec)")
			ax.set_ylim(0)

		else:
			if inc_cosmo_line is True:
				ax.plot(z_arr, np.log10(dilation_line(z_arr)), color="C1", alpha=1, linewidth=3)
			ax.set_ylabel("log(Duration / 1 sec)")
			ax.set_ylim(-1)

		ax.set_xlim(0, z_max)

	def redshift_fluence_evo(self, sim_results, ax=None, 
		F_true=None, F_max=None, F_min=None, f_bins=None, z_bins=None, z_true=None,
		fluence_frac=False, norm=mcolors.LogNorm, inc_cbar=False, 
		cmin = 1, inc_sensitivity_line = False, num_t_bins = None,
		inc_cosmo_line=False, specfunc=None, e_min=15, e_max=350, **kwargs):
		"""
		Method to plot the measured fluence of each synthetic light curve as a function redshift

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
		f_bins : None or int or array-like
			The bin specification:
				If `int`, the number of fluence bins `(ny = f_bins)`.
				If array-like, the bin edges for the fluence bins `(y_edges = f_bins)`.
		z_bins : None or int or array-like
			The bin specification:
				If `int`, the number of redshift bins `(nx = z_bins)`.
				If array-like, the bin edges for the redshift bins `(x_edges = z_bins)`.
		z_true : float
			Observed redshift of the source. This should be used in cases where the minimum redshift of the 
			simulations is greater than the observed redshift 
		fluence_frac : bool
			Indicates whether the y-axis should be a fraction of the true fluence
		norm : str or matplotlib.colors.Normalize
			Sets the normalization scale for the density plot according to the colormap 
		inc_cbar : bool
			Indicates whether to include a colorbar or not
		cmin : float
			Sets the minimum cut-off value for the density plot. Any bin with fewer counts than cmin are omitted. 
		inc_sensitivity_line : bool
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
		if (inc_sensitivity_line) and (num_t_bins is None):
			print("If a fluence sensitivity line is to be included, the number of Bayesian block measurements (i.e., time bins) must be given.")
			return;

		results = sim_results[sim_results['FLUENCE'] > 0]

		z_min, z_max = np.min(sim_results['z']), np.max(sim_results['z'])
		num_trials = len(results["DURATION"][results['z']==z_min])
		if z_true is None:
			z_true = z_min

		if F_max is None:
			F_max = np.log10(np.max(sim_results['FLUENCE']))

		if F_true is None:
			F_true = np.mean(sim_results['FLUENCE'][sim_results['z']==z_true])

		cmap = mpl.colormaps["viridis"].copy()
		cmap.set_bad(color="w")
		cmap.set_under(color="w")

		fluence_arr = np.log10(results["FLUENCE"])
		if fluence_frac is True:
			fluence_arr /= F_true

		if F_min is None:
			# Determine bins before any cuts are applied
			tmp_num_z_bins = len(np.unique(sim_results['z']))
			tmp_dz = (z_max - z_min)/tmp_num_z_bins
			tmp_z_bins = np.arange(start=z_min-(tmp_dz/2), stop=z_max+(tmp_dz/2), step=tmp_dz)

			tmp_f_bins = np.linspace(start=np.min(fluence_arr), stop=F_max, num=50)
			# Make a histogram
			hist, xedges, yedges = np.histogram2d(results['z'], fluence_arr, bins=[tmp_z_bins, tmp_f_bins])
			# Remove bins that fall under the cut-off limit
			hist[np.where(hist <= cmin)] = 0
			# Find F min of this histogram
			F_min = np.min([-1, yedges[np.min(np.where(hist>0)[1])] ])

		# Determine redshift bins 
		if isinstance(z_bins, int):
			num_z_bins = z_bins
		else:
			num_z_bins = len(np.unique(sim_results['z']))
		dz = (z_max - z_min)/num_z_bins
		if not isinstance(z_bins, np.ndarray):
			z_bins = np.linspace(start=z_min-(dz/2), stop=z_max+(dz/2), num=num_z_bins+1)

		# Determine fluence bins 
		if isinstance(f_bins, int):
			num_f_bins = f_bins
		else:
			num_f_bins = int( (F_max - F_min)*50)
		df = (F_max - F_min)/num_f_bins
		if not isinstance(f_bins, np.ndarray):
			f_bins = np.linspace(start=F_min-(df/2), stop=F_max+(df/2), num = num_f_bins)

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

		ax.set_xlabel("Redshift")

		if inc_cosmo_line is True:
			z_arr = np.linspace(z_min, z_max*1.1)
			
			ax.plot(z_arr, 
				np.log10(self._luminosity_distance(2, z_arr, specfunc, F_true, z_true, e_min, e_max)), 
				color="C1", alpha=1, linewidth=3)

		if inc_sensitivity_line is True:
			z_vals = np.unique(sim_results['z'])
			t_vals = np.zeros(shape=len(z_vals))
			for i in range(len(z_vals)):
				t_vals[i] = np.mean(results['DURATION'][results['z']==z_vals[i]])

			ax.plot(z_vals, np.log10(self._fluence_sens(num_t_bins, t_vals)), color="magenta", linewidth=2) # 5-sigma fluence limit 

		ax.set_ylabel("log(Photon Fluence /"+"\n"+r"1 cnt det$^{-1}$)")
		ax.set_ylim(F_min)

		ax.set_xlim(0, z_max)

		# cbar.set_label("Frequency")


	def _fluence_sens(self, n, time):
		"""
		Calculate fluence limit. 
		Based on the threshold of Bayesian block algorithm (Scargle et al 2013, Eq. 15) 
		and average background fluctuations of BAT

		Attributes:
		----------
		n : int
			number of measurements made by the Bayesian block algorithm
		time : float
			Duration of the emission
		"""
		return 1.12*10**(-2)* np.sqrt(2 * np.log10(n))* time**(1./2.)  # Units of counts / det

	def redshift_fpeak_evo(self, sim_results, ax=None, 
		fp_true=None, fp_max=None, fp_bins=None, z_bins=None, z_true=None,
		flux_frac = False, norm=mcolors.LogNorm, inc_cbar=False,
		cmin = 1, inc_sensitivity_line = False, num_t_bins = None, dt = None,
		inc_cosmo_line=False, specfunc=None, e_min=15, e_max=350, **kwargs):
		"""
		Method to plot the measured peak flux of each synthetic light curve as a function redshift

		Attributes:
		----------
		sim_results : dt_sim_res
			Simulation results to be plotted
		ax : matplotlib.axes
			Axis on which to create the figure
		fp_true : float
			Observed peak flux of the burst at the lowest redshift 
		fp_max : float
			y-axis maximum
		fp_bins : None or int or array-like
			The bin specification:
				If `int`, the number of peak flux bins `(ny = fp_bins)`.
				If array-like, the bin edges for the peak flux bins `(y_edges = fp_bins)`.
		z_bins : None or int or array-like
			The bin specification:
				If `int`, the number of redshift bins `(nx = z_bins)`.
				If array-like, the bin edges for the redshift bins `(x_edges = z_bins)`.
		z_true : float
			True redshift of the source. This should be used in cases where the minimum redshift of the 
			simulations is greater than the observed redshift 
		flux_frac : bool
			Indicates whether the y-axis should be a fraction of the true fluence
		norm : str or matplotlib.colors.Normalize
			Sets the normalization scale for the density plot according to the colormap 
		inc_cbar : bool
			Indicates whether to include a colorbar or not
		cmin : float
			Sets the minimum cut-off value for the density plot. Any bin with fewer counts than cmin are omitted. 
		inc_sensitivity_line : bool
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
			print("If a cosmological flux line is to be included, a spectral function of type SPECFUNC must also be given.")
			return;
		if (inc_sensitivity_line) and (num_t_bins is None):
			print("If a flux sensitivity line is to be included, the number of Bayesian block measurements (i.e., time bins) must be given.")
			return;
		if (inc_sensitivity_line) and (dt is None):
			print("If a flux sensitivity line is to be included, the time bin size must be given.")
			return;

		results = sim_results[sim_results['1sPeakFlux'] > 0]

		z_min, z_max = np.min(sim_results['z']), np.max(sim_results['z'])
		num_trials = len(results["DURATION"][results['z']==z_min])
		if z_true is None:
			z_true = z_min

		if fp_true is None:
			fp_true = np.mean(sim_results['1sPeakFlux'][sim_results['z']==z_min])

		if fp_max is None:
			fp_max = np.log10(np.max(sim_results['1sPeakFlux']))

		cmap = mpl.colormaps["viridis"].copy()
		cmap.set_bad(color="w")
		cmap.set_under(color="w")

		flux_arr = np.log10(results["1sPeakFlux"])
		if flux_frac is True:
			flux_arr /= fp_true

		# Determine redshift bins 
		if isinstance(z_bins, int):
			num_z_bins = z_bins
		else:
			num_z_bins = len(np.unique(sim_results['z']))
		dz = (z_max - z_min)/num_z_bins
		if not isinstance(z_bins, np.ndarray):
			z_bins = np.linspace(start=z_min-(dz/2), stop=z_max+(dz/2), num=num_z_bins+1)

		# Determine fluence bins 
		fp_min = -3 # Log 
		if isinstance(fp_bins, int):
			num_fp_bins = fp_bins
		else:
			num_fp_bins = int( (fp_max - fp_min)*50)
		df = (fp_max - fp_min)/num_fp_bins
		if not isinstance(fp_bins, np.ndarray):
			fp_bins = np.linspace(start=fp_min-(df/2), stop=fp_max+(df/2), num = num_fp_bins)

		im = ax.hist2d(results['z'], flux_arr, 
						bins=[z_bins, fp_bins], cmin=cmin, cmap=cmap, 
						norm=norm(vmin=cmin, vmax= num_trials), **kwargs)

		if inc_cbar == True:
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			cbar = fig.colorbar(im[3], cax=cax, orientation='vertical', ticks=[0.01*num_trials, 0.1*num_trials, num_trials])
			cbar.ax.set_yticklabels(["1%", "10%", "100%"])

		ax.axvline(x=z_min,color="k",linewidth=2, linestyle="dotted", label="Measured Redshift")
		# ax.axvline(x=z_max,color="C1",linewidth=2, label="Max. Simulated Redshift")

		ax.set_xlabel("Redshift")

		if inc_cosmo_line is True:
			z_arr = np.linspace(z_min, z_max*1.1)
			
			ax.plot(z_arr, 
				np.log10(self._luminosity_distance(1, z_arr, specfunc, fp_true, z_true, e_min, e_max)), 
				color="C1", alpha=1, linewidth=3)

		if inc_sensitivity_line is True:
			z_vals = np.unique(sim_results['z'])
			t_vals = np.zeros(shape=len(z_vals))

			ax.plot(z_vals, np.ones(shape=len(z_vals)) * np.log10(self._flux_sens(num_t_bins, dt)), color="magenta", linewidth=2) # 5-sigma flux limit 

		ax.set_ylabel("log(1s Peak Flux /"+"\n"+r"1 cnts s$^{-1}$ det$^{-1}$)")
		ax.set_ylim(-3)

		ax.set_xlim(0, z_max)

		# cbar.set_label("Frequency")


	def _flux_sens(self, n, dt):
		"""
		Calculate flux limit. 
		Based on the threshold of Bayesian block algorithm (Scargle et al 2013, Eq. 15) 
		and average background fluctuations of BAT

		Attributes:
		----------
		n : int
			Number of measurements made by the Bayesian block algorithm
		dt : float 
			Light curve time bin size 
		"""

		return 1.12*10**(-2) * np.sqrt(2 * np.log10(n)) * dt  # Units of counts / det / s

	def _luminosity_distance(self, N, z, specfunc, F_true, z_min, e_min, e_max):
		# Analytically calculate fluence or flux evolution across cosmological distances
		# N = 0 --> energy flux
		# N = 1 --> photon flux or energy fluence
		# N = 2 --> photon fluence

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

			arr[i] = F_true * k_corr_rat * ((1+z[i])/(1+z_min))**N * (lum_dis(z_min) / lum_dis(z[i]) )**2
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


		ax.set_xlabel("Redshift")
		ax.set_ylabel("Detection Fraction")
