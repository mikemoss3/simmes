"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the class and methods used for creating plots related to trigger algorithms 

"""
from fits.io import fits
import matplotlib.pyplot as plt

from simmes.PLOTS import PLOTS

class PLOTTRIGS(PLOTS):
	def __init__(self):
		PLOTS.__init__(self)

	def plot_BAT_det_plane(self, fn="src.mask", ax=None, show_quads=False, fn_prefix=None, dpi="figure"):
		"""
		Method to plot the BAT detector plane based on the given detector mask (created for a 
		specified source position on the detector plane).

		Attributes:
		--------------
		fn : str
			file path to the detector plane mask 
		ax : matplotlib.axes.Axes
			matplotlib.axes.Axes instance to plot to
		show_quad : boolean
			Indicate whether or not to display quad labels and division lines
		fn_prefix : str
			File path to where the figure should be saved
		dpi : float or 'figure'
			The resolution in dots per inch. If 'figure', use the figure's dpi value.

		Returns:
		--------------
		ax : matplotlib.axes.Axes
			Axes instance that the detector plane image has been made on
		"""

		if ax is None:
			ax = plt.figure().gca()

		mask = fits.getdata(fn)

		ax.imshow(mask, cmap="Greys", origin="lower")

		if show_quads == True:
			ax.axvline(x=int(len(mask[0])/2), ymin=0, ymax=1, color="r", alpha=0.5, linewidth=2)
			ax.axhline(y=int(len(mask)/2), xmin=0, xmax=1, color="r", alpha=0.5, linewidth=2)
			ax.text(x=0.05, y=0.05, s="Q0", color="k", transform=ax.transAxes,horizontalalignment='center',
				verticalalignment='center', bbox=dict(facecolor='white', alpha=0.75))
			ax.text(x=0.95, y=0.05, s="Q1", color="k", transform=ax.transAxes,horizontalalignment='center',
				verticalalignment='center', bbox=dict(facecolor='white', alpha=0.75))
			ax.text(x=0.05, y=0.95, s="Q2", color="k", transform=ax.transAxes,horizontalalignment='center',
				verticalalignment='center', bbox=dict(facecolor='white', alpha=0.75))
			ax.text(x=0.95, y=0.95, s="Q3", color="k", transform=ax.transAxes,horizontalalignment='center',
				verticalalignment='center', bbox=dict(facecolor='white', alpha=0.75))

		ax.set_ylabel("IMY")
		ax.set_xlabel("IMX")

		if fn_prefix is not None:
			plt.savefig(fn_prefix+"detector-plane.png", dpi=dpi)
		
		return ax

	def plot_quad_band_and_det_plane(self, quad_lc, mask_fn="src.mask", fn_prefix=None, dpi="figure"):
			"""
			Method to plot the given quad band light curve

			Attributes:
			--------------
			mask_fn : str
				File path to the source mask 
			fn_prefix : str
				File path to where the figure should be saved
			dpi : float or 'figure'
				The resolution in dots per inch. If 'figure', use the figure's dpi value.

			Returns:
			--------------
			ax1, ax2, ax3, ax4 : matplotlib.pyplot.Axes
				Axes instances with quad-band light curves for each quadrant 
			"""
			
			ax1 = plt.figure().gca()
			ax1.step(quad_lc['TIME'], quad_lc['RATE'], label="Tot")
			ax1.step(quad_lc['TIME'], quad_lc['1525']["q0"], label="1525")
			ax1.step(quad_lc['TIME'], quad_lc['1550']["q0"], label="1550")
			ax1.step(quad_lc['TIME'], quad_lc['25100']["q0"], label="25100")
			ax1.step(quad_lc['TIME'], quad_lc['50350']["q0"], label="50350")
			ax1.set_title("q0", fontsize=14)
			ax1.set_xlabel("TIME (sec)")
			ax1.set_ylabel("RATE (cnts/s)")
			ax1.legend(fontsize=14)
			if fn_prefix is not None:
				plt.savefig(fn_prefix+"quad-band-lc-q0.png", dpi=dpi)

			ax2 = plt.figure().gca()
			ax2.step(quad_lc['TIME'], quad_lc['RATE'], label="Tot")
			ax2.step(quad_lc['TIME'], quad_lc['1525']["q1"], label="1525")
			ax2.step(quad_lc['TIME'], quad_lc['1550']["q1"], label="1550")
			ax2.step(quad_lc['TIME'], quad_lc['25100']["q1"], label="25100")
			ax2.step(quad_lc['TIME'], quad_lc['50350']["q1"], label="50350")
			ax2.set_title("q1", fontsize=14)
			ax2.set_xlabel("TIME (sec)")
			ax2.set_ylabel("RATE (cnts/s)")
			ax2.legend(fontsize=14)
			if fn_prefix is not None:
				plt.savefig(fn_prefix+"quad-band-lc-q1.png", dpi=dpi)

			ax3 = plt.figure().gca()
			ax3.step(quad_lc['TIME'], quad_lc['RATE'], label="Tot")
			ax3.step(quad_lc['TIME'], quad_lc['1525']["q2"], label="1525")
			ax3.step(quad_lc['TIME'], quad_lc['1550']["q2"], label="1550")
			ax3.step(quad_lc['TIME'], quad_lc['25100']["q2"], label="25100")
			ax3.step(quad_lc['TIME'], quad_lc['50350']["q2"], label="50350")
			ax3.set_title("q2", fontsize=14)
			ax3.set_xlabel("TIME (sec)")
			ax3.set_ylabel("RATE (cnts/s)")
			ax3.legend(fontsize=14)
			if fn_prefix is not None:
				plt.savefig(fn_prefix+"quad-band-lc-q2.png", dpi=dpi)

			ax4 = plt.figure().gca()
			ax4.step(quad_lc['TIME'], quad_lc['RATE'], label="Tot")
			ax4.step(quad_lc['TIME'], quad_lc['1525']["q3"], label="1525")
			ax4.step(quad_lc['TIME'], quad_lc['1550']["q3"], label="1550")
			ax4.step(quad_lc['TIME'], quad_lc['25100']["q3"], label="25100")
			ax4.step(quad_lc['TIME'], quad_lc['50350']["q3"], label="50350")
			ax4.set_title("q3", fontsize=14)
			ax4.set_xlabel("TIME (sec)")
			ax4.set_ylabel("RATE (cnts/s)")
			ax4.legend(fontsize=14)

			if fn_prefix is not None:
				plt.savefig(fn_prefix+"quad-band-lc-q3.png", dpi=dpi)

			return ax1, ax2, ax3, ax4

	def plot_trig_alg(self, ax, t0, tbk1, tfg, tbk2, elapsedur, fg_color="C1", bk_color="grey", alpha=0.5):
		"""
		Method to plot the intervals of the input trigger algorithm. 

		Attributes:
		--------------
		ax : matplotlib.pyplot.Axes
			Axes instance to plot onto
		t0 : float
			Start time of the trigger algorithm
		tbk1 : float 
			Duration of the first background interval
		tfg : float
			Duration of the foreground interval
		tbk2 : float
			Duration of the Second background interval
		elapsedur : float
			Duration between the background and foreground intervals
		fg_color : str
			Face color of the foreground region
		bk_color : str
			Face color of the background region(s)
		alpha : float 
			Opacity of the regions 

		Returns:
		--------------
		ax : matplotlib.pyplot.Axes
			Axes instance
		"""

		ax.axvspan(xmin=t0-elapsedur-tbk1, xmax=t0-elapsedur, color=bk_color, alpha=alpha)
		ax.axvspan(xmin=t0, xmax=t0+tfg, color=fg_color, alpha=alpha)
		ax.axvspan(xmin=t0+tfg+elapsedur, xmax=t0+tfg+elapsedur+tbk2, color=bk_color, alpha=alpha)

		return ax