import numpy as np
from astropy.io import fits

def find_grid_id(imx,imy):
	"""
	Method to find the Swift/BAT response matrix GridID based on the position of the source on the detector plane according to Lien et al 2012.
	"""
	imx_imy_info = np.genfromtxt("./util_packages/files-det-ang-dependence/gridnum_imx_imy.txt",dtype=[("GRIDID","U3"),("imx",float),("imxmin",float),("imxmax",float),("imy",float),("imymin",float),("imymax",float),("thetacenter",float),("pcode",float)])
	# Based on imx and imy, determine which grid number to use
	try:
		gridid = imx_imy_info['GRIDID'][(imx>=imx_imy_info['imxmin']) & (imx<=imx_imy_info['imxmax']) & (imy>=imx_imy_info['imymin']) & (imy<=imx_imy_info['imymax'])][0]
	except:
		gridid = None

	return gridid

def find_inc_ang(imx,imy):
	"""
	Method to calculate the incidence angle (in radians) from a given position on the detector plane 
	"""

	""" # This is the correct way to find the incidence angle, but doesn't work with response function grid
	theta = np.arctan( np.sqrt( imx**2 + imy**2 ) )
	return theta
	"""

	imx_imy_info = np.genfromtxt("./util_packages/files-det-ang-dependence/gridnum_imx_imy.txt",dtype=[("GRIDID","U3"),("imx",float),("imxmin",float),("imxmax",float),("imy",float),("imymin",float),("imymax",float),("thetacenter",float),("pcode",float)])
	# Based on imx and imy, determine which grid number to use
	try:
		theta = imx_imy_info['thetacenter'][(imx>=imx_imy_info['imxmin']) & (imx<=imx_imy_info['imxmax']) & (imy>=imx_imy_info['imymin']) & (imy<=imx_imy_info['imymax'])][0]
		theta *= np.pi / 180
	except:
		theta = np.pi/2

	return theta

def find_pcode(imx,imy):
	"""
	Method to calculate the partial coding fraction on the detector plane for a given position on the detector plane 
	"""

	""" # This is the correct way to find the pcode, but doesn't work with response function grid	
	# Load pcode map image 
	pcode_img = fits.getdata("./util_packages/files-det-ang-dependence/pcode-map.img",ext=0) # indexing as pcode_img[y-index, x-index]
	# Load header from file
	pcode_img_header = fits.getheader("./util_packages/files-det-ang-dependence/pcode-map.img",ext=0)

	# Make (imx, imy) grid based on the indices (i,j)
	# i and j are the indices of each pixel
	# From headher files (the T at the end of the field name indicates tangent position): 
	# crpix: Reference pixel position
	# cdelt: Pixel spacing in physical units
	# crval: Coordinate value at reference pixel position (seems to be zero most of the time)
	i = int( ( ( imx - pcode_img_header["CRVAL1T"]) / -pcode_img_header["CDELT1T"]) + pcode_img_header["CRPIX1T"] )
	j = int( ( ( imy - pcode_img_header["CRVAL2T"]) / pcode_img_header["CDELT2T"]) + pcode_img_header["CRPIX2T"] )


	# The given imx,imy may be calculated to be in the center of a pixel. To make this compatible with calling an index, we force it to be a integer.
	# This has the result of rounding down.

	return pcode_img[j,i]
	"""

	imx_imy_info = np.genfromtxt("./util_packages/files-det-ang-dependence/gridnum_imx_imy.txt",dtype=[("GRIDID","U3"),("imx",float),("imxmin",float),("imxmax",float),("imy",float),("imymin",float),("imymax",float),("thetacenter",float),("pcode",float)])
	# Based on imx and imy, determine which grid number to use
	try:
		pcode = imx_imy_info['pcode'][(imx>=imx_imy_info['imxmin']) & (imx<=imx_imy_info['imxmax']) & (imy>=imx_imy_info['imymin']) & (imy<=imx_imy_info['imymax'])][0]
	except:
		pcode = 0

	return pcode


def fraction_correction(imx, imy):
	"""
	Method that calculates and returns a correction fraction that was found to be needed for off-axis bursts. 
	This factor is needed to correct for the FFT convolution that is used for Swift/BAT
	This correction was empirically fit with a quadratic function, which is how the parameter values in this method were determined.
	"""

	pcode = find_pcode(imx, imy)

	a=1.121
	b=-1.214
	c=0.618
	return a + (b*pcode) + (c*np.power(pcode,2))