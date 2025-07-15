"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the main class this code uses to store response matrices and the associated methods.
"""

import numpy as np
from astropy.io import fits
import fitsio
from scipy.stats import norm
import copy 
from pathlib import Path
import scipy.integrate as integrate

from simmes.util_packages.det_ang_dependence import find_grid_id
from simmes.util_packages.fluctuations import add_spec_fluctuations

path_here = Path(__file__).parent

class RSP(object):
	"""
	Response matrix class

	Attributes:
	----------
	E_phot_min, E_phot_max : float, float
		Minimum and maximum photon energy bins, keV
	num_phot_bins : int
		Number of photon bins
	E_chan_min, E_chan_max : float, float
		Minimum and maximum channel energies, keV
	num_chans : int
		Number of instrument channels
	"""

	def __init__(self, E_phot_min=1, E_phot_max=10, num_phot_bins=10, E_chan_min=1, E_chan_max=10, num_chans=10):
		"""
		RSP class initialization 
		
		Attributes:
		-----------
		E_phot_min : float
			Minimum photon energy, keV 
		E_phot_max : float
			Maximum photon energy, keV
		num_phot_bins : int
			Number of bins along the photon energy axis
		E_chan_min : float 
			Minimum channel energy, keV
		E_chan_max : float
			Maximum channel energy, keV
		num_chans : int
			Number of instrument channels 
		"""

		self.num_phot_bins = num_phot_bins
		self.num_chans = num_chans
		
		self.set_E_phot(E_phot_min=E_phot_min, E_phot_max=E_phot_max, num_phot_bins=num_phot_bins,verbose=False)
		self.set_E_chans(E_chan_min=E_chan_min, E_chan_max=E_chan_max, num_chans=num_chans, verbose=False)
		
		self.N_GRP = np.ones(shape=num_phot_bins) # The number of 'channel subsets' for for the energy bin
		self.F_CHAN = np.zeros(shape=num_phot_bins) # The channel number of the start of each "channel subset" for the energy bin
		self.N_CHAN = np.ones(shape=num_phot_bins)*num_chans # The number of channels within each "channel subset" for the energy bin

		# Initialize self.MATRIX as empty array
		self.make_empty_resp() # Contains all the response probability values for each
		# 						'channel subset' corresponding to the energy bin for a given row


	def __copy__(self):
		cls = self.__class__
		result = cls.__new__(cls)
		result.__dict__.update(self.__dict__)
		return result

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, copy.deepcopy(v, memo))
		return result

	def copy(self):
		return copy.deepcopy(self)

	def deepcopy(self):
		return copy.deepcopy(self)

	def copy_structure(self, templ_rsp):
		"""
		Attributes:
		----------
		templ_rsp : RSP
			RSP object to use copy matrix structure from 

		Returns:
		--------------
		None

		"""

		self.num_phot_bins = templ_rsp.num_phot_bins
		self.num_chans = templ_rsp.num_chans

		self.ENERG_LO = templ_rsp.ENERG_LO
		self.ENERG_MID = templ_rsp.ENERG_MID
		self.ENERG_HI = templ_rsp.ENERG_HI
		
		self.ECHAN_LO = templ_rsp.ECHAN_LO
		self.ECHAN_HI = templ_rsp.ECHAN_HI
		self.ECHAN_MID = templ_rsp.ECHAN_MID
		
		self.N_GRP = templ_rsp.N_GRP
		self.F_CHAN = templ_rsp.F_CHAN
		self.N_CHAN = templ_rsp.N_CHAN

		self.MATRIX = templ_rsp.MATRIX


	def set_E_phot(self, E_phot_min=0, E_phot_max=500, num_phot_bins=200, verbose=True):
		"""
		Class method to set the photon energy axis

		Attributes:
		----------
		E_phot_min, E_phot_max : float, float
			Minimum and maximum photon energy bins, keV
		num_phot_bins : int
			Number of photon bins
		verbose : bool
			Print completion message 

		Returns:
		--------------
		None
		""" 
		ENERG_AX = make_en_axis(E_phot_min, E_phot_max, num_phot_bins)
		self.num_phot_bins = num_phot_bins
		self.ENERG_LO = ENERG_AX['Elo'] # Incoming photon energy, lower bound
		self.ENERG_MID =  ENERG_AX['Emid'] # Incoming photon energy, bin center
		self.ENERG_HI =  ENERG_AX['Ehi'] # Incoming photon energy, upper bound
		if verbose is True:
			print("Response matrix has been reset to zeros.")
		self.make_empty_resp()

	def set_E_chans(self, E_chan_min=0, E_chan_max=200, num_chans=80, verbose=True):
		"""
		Class method to set the photon energy axis

		Attributes:
		-----------
		E_chan_min : float 
			Minimum channel energy, keV
		E_chan_max : float
			Maximum channel energy, keV
		num_chans : int
			Number of instrument channels 
		verbose : bool
			Print completion message 

		Returns:
		--------------
		None
		""" 
		ECHAN_AX = make_en_axis(E_chan_min, E_chan_max, num_chans)
		self.num_chans = num_chans
		self.ECHAN_LO = ECHAN_AX['Elo'] # Instrument energy channel lower bound
		self.ECHAN_MID = ECHAN_AX['Emid'] # Instrument energy channel center
		self.ECHAN_HI = ECHAN_AX['Ehi'] # Instrument energy channel upper bound
		if verbose is True:
			print("Response matrix has been reset to zeros.")
		self.make_empty_resp()

	def make_empty_resp(self, num_phot_bins=None, num_chans = None):
		""" 
		Create an empty response matrix. 
		Note: If the shape of the response matrix is changed, the response matrix is reset to zeros.

		Attributes:
		-----------
		num_phot_bins : int
			Number of photon bins
		num_chans : int
			Number of instrument channels 

		Returns:
		--------------
		None
		"""
		if num_phot_bins is not None:
			self.num_phot_bins = num_phot_bins
		if num_chans is not None:
			self.num_chans = num_chans

		self.ENERG_LO = np.zeros( shape=self.num_phot_bins )
		self.ENERG_HI = np.zeros( shape=self.num_phot_bins )
		self.ENERG_MID = np.zeros( shape=self.num_phot_bins )
		self.N_GRP = np.zeros( shape=self.num_phot_bins )
		self.F_CHAN = np.zeros( shape=self.num_phot_bins )
		self.N_CHAN = np.zeros( shape=self.num_phot_bins )

		self.ECHAN_LO = np.zeros( shape=self.num_chans )
		self.ECHAN_HI = np.zeros( shape=self.num_chans )
		self.ECHAN_MID = np.zeros( shape=self.num_chans )

		self.MATRIX = np.zeros(shape=(self.num_chans, self.num_phot_bins))

	def identity(self):
		""" Make identity matrix """
		if self.num_phot_bins != self.num_chans:
			print("Identity matrix must be a square matrix")
			return 1;

		for i in range(self.num_phot_bins):
			self.MATRIX[j, i] = 1

	def overDeltaE(self, alpha=2):
		"""
		Decrease as 1/DeltaE^alpha from E_true 

		Attributes:
		-----------
		alpha : float
			Power law index
		"""
		for i in range(self.num_phot_bins):
			for j in range(self.num_chans):
				self.MATRIX[i,j] = 1/(1+np.abs(self.ECHAN_MID[j] - self.ENERG_MID[i])**alpha)
			
			# Normalize this column
			self.MATRIX[:, i] /= np.sum(self.MATRIX[:, i])

	def gauss(self):
		""" Decrease as probability in a Gaussian behavior as channel energy moves away from photon energy """
		
		for i in range(self.num_phot_bins):
			dist = norm(self.ENERG_MID[i], np.sqrt(self.ENERG_MID[i]))
			for j in range(self.num_chans):
				self.MATRIX[j, i] = dist.pdf(self.ECHAN_MID[j])

	def load_rsp_from_file(self, file_name):
		"""
		Load response matrix from file

		Attributes:
		-----------
		file_name : str
			Path to file containing the response file to be loaded 

		Returns:
		--------------
		None
		""" 
		resp_data = fitsio.read(filename=file_name, ext=1)
		ebounds_data = fitsio.read(file_name, ext=2)
		
		self.num_phot_bins = len(resp_data)
		self.num_chans = len(ebounds_data)

		self.ENERG_LO = np.zeros(shape=self.num_phot_bins)
		self.ENERG_HI = np.zeros(shape=self.num_phot_bins)
		self.ENERG_MID = np.zeros(shape=self.num_phot_bins)
		self.N_GRP = np.zeros(shape=self.num_phot_bins)
		self.F_CHAN = np.zeros(shape=self.num_phot_bins)
		self.N_CHAN = np.zeros(shape=self.num_phot_bins)
		self.MATRIX = np.zeros(shape=(self.num_chans, self.num_phot_bins) )

		self.ECHAN_LO = np.zeros(shape=self.num_chans)
		self.ECHAN_HI = np.zeros(shape=self.num_chans)
		self.ECHAN_MID = np.zeros(shape=self.num_chans)

		for i in range(self.num_phot_bins):
			self.ENERG_LO[i] = resp_data[i][0] # Incoming photon energy, lower bound
			self.ENERG_HI[i] =  resp_data[i][1] # Incoming photon energy, upper bound
			self.ENERG_MID[i] =  (self.ENERG_LO[i]+self.ENERG_HI[i])/2. # Incoming photon energy, bin center
			self.N_GRP[i] = resp_data[i][2] # The number of 'channel subsets' for for the energy bin
			self.F_CHAN[i] = resp_data[i][3] # The channel number of the start of each "channel subset" for the energy bin
			self.N_CHAN[i] = resp_data[i][4] # The number of channels within each "channel subset" for the energy bin
			
			self.MATRIX[:,i] = resp_data[i][5] # Contains all the response probability values for each
			# 										'channel subset' corresponding to the energy bin for a given row
		
		for i in range(self.num_chans):
			self.ECHAN_LO[i] = ebounds_data[i][1] # Instrument energy channel lower bound
			self.ECHAN_HI[i] = ebounds_data[i][2] # Instrument energy channel upper bound
			self.ECHAN_MID[i] = (self.ECHAN_LO[i]+self.ECHAN_HI[i])/2. # Instrument energy channel center

	def _load_swift_bat_rsp_from_file(self, file_name):
		"""
		Load a Swift/BAT response matrix from file, this has hard coded key words

		Attributes:
		--------------
		file_name : str
			Path to file containing the response file to be loaded 

		Returns:
		--------------
		None
		""" 
		resp_data = fitsio.read(filename=file_name, ext=1)
		ebounds_data = fitsio.read(file_name, ext=2)
		
		self.num_phot_bins = len(resp_data)
		self.num_chans = len(ebounds_data)

		self.ENERG_LO = np.zeros(shape=self.num_phot_bins)
		self.ENERG_HI = np.zeros(shape=self.num_phot_bins)
		self.ENERG_MID = np.zeros(shape=self.num_phot_bins)
		self.N_GRP = np.zeros(shape=self.num_phot_bins)
		self.F_CHAN = np.zeros(shape=self.num_phot_bins)
		self.N_CHAN = np.zeros(shape=self.num_phot_bins)
		self.MATRIX = np.zeros(shape=(self.num_chans, self.num_phot_bins) )

		self.ECHAN_LO = np.zeros(shape=self.num_chans)
		self.ECHAN_HI = np.zeros(shape=self.num_chans)
		self.ECHAN_MID = np.zeros(shape=self.num_chans)

		self.ENERG_LO = resp_data["ENERG_LO"] # Incoming photon energy, lower bound
		self.ENERG_HI =  resp_data["ENERG_HI"] # Incoming photon energy, upper bound
		self.ENERG_MID =  (self.ENERG_LO+self.ENERG_HI)/2. # Incoming photon energy, bin center
		self.N_GRP = resp_data["N_GRP"] # The number of 'channel subsets' for for the energy bin
		self.F_CHAN = resp_data["F_CHAN"] # The channel number of the start of each "channel subset" for the energy bin
		self.N_CHAN = resp_data["N_CHAN"] # The number of channels within each "channel subset" for the energy bin
		self.MATRIX = resp_data["MATRIX"].T # Contains all the response probability values for each
		# 										'channel subset' corresponding to the energy bin for a given row

		self.ECHAN_LO = ebounds_data["E_MIN"] # Instrument energy channel lower bound
		self.ECHAN_HI = ebounds_data["E_MAX"] # Instrument energy channel upper bound
		self.ECHAN_MID = (self.ECHAN_LO+self.ECHAN_HI)/2. # Instrument energy channel center

	def load_SwiftBAT_resp(self, imx, imy):
		"""
		Method to load an (interpolated) Swift/BAT response matrix given the position of the source on the detector plane.

		Attributes:
		-----------
		imx, imy : float, float
			Source position on the detector plane in x and y coordinates

		Returns:
		--------------
		None
		"""
		# # Error prevention.
		# inft = 1e-10
		# imx+=inft
		# imy+=inft

		# If the imx, imy is off of the detector plane, we can just return an empty response matrix.
		gridid_test = find_grid_id(imx, imy)
		if gridid_test is None:
			self.num_chans=80
			self.num_phot_bins=204
			self.make_empty_resp()
			return;

		# Else, we will need to interpolate a response matrix from the response functions found at the center of the grids 
		def min_dif(x,y, x0, y0):
			""" Find the distance between two cartesian points """
			return np.sqrt((x-x0)**2 + (y-y0)**2)

		# Load information about detector plane
		imx_imy_info = np.genfromtxt(path_here.joinpath("util_packages/files-det-ang-dependence/gridnum_imx_imy.txt"),
														dtype=[("GRIDID","U3"),
														("imx",float),("imxmin",float),("imxmax",float),
														("imy",float),("imymin",float),("imymax",float),
														("thetacenter",float),("pcode",float)])

		# If this imx, imy combination is exactly at a grid center, we don't need to interpolate
		if np.any(imx == imx_imy_info['imx']) and np.any(imy == imx_imy_info['imy']):
			grid_id = imx_imy_info[ (imx_imy_info['imx']==imx) & (imx_imy_info['imy']==imy) ]['GRIDID'][0]
			# self._load_swift_bat_rsp_from_file(file_name = path_here.joinpath("util_packages/files-swiftBAT-resp-mats/BAT_alldet_grid_{}.rsp".format(grid_id)) )
			self._load_swift_bat_rsp_from_file(file_name = path_here.joinpath("util_packages/files-swift-bat-resps/bat_grid_{}.rsp".format(grid_id)) )

			# Else, we need to interpolate
		else:
			# Find the grids that surround the point at imx, imy 
			# Grid IMX's
			imx_arr = np.unique(imx_imy_info['imx'])
			# Grid IMY's
			imy_arr = np.unique(imx_imy_info['imy'])

			closest_imx_ind = None
			closest_imy_ind = None
			min_dist = 1e10
			for i in range(len(imx_arr)):
				for j in range(len(imy_arr)):
					curr_dist = min_dif(imx, imy, imx_arr[i], imy_arr[j])
					if curr_dist < min_dist:
						min_dist = curr_dist
						closest_imx_ind, closest_imy_ind = i, j

			# Sign for imx
			try:
				xsign = int((imx - imx_arr[closest_imx_ind])/np.abs(imx - imx_arr[closest_imx_ind]))
			except:
				xsign = 1
			try:
				ysign = int((imy - imy_arr[closest_imy_ind])/np.abs(imy - imy_arr[closest_imy_ind]))
			except:
				ysign = 1

			grid_ids_inds = np.array([
				(closest_imx_ind, closest_imy_ind),
				(closest_imx_ind, closest_imy_ind+ysign),
				(closest_imx_ind+xsign, closest_imy_ind),
				(closest_imx_ind+xsign, closest_imy_ind+ysign),
				])

			# Record the GridIDs
			# If the imx, imy point is beyond the final grid in either the x- or y- direction, 
			# then an empty response matrix will be used in that direction
			grid_ids = []
			for i in range(len(grid_ids_inds)):
				if (grid_ids_inds[i,0] < 0) or (grid_ids_inds[i,0] >= len(imx_arr)):
					grid_ids.append(None)
				elif (grid_ids_inds[i,1] < 0) or (grid_ids_inds[i,1] >= len(imy_arr)):
					grid_ids.append(None)
				else:
					grid_ids.append(find_grid_id(imx_arr[grid_ids_inds[i,0]], imy_arr[grid_ids_inds[i,1]]))

			# Load the four grids response functions. 
			grid_rsps = np.empty(shape=4, dtype=RSP)

			use_ind = None # Keeps track of which response to use as a template for the final interpolated matrix
			for i in range(len(grid_rsps)):
				if grid_ids[i] == None:
					grid_rsps[i] = RSP(num_phot_bins=204, num_chans=80)
					# The actual min and max values of the response matrix axes doesn't matter because they aren't considered later when interpolating
				else:
					use_ind = i
					grid_rsps[i] = RSP()
					# grid_rsps[i]._load_swift_bat_rsp_from_file(file_name = path_here.joinpath("util_packages/files-swiftBAT-resp-mats/BAT_alldet_grid_{}.rsp".format(grid_ids[i])) )
					grid_rsps[i]._load_swift_bat_rsp_from_file(file_name = path_here.joinpath("util_packages/files-swift-bat-resps/bat_grid_{}.rsp".format(grid_ids[i])) )

			# Initialize response matrix for imx, imy
			self.copy_structure(grid_rsps[use_ind])
			
			# Find surrounding imx, imy box
			imx1 = imx_arr[closest_imx_ind]
			try:
				imx2 = imx_arr[closest_imx_ind+xsign]
			except:
				imx2 = imx1+(0.5*xsign)
			imy1= imy_arr[closest_imy_ind]
			try:
				imy2 = imy_arr[closest_imy_ind+ysign]
			except:
				imy2 = imy1+(0.35*ysign)

			# Interpolate four grid response functions to create response at imx, imy
			norm = (1/(imx2-imx1)/(imy2-imy1))
			term1 = grid_rsps[0].MATRIX * (imx2 - imx)*(imy2 - imy)
			term2 = grid_rsps[1].MATRIX * (imx2 - imx)*(imy -  imy1)
			term3 = grid_rsps[2].MATRIX * (imx - imx1)*(imy2 - imy)
			term4 = grid_rsps[3].MATRIX * (imx - imx1)*(imy - imy1)
			self.MATRIX = ( norm * (term1 + term2 + term3 + term4) )
		
	
	def fold_spec(self, specfunc, add_fluc=False):
		"""
		Method to fold a spectrum through this response matrix

		Attributes:
		-----------
		specfunc : SPECFUNC
			Spectral function to fold with the loaded response matrix

		Returns:
		--------------
		folded_spec : np.ndarray with [("ENERGY",float),("RATE",float),("UNC",float)]
			Array holding a folded spectrum
		"""

		folded_spec = make_folded_spec(specfunc, self, add_fluc=add_fluc)

		return folded_spec


# class RSPARRAY(RSP):
# 	"""
# 	Class to store response matrices for more than one time interval
# 	"""
# 	def __init__(self):

def make_en_axis(Emin, Emax, num_en_bins):
	""" 
	Make energy axis 

	Attributes:
	-----------
	Emin, Emax : float, float
		Minimum and maximum energies to make the energy array across
	num_en_bins : int 
		Number of bins in the array 

	Returns:
	--------------
	en_axis : np.ndarray with [("Elo",float),("Emid",float),("Ehi",float)])
		Array storing the energy axis from Emin to Emax
	"""

	en_axis = np.zeros(shape=num_en_bins, dtype=[("Elo",float), ("Emid",float), ("Ehi",float)])
	en_axis['Elo'] = np.logspace(np.log10(Emin), np.log10(Emax), num_en_bins, endpoint=False)
	en_axis['Ehi'] = np.logspace(np.log10(en_axis["Elo"][1]), np.log10(Emax), num_en_bins, endpoint=True)
	en_axis['Emid'] = (en_axis['Ehi'] + en_axis['Elo'])/2
	return en_axis

def make_folded_spec(source_spec_func, rsp, add_fluc=False):
	""" 
	Convolve spectral function with instrument response to obtain observed spectrum

	Attributes:
	----------
	source_spec_func : SPECFUNC
		Unfolded source spectral function
	rsp : RSP
		Response matrix to convolve with
	add_fluc : boolean 
		Indicates whether or not to include random fluctuations to the spectrum.
		The fluctuations are taken from distributions of BAT statistical and systematic errors. 

	Returns:
	--------------
	folded_spec : np.ndarray with [("ENERGY",float),("RATE",float),("UNC",float)]
		Array holding a folded spectrum
	"""

	# Initialize folded spectrum 
	folded_spec = np.zeros(shape=rsp.num_chans,dtype=[("ENERGY",float), ("RATE",float)])
	# The folded spectrum will have the same energy bins as the response matrix
	folded_spec['ENERGY'] = rsp.ECHAN_MID

	# Initialize the binned source spectrum
	binned_source_spec = np.zeros(shape=rsp.num_phot_bins)  # photons / s / cm^2
	for i in range(rsp.num_phot_bins):
		binned_source_spec[i] = integrate.quad(source_spec_func, rsp.ENERG_LO[i], rsp.ENERG_HI[i])[0]

	# Fold the correctly binned source spectrum with the response matrix
	folded_spec['RATE'] = np.matmul(rsp.MATRIX, binned_source_spec)  # counts / s / bin / det area
	folded_spec['RATE'] /= (rsp.ECHAN_HI - rsp.ECHAN_LO) # counts / s / keV / det area (can be compared to XSPEC)

	if add_fluc is True:
		folded_spec = add_spec_fluctuations(folded_spec)

	return folded_spec
