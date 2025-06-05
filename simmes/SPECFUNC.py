"""
Author: Mike Moss
Contact: mikejmoss3@gmail.com	

Defines the all classes related to spectral functions used in this library to represent GRB spectra.

"""

import numpy as np
from scipy.integrate import romberg
import copy 


class SPECFUNC():
	"""
	Base class to hold variables associated with the loaded spectral model and its parameters

	Attributes
	----------
	"parameter name" : float 
		Defines the parameter value for the specific parameter argument. 
		The parameter names available depends on the spectral function currently being used (e.g., PL or CPL). 
	"""
	def __init__(self, **kwargs):
		
		self.color = "k"  # Define a color, used for plotting

		# If specific keyword parameters are defined
		for i, (key, val) in enumerate(kwargs.items()):
			self.params[key] = val

		self.param_names = list(self.params.keys())
		self.param_vals = list(self.params.values())

	def __call__(self, energy):
		"""
		Method to evaluate the spectral function at a given energy.

		Attributes
		----------
		energy : float 
			Energy to evaluate the spetrum at
		"""
		return self.evaluate(energy)

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

	def print_params(self):
		"""
		Print current parameters names and values
		"""

		for param in self.param_names:
			print("Name: {}\n\tdescription: {}\n\tvalue = {}\n".format(param,getattr(type(self), param)._description, getattr(type(self), param).value))

		return 0;

	def get_param_names(self):
		"""
		Method to get the current parameter names for this spectral function
		"""

		return list(self.params.keys())

	def get_param_vals(self):
		"""
		Method to get the current parameter values for this spectral function
		"""

		return list(self.params.values())

	def set_params(self, **kwargs):
		"""
		Method to set the current parameters for the spectral function being used.
		
		Attributes
		----------
		"parameter name" : float 
			Defines the parameter value for the specific parameter argument. 
			The parameter names available depends on the spectral function currently being used (e.g., PL or CPL). 
		"""

		# If specific keyword parameters are defined
		for i, (key, val) in enumerate(kwargs.items()):
			self.params[key] = val

		self.param_vals = list(self.params.values())


	def make_spectrum(self, emin, emax, num_bins = None):
			"""
			Method to evaluate the spectrum over the defined energy interval using the GRB object's spectral model and parameters

			Attributes:
			----------
			emin, emax : float, float
				Defines the lower and upper bounds of the energy interval over which to evaluate the spectrum
			num_bins : int
				Number of energy bins to use, default is 10*log(emax/emin)
			"""
			if num_bins is None:
				num_bins = int(np.log10(emax/emin)*20)

			# Initialize array
			spectrum = np.zeros(shape=num_bins, dtype=[("ENERGY", float), ("RATE", float)])
			# Evaluate energy array
			spectrum['ENERGY'] = np.logspace(np.log10(emin),np.log10(emax), num=num_bins)

			# Evaluate spectrum rate
			spectrum['RATE'] = self.evaluate(spectrum['ENERGY'])

			return spectrum

	def _calc_energy_flux(self, emin, emax):
		"""
		Method to find the total energy flux of the spectral model within the given energy range

		Attributes:
		----------
		emin, emax : float, float
			Defines the lower and upper bounds of the energy interval over which to evaluate the energy flux. Unit of keV)
		"""
		energy_flux_kev = romberg(function=lambda x: x*self.evaluate(x), a=emin, b=emax)  ## [keV/s/cm2]

		kev2erg = 1000*1.60217657e-12

		energy_flux = energy_flux_kev*kev2erg  ## [erg/s/cm2]

		return energy_flux

	def _find_norm(self, flux, emin, emax):
		"""
		Method to find the spectrum normalization based on observed flux

		Attributes:
		----------
		flux : float
			Observed flux to normalize to
		emin, emax : float, float
			Defines the lower and upper bounds of the energy interval over which to evaluate the energy flux. Unit of keV)
		"""

		return flux/self._energy_flux(emin, emax)

	def _calc_phot_flux(self, emin, emax):
		"""
		Method to find the total photon flux of the spectral model within the given energy range

		Attributes:
		----------
		emin, emax : float, float
			Defines the lower and upper bounds of the energy interval over which to evaluate the energy flux. Unit of keV)
		"""
		return romberg(function=self.evaluate, a=emin, b=emax)  ## [count/s/cm2]


class PL(SPECFUNC):
	"""
	Power Law

	Parameters
	----------
	alpha : float
		Power law index
	norm : float
		Model normalization
	enorm : float
		Normalization energy
	"""

	def __init__(self, **kwargs):
		self.name = "Power Law"
		
		# Default values
		def_alpha = -1.
		def_norm = 1.
		def_enorm = 1.
		
		self.params = {"alpha" : def_alpha , "norm" : def_norm , "enorm" : def_enorm}
		
		super().__init__(**kwargs)

	def evaluate(self, energy):
		"""
		Compute the power law spectrum at a particular energy given the current spectral parameters

		Attributes
		----------
		energy : float 
			Energy to evaluate the spetrum at
		"""

		flux_value = self.params['norm'] * np.power(energy/self.params['enorm'], self.params['alpha'])
		
		return flux_value

class CPL(SPECFUNC):
	"""
	Cut-off Power Law 

	Parameters
	----------
	ep : float
		Peak energy
	alpha : float
		Power law index
	norm : float
		Model normalization
	enorm : float
		Normalization energy
	"""

	def __init__(self, **kwargs):
		self.name = "Cut-off Power Law"

		# Default values
		def_ep = 100.
		def_alpha = -1.
		def_norm = 1.
		def_enorm = 1.

		self.params = {"ep" : def_ep, "alpha" : def_alpha, "norm" : def_norm, "enorm" : def_enorm}

		super().__init__(**kwargs)

	def evaluate(self, energy):
		"""
		Compute the cut-off power law spectrum at a particular energy given the current spectral parameters

		Attributes
		----------
		energy : float 
			Energy to evaluate the spetrum at
		"""
		flux_value = self.params['norm'] * np.power(energy/self.params['enorm'], self.params['alpha']) * np.exp(- energy / self.params['ep'])

		return flux_value

class CPLSwift(SPECFUNC):
	"""
	Cut-off Power Law 

	Parameters
	----------
	ep : float
		Peak energy
	alpha : float
		Power law index
	norm : float
		Model normalization
	enorm : float
		Normalization energy
	"""

	def __init__(self, **kwargs):
		self.name = "Swift/BAT Cut-off Power Law"

		# Default values
		def_ep = 100.
		def_alpha = -1.
		def_norm = 1.

		self.params = {"ep" : def_ep, "alpha" : def_alpha, "norm" : def_norm}

		super().__init__(**kwargs)

	def evaluate(self, energy):
		"""
		Compute the cut-off power law spectrum at a particular energy given the current spectral parameters

		Attributes
		----------
		energy : float 
			Energy to evaluate the spetrum at
		"""
		flux_value = self.params['norm'] * np.power(energy/50.0, self.params['alpha']) * np.exp(- energy * (2.0 + self.params['alpha']) / self.params['ep'])

		return flux_value

class Blackbody(SPECFUNC):
	"""
	Blackbody function.

	Parameters
	----------
	temp : float
		Blackbody temperature (in units of energy, i.e., k_B*T where k_B is the Boltzmann constant)
	alpha : float
		Index of the power law below temperature
	norm : float
		Model normalization
	"""
	def __init__(self, **kwargs):
		self.name = "Blackbody"

		# Default values
		def_temp = 20.
		def_alpha = 0.4
		def_norm = 1

		self.params = {"temp" : def_temp,"alpha" : def_alpha,"norm" : def_norm}

		super().__init__(**kwargs)

	def evaluate(self, energy):
		"""
		Compute the blackbody spectrum at a particular energy

		Attributes
		----------
		energy : float 
			Energy to evaluate the spetrum at
		"""

		# Initialize the return value
		flux_value = np.zeros_like(energy, subok=False)

		if hasattr(energy, '__len__'):
			i = energy < 2e3
			if i.max():
				# If the energy is less than 2 MeV
				flux_value[i] = self.params['norm'] * np.power(energy[i]/self.params['temp'],1.+self.params['alpha'])/(np.exp(energy[i]/self.params['temp']) - 1.)
			i = energy >= 2e3
			if i.max():
				flux_value[i] = 0
		else:
			if energy < 2e3:
				# If the energy is less than 2 MeV
				flux_value = self.params['norm'] * np.power(energy/self.params['temp'],1.+self.params['alpha'])/(np.exp(energy/self.params['temp']) - 1.)
			else:
				# energy >= 2e3
				flux_value = 0

		return flux_value

class Band(SPECFUNC):
	"""
	Band function (see Band et al. 1993)

	Parameters
	----------
	ep : float
		Peak energy
	alpha : float
		Low energy power law index
	beta : float
		High energy power law index
	norm : float
		Model normalization
	"""
	def __init__(self, **kwargs):
		self.name = "Band"

		# Default values
		def_ep = 400.
		def_alpha = -1.
		def_beta = -2.5
		def_norm = 1.
		def_enorm = 100. # keV

		self.params = {"ep" : def_ep, "alpha" : def_alpha, "beta" : def_beta, "norm" : def_norm, "enorm" : def_enorm}

		super().__init__(**kwargs)

	def evaluate(self, energy):
		"""
		Compute the Band spectrum at a particular energy given the current spectral parameters

		Attributes
		----------
		energy : float 
			Energy to evaluate the spetrum at
		"""
		# Initialize the return value
		flux_value = np.zeros_like(energy, subok=False)

		# Calculate break energy
		e0 = self.params['ep'] / (self.params['alpha'] - self.params['beta'])

		if hasattr(energy, '__len__'):
			i = energy <= self.params['ep']
			if i.max():
				flux_value[i] = self.params['norm'] * np.power(energy[i]/self.params['enorm'], self.params['alpha']) * np.exp(- energy[i] / e0)
			
			i = energy > self.params['ep']
			if i.max():
				flux_value[i] = self.params['norm'] * np.power((self.params['alpha'] - self.params['beta']) * e0/self.params['enorm'], self.params['alpha'] - self.params['beta']) * np.exp(self.params['beta'] - self.params['alpha']) * np.power(energy[i]/self.params['enorm'],self.params['beta'])
		else:
			if energy <= self.params['ep']:
				flux_value = self.params['norm'] * np.power(energy/self.params['enorm'], self.params['alpha']) * np.exp(- energy / e0)
			else: 
				# energy > self.params['ep']
				flux_value = self.params['norm'] * np.power((self.params['alpha'] - self.params['beta']) * e0/self.params['enorm'], self.params['alpha'] - self.params['beta']) * np.exp(self.params['beta'] - self.params['alpha']) * np.power(energy/self.params['enorm'],self.params['beta'])


		return flux_value
