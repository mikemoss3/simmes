# GRB Measurement Simulations (simmes)
Author: Mike Moss
Contact: mikejmoss3@gmail.com

This code is pip-installable.

## Purpose

This project allows a user to measure the duration and gamma-ray fluence of simulated gamma-ray burst (GRB) prompt observations while taking into consideration observation conditions, such as the angle of the simulated GRB with respect to the detector bore-sight. This code is based on the work of [Moss et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...927..157M/abstract). This library currently focuses on analysis with the Swift/BAT instrument. Support for non-coded-aperture mask instruments is in development. 

## Procedure and How-to-Use

Here are short descriptions of each file and directory in the project:
```
simmes/					
├── GRB.py				# Defines a GRB object to store observed and simulated light curve and spectral information
├── PLOTS.py				# Defines the class and methods used for plotting simulation results
├── RSP.py				# Defines the main class this code uses to store response matrices and the associated methods
├── SPECFUNC.py				# Defines the all classes related to spectral functions used in this library to represent GRB spectra
├── bayesian_block.py 			# Defines the Bayesian block method to calculate the duration of a GRB from a supplied light curve
├── fluence.py				# Defines scripts used to calculate fluence of a light curve
├── simulations.py			# Defines top-level functions necessary to simulate a GRB prompt emission observation
├── threshold_search.py			# Defines scripts related to searching for redshifts that have a given detection threshold (in development!)
├── unit_tests/ 			# Holds all unit tests (which still haven't been written, sorry!)
├── util_packages/			# Holds the support packages and libraries for the main code 
├────── files-det-ang-dependence/	# Holds files relating incident angle, PCODE, and detector plane positions
├────── files-spec-unc/		# Holds files relating to the statistical and systematic errror of Swift/BAT measuremed spectra
├────── files-swift-bat-resps/	# Holds response files for Swift/BAT detector, broken into 31 grid positions (see Lien et al 2014)
├────── cosmology.py 			# Defines functions used for cosmology calculations 
├────── fluctuations.py 			# Defines functions used for to add fluctuations/variations to light curves and spectra
├────── datatypes.py 			# Defines data structures
├────── det_ang_dependence.py 		# Defines functions relating incident angle, PCODE, and detector plane positions
└────── globalconstants.py 		# Defines global constants 
```

### Basic Outline:
To run these simulations, a template GRB must be created using an observed/theoretical light curve and specrtum. In addition, the observation conditions, e.g., the redshifts and response matrices to use, must be defined. Then perform the simulations! Finally, there are various functions available to analyze and visualize the data (e.g., measuring durations with Bayesian blocks or plotting redshift evolution trends).

### Loading a Template GRB
First, create a GRB object that will act as a template for our simulations.
```
from simmes.GRB import GRB

template_grb = GRB()
```

Next load a light curve for `template_grb`. Currently, the `GRB` class can load light curves from either .txt files or .fits files. In this example, the Swift/BAT single-channel, 1-second time bin, mask-weighted light curve for GRB 081007 is used[^1].
```
template_grb.load_light_curve("grb_081007_1chan_1s.lc", rm_trigtime=True, norm=True)
```
where `rm_trigtime=True` indicates that we subtract the detector trigger time from the time axis of the light curve so that the burst trigger occurs at T=0 s. Additionally, `norm=True` indicates that this mask-weighted light curve should be normalized. The normalization factor depends on what interval the given spectrum (see below) is valid over. For example, if a 1-second peak spectrum is provided, then the light curve should be normalized by the count rate in the peak flux bin. In this way, the light curve becomes unitless and only provides the relative fluxes between each time bin, not raw count rates.

Now a spectral function must be defined for the GRB. In this example, a cut-off power law spectral function with a spectral index $\alpha = -1$, peak energy $\E_{peak} = 100$ keV and $norm = 4$ (note: the normalization energy is set to $e_{norm} = 1$ keV by default[^2])
```
from packages.SPECFUNC import CPL

spectral_function = CPL( alpha= -1., ep=100., norm=4. )
template_grb.load_specfunc( spectral_function )
```
This can be shortened to a single line, like so,
```
template_grb.load_specfunc( CPL( alpha= -1., ep=100., norm=4.) )
```

Currently, the power law (PL), cut-off power law (CPL), and Band (Band) spectral functions are implemented. If you plan to simulate a burst out to higher redshifts, it is recommended to assume a CPL with an observed/typical peak energy so that the peak energy realistically passes through an instrument's energy band as the spectrum is redshifted to lower energies with increasing distance.

Since this GRB was observed at a redshift of `z=0.5295`, we can indicate a redshift for this mock GRB (note: the indicated redshift does not need to be associated any observation, it can be any value, however using the observed redshift may allow results to be directly comparable to the respective observation) 
```
template_grb.z = 0.5295
```
Alternatively, a rest-frame light curve could have been used for `template_grb` and no redshift would have been needed.

[^1]: Light curves and spectral parameters for all Swift/BAT GRBs can be found on the online [Swift/BAT Catalog](https://swift.gsfc.nasa.gov/results/batgrbcat/)
[^2]: However, the spectral parameters found on the Swift/BAT catalog assume that the normalization energy is 50 keV (see the page 11 of the Third Swift/BAT GRB Catalog, [Lien et al. 2014](https://swift.gsfc.nasa.gov/results/batgrbcat/3rdBATcatalog.pdf))

### Simulating A GRB
Before a mock GRB can be simulated, the desired observing condition parameter values should be assigned and a response matrix should be created. The following example will focus on the Swift/BAT instrument, but can rather easily be applied to any coded-aperture mask instrument.
```
from simmes.RSP import RSP

z_p = 1 # redshift to the synthetic GRB
imx = 0 # x-axis location of the source on the detector plane
imy = 0 # y-axis location of the source on the detector plane
ndets = 32768 # number of detectors enable on the detector plane at the time of observation

resp = RSP() # Make response matrix object instance
```
There are several methods included for loading response matrices. In the example below, a Swift/BAT response matrix will be loaded. The specific response matrix loaded depends on where GRB is located on the instrument detector plane (i.e., specified with the coordinates ($imx$, $imy$)). 

Instead of properly remaking BAT response matrix using standard BAT tools for each simulation, 30 response matrices have been generated that provide a rough discretization of Swift/BAT's continuous angle-dependent response function. The response matrices roughly separate the BAT detector plane into 30 regions (see method from [Lien et al 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...783...24L/abstract). An interpolated response matrix is generated from these grid-based matrices for any detector plane position.
```
resp.load_SwiftBAT_resp(imx, imy)
```

To simulate a mock GRB observation and measure it's duration, import and run the command 
```
from simmes.simulations import simulate_observation

synth_grb = simulate_observation( template_grb, z_p, imx, imy, ndets, resp)
```
And there you have it, `synth_grb` is a `GRB` object that has a mask-weighted light curve and spectrum generated from a known template input light curve and spectrum that have been folded through the Swift/BAT response. 

To obtain a duration measurement of the light curve, the `bayesian_t_blocks` method can be used. To find the T<sub>90</sub> duration, the duration percentage keyword should be set to 90, i.e., `dur_per=90` (note, this is the default value).
```
from simmes.bayesian_block import bayesian_t_blocks

duration, t_start, fluence = bayesian_t_blocks(synth_grb, dur_per=90)
```

### Simulating Many GRBs
Repeating the above steps for many observing condition combinations can be tedious, so the `simulations` package was developed to perform many simulations based on a given list of unique parameter combinations. Create a list of unique parameters by defining the all the values of $z$, $imx$, $imy$, and $ndets$ desired,
```
from simmes.simulations import make_param_list, many_simulations

z_arr = np.array([1, 2, 3])
imx_arr = np.array([0, 0.5])
imy_arr = np.array([0, 0.5])
ndets_arr = np.array([30000, 20000, 10000])
param_list = make_param_list(z_arr, imx_arr, imy_arr, ndets_arr)
```
`param_list` now contains a list of every unique combination of parameters possible from the four parameter lists `z_arr`, `imx_arr`, `imy_arr`, and `ndets_arr`.

Now, call the `many_simulations()` method. This requires specifying a template GRB that holds a user-defined light curve and spectral function, the parameter combination list that was just created, and a number of trials to simulate each parameter combination for. 
```
trials = 10
sim_results = many_simulations(template_grb, param_list, trials)
```

This tutorial is not exhaustive. The code has plenty of other functionality that is not overviewed here (i.e., using time-resolved spectra), but this the basic procedure to follow. Please contact me about any questions or concerns.