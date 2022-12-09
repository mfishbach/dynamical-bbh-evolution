import numpy as np

import jax.numpy as jnp

import astropy.units as u
from astropy.cosmology import Planck15

#cosmology helper functions to convert between redshift and lookback time
zs_i = jnp.linspace(0, 20, 1000)
tLs_i = Planck15.lookback_time(zs_i).to(u.Gyr).value
tL_at_z_interp = lambda z: jnp.interp(z, zs_i, tLs_i)
z_at_tL_interp = lambda t: jnp.interp(t, tLs_i, zs_i)

#simulated grid parameters
zeta_grid = [0.0002, 0.002, 0.02] #metallicities (0.01, 0.1, 1 Zsun)
rv_grid = [0.5, 1, 2, 4] #virial radii in pc
ncl_grid = [2e5, 4e5, 8e5, 1.6e6] #number of particles. Stellar mass is 0.6 Msun * ncl.

def mean_log10metallicity(z):
    '''Returns the mean log10Z as a function of z
    Assumption is that star-forming gas in GCs has the same metallicity as the rest of the galaxy
    From Madau & Fragos 2017'''
     #this predicts fairly high metallicities
     #need to go beyond z = 7 to get mean metallicity below 0.1 solar, even though GCs in MW have metallicities below 0.1 solar... 
    #probably doesn't matter too much though because metallicity doesn't seem to affect cluster rates as much as other things 
    return 0.153 - 0.074 * z ** 1.34

def metallicity_weights(metals, redshift, sigma_dex = 0.5, Zsun = 0.02):
    '''
    metals_Zsun: metallicity in units of solar metallicity
    redshift: formation redshift
    sigma_dex: scatter in log10Z
    Returns fraction of star formation in a given metallicity bin at a given redshift
    Assumes the metallicity distribution at each redshift is lognormal, truncated between maximum and minimum simulated metallicity
    assumes metallicity bins are log-spaced
    '''

    log10mean = mean_log10metallicity(redshift)
   
    x = jnp.log10(metals/Zsun) 
    x_grid = jnp.log10(zeta_grid/Zsun)
    
    w = jnp.exp(-(x-log10mean)**2/(2*sigma_dex**2))
    w_grid = jnp.exp(-(x_grid-log10mean)**2/(2*sigma_dex**2)) #for normalization
    norm = jnp.sum(w_grid)

    return w/norm

def mass_weights_powerlaw(cluster_mass, beta = -2, missing_cluster_factor = 4.0):
    '''
    assume cluster mass distribution is a power law with slope beta
    note that Kremer+ 2020 assumes it is lognormal with mean log10M = 5.54 (approximately center of simulated range) and width sigma(log10M) = 0.52
    missing_cluster_factor: contribution from the clusters too big to model directly. Kremer+ 2020 find that this gives a factor of 4 regardless of radius distribution (but probably this also affects t_gw)? 
    '''
    w = cluster_mass**(beta + 1) #must take into account that cluster mass is log-spaced, this is dM/dlogM 
    w_grid = ncl_grid**(beta + 1)
    norm = jnp.sum(w_grid)
    
    return w/norm * missing_cluster_factor

def radius_weights(cluster_radius, mu_rv = 1, sigma_rv = 1.5):
    '''
    assume cluster size distribution is Gaussian
    cluster_radius: virial radius of given cluster (pc)
    mu_rv: mean radius (pc)
    sigma_rv: standard deviation (pc)
    returns: fractional contribution from the given cluster radius
    '''
    w = jnp.exp(-(cluster_radius - mu_rv) ** 2. / (2. * sigma_rv ** 2.)) * cluster_radius #must take into account that cluster radius is log-spaced
    w_grid = jnp.exp(-(rv_grid - mu_rv) ** 2. / (2. * rv_grid ** 2.)) * rv_grid
    
    return w/jnp.sum(w_grid)


def sfr_at_z(z, dNdV0 = 2.31e9, z_gc = 3.2, sigma_gc = 0.5, disrupted_factor = 1.0): 
    '''
    assume cluster star formation history is Gaussian in redshift (e.g. Fig 5 in Rodriguez & Loeb 2018)
    reasonably centered at z = 3.2 with sigma(z) = 1.5. 
    dNdV0: number density in comoving Gpc^-3 at z = 0, found by integrating the sfr dN/dVdt over all t. Kremer+ 2020 assumes volumetric number density of 2.31e9 Gpc^-3. In terms of mass density, would be typical cluster mass * 2.31e9 Gpc^-3 yr^-1 or ~5e5 Msun Mpc^-3 yr^-1. If mass density is better known than number density, replace this with dM/dV and then divide by typical cluster mass according to assumed mass distribution.
    disrupted_factor: accounts for clusters that were disrupted before the present day, which would have the same effect as increasing the cluster number density. Typical factor here is 2.6, but default Kremer+ 2020 models do not include it.
    returns: number density (comoving Gpc^-3 yr^-1) evaluated at z
    '''
    
    dNdVdt_unnorm = jnp.exp(-(z - z_gc)**2/ 2*sigma_gc**2) #dN/dVcdt(z) 
    dNdV0_unnorm = jnp.trapz(jnp.exp(-(zs_i - z_gc)**2/ 2*sigma_gc**2), tLs_i) #integrate over lookback time 
    dNdVdt_norm = dNdVdt_unnorm * dNdV0/dNdV0_unnorm
    
    return dNdVdt_norm

def read_data():
    
    #load in data
    data = np.loadtxt('data.txt')
    
    #cluster params and merger time for each BBH merger(?) -- total is 4330 mergers
    numsim = data[:,0] #simulation number
    rvv = data[:,1] #virial radii
    zb = data[:,3] #metallicity
    ncll = data[:,4] #number of particles. multiply by 0.6 Msun to get stellar mass. 
    tgw = data[:,6] * 1e-3 #merger times in Gyr

    # limit to ncll<2.e6 to have a uniform grid
    (Idx,) = np.where(ncll < 2.e6) 
    numsim = numsim[Idx] 
    rvv = rvv[Idx]
    zb = zb[Idx]
    ncll = ncll[Idx]
    tgw = tgw[Idx]

    #141 different GC simulations -- supposed to be 144 (4 ncl, 4rv, 3 zeta, 3 different galactocentric radii), 
    #but missing the 3 corresponding to (ncll/2e5==8) & (rvv == 0.5) & (zb == 0.0002). 
    #It's probably fine because such low metallicities are very rare at relevant redshifts.
    #To make the grid consistent, add in these missing simulations assuming they are identical to the zb == 0.002 versions. 

    #select sims we want to copy
    copy_sel = (ncll/2e5==8) & (rvv == 0.5) & (zb == 0.002)
    
    ncopy = len(numsim[copy_sel])
    
    #get numsim of missing sims
    #missing_sims = list(set(1+np.arange(143)).difference(set(numsim)))
    
    #make the copies
    rvv_copy = rvv[copy_sel]
    ncll_copy = ncll[copy_sel]
    tgw_copy = tgw[copy_sel]
    
    #pretend they correspond to the missing zb
    zb_copy = 0.0002 * np.ones(ncopy)
    
    #label the fake sims with -1* original numsim so we hopefully remember we did something sketchy 
    numsim_copy = -numsim[copy_sel]

    rvv_new = np.concatenate((rvv, rvv_copy))
    ncll_new = np.concatenate((ncll, ncll_copy))
    tgw_new = np.concatenate((tgw, tgw_copy))

    zb_new = np.concatenate((zb, zb_copy))
    
    numsim_new = np.concatenate((numsim, numsim_copy))
    
    return numsim_new, rvv_new, zb_new, ncll_new, tgw_new

#each BBH came from a cluster that represents a rate density at some z/time. 

def merger_rate_at_z(zmerge, formation_rate_at_z, tgw, cluster_weight, metal, metal_frac_at_z, sfr_kwargs = {}, metal_kwargs = {}):

    ''' 
    zmerge: desired merger redshift 
    formation_rate_at_z: a function that returns the formation rate (dN/dVcdt) at a given redshift
    tgw: array of delay times between formation and merger (Gyr), each delay time coresponds to one BBH
    cluster_weight: weight assigned to specific cluster, same dimensions as tgw
    metal: metallicity assigned to specific cluster, same dimensions as tgw
    metal_frac_at_z: a function that returns the metallicity fraction at a given formation redshift and metallicity
    sfr_kwargs: other params called by formation_rate_at_z
    metal_kwargs: other params called by metal_pdf_at_z
    returns: merger rate at given zmerge
    '''
   
    tL_merge = tL_at_z_interp(zmerge) #lookback time at merger in Gyr
    tL_form = tL_merge + tgw #lookback time at formation
    z_form = z_at_tL_interp(tL_form) #redshift at formation

    metal_weight = metal_frac_at_z(metal, z_form, **metal_kwargs)
    
    return jnp.sum(cluster_weight * metal_weight * formation_rate_at_z(z_form, **sfr_kwargs)) #sum over all mergers

def merger_rate_at_z_pop(data, zmerge, formation_rate_at_z = sfr_at_z, metal_frac_at_z = metallicity_weights, sfr_kwargs = {'dNdV0': 2.31e9, 'z_gc': 3.2, 'sigma_gc' : 0.5, 'disrupted_factor' : 1.0}, metal_kwargs = {'sigma_dex' : 0.5, 'Zsun' : 0.02}, mass_dist_kwargs = {'beta':-2, 'missing_cluster_factor':4.0}, r_dist_kwargs = {'mu_rv': 1, 'sigma_rv': 1.5}): 
    
    '''
    data: output of read_data() -- list of numsim, rvv, zb, ncll, tgw
    '''
    
    numsim, rvv, zb, ncll, tgw = data[0], data[1], data[2], data[3], data[4]
    
    #compute mass and radius weights for each simulation based on ncl, rv. 
    mweights = mass_weights_powerlaw(ncll, **mass_dist_kwargs)
    rweights = radius_weights(rvv, **r_dist_kwargs)
    
    cluster_weight = mweights * rweights
    
    out = jnp.sum(merger_rate_at_z(zmerge, formation_rate_at_z, tgw, cluster_weight, zb, metal_frac_at_z, sfr_kwargs, metal_kwargs))
    
    return out 



    