import numpy as np

import jax.numpy as jnp

from jax.scipy.special import gammaincc as jax_gammainc
from jax.scipy.special import gammaln as jax_gammaln

import astropy.units as u
from astropy.cosmology import Planck15

#cosmology helper functions to convert between redshift and lookback time
#zmax is the maximum redshift of star formation 
zmax = 20
zs_i = jnp.linspace(0, zmax, 1000)
tLs_i = Planck15.lookback_time(zs_i).to(u.Gyr).value
tL_at_z_interp = lambda z: jnp.interp(z, zs_i, tLs_i)
z_at_tL_interp = lambda t: jnp.interp(t, tLs_i, zs_i)

#simulated grid parameters
zeta_grid = jnp.array([0.0002, 0.002, 0.02]) #metallicities (0.01, 0.1, 1 Zsun)
rv_grid = jnp.array([0.5, 1, 2, 4]) #virial radii in pc
ncl_grid = jnp.array([2e5, 4e5, 8e5, 1.6e6]) #number of particles. Stellar mass is 0.6 Msun * ncl.

#helper function because jax doesn't have a gamma function defined
def jax_gamma(x):
    return jnp.exp(jax_gammaln(x))

def schechter_lower_int(beta, logMstar, logMlo):
    '''
    inputs: power law slope beta, log10 Schechter mass Mstar, log10 minimum integration bound Mlo
    returns the integral M^beta exp(-M/Mstar) dM from Mlo to infinity
    '''
    #change of variables x = M/Mstar 
    #M = x*Mstar, dx = dM/Mstar, dM = dx * Mstar, xlo = Mlo/ Mstar
    # Mstar^(beta + 1) integral [x^beta exp(-x) dx] from xlo to infinity
    lnMstar = logMstar * jnp.log(10)
    lnMlo = logMlo * jnp.log(10)
    xlow = jnp.exp(lnMlo - lnMstar)
    ln_out = (beta + 1) * lnMstar + jnp.log(jax_gammainc(beta + 1, xlow)) + jax_gammaln(beta + 1) #this last term is because we don't want the normalized version
    return jnp.exp(ln_out)
    #return jax_gammainc(beta + 1, xlow) #unnormalized

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
    metals: metallicity
    redshift: formation redshift
    sigma_dex: scatter in log10Z
    Returns fraction of star formation in a given metallicity bin at a given redshift
    Assumes the metallicity distribution at each redshift is lognormal, truncated between maximum and minimum simulated metallicity
    assumes metallicity bins are log-spaced
    '''

    log10mean = mean_log10metallicity(redshift) #an array if redshift is an array
   
    x = jnp.log10(metals/Zsun) 
    x_grid = jnp.log10(zeta_grid/Zsun)
    
    w = jnp.exp(-(x - log10mean)**2/(2*sigma_dex**2))
    
    w_grid = jnp.array([jnp.exp(-(xg - log10mean)**2/(2*sigma_dex**2)) for xg in x_grid]) #needs to be normalized at every redshift
    
    norm = jnp.sum(w_grid, axis = 0)

    return w/norm

def mass_weights_powerlaw(cluster_mass, beta = -2, missing_cluster_factor = 4.0):
    '''
    assume cluster mass distribution is a power law with slope beta
    note that Kremer+ 2020 assumes it is lognormal with mean log10M = 5.54 (approximately center of simulated range) and width sigma(log10M) = 0.52
    missing_cluster_factor: contribution from the clusters too big to model directly. Kremer+ 2020 find that this gives a factor of 4 regardless of radius distribution, but they assume a mass distribution much more skewed to heavy systems.
    '''
    w = cluster_mass**(beta + 1) #must take into account that cluster mass is log-spaced, this is dM/dlogM 
    w_grid = (0.6*ncl_grid)**(beta + 1)
    norm = jnp.sum(w_grid)
    
    return w/norm * missing_cluster_factor

def mass_weights_schechter(cluster_mass, beta = -2, logMstar0 = 6.26):
    '''
    Following section II.B of Antonini & Gieles 2020, this is the *initial* cluster mass function
    not to be confused with present (evolved) MW cluster mass function 
    cluster_mass: initial cluster mass
    beta: power law slope
    logMstar0: in log10(Msun), initial Schechter mass, 2Mc from Antonini & Gieles 2020
    '''
    x = cluster_mass/ 10**logMstar0
    w = x**(beta + 1) * jnp.exp(-x)
    
    x_grid = 0.6 * ncl_grid/ 10**logMstar0
    w_grid = x_grid**(beta + 1) * jnp.exp(-x_grid)
    norm = jnp.sum(w_grid)  
    
    return w/norm

def compute_missing_cluster_factor(beta = -2, logMstar0 = 6.26, logMlo = 2, logMhi = 8, res = 100):
    '''
    beta: power law slope
    logMstar0: in log10(Msun), initial Schechter mass, 2Mc from Antonini & Gieles 2020
    logMlo: in log10(Msun), minimum initial cluster mass
    logMhi: in log10(Msun), maximum initial cluster mass
    returns: factor by which to multiply BBH merger rate to account for cluster masses not simulated. This corresponds to the average number of mergers over the simulated mass range, divided by the average number of mergers over the full mass range from logMlo to logMhi. Note it is usually smaller than 1!
    '''
    
    #asumption is that number of mergers as a function of cluster mass [for a fixed radius] scales as M^1.6 (from Antonini & Gieles 2020)
    
    x_grid_full = jnp.logspace(logMlo-logMstar0, logMhi-logMstar0, res) #log spaced bins between 100 and 10^8 Msun
    w_grid_full = x_grid_full**(beta + 1) * jnp.exp(-x_grid_full) #cluster weight according to mass distribution (not normalized)
    norm_full = jnp.sum(w_grid_full)
    
    average_merge_full = jnp.sum(x_grid_full**1.6 * w_grid_full/norm_full) #weighted sum of (m/Mstar)**1.6, corresponding to average number of mergers per cluster over the full mass range 
    
    x_grid = 0.6 * ncl_grid/ 10**logMstar0
    w_grid = x_grid**(beta + 1) * jnp.exp(-x_grid)
    norm = jnp.sum(w_grid)
    average_merge_sim = jnp.sum(x_grid**1.6 * w_grid/norm) #weighted sum of (m/Mstar)**1.6 in the simulated mass range, corresponding to average number of mergers per simulated cluster
    
    missing_cluster_factor = average_merge_full/average_merge_sim
    
    return missing_cluster_factor

def compute_disrupted_cluster_factor(beta = -2, logMstar0 = 6.26, logMlo = 2, logMhi = 8, logDelta = 5.33):
    '''
    beta: power law slope of Schecther function describing GC birth mass distribution
    logMstar0: log10 Schechter mass of GC birth mass distribution
    logMlo: log10 minimum cluster mass
    logDelta: log10 of mass lost by clusters between birth and now (excluding stellar mass loss). 
    returns: factor by which to multiply BBH merger density to account for cluster disruption/ evaporation mass loss. 
    '''
    
    #Following Section II of Antonini & Gieles 2020
    #Assumes all clusters lost the same mass Delta (excluding stellar mass loss).
    #Delta is typically inferred by comparing evolved GC mass distribution to birth GC mass distribution. 
    #Integral must be evaluated numerically. 
     
    logMc = logMstar0 - jnp.log10(2) #log(Mstar0/2)
    
    logm_grid = jnp.logspace(logMlo, logMhi, 20) #log spaced bins between 100 and 10^8 Msun, preliminary tests suggest 20 is enough
    
    phi_cl0 = 2**(-1-beta) * logm_grid**beta * jnp.exp(-logm_grid/ 10**logMstar0) #birth mass function 

    phi_cl = (logm_grid + (10**logDelta))**beta * jnp.exp(-(logm_grid+10**logDelta)/ 10**logMc)

    NBH_initial = jnp.trapz(phi_cl0 * logm_grid**1.6, logm_grid)
    
    NBH_final = jnp.trapz(phi_cl * logm_grid**1.6, logm_grid)
    
    K_merge = NBH_initial / NBH_final / 2**1.6 #Divide by M = 2 because factor of 2 just from stellar mass loss so doesn't contribute to BBH rate. check that this is still 2 for arbitrary beta, or does it become 2**(-1-beta). 
    
    return K_merge

def average_mass_schechter(beta = -2, logMstar0 = 6.26, logMlo = 2, logMhi = 8, res = 20):
    
    m_grid = jnp.logspace(logMlo, logMhi, res)
    
    x = m_grid / 10**logMstar0
    
    pdf_mass = x**(beta + 1) * jnp.exp(-x)
    
    pdf_mass /= jnp.trapz(pdf_mass, m_grid)
    
    average_mass = jnp.trapz(pdf_mass * m_grid, m_grid)
    
    return average_mass

def cluster_number_density_from_mass_density(rho_GC = 7.3e14, beta = -2, logMstar0 = 6.26, logMlo = 2, logMhi = 8, logDelta = 5.33):
    '''
    rho_GC: mass density of GCs *today*, units Msun/ Gpc^3 (e.g. Antonini & Gieles 2020 Sec IIA)
    beta: power law slope of Schecther function describing GC birth mass distribution
    logMstar0: log10 Schechter mass of GC birth mass distribution
    logMlo: log10 minimum cluster mass
    logDelta: log10 of mass lost by clusters between birth and now (excluding stellar mass loss). 
    returns: cluster number density given a mass density, assuming mass distribution follows evolved Schechter function. Units 1/ Gpc^3 (or the units of rho_GC/ Msun)
    '''
    logMc = logMstar0 - jnp.log10(2) #log(Mstar0/2)
    
    logm_grid = jnp.logspace(logMlo, logMhi, 20)
    #logm_grid = ncl_grid * 0.6
    
    phi_cl = (logm_grid + (10**logDelta))**beta * jnp.exp(-(logm_grid+10**logDelta)/ 10**logMc)
    
    #number density = rho/<M> where <M> is average cluster mass \int M p(M) dM 
    average_mass = jnp.trapz(phi_cl * logm_grid, logm_grid)/ jnp.trapz(phi_cl, logm_grid)
    
    #average_mass = jnp.sum(phi_cl * logm_grid)/jnp.sum(phi_cl)
    
    return rho_GC/ average_mass 
                             
def radius_weights(cluster_radius, mu_rv = 1, sigma_rv = 1.5):
    '''
    assume cluster size distribution is Gaussian
    cluster_radius: virial radius of given cluster (pc)
    mu_rv: mean radius (pc)
    sigma_rv: standard deviation (pc)
    returns: fractional contribution from the given cluster radius (normalized so that the sum over the radius grid is unity)
    '''
    w = jnp.exp(-(cluster_radius - mu_rv) ** 2. / (2. * sigma_rv ** 2.)) * cluster_radius #must take into account that cluster radius is log-spaced
    w_grid = jnp.exp(-(rv_grid - mu_rv) ** 2. / (2. * sigma_rv ** 2.)) * rv_grid
    
    return w/jnp.sum(w_grid)

def redshift_peak(z, a, b, zp):
    '''
    Madau-like redshift distribution
    a: low redshift is approximately (1 + z)^a
    b: high redshift is approximately (1 + z)^-b
    zp: approximate peak redshift
    '''
    return (1.0+(1.0+zp)**(-a-b))*(1+z)**a/(1.0+((1.0+z)/(1.0+zp))**(a+b))

def sfr_at_z_norm(z, z_gc = 4.5, a = 2.5, b = 2.5):
    '''
    cluster star formation history, normalized to give volumetric number density of 1 Gpc^-3 yr^-1 today
    Assume it is Madau-like with params z_gc, a, b
    z_gc: peak redshift
    a: low redshift power-law slope in (1 + z)
    b: high redshift power-law slope slope in (1 + z)
    '''
    dNdVdt_unnorm = redshift_peak(z, a, b, z_gc) #dN/dVcdt(z) 
    dNdV0_unnorm = jnp.trapz(redshift_peak(zs_i, a, b, z_gc), tLs_i*1e9) #integrate over lookback time, recall that tLs_i is in Gyr
    dNdVdt = dNdVdt_unnorm/dNdV0_unnorm
    
    return dNdVdt

def sfr_at_z(z, dNdV0 = 2.31e9, z_gc = 4.5, a = 2.5, b = 2.5, disrupted_factor = 1.0): 
    '''
    cluster star formation history (e.g. Fig 5 in Rodriguez & Loeb 2018)
    Assume it is Madau-like with params z_gc, a, b
    z_gc: peak redshift
    a: low redshift power-law slope in (1 + z)
    b: high redshift power-law slope slope in (1 + z)
    dNdV0: number density in comoving Gpc^-3 at z = 0, found by integrating the sfr dN/dVdt over all t. Kremer+ 2020 assumes volumetric number density of 2.31e9 Gpc^-3. In terms of mass density, would be typical cluster mass * 2.31e9 Gpc^-3 yr^-1 or ~5e5 Msun Mpc^-3 yr^-1. If mass density is better known than number density, replace this with dM/dV and then divide by typical cluster mass according to assumed mass distribution.
    disrupted_factor: accounts for contribution from clusters that were disrupted/ evaporated before the present day, which has the same effect as adjusting the cluster number density 
    returns: number density (comoving Gpc^-3 yr^-1) evaluated at z
    '''
    dNdVdt = sfr_at_z_norm(z, z_gc, a, b)
    
    dNdVdt_norm = dNdVdt * dNdV0 * disrupted_factor
    
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

def merger_rate_at_z_pop(data, zmerge, z_gc = 4.5, a = 2.5, b = 2.5, dNdV0 = 2.31e9, logf_disrupted_cluster = 0.5, sigma_dex = 0.5, Zsun = 0.02, mu_rv = 1, sigma_rv = 1.5, beta = -2, logMstar0 = 6.26, logMlo = 2, logMhi = 8):
    '''
    data: output of read_data() -- list of numsim, rvv, zb, ncll, tgw
    zmerge: merger redshift
    z_gc: peak formation redshift
    a: formation rate follows (1 + z)^a at low z
    b: formation rate follows (1 + z)^-b at high z
    dNdV0: number density of GCs today in units Gpc^-3
    logf_disrupted_cluster: log10 of the contribution to formation rate at each z from cluster mass lost between formation and today
    sigma_dex: scatter in metallicity-redshift relation
    Zsun: solar metallicity
    mu_rv: mean cluster radius (pc)
    sigma_rv: standard deviation of cluster radius distripution (pc)
    beta: power law slope of birth cluster mass distribution
    logMstar0: log10 Schechter mass of birth cluster mass distribution
    logMlo: log10 minimum GC mass (Msun)
    logMhi: log10 maximum GC mass (Msun)
    '''

    numsim, rvv, zb, ncll, tgw = data[0], data[1], data[2], data[3], data[4]
            
    #compute mass and radius weights for each simulation based on ncl, rv. 
    mweights = mass_weights_schechter(ncll*0.6, beta, logMstar0)
    rweights = radius_weights(rvv, mu_rv, sigma_rv)
    
    f_missing_cluster = compute_missing_cluster_factor(beta, logMstar0, logMlo, logMhi)
    
    cluster_weight = mweights * rweights * dNdV0 * f_missing_cluster * 10**logf_disrupted_cluster
    
    merger_rate_array = merger_rate_at_z(zmerge, sfr_at_z_norm, tgw, cluster_weight, zb, metallicity_weights, sfr_kwargs = {'z_gc': z_gc, 'a': a, 'b': b}, metal_kwargs = {'sigma_dex': sigma_dex, 'Zsun': Zsun})
    
    out = jnp.sum(merger_rate_array)
    
    return out

def merger_rate_at_z_pop_selfconsistentfactors(data, zmerge, z_gc = 4.5, a = 2.5, b = 2.5, sigma_dex = 0.5, Zsun = 0.02, mu_rv = 1, sigma_rv = 1.5, beta = -2, logMstar0 = 6.26, rho_GC = 7.3e14, logDelta = 5.33, logMlo = 2, logMhi = 8, average_M_evolved = None):
    '''
    data: output of read_data() -- list of numsim, rvv, zb, ncll, tgw
    zmerge: merger redshift
    z_gc: peak formation redshift
    a: formation rate follows (1 + z)^a at low z
    b: formation rate follows (1 + z)^-b at high z
    sigma_dex: scatter in metallicity-redshift relation
    Zsun: solar metallicity
    mu_rv: mean cluster radius (pc)
    sigma_rv: standard deviation of cluster radius distripution (pc)
    beta: power law slope of birth cluster mass distribution
    logMstar0: log10 Schechter mass of birth cluster mass distribution
    rho_GC: mass density of GCs today (Msun/ Gpc^3)
    logDelta: log10 mass (Msun) lost by GCs between formation and today (excluding stellar mass loss)
    logMlo: log10 minimum GC mass (Msun)
    logMhi: log10 maximum GC mass (Msun)
    average_M_evolved: average GC mass of evolved clusters (Msun) used to compute number density dNdV0 from rho_GC. If None, then average evolved mass is computed from other parameters assuming model of Antonini & Gieles 2020. Typical value is 3e5. 
    '''
    
    if average_M_evolved:
        dNdV0 = rho_GC/ average_M_evolved
    else:
        dNdV0 = cluster_number_density_from_mass_density(rho_GC, beta, logMstar0, logMlo, logMhi, logDelta)
    
    f_missing_cluster = compute_missing_cluster_factor(beta, logMstar0, logMlo, logMhi)
    
    f_disrupted_cluster = compute_disrupted_cluster_factor(beta, logMstar0, logMlo, logMhi, logDelta)

    numsim, rvv, zb, ncll, tgw = data[0], data[1], data[2], data[3], data[4]
            
    #compute mass and radius weights for each simulation based on ncl, rv. 
    mweights = mass_weights_schechter(ncll*0.6, beta, logMstar0)
    rweights = radius_weights(rvv, mu_rv, sigma_rv)
    
    cluster_weight = mweights * rweights * dNdV0 * f_missing_cluster * f_disrupted_cluster
    
    merger_rate_array = merger_rate_at_z(zmerge, sfr_at_z_norm, tgw, cluster_weight, zb, metallicity_weights, sfr_kwargs = {'z_gc': z_gc, 'a': a, 'b': b}, metal_kwargs = {'sigma_dex': sigma_dex, 'Zsun': Zsun})
    
    out = jnp.sum(merger_rate_array)
    
    return out


    