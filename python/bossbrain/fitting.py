import os
import numpy as np
from scipy.optimize import curve_fit
from dlnpyutils import utils as dln
import doppler
from doppler import rv
from .ann import BOSSANNModel

# Initialize the objects
bsynfluxed = BOSSANNModel(fluxed=True)
bsynnorm = BOSSANNModel(fluxed=False)

def fitting(spec,verbose=True):
    """ Fit a BOSS spectrum """

    sp = spec.copy()
    
    # Fit with Doppler first
    print('Step 1: Running Doppler')

    #flag = False
    #count = 0
    #while (flag==False):
    #    # Run doppler
    #    if count==0:
    #        out,cmodel,specm = doppler.fit(sp,verbose=False)
    #    else:
    #        # after the first time, we should be able to give
    #        # it an initial estimate
    #        # rv.fit_lsq_cannon()
    #        estimates = {'TEFF':pars['teff'],'LOGG':pars['logg'],'FE_H':pars['mh'],'RV':pars['vrel']}
    #        out,cmodl,specm = rv.fit_cannon(sp,estimates=estimates)
    #    
    #    # Get better continuum
    #    pars = {'teff':out['teff'][0],'logg':out['logg'][0],
    #            'mh':out['feh'][0],'vrel':out['vrel'][0]}
    #    sp = spec.copy()
    #    cont = continuum(sp,pars)
    #    sp._cont = cont
    #    sp.flux /= cont
    #    sp.err /= cont
    #    sp.normalized = True

    #    count += 1
    #    if count>1:
    #        dpars = np.array(list(pars.values()))-np.array(list(last_pars.values()))
    #        drelpars = dpars/np.array(list(pars.values()))
    #        maxdrelpars = np.max(np.abs(drelpars))*100
    #        print(count,pars,maxdrelpars)
    #        if maxdrelpars<2 or count>10: flag=True
    #    else:
    #        print(count,pars)
    #    last_pars = pars.copy()
    #dopout = out
    #vrel = dopout['vrel'][0]


    out,cmodel,specm = doppler.fit(sp,verbose=False)
    dopout = out
    vrel = dopout['vrel'][0]
        
    # Use APOGEE logg relationship??


    # Do NOT the parameters and the sigma scaling factor
    # just use the gaia xp parameters and fit the sigma scaling factor in wavelength bins
    # and use doppler vrel
    
    # Fit LSF sigma scaling values in wavelength windows
    print('Step 2: Fitting LSF scaling')
    estimates = {'teff':dopout['teff'][0],'logg':dopout['logg'][0],
                 'mh':dopout['feh'][0]}
    # fit in 1000A chunks
    wchunks = [5000.0,5500.0,6000.0,6500.0,7000.0,7500.0,8000.0,8500.0,9000.0]
    wstep = 500.0
    scalings = np.array([0.70,0.85,1.0,1.15,1.30])
    chisq = np.zeros((len(wchunks),len(scalings)),float)
    bestchisq = np.zeros(len(wchunks),float)
    bestscalings = np.zeros(len(wchunks),float)
    #fitparams = ['teff','logg','mh','om','cam','sim','tim','rv']
    fitparams = ['teff','logg','mh','alpham','rv']
    for i in range(len(wchunks)):
        w0 = wchunks[i]-wstep
        w1 = wchunks[i]+wstep
        # Scaling factor loop
        for j in range(len(scalings)):
            scl = scalings[j]
            spec1 = spec.copy()
            spec1.trim(w0,w1)
            spec1.lsf._sigma *= scl
            out1 = bsynnorm.fit(spec1,vrel,estimates=estimates,
                                skipdoppler=True,initgrid=False)
            chisq[i,j] = out1['chisq']
            print('{:d} {:d} {:.2f} {:.2f} {:.3f} {:.3f}'.format(i+1,j+1,w0,w1,scl,chisq[i,j]))
        scalings2 = np.linspace(np.min(scalings),np.max(scalings),100)
        chisq2 = dln.interp(scalings,chisq[i,:],scalings2)
        bestind = np.argmin(chisq2)
        bestchisq[i] = chisq2[bestind]
        bestscalings[i] = scalings2[bestind]
        print('{:d} {:.2f} {:.2f} {:.3f} {:.3f}'.format(i+1,w0,w1,bestchisq[i],bestscalings[i]))


    #ww = np.array(wchunks)+500
    allscalings = dln.interp(np.array(wchunks),bestscalings,spec.wave,kind='quadratic',extrapolate=True)
    spec1 = spec.copy()
    spec1.lsf._sigma *= allscalings
    
    import pdb; pdb.set_trace()

    # Now fit all of the stellar parameters
    fitparams = ['teff','logg','mh','alpham','rv']
    out2 = bsynnorm.fit(spec1,vrel,fitparams=fitparams,
                        estimates=estimates,skipdoppler=True,initgrid=False)


    # Now fit all of the abundances as wel
    #fitparams = ['teff','logg','mh','alpham','rv']
    out3 = bsynnorm.fit(spec1,vrel,skipdoppler=True,initgrid=False)
    
    import pdb; pdb.set_trace()
    
    
    # Refit
    out2,model2,specm2 = doppler.fit(sp,verbose=True)
    # Get better continuum
    pars2 = {'teff':out2['teff'][0],'logg':out2['logg'][0],
             'mh':out2['feh'][0],'vrel':out2['vrel'][0]}
    print(pars2)
    sp = spec.copy()
    cont = continuum(sp,pars2)
    sp._cont = cont
    sp.flux /= cont
    sp.err /= cont
    sp.normalized = True
    
    # Refit again
    out3,model3,specm3 = doppler.fit(sp,verbose=True)
    # Get better continuum
    pars3 = {'teff':out3['teff'][0],'logg':out3['logg'][0],
             'mh':out3['feh'][0],'vrel':out3['vrel'][0]}
    print(pars3)    
    sp = spec.copy()
    cont = continuum(sp,pars3)
    sp._cont = cont
    sp.flux /= cont
    sp.err /= cont
    sp.normalized = True
    
    # Refit 3
    out4,model4,specm4 = doppler.fit(sp,verbose=True)
    # Get better continuum
    pars4 = {'teff':out4['teff'][0],'logg':out4['logg'][0],
             'mh':out4['feh'][0],'vrel':out4['vrel'][0]}
    print(pars4)  

    # Now fit abundances that change the entire spectrum
    # alpha, C and N

    fitparams = ['teff','logg','mh']
    estimates = [pars4[c] for c in fitparams]
    vrel = pars4['vrel']

    import pdb; pdb.set_trace()
    
    out = bsynnorm.fit(sp,fitparams=fitparams,initgrid=False,
                       normalize=False,estimates=estimates,vrel=vrel,
                       verbose=verbose)
    
    #bsynnorm._wobs = bsp.wave
    #estimates = [teff,logg,mh,alpha]
    #pars,pcov = curve_fit(bsynnorm.model,bsp.wave,bsp.flux,p0=estimates,
    #                      sigma=bsp.err,bounds=bounds) #,jac=self.jac)
    #perror = np.sqrt(np.diag(pcov))
    #bestmodel = self.model(spec.wave,*pars)
    #chisq = np.sum((spec.flux-bestmodel)**2/spec.err**2)/spec.size

    import pdb; pdb.set_trace()
    

    # Get better continuum
    
    
    # Now fit individual elements


    return out
    
def continuum(spec,params=None):
    """ Fit the continuum """

    # Fit smooth function to the ratio of the model and the data

    # No parameters input
    # Use Doppler to get initial estimates
    if params is None:
        out,model,specm = doppler.fit(spec)
        params = {'teff':out['teff'][0],'logg':out['logg'][0],
                  'mh':out['feh'][0],'vrel':out['vrel'][0]}

    labels = params.copy()
    # Vrel
    if 'vrel' in labels.keys():
        vrel = labels['vrel']
        del labels['vrel']
    else:
        vrel = 0.0
    
    # Create fluxed and normalized model spectrum
    spf = bsynfluxed(labels,vrel=vrel)
    sp = bsynnorm(labels,vrel=vrel)
    spf2 = spf.interp(spec.wave)
    sp2 = sp.interp(spec.wave)
    #scont = spf2.flux/sp2.flux  # model continuum

    #ratio = spec.flux/(spf2.flux/np.nanmedian(spf2.flux))
    #medratio = dln.medfilt(ratio,101)

    # Mask bad pixels!!

    ratio = spec.flux/(sp2.flux)
    medratio = dln.medfilt(ratio,201)
    smedratio = dln.gsmooth(medratio,50)
    cont = smedratio

    # flux/cont will be the "properly" normalized spectrum
    # i.e., the fluxed spectrum divided by the true continuum
    
    return cont
