import os
import copy
import time
import numpy as np
import traceback
from astropy.table import Table
from scipy.optimize import curve_fit
from theborg.emulator import Emulator
from dlnpyutils import utils as dln,lsqr
import emcee
import corner
import doppler
from doppler.spec1d import Spec1D
import matplotlib
import matplotlib.pyplot as plt
from . import utils

cspeed = 2.99792458e5  # speed of light in km/s

class BOSSANNModel():

    # Model BOSS spectra using ANN model
    
    def __init__(self,spobs=None,loggrelation=False,fluxed=False,verbose=False):
        # Load the ANN models
        em1 = Emulator.read(utils.datadir()+'ann_22pars_3500-4200.pkl')
        em2 = Emulator.read(utils.datadir()+'ann_22pars_4000-5000.pkl')
        em3 = Emulator.read(utils.datadir()+'ann_22pars_4900-6000.pkl')
        self._models = [em1,em2,em3]
        self.nmodels = len(self._models)
        self.labels = self._models[0].label_names
        self.nlabels = len(self.labels)
        self._ranges = np.zeros((self.nmodels,self.nlabels,2),float)
        for i in range(self.nmodels):
            for j in range(self.nlabels):
                self._ranges[i,j,:] = [np.min(self._models[i].training_labels[:,j]),
                                       np.max(self._models[i].training_labels[:,j])]
        self._ranges[0,0,1] = 4100.0  # use 3500-4200 model up to 4100
        self._ranges[1,0,0] = 4100.0  # use 4000-5000 model from 4100 to 4950
        self._ranges[1,0,1] = 4950.0        
        self._ranges[2,0,0] = 4950.0  # use 4900-6000 model from 4950
        self.ranges = np.zeros((self.nlabels,2),float)
        self.ranges[0,0] = np.min(self._ranges[:,0,:])
        self.ranges[0,1] = np.max(self._ranges[:,0,:])        
        for i in np.arange(1,self.nlabels):            
            self.ranges[i,:] = [np.max(self._ranges[:,i,0]),np.min(self._ranges[:,i,1])]
        
        # Alpha element index
        alphaindex = []
        for i,l in enumerate(self.labels):
            if l in ['om','cam','mgm','tim','sm','sim']:
                alphaindex.append(i)
        self._alphaindex = np.array(alphaindex)

        # Input observed spectrum information
        self._spobs = spobs
        
        # ANN model wavelengths
        npix_model = 14001
        self._dispersion = np.arange(npix_model)*0.5+3500.0

        # Get logg label
        loggind, = np.where(np.char.array(self.labels).lower()=='logg')
        if len(loggind)==0:                
            raise ValueError('No logg label')
        self.loggind = loggind[0]
        # Get temperature label
        teffind, = np.where(np.char.array(self.labels).lower()=='teff')
        if len(teffind)==0:
            teffind, = np.where(np.char.array(self.labels).lower().find('temp')>-1)
        self.teffind = teffind[0]
        # Get metallicity label
        mhind, = np.where(np.char.array(self.labels).lower()=='mh')                
        self.mhind = mhind[0]
            
        # Use logg relation
        if loggrelation:
            # Get alpha abundance label
            alphaind, = np.where(np.char.array(self.labels).lower()=='[alpha/fe]')
            if len(alphaind)==0:
                alphaind, = np.where(np.char.array(self.labels).lower()=='alphafe')
            if len(alphaind)==0:
                alphaind, = np.where(np.char.array(self.labels).lower()=='alpha_fe')
            if len(alphaind)==0:
                alphaind, = np.where(np.char.array(self.labels).lower().find('alpha')>-1)
            if len(alphaind)==0:
                self.alphaind = None
            else:
                self.alphaind = alphaind[0]
            # Load the ANN model
            logg_model = Emulator.load(utils.datadir()+'apogeedr17_rgb_logg_ann.npz')
            logg_model.ranges = np.array([np.min(logg_model.training_labels,axis=0),
                                          np.max(logg_model.training_labels,axis=0)]).T
            self.logg_model = logg_model
        
        self.loggrelation = loggrelation
        self.fluxed = fluxed
        self.verbose = verbose
        self.fitparams = None
        self.allparams = None
        self.nmodel = 0
        self.njac = 0


    @property
    def nfitparams(self):
        if self.fitparams is None:
            return 0
        else:
            return len(self.fitparams)

    @property
    def nallparams(self):
        if self.allparams is None:
            return 0
        else:
            return len(self.allparams)
        
    def mklabels(self,pars):
        """ Make the ANN labels array from a dictionary."""
        # Dictionary input
        if type(pars) is dict:
            labels = np.zeros(self.nlabels)
            # Must at least have Teff and logg
            for k in pars.keys():
                if k=='alpham':   # mean alpha
                    ind = self._alphaindex.copy()
                elif k=='rv' or k=='vrel':
                    pass
                else:
                    ind, = np.where(np.array(self.labels)==k.lower())
                if len(ind)==0:
                    raise ValueError(k+' not found in labels: '+','.join(self.labels))
                labels[ind] = pars[k]
            if labels[0]<=0 or labels[1]<=0:
                raise ValueError('pars must at least have teff and logg')
        # List or array input
        else:
            # Make full parameter array
            #  input array should have same elements as allparams
            #  or one less if 'rv' was removed
            if self.allparams is not None and ((len(pars) == len(self.allparams)) |
                                               (len(pars) == len(self.allparams)-1)):
                fitlabels = [p.lower() for p in self.allparams if p.lower() != 'rv']
                fitlabels = np.array(fitlabels)
                if len(pars) != len(fitlabels):
                    import pdb; pdb.set_trace()
                    raise ValueError('pars size not consistent with allparams')
                labels = np.zeros(self.nlabels)
                for i in range(len(fitlabels)):
                    if fitlabels[i]=='alpham':   # mean alpha
                        ind = self._alphaindex.copy()
                    else:
                        ind, = np.where(np.array(self.labels)==fitlabels[i])
                    labels[ind] = pars[i]
            else:
                labels = pars

        return labels

    def meanlabels(self):
        """ Return mean labels."""
        return np.mean(self.ranges,axis=1)

    def get_best_model(self,labels):
        """ This returns the first ANN model that has the right range."""
        # These labels should be in order
        for m in range(self.nmodels):
            ranges = self._ranges[m,:,:]
            inside = True
            for i in range(3):
                inside &= (labels[i]>=ranges[i,0]) & (labels[i]<=ranges[i,1])
            if inside:
                return m
        return None
    
    def mkbounds(self,params,fixparams={}):
        """ Make bounds for input parameter names."""
        bounds = [np.zeros(len(params)),np.zeros(len(params))]
        for i in range(len(params)):
            if params[i] in fixparams.keys():
                bounds[0][i] = fixparams[params[i]]-1e-7
                bounds[1][i] = fixparams[params[i]]+1e-7
                continue
            if params[i].lower()=='alpham':   # mean alpha
                ind = self._alphaindex.copy()
                bounds[0][i] = np.max(self.ranges[ind,0])+0.01
                bounds[1][i] = np.min(self.ranges[ind,1])-0.01
            elif params[i].lower()=='rv':
                bounds[0][i] = -1000
                bounds[1][i] = 1000
            else:
                ind, = np.where(np.array(self.labels)==params[i].lower())
                bounds[0][i] = self.ranges[ind,0]+0.01
                bounds[1][i] = self.ranges[ind,1]-0.01
        return bounds
    
    def inrange(self,pars):
        """ Check that the parameters are in range."""
        # pull out rv
        if self.allparams is not None:
            allparams = self.allparams
        else:
            allparams = self.labels
        if len(pars)==len(allparams) and 'rv' in allparams:
            ind, = np.where(allparams!='rv')
            labels = self.mklabels(np.array(pars)[ind])
        else:
            labels = self.mklabels(pars)
        # These "labels" are now in the right order
        # Get the right model to use based on input Teff/logg/feh
        modelindex = self.get_best_model(labels)
        if modelindex is None:
            return False,0,[self.ranges[0,0],self.ranges[0,1]]
        # Check other ranges
        for i in np.arange(1,self.nlabels):
            rr = [self._ranges[modelindex,i,0],self._ranges[modelindex,i,1]]
            if labels[i]<=rr[0] or labels[i]>=rr[1]:
                return False,i,rr
        return True,None,None

    def getlogg(self,pars):
        """ Get logg from the logg relation and fill it into the parameter array where it belongs."""
        # The only parameters should be input, with the logg one missing/excluded
        # Insert a dummy value for logg
        newpars = np.insert(pars,self.loggind,0.0)
        teff = newpars[self.teffind]
        teff = np.clip(teff,self.logg_model.ranges[0,0],self.logg_model.ranges[0,1])
        feh = newpars[self.mhind]
        feh = np.clip(feh,self.logg_model.ranges[1,0],self.logg_model.ranges[1,1])
        if self.alphaind is not None:
            alpha = newpars[self.alphaind]
        else:
            alpha = 0.0
        alpha = np.clip(alpha,self.logg_model.ranges[2,0],self.logg_model.ranges[2,1])
        logg = self.logg_model([teff,feh,alpha],border='extrapolate')
        newpars[self.loggind] = logg
        return newpars

    def printpars(self,pars,perror):
        """ Print out parameters and errors."""
        
        for i in range(len(pars)):
            if self.allparams is not None and len(pars) != self.nlabels:
                name = self.allparams[i]
            else:
                name = self.labels[i]
            if name=='teff':
                print('{:6s}: {:10.1f} +/- {:5.2g}'.format(name,pars[i],perror[i]))
            else:
                print('{:6s}: {:10.4f} +/- {:5.3g}'.format(name,pars[i],perror[i]))

    def randompars(self,params=None,fixparams={},n=100):
        """ Create random parameters for initial guesses."""
        if params is None:
            params = self.fitparams
        # Do NOT include RV, that is handled separately
        labelparams = [p.lower() for p in params if p.lower() != 'rv']
        nlabelparams = len(labelparams)
        rndpars = np.zeros((n,nlabelparams),float)
        for i in range(nlabelparams):
            if labelparams[i] in fixparams.keys():
                rndpars[:,i] = fixparams[labelparams[i]]
                continue
            if labelparams[i].lower() == 'alpham':
                ind = self._alphaindex.copy()
                vmin = np.max(self.ranges[ind,0])
                vmax = np.min(self.ranges[ind,1])
            else:
                ind, = np.where(np.array(self.labels)==params[i])
                vmin = self.ranges[ind,0]
                vmax = self.ranges[ind,1]
            vrange = vmax-vmin
            # make a small buffer
            vmin += vrange*0.01
            vrange *= 0.98
            rndpars[:,i] = np.random.rand(n)*vrange+vmin

        return rndpars

    def fiducialspec(self):
        """ Default BOSS resolution and wavelength spectrum."""
        # Default observed spectrum
        wobs_coef = np.array([0.00010, 3.55230])  # log wavelength
        # 4648 observed pixels
        npix_obs = 4648
        x = np.arange(npix_obs)
        wobs = 10**np.polyval(wobs_coef,x)
        # resolution in three parts
        # these are piece-wise polynomial fits to WRESL which is FWHM
        # 1: 400-2165
        lpars1 = np.array([ 6.83669706e-14, -1.10388358e-10, -4.21884602e-07,
                            8.35079141e-04, 2.67629137e+00])
        # 2: 2165-2519
        lpars2 = np.array([-1.82241977e-11,  1.81121293e-07, -6.74495096e-04,
                           1.11666990e+00, -6.90795349e+02])
        # 3: 2519-4625
        lpars3 = np.array([ 9.84869119e-14, -1.31647649e-09,  6.58716558e-06,
                            -1.44335837e-02, 1.50830710e+01])
        lsfsigma = np.zeros(npix_obs,float)
        lsfsigma[400:2165] = np.polyval(lpars1,x[400:2165])
        lsfsigma[2165:2519] = np.polyval(lpars2,x[2165:2519])
        lsfsigma[2519:4625] = np.polyval(lpars3,x[2519:4625])
        lsfsigma[:400] = lsfsigma[400]
        lsfsigma[4625:] = lsfsigma[4624]
        lsfsigma /= 2.35    # convert from FWHM to SIGMA
        spobs = Spec1D(np.zeros(npix_obs),wave=wobs,err=np.ones(npix_obs),
                       lsfsigma=lsfsigma,lsftype='Gaussian',lsfxtype='wave')        
        return spobs
    
    def __call__(self,pars=None,spobs=None,snr=None,vrel=None,normalize=False,
                 fluxed=False,fiducial=False):
        """
        Returns BOSS model spectrum.

        Parameters
        ----------
        pars : list or array
           Parameters for the labels.
        spobs : Spec1D, optional
           Observed spectrum to model, using the resolution and wavelength.
        snr : float, optional
           Add random noise to the spectrum at the "snr" level.
        vrel : float, optional
           Doppler shift the spectrum by vrel (km/s).
        normalize : bool, optional
           Perform continuum normalization on the spectrum.
        fluxed : bool, optional
           Make the spectrum fluxed, i.e. add continuum.  Default is False.
        fiducial : bool, optional
           Use fiducial BOSS resolution and wavelength values.

        Returns
        -------
        spec : Spec1D
           BOSS model spectrum.

        Example
        -------

        spec = jw(pars)

        """
        # If no pars input, use mean values
        if pars is None:
            pars = self.meanlabels()

        # PARS must be
        # -dictionary
        # -array of length Nlabels or Nlabels+1 (plus RV)
        if isinstance(pars,dict):
            labels = self.mklabels(pars)
        elif isinstance(pars,np.ndarray):
            if len(pars) != self.nlabels and len(pars) != self.nlabels+1:
                raise Exception('pars numpy array must be Nlabels or Nlabels+1 (with RV)')
            labels = self.mklabels(pars)
        else:
            raise Exception('pars must be dict or numpy array')
        
        # Get label array
        #  vrel is not in "pars", input separately
        #labels = self.mklabels(pars)

        # Break out RV if needed
        
        #import pdb; pdb.set_trace()
        
        # Are we making a fluxed spectrum?
        if fluxed is None:
            fluxed = self.fluxed

        if fiducial and spobs is None:   # use fiducial values
            spobs = self.fiducialspec()
            
        # Check that the labels are in range
        flag,badindex,rr = self.inrange(labels)
        if flag==False:
            srr = '[{:.4f},{:.3f}]'.format(*rr)
            error = 'parameters out of range: '
            error += '{:s}={:.4f}'.format(self.labels[badindex],labels[badindex])
            error += ' '+srr
            if spobs is None:
                return np.zeros(self._spobs.size)+1e30
            else:
                return np.zeros(spobs.size)+1e30
            #raise ValueError(error)

        # Get the right model to use based on input Teff/logg/feh
        modelindex = self.get_best_model(labels)
            
        # Get the ANN model spectrum
        flux = self._models[modelindex](labels)
        wave = self._dispersion

        # Fluxed
        if fluxed:
            m = doppler.models.get_best_model(labels[0:3])
            c = m._data[0].continuum
            fc = c(labels[0:3])
            wc = c.dispersion
            # interpolate to the wavelength array
            cont = dln.interp(wc,fc,wave,kind='quadratic')
            cont = 10**cont
            flux *= cont
            
        # Doppler shift
        if (vrel is None and hasattr(self,'_spobs') and self._spobs is not None and
            hasattr(self._spobs,'vrel') and self._spobs.vrel is not None):
            vrel = self._spobs.vrel
        if vrel is not None and vrel != 0.0:
            redwave = wave*(1+vrel/cspeed)  # redshift the wavelengths
            orig_flux = flux.copy()
            flux = dln.interp(redwave,flux,wave,extrapolate=False,fill_value=np.nan)
            flux[~np.isfinite(flux)] = 1.0
                
        # Make the model Spec1D object
        #  synspec wavelengths are in air units
        spsyn = Spec1D(flux,err=np.zeros(len(flux)),wave=wave,wavevac=False)
        # Say it is normalized
        if fluxed==False:
            spsyn.normalized = True
            spsyn._cont = np.ones(spsyn.flux.shape)        
        # Convolve to observed resolution and wavelength
        if fiducial and spobs is None:   # use fiducial values
            spobs = self.fiducialspec()
        if spobs is not None or self._spobs is not None:
            if spobs is None:
                spmonte = spsyn.prepare(self._spobs)
                spmonte.continuum_func = self._spobs.continuum_func
            else:
                spmonte = spsyn.prepare(spobs)
                spmonte.continuum_func = spobs.continuum_func
        # No observed spectrum input
        else:
            spmonte = spsyn
        # Now normalize if requested
        if normalize:
            spmonte.normalize()
        # Add labels to spectrum object
        spmonte.labels = labels
        # Add noise
        if snr is not None:
            spmonte.flux += np.random.randn(*spmonte.err.shape)*1/snr
            spmonte.err += 1/snr
        # Deal with any NaNs
        bd, = np.where(~np.isfinite(spmonte.flux))
        if len(bd)>0:
            spmonte.flux[bd] = 1.0
            spmonte.err[bd] = 1e30
            spmonte.mask[bd] = True
            
        return spmonte
    
    def model(self,wave,*pars,**kwargs):
        """ Model function for curve_fit."""
        # PARS must be a numpy array or list and have elements equal to allparams
        if self.verbose:
            print('model: ',pars)
        # 
        out = self(pars,**kwargs)
        self.nmodel += 1
        # Only return the flux
        if isinstance(out,Spec1D):
            return out.flux
        else:
            return out
        
    def jac(self,wave,*args,retmodel=False,**kwargs):
        """
        Method to return Jacobian matrix.
        This includes the contribution of the lookup table.

        Parameters
        ----------
        args : float
            Model parameter values as separate positional input parameters.
        retmodel : boolean, optional
            Return the model as well.  Default is retmodel=False.

        Returns
        -------
        if retmodel==False
        jac : numpy array
          Jacobian matrix of partial derivatives [N,Npars].
        model : numpy array
          Array of (1-D) model values for the input xdata and parameters.
          If retmodel==True, then (model,jac) are returned.

        Example
        -------

        jac = BOSSSyn.jac(wave,*pars)

        """

        ## add RV, if we are fitting it
        #if 'rv' in self.allparams:
        #    # remove vrel at the end of args array
        #    rvind, = np.where(self.allparams=='rv')
        #    labelargs = np.array(args).copy()
        #    labelargs = np.delete(labelargs,rvind)
        #    fullargs = self.mklabels(labelargs)
        #    # now add vrel value to it at the end
        #    fullargs = np.hstack((fullargs,args[rvind[0]]))
        #
        #else:
        fullargs = self.mklabels(args)
            
        # logg relation
        #  add a dummy logg value in
        #if self.loggrelation:
        #    fullargs = np.insert(args,self.loggind,0.0)
        #else:
        #    fullargs = args

        if self.verbose:
            print('jac: ',args)
            
        # Initialize jacobian matrix
        npix = len(wave)
        fjac = np.zeros((npix,len(self.allparams)),np.float64)
        
        # Loop over parameters
        pars = np.array(copy.deepcopy(args))
        f0 = self.model(wave,*pars,**kwargs)        
        steps = np.zeros(len(self.allparams))
        for i in range(len(self.allparams)):
            ind, = np.where(np.array(self.labels)==self.allparams[i])
            if self.loggrelation and i==self.loggind:
                continue
            targs = np.array(copy.deepcopy(fullargs))
            if self.allparams[i]=='teff':
                step = 10.0                
            else:
                step = 0.02
            if self.allparams[i]=='rv':
                isrv = True
                step = 1.0
            else:
                isrv = False
            steps[i] = step
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if isrv:
                if targs[i]+step > 1000:
                    step *= -1
            else:
                if targs[i]+step>=self.ranges[ind,1]-0.01:
                    step *= -1
            targs[i] += step
            # Remove dummy logg if using logg relation
            if self.loggrelation:
                targs = np.delete(targs,self.loggind)
            #print(i,step,targs)
            f1 = self.model(wave,*targs,**kwargs)
            fjac[:,i] = (f1-f0)/steps[i]
                
        self.njac += 1
            
        return fjac

   
    def emcee_lnlike(self, theta, spec):
        """
        This helper function calculates the log likelihood for the MCMC portion of fit().
    
        Parameters
        ----------
        theta : array
           Input parameters [teff, logg, feh, rv].
        x : array
           Array of x-values for y.  Not really used.
        y : array
           Observed flux array.
        yerr : array
           Uncertainties in the observed flux.
        spec : Spec1D
           The observed spectrum.  Needed to run cannon.model_spectrum().

        Outputs
        -------
        lnlike : float
           The log likelihood value.

        """
        m = self.model(spec.wave,*theta,spobs=spec)
        inv_sigma2 = 1.0/spec.err**2
        return -0.5*(np.sum((spec.flux-m)**2*inv_sigma2))


    def emcee_lnprior(self, theta):
        """
        This helper function calculates the log prior for the MCMC portion of fit().
        It's a flat/uniform prior across the stellar parameter space covered by the
        Cannon models.
        
        Parameters
        ----------
        theta : array
        Input parameters [teff, logg, feh, rv].
        models : list of Cannon models
        List of Cannon models to use
        
        Outputs
        -------
        lnprior : float
         The log prior value.
        
        """
    
        flag,_,_ = self.inrange(theta)
        if flag:
            return 0.0
        else:
            return -np.inf

    def emcee_lnprob(self, theta, spec):
        """
        This helper function calculates the log probability for the MCMC portion of fit().
        
        Parameters
        ----------
        theta : array
           Input parameters/labels.
        x : array
           Array of x-values for y.  Not really used.
        y : array
           Observed flux array.
        yerr : array
           Uncertainties in the observed flux.
        spec : Spec1D
           The observed spectrum.  Needed to run cannon.model_spectrum().

        Outputs
        -------
        lnprob : float
           The log probability value, which is the sum of the log prior and the
           log likelihood.

        """
        lp = self.emcee_lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.emcee_lnlike(theta, spec)

    
    def mcmc(self,spec,vrel=None,fitparams=None,fixparams={},loggrelation=False,
             normalize=False,initgrid=True,estimates=None,outlier=False,steps=100,
             cornername=None,skipdoppler=False,verbose=False):
        """
        Run MCMC on the spectrum
    
        Parameters
        ----------
        spec : Spec1D object
            The observed spectrum to match.
        models : list of Cannon models, optional
            A list of Cannon models to use.  The default is to load all of the Cannon
            models in the data/ directory and use those.
        initpar : numpy array, optional
            Initial estimate for [teff, logg, feh, RV], optional.
        steps : int, optional
            Number of steps to use.  Default is 100.
        cornername : string, optional
            Output filename for the corner plot.  If a corner plot is requested, then the
            minimum number of steps used is 500.
        verbose : bool, optional
            Verbose output of the various steps.  This is False by default.

        Returns
        -------
        out : numpy structured array
            The output structured array of the final derived RVs, stellar parameters and errors.
        bmodel : Spec1D object
            The best-fitting Cannon model spectrum (as Spec1D object).

        Example
        -------

        .. code-block:: python
        
         out, bmodel = fit_mcmc(spec)
        
        """

        if hasattr(spec,'vrel') == False and vrel is not None:
            spec.vrel = vrel
        if verbose:
            if vrel is not None:
                print('Vrel: {:.2f} km/s'.format(vrel))
            print('S/N: {:.2f}'.format(spec.snr))
        # Now normalize
        if normalize:
            if spec.normalized==False:
                spec.normalize()

        if fitparams is None:
            fitparams = self.labels+['rv']
        fitparams = [p.lower() for p in fitparams]
        self.fitparams = np.array(fitparams)
        nfitparams = len(fitparams)
        nfixparams = len(fixparams)
        self.fixparams = fixparams
        # check that fixparam names are correct
        if nfixparams>0:
            for c in fixparams.keys():
                if c.lower() not in self.labels+['rv']:
                    raise Exception(str(c)+' not one of the labels or rv')
        nallparams = nfitparams+nfixparams
        if nfixparams>0:
            allparams = np.unique(fitparams+list(fixparams.keys()))
        else:
            allparams = np.unique(fitparams)
        allparams = [a.lower() for a in allparams]   # lowercase
        # put allparams in the same order as labels
        allparams = [c for c in self.labels+['rv'] if c in allparams]
        self.allparams = np.array(allparams)

        # Make bounds
        bounds = self.mkbounds(allparams,fixparams)

        # Use spec.vrel
        if vrel is None and hasattr(spec,'vrel') and getattr(spec,'vrel') is not None:
            vrel = spec.vrel
        
        # Run doppler if no velocity input
        if vrel is None or skipdoppler==False:
            dopout, dopfmodel, dopspecm = doppler.fit(spec,verbose=verbose)
            vrel = dopout['vrel'][0]
            if estimates is None:
                estimates = {'teff':dopout['teff'][0],'logg':dopout['logg'][0],
                             'mh':dopout['feh'][0]}
                tind, = np.where(self.allparams=='teff')[0]
                estimates['teff'] = dln.limit(dopout['teff'][0],bounds[0][tind]+1,bounds[1][tind]-1)
                lind, = np.where(self.allparams=='logg')[0]
                estimates['logg'] = dln.limit(dopout['logg'][0],bounds[0][lind]+0.01,bounds[1][lind]-0.01)
                mind, = np.where(self.allparams=='mh')[0]
                estimates['mh'] = dln.limit(dopout['feh'][0],bounds[0][mind]+0.01,bounds[1][mind]-0.01)
        # Using input vrel or default of 0.0
        else:
            if vrel is None:
                vrel = 0.0
        spec.vrel = vrel

        # initial parameter estimates
        initpar = np.zeros(nallparams,float)
        if estimates is not None and len(estimates)>0:
            for i in range(len(estimates)):
                ind, = np.where(self.allparams==list(estimates.keys())[i])
                initpar[ind] = estimates[list(estimates.keys())[i]]

        # Set up the MCMC sampler
        ndim, nwalkers = nallparams, 50
        delta = np.ones(nallparams)*0.1
        tind, = np.where(self.allparams=='teff')[0]
        delta[tind] = 50.0
        #pos = [initpar + delta*np.random.randn(ndim) for i in range(nwalkers)]
        pos = nwalkers*[None]
        for i in range(nwalkers):
            pos1 = initpar + delta*np.random.randn(ndim)
            for j in range(nallparams):
                pos1[j] = np.clip(pos1[j],bounds[0][j],bounds[1][j])
            pos[i] = pos1

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.emcee_lnprob,
                                        args=(spec),live_dangerously=True)

        if cornername is not None: steps=np.maximum(steps,500)  # at least 500 steps
        out = sampler.run_mcmc(pos, steps, skip_initial_state_check=True)
        
        samples = sampler.chain[:, int(steps/2):, :].reshape((-1, ndim))

        # Get the median and stddev values
        pars = np.zeros(ndim,float)
        parerr = np.zeros(ndim,float)
        if verbose is True: print('MCMC values:')
        names = self.allparams
        for i in range(ndim):
            t = np.percentile(samples[:,i],[16,50,84])
            pars[i] = t[1]
            parerr[i] = (t[2]-t[0])*0.5
        if verbose is True: self.printpars(pars,parerr)
        
        # The maximum likelihood parameters
        bestind = np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
        pars_ml = sampler.chain[bestind[0],bestind[1],:]

        # treat rv separately
        if len(pars_ml)==len(self.allparams) and 'rv' in self.allparams:
            rvind, = np.where(self.allparams=='rv')
            rv = pars_ml[rvind][0]
            ind, = np.where(self.allparams!='rv')
            labels = np.array(pars_ml)[ind]
            mcmodel = self(labels,spobs=spec,vrel=rv)
        else:
            mcmodel = self(pars_ml,spobs=spec)
        mcchisq = np.sqrt(np.sum(((spec.flux-mcmodel.flux)/spec.err)**2)/(spec.npix*spec.norder))

        # Put it into the output structure
        dtype = np.dtype([('label',(str,50),ndim),('pars',float,ndim),('pars_ml',float,ndim),('parerr',float,ndim),('chisq',float)])
        out = np.zeros(1,dtype=dtype)
        out['label'] = self.allparams
        out['pars'] = pars
        out['pars_ml'] = pars_ml    
        out['parerr'] = parerr
        out['chisq'] = mcchisq
    
        # Corner plot
        if cornername is not None:
            backend = matplotlib.rcParams['backend']        
            matplotlib.use('Agg')
            fig = corner.corner(samples, labels=self.allparams, truths=pars)
            plt.savefig(cornername)
            plt.close(fig)
            print('Corner plot saved to '+cornername)
            matplotlib.use(backend)  # back to the original backend
        
        return out,mcmodel
    
    
    
    def fit(self,spec,vrel=None,fitparams=None,fixparams={},loggrelation=False,normalize=False,
            initgrid=True,estimates={},outlier=False,skipdoppler=False,verbose=False):
        """
        Fit an observed spectrum with the ANN models and curve_fit.

        Parameters
        ----------
        spec : Spec1D 
           Spectrum to fit.
        vrel : float, optional
           Input list of doppler shifts for the spectra.
        fitparams : list, optional
           List of names of parameters to fit.  By default, all ANN labels
              and RV are fit.
        fixparams : dict, optional
           Dictionary of parameter names and the values to fix them to.
        loggrelation : bool, optional
           Use the logg-relation as a function of Teff/feh/alpha.
        normalize : bool, optional
           Use the normalized spectra.  Default is False.
        initgrid : bool, optional
           Try an initial grid of stellar parameters. Default is True.
        estimates : list/dict, optional
           List of initial estimates for fitparams, or dict of initial estimates.
        outlier : bool, optional
           Mask outlier pixels in the spectrum. Default is False.
        skipdoppler : bool, optional
           Skip running Doppler at the beginning. Default is False.
        verbose : bool, optional
           Verbose output to screen.  Default is False.

        Returns
        -------
        out : table
           Output table of values and spectra.

        Example
        -------

        out = fit(spec)

        """

        if hasattr(spec,'vrel') == False and vrel is not None:
            spec.vrel = vrel
        if verbose:
            if vrel is not None:
                print('Vrel: {:.2f} km/s'.format(vrel))
            print('S/N: {:.2f}'.format(spec.snr))
        # Now normalize
        if normalize:
            if spec.normalized==False:
                spec.normalize()

        if fitparams is None:
            fitparams = self.labels+['rv']
        fitparams = [p.lower() for p in fitparams]
        self.fitparams = np.array(fitparams)
        nfitparams = len(fitparams)
        nfixparams = len(fixparams)
        self.fixparams = fixparams
        # check that fixparam names are correct
        if nfixparams>0:
            for c in fixparams.keys():
                if c.lower() not in self.labels+['rv']:
                    raise Exception(str(c)+' not one of the labels or rv')
        nallparams = nfitparams+nfixparams
        if nfixparams>0:
            allparams = np.unique(fitparams+list(fixparams.keys()))
        else:
            allparams = np.unique(fitparams)
        allparams = [a.lower() for a in allparams]   # lowercase
        # put allparams in the same order as labels
        allparams = [c for c in self.labels+['rv'] if c in allparams]
        self.allparams = np.array(allparams)
        
        # Make bounds
        bounds = self.mkbounds(allparams,fixparams)

        # Use spec.vrel
        if vrel is None and hasattr(spec,'vrel') and getattr(spec,'vrel') is not None:
            vrel = spec.vrel
        
        # Run doppler if no velocity input
        if vrel is None and skipdoppler==False:
            dopout, dopfmodel, dopspecm = doppler.fit(spec,verbose=verbose)
            vrel = dopout['vrel'][0]
            # add estimates for teff/logg/mh/rv
            if estimates is None:
                estimates = {'teff':dopout['teff'][0],'logg':dopout['logg'][0],
                             'mh':dopout['feh'][0],'rv':dopout['vrel'][0]}
            else:
                if 'teff' not in estimates.keys():
                    estimates['teff'] = dopout['teff'][0]
                if 'logg' not in estimates.keys():
                    estimates['logg'] = dopout['logg'][0]
                if 'mh' not in estimates.keys():
                    estimates['mh'] = dopout['feh'][0]
                if 'rv' not in estimates.keys():
                    estimates['rv'] = dopout['vrel'][0] 
            # Limit the estimates to be within the bounds
            for k in list(estimates.keys()):
                ind, = np.where(self.allparams==k)
                if len(ind)>0:
                    estimates[k] = dln.limit(estimates[k],bounds[0][ind[0]]+1,bounds[1][ind[0]]-1)
                    if k=='rv':
                        vrel = estimates[k]
        # Using input vrel or default of 0.0
        else:
            if vrel is None:
                vrel = 0.0
        spec.vrel = vrel
            
        # Save the spectrum to fit
        self._spobs = spec

        # ADD AN OPTION TO AUTOMATICALLY SCALE THE LSF SIGMA ARRAY
        
        # Run set of ~100 points to get first estimate
        ngrid = 100
        if initgrid:
            
            #if loggrelation:
            #    nsample = 5
            #    tstep = np.ptp(self._ranges[:,0,:])/nsample
            #    tgrid = np.arange(nsample)*tstep+self._ranges[0,0,0]+tstep*0.5
            #    mstep = np.ptp(self._ranges[:,2,:])/nsample
            #    mgrid = np.arange(nsample)*mstep+self._ranges[0,2,0]+mstep*0.5
            #    astep = np.ptp(self._ranges[:,3,:])/nsample
            #    agrid = np.arange(nsample)*astep+self._ranges[0,3,0]+astep*0.5
            #    tgrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,mgrid,agrid)
            #    gridpars = np.vstack((tgrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            #else:
            #    nsample = 4
            #    tstep = np.ptp(self._ranges[:,0,:])/nsample/1.1
            #    tgrid = np.arange(nsample)*tstep+self._ranges[0,0,0]+tstep*0.5
            #    gstep = np.ptp(self._ranges[:,1,:])/nsample/1.1
            #    ggrid = np.arange(nsample)*gstep+self._ranges[0,1,0]+gstep*0.5            
            #    mstep = np.ptp(self._ranges[:,2,:])/nsample/1.1
            #    mgrid = np.arange(nsample)*mstep+self._ranges[0,2,0]+mstep*0.5
            #    astep = np.ptp(self._ranges[:,3,:])/nsample/1.1
            #    agrid = np.arange(nsample)*astep+self._ranges[0,3,0]+astep*0.5
            #    tgrid2d,ggrid2d,mgrid2d,agrid2d = np.meshgrid(tgrid,ggrid,mgrid,agrid)
            #    gridpars = np.vstack((tgrid2d.flatten(),ggrid2d.flatten(),mgrid2d.flatten(),agrid2d.flatten())).T
            
            # These are only for the LABELS, not RV
            gridpars = self.randompars(self.allparams,fixparams=fixparams,n=ngrid)
            fitlabels = [p.lower() for p in self.allparams if p.lower() != 'rv']
            fitlabels = np.array(fitlabels)
            if verbose:
               print('Testing an initial set of '+str(gridpars.shape[0])+' random parameters')
            # Use input estimates
            if estimates is not None:
                for i in range(len(estimates.keys())):
                    key = list(estimates.keys())[i]
                    if key in fixparams.keys():   # skip fixed parameters
                        continue
                    ind, = np.where(fitlabels == key.lower())
                    if len(ind)>0:
                        gridpars[:,ind[0]] = estimates[key]
                        
            # Make the models
            for i in range(gridpars.shape[0]):
                tpars1 = {}
                for j in range(len(fitlabels)):
                    tpars1[fitlabels[j]] = gridpars[i,j]
                sp1 = self(tpars1,vrel=vrel)
                if i==0:
                    synflux = np.zeros((gridpars.shape[0],sp1.size),float)
                synflux[i,:] = sp1.flux
            chisqarr = np.sum((synflux-spec.flux)**2/spec.err**2,axis=1)/spec.size
            bestind = np.argmin(chisqarr)
            estimates = gridpars[bestind,:]
            # add rv to the end, if necessary
            if 'rv' in self.fitparams:
                estimates = np.hstack((estimates,vrel))
        else:
            estimates = np.zeros(len(self.fitparams))
            ind, = np.where(np.array(self.fitparams)=='teff')
            if len(ind)>0:
                estimates[ind] = 4200.0
            ind, = np.where(np.array(self.fitparams)=='logg')
            if len(ind)>0:
                estimates[ind] = 1.5
            ind, = np.where(np.array(self.fitparams)=='rv')
            if len(ind)>0:
                estimates[ind] = vrel
            
        if verbose:
            print('Initial estimates: ',estimates)

        try:
            #try:
            pars,pcov = curve_fit(self.model,spec.wave,spec.flux,p0=estimates,
                                  sigma=spec.err,bounds=bounds,jac=self.jac)
            #except:
            #    print('curve_fit problem')
            #    import pdb; pdb.set_trace()
            perror = np.sqrt(np.diag(pcov))
            bestmodel = self.model(spec.wave,*pars)
            chisq = np.sum((spec.flux-bestmodel)**2/spec.err**2)/spec.size
            
            # Get full parameters
            if loggrelation:
                fullpars = self.getlogg(pars)
                fullperror = np.insert(perror,self.loggind,0.0)
            else:
                fullpars = pars
                fullperror = perror

            if verbose:
                print('Best parameters:')
                self.printpars(fullpars,fullperror)
                print('Chisq: {:.3f}'.format(chisq))

            # Construct the output dictionary
            out = {'vrel':vrel,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':spec.wave,
                   'flux':spec.flux,'err':spec.err,'model':bestmodel,'chisq':chisq,
                   'loggrelation':loggrelation,'success':True}
            success = True
        except:
            traceback.print_exc()
            success = False
            out = {'success':False}
                
        # Remove outliers and refit
        if success and outlier:
            diff = spec.flux-bestmodel
            med = np.median(diff)
            sig = dln.mad(diff)
            bd, = np.where(np.abs(diff) > 3*sig)
            nbd = len(bd)
            if nbd>0:
                if verbose:
                    print('Removing '+str(nbd)+' outliers and refitting')
                err = spec.err.copy()
                flux = spec.flux.copy()
                err[bd] = 1e30
                flux[bd] = bestmodel[bd]
                # Save original values
                pars0 = pars
                estimates = pars0
                # Run curve_fit
                try:
                    pars,pcov = curve_fit(self.model,spec.wave,flux,p0=estimates,
                                          sigma=err,bounds=bounds,jac=self.jac)
                    perror = np.sqrt(np.diag(pcov))
                    bestmodel = self.model(spec.wave,*pars)
                    chisq = np.sum((flux-bestmodel)**2/err**2)/len(flux)

                    # Get full parameters
                    if loggrelation:
                        fullpars = self.getlogg(pars)
                        fullperror = np.insert(perror,self.loggind,0.0)
                    else:
                        fullpars = pars
                        fullperror = perror

                    if verbose:
                        print('Best parameters:')
                        self.printpars(fullpars,fullperror)
                        print('Chisq: {:.3f}'.format(chisq))                        
                        
                    # Construct the output dictionary
                    out = {'vrel':vrel,'snr':spec.snr,'pars':fullpars,'perror':fullperror,'wave':spec.wave,
                           'flux':spec.flux,'err':spec.err,'mflux':flux,'merr':err,'noutlier':nbd,'model':bestmodel,
                           'chisq':chisq,'loggrelation':loggrelation,'success':True}
                    success = True
                except:
                    traceback.print_exc()
                    success = False
                    out = {'success':False}

        return out
