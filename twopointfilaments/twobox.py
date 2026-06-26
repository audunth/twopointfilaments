import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize
from twopointfilaments import constants

class TBTPW:
    """
    Two-Box Two-Point model with Wall fluxes.

    Two-box extention of the two-point model including near-SOL and far-SOL,
    ballooning and fluxes to the walls.

    Input parameters:
        coll: Reference collisionality
        what: Perpendicular to parallel transport time
        gamma: Sheath heat coeffificient
        tau: Ion to electron temperature ratio. 
        ion_species: Default to deuterium.
        doL: normalized parallel ballooning length.
    """
    def __init__(self,coll,what,gamma=7,tau=1,ion_species='deuterium',doL=0.33):
        self.coll = coll
        self.xi = constants.xi_factor(tau,ion_species)*coll
        self.what = what
        self.ghat = gamma/(1+tau)
        self.doL = doL

        self._calculate_near_sol()

        if what>0:
            self._calculate_far_sol()
        else:
            self.nuF=np.nan
            self.ndF=np.nan
            self.TuF=np.nan
            self.TdF=np.nan

    def get_far_sol_profiles(self, xpoints=100):
        assert self.what>0, "No far SOL for chosen parameters"
        X = np.linspace(0,1,xpoints)[::-1]

        def f(x):
            return (1-np.exp(-x/self.doL))/(1-np.exp(-1./self.doL))
        def M2(x,T):
            # T is T(x,F)/TdF
            return (1-(1-T*f(x)**2)**0.5)/(1+(1-T*f(x)**2)**0.5)
            
        def diffq(x,T):
            c = 2*self.xi*self.ndF / (7*self.TdF**2)
            return c*T**(-2.5)*f(x) * (0.5*T*(5+M2(x,T)) - self.ghat)
            
        res=solve_ivp(diffq, [1.,0.], [1.,], t_eval=X)
        print(res)
        T = res.y[0,:]*self.TdF
        M = np.sqrt(M2(X,res.y[0,:]))
        n = 2*self.ndF/(np.sqrt(res.y[0,:])*(M2(X,res.y[0,:])+1))

        return X, n, M, T

    def _calculate_near_sol(self):
        # The three equations combined give for the upstream temperature:
        def g(TuN):
            z0 = 2.5*self.what*self.doL*self.xi*TuN
            e0 = 1-np.exp(-1/self.doL)
            e1 = 1-self.doL*e0
            
            return TuN**3.5-(2*(1-z0*e0)/(self.ghat*self.xi*TuN))**(7) - 1 + z0*e1

        res = root_scalar(g,bracket=(1e-5,1e5),x0=1)
        self.TuN = res.root
        self.TdN = (2*(1-2.5*self.what*self.doL*self.xi*self.TuN*(1-np.exp(-1/self.doL)))/(self.ghat*self.xi*self.TuN))**(2) 
        self.ndN = 0.5*self.TuN/self.TdN
        self.nuN = 1.



    def _calculate_far_sol(self, x0=1.1):
        theta = self._optimize_TuF(x0).x[0]
        self.ndF, self.TdF = self._ndF_TdF(theta)
        self.TuF = theta*self.TdF
        self.nuF = 2*self.ndF/theta

    def _ndF_TdF(self,theta):
        # Returns ndF and TdF for given theta=TuF/TdF

        a = self.what*self.doL*(1-np.exp(-1./self.doL))
        c = 2.5*a/(self.ghat)
        b = c*self.TuN

        # Canclulate sqrt(TdF)
        P = np.polynomial.Polynomial([-2*b/theta, -b/a, 2*c, 1.])
        roots = P.roots()
        good0 = np.isreal(roots)
        good1 = roots[good0]>0

        TdFsq = np.real(roots[good0][good1])
        assert len(TdFsq)==1, 'Too many positive TdF roots'
        TdF = TdFsq[0]**2
        ndF = a/(TdFsq[0]+2*a/theta)

        return ndF, TdF

    def _optimize_TuF(self,x0):
        def f(x):
            return (1-np.exp(-x/self.doL))/(1-np.exp(-1./self.doL))
        def M2(x,T):
            # T is T(x,F)/TdF
            return (1-(1-T*f(x)**2)**0.5)/(1+(1-T*f(x)**2)**0.5)
            
        def diffq(x,T,ndF,TdF):
            c = 2*self.xi*ndF / (7*TdF**2)
            return c*T**(-2.5)*f(x) * (0.5*T*(5+M2(x,T)) - self.ghat)
            
        def fitfun(theta):
            ndF, TdF = self._ndF_TdF(theta[0])
            res=solve_ivp(diffq, [1.,0.], [1.,], args=(ndF, TdF))
            assert res.success, f'Did not find a solution to the energy equation, coll={self.coll}, what={self.what}'
            return np.abs(theta-res.y[0,-1])

        return minimize(fitfun, x0, bounds=((1e-1,10.),))
