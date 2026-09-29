import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize
from twopointfilaments import constants
import sympy as sy

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

    def _calculate_near_sol(self):
        # The three equations combined give for the upstream temperature:
        def find_TuN():
            '''
            Uses sympy to find the root of the polynomial for t=T_uN^0.5. 
            With count_roots, we ensure there is only one positive real root.

            The variables z0, z1, e0, e1 are collections of model parameters as defined below.
            '''
            t, z0, z1, e0, e1 = sy.symbols('t z0 z1 e0 e1')
            
            # Define polynomial
            P0 = sy.poly(t**21-z1*(1-z0*e0*t**2)**7+z0*e1*t**16-t**14)
            # Insert numerical values for the variables
            P = P0.subs([
                (z0, 2.5*self.what*self.doL*self.xi),
                (z1, (2/(self.ghat*self.xi))**7),
                (e0, 1-np.exp(-1/self.doL)),
                (e1, 1-self.doL*(1-np.exp(-1/self.doL))),
                ])
            
            real_roots = sy.real_roots(P)
            assert(len(real_roots)==1), 'Near-SOL does not have a unique solution for TuN.'

            return float(real_roots[0].evalf()**2)
        self.TuN = find_TuN() 
        self.TdN = (2*(1-2.5*self.what*self.doL*self.xi*self.TuN*(1-np.exp(-1/self.doL)))/(self.ghat*self.xi*self.TuN))**(2) 
        self.ndN = 0.5*self.TuN/self.TdN
        self.nuN = 1.



    def _calculate_far_sol(self):
        def f(TdF):
            a = 5*self.what*self.doL*(1-np.exp(-1./self.doL))/self.ghat
            b = (5/14)*self.xi*(self.ghat-3)*self.what*self.doL**2*(1-np.exp(-1./self.doL))**2*np.exp(1./self.doL)*self.TuN/self.ghat

            return TdF**3.5+a*TdF**3-b
        
        res = root_scalar(f,bracket=(1e-6,1e5),x0=1)
        self.TdF = res.root
        self.ndF = 7*np.exp(-1./self.doL)*self.TdF**2/( self.xi*(self.ghat-3)*self.doL*(1-np.exp(-1./self.doL)) )
        self.nuF = 1.- self.ndF*self.TdF**0.5/(self.what*self.doL*(1.-np.exp(-1./self.doL)))
        self.TuF = 2*self.ndF*self.TdF/self.nuF

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
        n = 2*self.ndF/(res.y[0,:]*(M2(X,res.y[0,:])+1))

        return X, n, M, T
