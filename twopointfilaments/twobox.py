import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize
from twopointfilaments import constants, TwoPoint
import sympy as sy
import mpmath as mp
import warnings

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
        lhat: far-SOL width divided by near-SOL width.
    """
    def __init__(self,coll,what,gamma=7,tau=1,ion_species='deuterium',doL=0.33,lhat=5):
        self.coll = mp.mpf(coll)
        self.xi = mp.mpf(constants.xi_factor(tau,ion_species)*coll)
        self.what = mp.mpf(what)
        self.ghat = mp.mpf(gamma/(1+tau))
        self.doL = mp.mpf(doL)
        self.lhat = mp.mpf(lhat)
        self.status = np.nan
        self.message = ''

        self.ndN=np.nan
        self.TuN=np.nan
        self.TdN=np.nan

        self.nuF=np.nan
        self.ndF=np.nan
        self.TuF=np.nan
        self.TdF=np.nan


        if what>0:
            try:
                self._calculate_near_sol()
                self._calculate_far_sol()
                self.status = 0
            except AssertionError as e:
                self.message = e.args[0][0]
                self.status = e.args[0][1]

        else:
            TP = TwoPoint(coll, gamma, tau, ion_species)
            self.nuN = 1
            self.ndN = TP.ndN
            self.TuN = TP.TuN
            self.TdN = TP.TdN
            self.status = 0

    def _calculate_near_sol(self):
        # The three near-SOL equations combined give for the upstream temperature:
        def find_TuN():
            '''
            Uses mpmath to find the root of the polynomial for t=T_uN^0.5. 
            With count_roots, we ensure there is only one positive real root.

            The variables z0, z1, e0, e1 are collections of model parameters as defined below.
            '''
            # Define constants
            Z0=2.5*self.what*(1+self.lhat)*self.doL*self.xi
            Z1=(2/(self.ghat*self.xi))**7
            E0=1-mp.exp(-1/self.doL)
            E1=1-self.doL*(1-mp.exp(-1/self.doL))

            pol = lambda t:t**21-Z1*(1-Z0*E0*t**2)**7+Z0*E1*t**16-t**14
            dpol = lambda t: 21*t**20-Z1*7*(1-Z0*E0*t**2)**6*(-2*Z0*E0*t)+16*Z0*E1*t**15-14*t**13
            d2pol = lambda t: 21*20*t**19-Z1*7*( 6*(1-Z0*E0*t**2)**5*(-2*Z0*E0*t)**2 + (1-Z0*E0*t**2)**6*(-2*Z0*E0) )+16*15*Z0*E1*t**14-14*13*t**12
            t, z0, z1, e0, e1 = sy.symbols('t, z0, z1, e0, e1')
            P0 = sy.poly(t**21-z1*(1-z0*e0*t**2)**7+z0*e1*t**16-t**14)
            P = P0.subs([
                (z0,Z0),
                (z1,Z1),
                (e0,E0),
                (e1,E1),
                ])
            positive_roots = []
            for r in sy.real_roots(P):
                if r>0:
                    positive_roots.append(r)
            assert len(positive_roots)==1, (f'far-SOL does not have a unique solution for TdF. Roots: {[r.evalf() for r in positive_roots]}',-3)
            try:
                res = mp.findroot(pol,positive_roots[0].evalf(),solver='mnewton',df=dpol, d2f=d2pol, tol=1e-6, maxsteps=1000)
                diff = np.abs(res-positive_roots[0].evalf())/positive_roots[0].evalf()
                if diff>1e-2:
                    print(diff)
                return res**2
            except ValueError:
                assert False, ('Root not numerically verified',-8)

        self.TuN = find_TuN()
        assert (1-2.5*self.what*(1+self.lhat)*self.doL*self.xi*self.TuN*(1-mp.exp(-1/self.doL))>0), ('Near-SOL sheath dissipation is negative.', -2)
        self.TdN = (2*(1-2.5*self.what*(1+self.lhat)*self.doL*self.xi*self.TuN*(1-mp.exp(-1/self.doL)))/(self.ghat*self.xi*self.TuN))**(2) 
        self.ndN = 0.5*self.TuN/self.TdN
        self.nuN = 1.

    def _calculate_far_sol(self):
        def find_TdF():
            """
            See find_TuN. Same principle.
            """
            t, a, b = sy.symbols('t a b')
            P0 = sy.poly(t**7 + a*t**6-b)
            P = P0.subs([
                (a,5*self.what*(1+1./self.lhat)*self.doL*(1-mp.exp(-1./self.doL))/self.ghat),
                (b,(5/14)*self.xi*(self.ghat-3)*self.what*(1+self.lhat)*self.doL**2*(1-mp.exp(-1./self.doL))**2*mp.exp(1./self.doL)*self.TuN/self.ghat)
                ])
            positive_roots = []
            for r in sy.real_roots(P):
                if r>0:
                    positive_roots.append(r)
            assert len(positive_roots)==1, (f'far-SOL does not have a unique solution for TdF. Roots: {[r.evalf() for r in positive_roots]}',-3)
            if P.subs(t,positive_roots[0])>1e-6:
                return self._improve_root(P, float(positive_roots[0]))
            return float(positive_roots[0].evalf()**2)
        
        self.TdF = find_TdF()
        self.ndF = 7*mp.exp(-1./self.doL)*self.TdF**2/( self.xi*(self.ghat-3)*self.doL*(1-mp.exp(-1./self.doL)) )
        self.nuF = (1.+self.lhat- self.ndF*self.TdF**0.5/(self.what*self.doL*(1.-mp.exp(-1./self.doL))))/(1+1./self.lhat)
        assert (self.nuF>0), ('Far-SOL upstream density is negative.',-4)
        assert (self.nuF<self.lhat), ('Far-SOL particle source is negative.',-5)
        assert (self.nuF<1), ('Far-SOL upstream density is higher than near-SOL', -6)
        self.TuF = 2*self.ndF*self.TdF/self.nuF
        assert (self.TuN>self.nuF*self.TuF/self.lhat), ('Far-SOL heat source is negative.',-7)

    def get_far_sol_profiles(self, xpoints=100):
        """
        Likely bad. 
        """
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


    def _improve_root(self,Psy, root):
        # Attempt to improve the found root of the sympy polynomial Psy by numerical iteration
        P = sy.lambdify(Psy.gens, Psy.as_expr(), modules='mpmath')
        res = mp.findroot(P,(mp.mpf(0.),mp.mpf(1e7)),solver='bisect',tol=1e-6)
        return res.root
