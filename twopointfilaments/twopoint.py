import numpy as np

class TwoPoint:
    """
    Classical Two-Point model. Normalized in the same way as twobox.TBTPW.

    Input parameters:
        coll: Reference collisionality
        gamma: Sheath heat coeffificient
        tau: Ion to electron temperature ratio. 
        sqmass: Square root of electron to total mass ratio. 
                Default for deuterium.
    """
    def __init__(self,coll,gamma=7,tau=1,sqmass=0.023):
        tautmp = (1+tau)
        
        self.xi = 7*tautmp**1.5*sqmass*coll/(3.2*2)
        self.ghat = gamma/tautmp

        # Canclulate Td^{7/4} 
        P = np.polynomial.Polynomial([-(2/(self.ghat*self.xi))**3.5, 1, 0, 1.])
        roots = P.roots()
        good0 = np.isreal(roots)
        good1 = roots[good0]>0
        self.TdN = np.real(roots[good0][good1])**(4/7)
        self.TuN = 2/(self.ghat*self.xi*self.TdN**0.5)
        self.ndN = 2*self.TuN/self.TdN
