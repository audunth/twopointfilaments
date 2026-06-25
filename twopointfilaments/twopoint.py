import numpy as np
from twopointfilaments import constants

class TwoPoint:
    """
    Classical Two-Point model. Normalized in the same way as twobox.TBTPW.

    Input parameters:
        coll: Reference collisionality
        gamma: Sheath heat coeffificient
        tau: Ion to electron temperature ratio. 
        ion_species: Default to deuterium.
    """
    def __init__(self,coll,gamma=7,tau=1,ion_species='deuterium'):
        self.coll = coll
        self.xi = constants.xi_factor(tau,ion_species)*coll
        self.ghat = gamma/(1+tau)

        # Canclulate Td^{7/4} 
        P = np.polynomial.Polynomial([-(2/(self.ghat*self.xi))**3.5, 1, 0, 1.])
        roots = P.roots()
        good0 = np.isreal(roots)
        good1 = roots[good0]>0
        self.TdN = np.real(roots[good0][good1][0])**(4/7)
        self.TuN = 2/(self.ghat*self.xi*self.TdN**0.5)
        self.ndN = 0.5*self.TuN/self.TdN
