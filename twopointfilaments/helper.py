import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def plot_far_sol_profiles(TBTPW):
    def f(x):
        return (1-np.exp(-x/TBTPW.doL))/(1-np.exp(-1./TBTPW.doL))
    def M2(x,T):
        # T is T(x,F)/TdF
        return (1-(1-T*f(x)**2)**0.5)/(1+(1-T*f(x)**2)**0.5)
        
    def diffq(x,T):
        c = 2*TBTPW.xi*TBTPW.ndF / (7*TBTPW.TdF**2)
        return c*T**(-2.5)*f(x) * (0.5*T*(5+M2(x,T)) - TBTPW.ghat)

    x = np.linspace(0,1,1000)[::-1]
    res=solve_ivp(diffq, [1.,0.], [1.,], t_eval=x)
    n = 2/(res.y[0,:]*(M2(x,res.y[0,:])+1))

    plt.figure()
    plt.plot(x, res.y[0,:], label='T/Td')
    plt.plot(x, np.sqrt(M2(x,res.y[0,:])), label='M')
    plt.plot(x,n,label='n/nd')
