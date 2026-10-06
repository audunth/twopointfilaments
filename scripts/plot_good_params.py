"""
Make a figure over which parameter regimes are successfull
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cosmoplots
from twopointfilaments import TBTPW, xi_factor
plt.style.use(["cosmoplots.default"])

saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

# These values are assumed for all plots
GHAT = 3.5
XI_FACTOR = xi_factor(1,'deuterium')

def plot_good_params(coll, what, lhat, doL):

    P = np.zeros([len(coll),len(what),len(lhat),len(doL)])

    for ci in range(len(coll)):
        print(f'{coll[ci]}',flush=True)
        for wk in range(len(what)):
            for Lj in range(len(lhat)):
                for dl in range(len(doL)):
                    P[ci,wk,Lj,dl]=TBTPW(coll[ci], what[wk], lhat=lhat[Lj], doL=doL[dl]).status

    X, Y = np.meshgrid(coll, np.log10(what), indexing='ij')
    
    fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
    for li in range(len(lhat)):
        for dj in range(len(doL)):
            ind = li + len(lhat)*dj
            ax[ind].pcolormesh(X, Y, P[:,:,li,dj], cmap='cividis', vmin=-6,vmax=0)
            ax[ind].text(38, 1.5, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'})
            if li == 0:
                ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
            if dj == len(doL)-1:
                ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

            #ax[ind].plot(coll, (2.5*XI_FACTOR*coll*doL[dj]*(1-np.exp(-1/doL[dj]))*(1+doL[dj]-1/(1-np.exp(-1/doL[dj])))**(2/7))**(-1))
            #ax[ind].plot(coll, 7*(2.5/GHAT)**2.5/( XI_FACTOR*coll*(GHAT-3)*doL[dj]**2*(1-np.exp(-1/doL[dj]))**2*np.exp(1/doL[dj]) ))
            ax[ind].set_xlim(coll[0],coll[-1])
            ax[ind].set_ylim(np.log10(what[0]),np.log10(what[-1]))
    fig.savefig(saveloc+'good_params.eps')

if __name__=="__main__":
    coll = np.linspace(1,60,100)
    what = 10**np.linspace(-1,2,50)
    lhat = [1, 10]
    doL = [1e-1, 2e-1, 5e-1]

    plot_good_params(coll, what, lhat, doL)
