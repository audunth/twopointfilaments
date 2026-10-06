"""
Make figures of all variables
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cosmoplots
from twopointfilaments import TBTPW, xi_factor
plt.style.use(["cosmoplots.default"])

dataloc = '/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'


def make_figure(data, coll, what, doL, lhat, dname, log):
    X, Y = np.meshgrid(np.log10(coll), np.log10(what), indexing='ij')
    
    fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
    for li in range(len(lhat)):
        for dj in range(len(doL)):
            ind = li + len(lhat)*dj
            plot = ax[ind].contourf(X, Y, np.log(data[:,:,li,dj]), cmap='cividis')#, vmin=0,vmax=10, levels = 10)
            fig.colorbar(plot, ax=ax[ind])
            ax[ind].text(1., 2, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'})
            if li == 0:
                ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
            if dj == len(doL)-1:
                ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
    fig.savefig(saveloc+dname+'vs_coll_what.eps')

def plot_vars():
    D = np.load(dataloc+'TBTPW_scan.npz')
    for dname in ['ndN', 'TuN', 'TdN', 'nuF', 'ndF', 'TuF', 'TdF']:
        make_figure(D[dname], D['coll'], D['what'], D['doL'], D['lhat'], dname, log=True)

    for data, dname in zip([1./D['TuN']**2, D['ndN']/D['TdN']**2, D['nuF']/D['TuF']**2, D['ndF']/D['TdF']**2],
                           ['uN','dN','uF','dF']):
        make_figure(data, D['coll'], D['what'], D['doL'], D['lhat'], 'coll_'+dname, log=True)


def plot_good():
    D = np.load(dataloc+'TBTPW_scan.npz')
    data = D['P']
    coll = D['coll']
    what = D['what']
    doL = D['doL']
    lhat = D['lhat']

    X, Y = np.meshgrid(np.log10(coll), np.log10(what), indexing='ij')
    
    fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
    for li in range(len(lhat)):
        for dj in range(len(doL)):
            ind = li + len(lhat)*dj
            plot = ax[ind].pcolormesh(X, Y, data[:,:,li,dj], cmap='cividis', vmin=-7,vmax=0)
            fig.colorbar(plot, ax=ax[ind])
            ax[ind].text(1., 2, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'})
            if li == 0:
                ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
            if dj == len(doL)-1:
                ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
    fig.savefig(saveloc+'good_params.eps')

def compare_near_standard(what_ind):
    D = np.load(dataloc+'TBTPW_scan.npz')
    coll = D['coll']
    what = D['what']
    doL = D['doL']
    lhat = D['lhat']

    TP = np.load(dataloc+'TwoPoint.npz')

    fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
    for li in range(len(lhat)):
        for dj in range(len(doL)):
            ind = li + len(lhat)*dj
            ax[ind].plot(np.log10(coll), D['ndN'][:,what_ind,li,dj]/TP['ndN'], label=r'$n_{d,N}$')
            ax[ind].plot(np.log10(coll), D['TuN'][:,what_ind,li,dj]/TP['TuN'], label=r'$T_{u,N}$')
            ax[ind].plot(np.log10(coll), D['TdN'][:,what_ind,li,dj]/TP['TdN'], label=r'$T_{d,N}$')
            ax[ind].text(1., 2, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'})
            ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
    fig.savefig(saveloc+f'tp_vs_what{what[what_ind]:.2f}.eps')

def test_vs_ofar_nustar():
    D = np.load(dataloc+'TBTPW_scan.npz')
    coll = D['coll']
    what = D['what']
    doL = D['doL']
    lhat = D['lhat']

    X, Y = np.meshgrid(np.log10(coll), np.log10(what), indexing='ij')
    
    fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
    for li in range(len(lhat)):
        for dj in range(len(doL)):
            ind = li + len(lhat)*dj
            ofar = 10**(Y)/D['TuF'][:,:,li,dj]**0.5
            colluN = 10**(X)/D['TuN'][:,:,li,dj]**2
            plot = ax[ind].scatter(np.log10(colluN), np.log10(ofar), c= np.log10(D['nuF'][:,:,li,dj]), cmap='cividis')
            fig.colorbar(plot, ax=ax[ind])
            ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'}, transform=ax[ind].transAxes)
            if li == 0:
                ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}/\sqrt{T_\mathrm{u,F}}$')
            if dj == len(doL)-1:
                ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{u,N}}$')
                ax[ind].plot(np.log10(colluN), -0.33*np.log10(colluN), 'k')
            if dj == len(doL)-2:
                ax[ind].plot(np.log10(colluN),1./lhat[li] -0.5*np.log10(colluN), 'k')

    fig.savefig(saveloc+'test_ofar.eps')

def make_fig_ofar_nustar():
    D = np.load(dataloc+'TBTPW_scan.npz')
    coll = D['coll']
    what = D['what']
    doL = D['doL']
    lhat = D['lhat']

    X, Y = np.meshgrid(np.log10(coll), np.log10(what), indexing='ij')
    
    fig = plt.figure()
    for li in range(1,len(lhat)-1):
        for dj in range(2,len(doL)):
            ind = li + len(lhat)*dj
            ofar = 10**(Y)/D['TuF'][:,:,li,dj]**0.5
            colluN = 10**(X)/D['TuN'][:,:,li,dj]**2

            good = (ofar>0.95)&(ofar<1.05)
            
            C = colluN[good]
            nuF = D['nuF'][:,:,li,dj][good]
            nuF = D['nuF'][:,:,li,dj][good]
            ind = np.argsort(C)
            plt.plot(C[ind], nuF[ind], label=f'{lhat[li]}, {doL[dj]}')
    plt.legend()

    fig.savefig(saveloc+'ofar.eps')


if __name__=="__main__":
    test_vs_ofar_nustar()
    #make_fig_ofar_nustar()
    #plot_good()
    #plot_vars()
    #compare_near_standard(0)
    #compare_near_standard(10)
    #nuf_dol_lhat([10**1.5,],[1.])
