"""
Make figures of all variables
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import cosmoplots
from twopointfilaments import TBTPW, xi_factor
plt.style.use(["cosmoplots.default"])

dataloc = '/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'


def make_figure(data, coll, what, doL, lhat, dname, log=True):
    X, Y = np.meshgrid(np.log10(coll), np.log10(what), indexing='ij')
    
    fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
    for li in range(len(lhat)):
        for dj in range(len(doL)):
            ind = li + len(lhat)*dj
            plot = ax[ind].contourf(X, Y, np.log10(data[:,:,li,dj]), cmap='cividis')#, vmin=0,vmax=10, levels = 10)
            fig.colorbar(plot, ax=ax[ind])
            ax[ind].text(1., 2, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'})
            if li == 0:
                ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
            if dj == len(doL)-1:
                ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
    fig.savefig(saveloc+dname+'vs_coll_what.eps')

def plot_vars():
    D = np.load(dataloc+'TBTPW_scan.npz')
    coll, what = np.meshgrid(D['coll'], D['what'],indexing='ij')
    what = np.stack((np.stack((what,)*D['lhat'].size,axis=-1),)*D['doL'].size,axis=-1)
    coll = np.stack((np.stack((coll,)*D['lhat'].size,axis=-1),)*D['doL'].size,axis=-1)
    for dname in ['ndN', 'TuN', 'TdN', 'nuF', 'ndF', 'TuF', 'TdF']:
        make_figure(D[dname], D['coll'], D['what'], D['doL'], D['lhat'], dname+'_', log=True)

    for data, dname in zip([1./D['TuN']**2, D['ndN']/D['TdN']**2, D['nuF']/D['TuF']**2, D['ndF']/D['TdF']**2],
                           ['uN','dN','uF','dF']):
        make_figure(data*coll, D['coll'], D['what'], D['doL'], D['lhat'], 'coll_'+dname+'_', log=True)

    
    make_figure(what/D['TuN']**0.5, D['coll'], D['what'], D['doL'], D['lhat'], 'what_uN_', log=True)
    make_figure(what/D['TuF']**0.5, D['coll'], D['what'], D['doL'], D['lhat'], 'what_uF_', log=True)

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
            plot = ax[ind].pcolor(X, Y, data[:,:,li,dj], cmap='cividis', vmin=-8,vmax=0)
            fig.colorbar(plot, ax=ax[ind])
            ax[ind].text(1., 2, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'})
            if li == 0:
                ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
            if dj == len(doL)-1:
                ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
            ax[ind].set_xlim(-1,2)
            ax[ind].set_ylim([-2,2])
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
    fig.savefig(saveloc+f'tp_vs_what{what[what_ind]:.4f}.eps')

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
            #colluF = 10**(X)*D['nuF'][:,:,li,dj]/D['TuF'][:,:,li,dj]**2
            #plot = ax[ind].scatter(np.log10(colluF), np.log10(ofar), color = mpl.colormaps['cividis'](300))
            plot = ax[ind].pcolor(X, np.log10(ofar), Y, cmap = 'cividis')
            ax[ind].axhline(np.log10(0.1), c='k')
            ax[ind].axhline(np.log10(1.), c='grey')
            fig.colorbar(plot, ax=ax[ind])
            ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'}, transform=ax[ind].transAxes)
            ax[ind].set_xlim(-1,2)
            ax[ind].set_ylim([-2,2])
            ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}/\sqrt{T_\mathrm{u,F}}$')
            ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')
                #ax[ind].plot(np.log10(colluN), -0.33*np.log10(colluN), 'k')
            #if dj == len(doL)-2:
            #ax[ind].plot(np.log10(coll),0.0 -0.66*np.log10(coll), 'r')

    fig.savefig(saveloc+'ofar_coll_ref.eps')

def make_fig_ofar():
    D = np.load(dataloc+'TBTPW_scan.npz')
    coll = D['coll']
    what = D['what']
    doL = D['doL']
    lhat = D['lhat']

    X, Y = np.meshgrid(np.log10(coll), np.log10(what), indexing='ij')
    
    figN,axN = plt.subplots()
    figF,axF = plt.subplots()
    figP,axP = plt.subplots()
    figC,axC = plt.subplots()
    for li in range(1,len(lhat)-2):
        for dj in range(2,len(doL)-1):
            ofar = 10**(Y)/D['TuF'][:,:,li,dj]**0.5
            #colluN = 10**(X)/D['TuN'][:,:,li,dj]**2

            #good = (ofar>0.95)&(ofar<1.05)
            good = (ofar>0.45)&(ofar<0.55)
            
            ndN = D['ndN'][:,:,li,dj][good]
            TuN = D['TuN'][:,:,li,dj][good]
            TdN = D['TdN'][:,:,li,dj][good]
            nuF = D['nuF'][:,:,li,dj][good]
            ndF = D['ndF'][:,:,li,dj][good]
            TuF = D['TuF'][:,:,li,dj][good]
            TdF = D['TdF'][:,:,li,dj][good]

            axN.scatter(10**X[good], np.log10(ndN), label='ndN/nuN')
            axN.scatter(10**X[good], np.log10(TuN), label='TuN/Tref')
            axN.scatter(10**X[good], np.log10(TdN/TuN), label='TdN/TuN')

            axF.scatter(10**X[good], np.log10(ndF/nuF), label='ndF/nuF')
            axF.scatter(10**X[good], np.log10(TdF/TuF), label='TdF/TuF')

            axP.scatter(10**X[good], np.log10(TuN), label='u,N')
            axP.scatter(10**X[good], np.log10(nuF*TuF), label='u, F')

            axC.scatter(10**X[good], np.log10(1./TuN**2), label='uN')
            axC.scatter(10**X[good], np.log10(ndN/TdN**2), label='dN')
            axC.scatter(10**X[good], np.log10(nuF/TuF**2), label='uF')
            axC.scatter(10**X[good], np.log10(ndF/TdF**2), label='dF')

    figN.legend()
    figF.legend()
    figP.legend()
    figC.legend()

    figN.savefig(saveloc+f'nearsol_ofar{0.5}_lhat{lhat[li]}_doL{doL[dj]}.eps')
    figF.savefig(saveloc+f'farsol_ofar{0.5}_lhat{lhat[li]}_doL{doL[dj]}.eps')
    figP.savefig(saveloc+f'pressure_ofar{0.5}_lhat{lhat[li]}_doL{doL[dj]}.eps')
    figC.savefig(saveloc+f'coll_ofar{0.5}_lhat{lhat[li]}_doL{doL[dj]}.eps')


if __name__=="__main__":
    #test2()
    test_vs_ofar_nustar()
    #make_fig_ofar()
    plot_good()
    #plot_vars()
    #compare_near_standard(0)
    #compare_near_standard(20)
