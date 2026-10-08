import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import cosmoplots
plt.style.use(["cosmoplots.default"])
from twopointfilaments import xi_factor
from scipy.integrate import quad

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

# Linestyle for plotting wrt what
linestyle = ['--', ':']
symbstyle = ['v','^']

class Plot:
    """
    Container class to make plots. 
    *_values indicates which values (of the full scans) to use.
    The full set of coll-values is always assumed, although plotting may truncate.

    Although the computation is done with respect to w_hat = v_perp L_par/ (c_{s,ref} lambda), 
    the what_values used here are 
    the plotting is done with respect
    """
    def __init__(self,lhat_vals, doL_vals,
                 gamma=7,tau=1,ion_species='deuterium',
                 ):
        """
        lhat_vals and doL_vals are lists of indices.
        """
        self.xi_factor = xi_factor(tau,ion_species)
        self.ghat = gamma/(1+tau)

        self.standard_twopoint = np.load(dataloc+"TwoPoint.npz")
        self.scan = np.load(dataloc+"TBTPW_scan.npz")
        
        self.coll = self.scan['coll']
        self.what = self.scan['what']
        self.lhat = self.scan['lhat']
        self.doL = self.scan['doL']

        self.ndN = self.scan['ndN']
        self.TuN = self.scan['TuN']
        self.TdN = self.scan['TdN']
        self.nuF = self.scan['nuF']
        self.ndF = self.scan['ndF']
        self.TuF = self.scan['TuF']
        self.TdF = self.scan['TdF']

        self.P = self.scan['P']

        self._select_scan_indices(lhat_vals, doL_vals)

    def _select_scan_indices(self, lhat_vals, doL_vals):
        lhat_ind = [np.argmin(np.abs(self.lhat-l)) for l in lhat_vals]
        doL_ind = [np.argmin(np.abs(self.doL-d)) for d in doL_vals]
        self.lhat = self.lhat[lhat_ind]
        self.doL = self.doL[doL_ind]
        
        attributes = [
            "ndN", "TuN", "TdN", "nuF",
            "ndF", "TuF", "TdF", "P",
        ]

        for name in attributes:
            d = getattr(self, name)
            d = np.take(d, lhat_ind, axis=2)
            d = np.take(d, doL_ind, axis=3)
            setattr(self, name, d)

    def find_what_uF_indices(self, 
                          what_vals = None, what_err = 5e-2):
        # Find combined nuref, what_ref indices which correspond to what_uf in what_vals, with 
        # error as a percentage set by what_err.

        # Returns an array of the same shape as ndN, TdN etc of booleans
        _, W = np.meshgrid(self.coll, self.what, indexing='ij')
        
        self.what_ind = np.array([np.argmin(np.abs(self.what-w)) for w in what_vals])
        self.what_uF_good_ind = {key:np.zeros_like(self.scan['P'], dtype=bool) for key in self.what_ind}
        self.what_uF = np.zeros_like(self.scan['P'])

        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                self.what_uF[:,:,li,dj] = W/self.TuF[:,:,li,dj]**0.5
                
                for wk in self.what_ind:
                    self.what_uF_good_ind[wk][:,:,li,dj] = (
                                                            (self.what_uF[:,:,li,dj]>(1-what_err)*self.what[wk])
                                                            &(self.what_uF[:,:,li,dj]<(1+what_err)*self.what[wk])
                                                            )
    
    def _get_plot_params(self, what_uF):
        if what_uF == True:
            good = lambda data, wk, li, dj: data[:,:,li,dj][self.what_uF_good_ind[wk][:,:,li,dj]]
            C = lambda wk, li, dj: np.stack((self.coll,)*len(self.what),axis=1)[self.what_uF_good_ind[wk][:,:,li,dj]]
            def plot(Ax, X, Y, col, styleind, label=None):
                Ax.scatter(X, Y, c=col, marker=symbstyle[styleind], label=label)
            fname = '_what_uF'
        else:
            C = lambda wk, li, dj: self.coll
            good = lambda data, wk, li, dj: data[:,wk,li,dj]
            def plot(Ax, X, Y, col, styleind, label=None):
                Ax.plot(X, Y, c=col, ls=linestyle[styleind], label=label)
            fname = ''

        return C, good, plot, fname

    def plot_good_params(self, truncated=True):   
        X, Y = np.meshgrid(np.log10(self.scan['coll']), 
                           np.log10(self.scan['what']), indexing='ij')
        
        if truncated:
            lhat = self.lhat
            doL = self.doL
            P = self.P
            fname = '_trunc'
        else:
            lhat = self.scan['lhat']
            doL = self.scan['doL']
            P = self.scan['P']
            fname = ''

        fig, ax = cosmoplots.figure_multiple_rows_columns(len(doL),len(lhat))
        for li in range(len(lhat)):
            for dj in range(len(doL)):
                ind = li + len(lhat)*dj
                ax[ind].pcolormesh(X, Y, P[:,:,li,dj], cmap='cividis', vmin=-7,vmax=0)
                #fig.colorbar(plot, ax=ax[ind])
                ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={lhat[li]},\, \hat{{\delta}}={doL[dj]}$', bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)
                if li == 0:
                    ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
                if dj == len(doL)-1:
                    ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')

                if truncated:
                    ax[ind].text(0.1, 0.1, r'$\mathrm{I}$',transform = ax[ind].transAxes)
                    ax[ind].text(0.9, 0.9, r'$\mathrm{IV}$', transform = ax[ind].transAxes)
                    if li==0:
                        ax[ind].text(0.5, 0.6, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                    if ind==1:
                        ax[ind].text(0.4, 0.4, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                        ax[ind].text(0.6, 0.6, r'$\mathrm{III}$', c = 'white',transform = ax[ind].transAxes)
                    if ind==3:
                        ax[ind].text(0.5, 0.55, r'$\mathrm{II}$',transform = ax[ind].transAxes)
                        ax[ind].text(0.05, 0.75, r'$\mathrm{III}$',c='white',transform = ax[ind].transAxes)

        fig.savefig(saveloc+'good_params'+fname+'.eps')

    def plot_good_params_what_uF(self):   
        X, Y = np.meshgrid(np.log10(self.scan['coll']), 
                           np.log10(self.scan['what']), indexing='ij')
        
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                ax[ind].scatter(X, np.log10(self.what_uF[:,:,li,dj]), color=mpl.colormaps['cividis'](300))
                ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)
                if li == 0:
                    ax[ind].set_ylabel(r'$\log_{10} \hat{\omega}$')
                if dj == len(self.doL)-1:
                    ax[ind].set_xlabel(r'$\log_{10} \nu_{\mathrm{ref}}$')

                #ax[ind].set_xlim(0,2)
                #ax[ind].set_ylim(-3,2)
        fig.savefig(saveloc+'good_params_ofar.eps')

    def plot_near_SOL(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                ax[ind].loglog(self.coll, self.standard_twopoint['ndN'], 'C0', label = r'$n_{d,N}/n_\mathrm{u,N}$')
                ax[ind].plot(self.coll, self.standard_twopoint['TdN']/self.standard_twopoint['TuN'], 'C1', label = r'$T_{d,N}/T_{u,N}$')
                ax[ind].plot(self.coll, self.standard_twopoint['TuN'], 'C2', label = r'$T_{u,N}/T_\mathrm{ref}$')

                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), good(self.ndN,wk,li,dj), 'C0', styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TdN/self.TuN,wk,li,dj), 'C1', styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TuN,wk,li,dj), 'C2', styleind)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim(0,4)
                ax[ind].legend()
        fig.savefig(saveloc+'near'+fname+'.eps')

    def plot_far_SOL(self, what_uF=False):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        C, good, plot, fname = self._get_plot_params(what_uF)
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                cosmoplots.change_log_axis_base(ax[ind],)
                for styleind, wk in enumerate(self.what_ind):
                    plot(ax[ind], C(wk,li,dj), good(self.ndF/self.nuF,wk,li,dj), 'C0',styleind)
                    plot(ax[ind], C(wk,li,dj), good(self.TdF/self.TuF,wk,li,dj), 'C1',styleind)
                    #plt.plot(Data['coll'], Data['TuF']/Data['TuN'], 'C2'+ls)
                    #plt.plot(Data['coll'], Data['nuF'], 'C3'+ls)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim(0,4)
                lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$n_\mathrm{d,F}/n_\mathrm{u,F}$'),
                         mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,F}$')]
                         #mlines.Line2D([], [], color='C2', ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
                         #mlines.Line2D([], [], color='C3', ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'far'+fname+'.eps')

    def plot_far_SOL_what_uF(self):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                for wk, w in enumerate(self.what_uF_good_ind.keys()):
                    C = np.stack((self.coll,)*len(self.what),axis=0)[self.what_uF_good_ind[w][:,:,li,dj]]
                    good = self.what_uF_good_ind[w][:,:,li,dj]
                    ax[ind].scatter(C, self.ndF[:,:,li,dj][good]/self.nuF[:,:,li,dj][good], c='C0', marker=symbstyle[wk])
                    ax[ind].scatter(C, self.TdF[:,:,li,dj][good]/self.TuF[:,:,li,dj][good], c='C1', marker=symbstyle[wk])
                    #plt.plot(Data['coll'], Data['TuF']/Data['TuN'], 'C2'+ls)
                    #plt.plot(Data['coll'], Data['nuF'], 'C3'+ls)

                    ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                                 transform = ax[ind].transAxes)

                    if dj == len(self.doL)-1:
                        ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')

                    ax[ind].set_xlim(self.coll[0],self.coll[-1])
                    #ax[ind].set_ylim(0,4)
                lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$n_\mathrm{d,F}/n_\mathrm{u,F}$'),
                         mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,F}$')]
                         #mlines.Line2D([], [], color='C2', ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
                         #mlines.Line2D([], [], color='C3', ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
                ax[ind].legend(handles=lines)

        fig.savefig(saveloc+'far_coll_what.eps')

    def plot_coll(self):
        fig, ax = cosmoplots.figure_multiple_rows_columns(len(self.doL),len(self.lhat))
        lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
                 mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
                 mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
                 mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$'),
                 mlines.Line2D([], [], color='k', ls = '-.', label=r'$\mathrm{ref}$')]
        for li in range(len(self.lhat)):
            for dj in range(len(self.doL)):
                ind = li + len(self.lhat)*dj
                ax[ind].loglog(self.coll, self.coll, 'k')
                ax[ind].loglog(self.coll, self.coll/self.standard_twopoint['TuN']**2, 'C0')
                ax[ind].plot(self.coll, self.coll*self.standard_twopoint['ndN']/self.standard_twopoint['TdN']**2, 'C1')
                for wk, w in enumerate(self.what):
                    ax[ind].plot(self.coll, self.coll*self.nuF[:,wk,li,dj]/self.TuF[:,wk,li,dj]**2, 'C2'+linestyle[wk])
                    ax[ind].plot(self.coll, self.coll*self.ndF[:,wk,li,dj]/self.TdF[:,wk,li,dj]**2, 'C3'+linestyle[wk])

                ax[ind].text(0.62, 0.1, rf'$\hat{{\lambda}}={self.lhat[li]},\, \hat{{\delta}}={self.doL[dj]}$', bbox={'facecolor':'white'},
                             transform = ax[ind].transAxes)

                if dj == len(self.doL)-1:
                    ax[ind].set_xlabel(r'$\nu_{\mathrm{ref}}$')
                if li == 0:
                    ax[ind].set_ylabel(r'$\nu$')

                ax[ind].set_xlim(self.coll[0],self.coll[-1])
                ax[ind].legend(handles=lines)
        #cosmoplots.change_log_axis_base(ax, base=10)
        #plt.xlim(1,32)
        #plt.ylim(0,4)
        fig.savefig(saveloc+'coll.eps')

def plot_near_SOL(what_arr, doL=0.33):
    plt.figure('near sol')
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.plot(Data['coll'], Data['ndN'], 'C0'+ls)
        plt.plot(Data['coll'], Data['TdN']/Data['TuN'], 'C1'+ls)
        plt.plot(Data['coll'], Data['TuN'], 'C2'+ls)

    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$n_\mathrm{d,N}/n_\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,N}/T_\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$T_\mathrm{u,N}/T_\mathrm{ref}$')]
    plt.legend(handles=lines)
    #plt.xlim(1,60)
    plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'near.eps')

def plot_far_SOL(what_arr, doL=0.33):
    plt.figure('far sol')
    # Case what=0 makes no sense in far SOL
    for i,ls in zip([1,2],['--',':']):
        Data = load_data(what_arr[i],doL)
        plt.plot(Data['coll'], Data['ndF']/Data['nuF'], 'C0'+ls)
        plt.plot(Data['coll'], Data['TdF']/Data['TuF'], 'C1'+ls)
        #plt.plot(Data['coll'], Data['TuF']/Data['TuN'], 'C2'+ls)
        #plt.plot(Data['coll'], Data['nuF'], 'C3'+ls)

    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$n_\mathrm{d,F}/n_\mathrm{u,F}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,F}$')]
             #mlines.Line2D([], [], color='C2', ls = '-', label=r'$T_\mathrm{u,F}/T_\mathrm{u,N}$'),
             #mlines.Line2D([], [], color='C3', ls = '-', label=r'$n_\mathrm{u,F}/n_\mathrm{u,N}$')]
    plt.legend(handles=lines)
    #plt.xlim(1,32)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'far.eps')

def plot_TdF_vs_TuN(what_arr, doL=0.33):
    # ghat and xi_factor are for tau=1 and deuterium
    fig=plt.figure('TdF vs TuN')
    ax = fig.gca()
    for i,ls in zip([1,2],['--',':']):
        Data = load_data(what_arr[i],doL)
        plt.loglog(Data['coll'], Data['TdF']/Data['TuN'], 'C0'+ls)
        #plt.loglog(Data['coll'], Data['TdN']/Data['TuN'], 'C1'+ls)
        plt.loglog(Data['coll'], Data['TdF'], 'C1'+ls)
    plt.axhline(2.5/(GHAT),c='k', ls='-.')
    #plt.loglog(Data['coll'][Data['coll']<7],3*Data['coll'][Data['coll']<7]**(-0.7),c='grey',ls='-.')
    #ax.text(3,0.7,r'$\nu_\mathrm{ref}^{-0.7}$',c='grey')
    cosmoplots.change_log_axis_base(ax, base=10)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{ref}$'),
             mlines.Line2D([], [], color='k', ls = '-.', label=r'$5 /(2 \hat{\gamma}\xi)$')]
    plt.legend(handles=lines)
    #plt.xlim(1,32)
    #plt.ylim(1e-1,10)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$T$')
    plt.savefig(saveloc+'tdfvtun.eps')

def plot_wall_flux(what_arr, doL=0.33):
    fig=plt.figure('wall flux')
    ax = fig.gca()
    # Case what=0 makes no sense in far SOL
    for i,ls in zip([1,2],['--',':']):
        Data = load_data(what_arr[i],doL)
        ax.semilogy(Data['coll'], Data['nuF'], 'C0'+ls)
        ax.plot(Data['coll'], XI_FACTOR*Data['coll']*Data['nuF']*Data['TuF'], 'C1'+ls)

    cosmoplots.change_log_axis_base(ax, base=10)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\Gamma_{\perp,F}(0)/\hat{\omega}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$2 q_{\perp,F}(0)/(5\hat{\omega})$')]
    plt.legend(handles=lines)
    #plt.xlim(1,32)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'wall_flux.eps')

def plot_coll(what_arr, doL=0.33):
    fig=plt.figure('collisionality')
    ax = fig.gca()
    plt.axhline(1,c='k',ls='-.')
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.semilogy(Data['coll'], 1/Data['TuN']**2, 'C0'+ls)
        plt.plot(Data['coll'], Data['ndN']/Data['TdN']**2, 'C1'+ls)
        if i>0:
            plt.plot(Data['coll'], Data['nuF']/Data['TuF']**2, 'C2'+ls)
            plt.plot(Data['coll'], Data['ndF']/Data['TdF']**2, 'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$'),
             mlines.Line2D([], [], color='k', ls = '-.', label=r'$\mathrm{ref}$')]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,32)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$\nu/\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'coll.eps')

def plot_nvt(what_arr, doL=0.33):
    # We make an artificial density ramp by letting T_ref be constant. Then n_{u,N} ~ nu_ref,
    # and we can simulate the density ramp by multiplying all densities by nu_ref = Data['coll']
    fig=plt.figure('nvt')
    ax = fig.gca()
    n = 10**np.linspace(-3,3,100)
    plt.plot(n, 6*n**(-0.68),c='k',ls='-.')
    plt.plot(n, 0.1*n**(-0.9),c='grey',ls='-.')
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.loglog(Data['coll'],Data['TuN'], 'C0'+ls)
        plt.plot(Data['ndN']*Data['coll'],Data['TdN'],  'C1'+ls)
        if i>0:
            plt.plot(Data['nuF']*Data['coll'],Data['TuF'],  'C2'+ls)
            plt.plot(Data['ndF']*Data['coll'],Data['TdF'],  'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$')]
             #mlines.Line2D([], [], color='k', ls = '-.', label=r'$n^{-0.68}$'),
             #mlines.Line2D([], [], color='grey', ls = '-.', label=r'$n^{-0.9}$')]
    ax.text(1,10,r'$n^{-0.68}$')
    ax.text(1e-1,1e-1,r'$n^{-0.9}$')
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(2e-3,6e2)
    #plt.ylim(3e-2,5e1)
    plt.xlabel(r'$n \nu_\mathrm{ref}/n_\mathrm{u,N}$')
    plt.ylabel(r'$T/T_\mathrm{ref}$')
    plt.savefig(saveloc+'nvt.eps')

def plot_nvtT(what_arr, doL=0.33):
    # We make an artificial density ramp by letting T_ref be constant. Then n_{u,N} ~ nu_ref,
    # and we can simulate the density ramp by multiplying all densities by nu_ref = Data['coll']
    fig=plt.figure('nvt')
    ax = fig.gca()
    n = 10**np.linspace(-2,2,100)
    plt.plot(n, n**(-0.68),c='k',ls='-.')
    plt.plot(n, n**(-0.9),c='grey')
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.loglog(Data['ndN'],Data['TdN']*Data['coll']**(-0.5),  'C1'+ls)
        if i>0:
            plt.plot(Data['nuF'],Data['TuF']*Data['coll']**(-0.5),  'C2'+ls)
            plt.plot(Data['ndF'],Data['TdF']*Data['coll']**(-0.5),  'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$')]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,32)
    #plt.ylim(0,4)
    plt.xlabel(r'$n/n_\mathrm{u,N}$')
    plt.ylabel(r'$T \nu_\mathrm{ref}^{-1/2}/T_\mathrm{ref}$')
    plt.savefig(saveloc+'nvt_T.eps')

def plot_pressure(what_arr, doL=0.33):
    fig=plt.figure('pressure')
    ax = fig.gca()
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.semilogy(Data['coll'], Data['TuN'], 'C0'+ls)
        #plt.plot(Data['coll'], Data['ndN']*Data['TdN'], 'C1'+ls)
        if i>0:
            plt.plot(Data['coll'], Data['nuF']*Data['TuF'], 'C1'+ls)
        #plt.plot(Data['coll'], Data['ndF']*Data['TdF'], 'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             #mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{u,F}$'),
             #mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$'),
             #mlines.Line2D([], [], color='k', ls = '-.', label=r'$\mathrm{ref}$')]
             ]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,32)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$n T / n_\mathrm{u,N} T_\mathrm{ref}$')
    plt.savefig(saveloc+'pressure.eps')

def plot_heat_fluxes(what_arr, doL=0.33):
    fig=plt.figure('heat_fluxes')
    ax = fig.gca()
    for i,ls in zip([1,2],['--',':']):
        Data = load_data(what_arr[i],doL)
        ax.semilogy(Data['coll'], 2.5*what_arr[i]*XI_FACTOR*Data['coll']*Data['nuF']*Data['TuF'], 'C0'+ls)
        ax.plot(Data['coll'], Data['Qpar'], 'C1'+ls)
        ax.plot(Data['coll'], 3*XI_FACTOR*Data['coll']*Data['ndF']*Data['TdF']**1.5, 'C2'+ls)
        ax.plot(Data['coll'], Data['Qparcond'], 'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$q_{\perp,F}(0)$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$q_{\parallel\mathrm{, tot}}(1)$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$q_{\parallel\mathrm{, conv}}(1)$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$q_{\parallel\mathrm{, cond}}(1)$'),
             ]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,32)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$Q A_{q \parallel}/P_\mathrm{SOL}$')
    plt.savefig(saveloc+'heat_fluxes.eps')


def plot_flux_error(what_arr, doL=0.33,polyorder=5):
    # Compare integrated fluxes to the approximated counterparts
    fig=plt.figure('heat_fluxes')
    ax = fig.gca()
    for i,ls in zip([1,2],['--',':']):
        Data = load_data(what_arr[i],doL)
        
        ax.plot(Data['coll'], np.abs((doL*Data['nuF']*(1-np.exp(-1/doL)))-Data['SFint'])/Data['SFint'], 'C0'+ls)
        ax.plot(Data['coll'], np.abs((doL*Data['nuF']*Data['TuF']*(1-np.exp(-1/doL)))-Data['QFint'])/Data['QFint'], 'C1'+ls)

        SN_approx = (doL*(1-np.exp(-1/doL)))
        SN_linear = doL*(doL*(Data['ndN']-1)+1+np.exp(-1/doL)*(doL-(1+doL)*Data['ndN']))
        SN_polynomial = np.zeros(Data['coll'].size)
        for i in range(Data['coll'].size):
            def integrand(x):
                return np.exp(-x/doL)*(1+(Data['ndN'][i]-1)*x**polyorder)
            SN_polynomial[i],_ = quad(integrand,0,1)
        QN_approx = (doL*Data['TuN']*(1-np.exp(-1/doL)))
        QN_linear = doL*(doL*(Data['ndN']*Data['TdN']-Data['TuN'])+Data['TuN']+np.exp(-1/doL)*(doL*Data['TuN']-(1+doL)*Data['ndN']*Data['TdN']))


        ax.plot(Data['coll'], np.abs(SN_approx-SN_linear)/SN_linear, 'C2'+ls)
        ax.plot(Data['coll'], np.abs(SN_approx-SN_polynomial)/SN_polynomial, 'C3'+ls)
        ax.plot(Data['coll'], np.abs(QN_approx-QN_linear)/QN_linear, 'C4'+ls)
        ax.plot(Data['coll'], np.abs(SN_linear-SN_polynomial)/SN_polynomial, 'C5'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathcal{S}_\mathrm{F}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathcal{Q}_\mathrm{F}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathcal{S}_\mathrm{N,1}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=rf'$\mathcal{{S}}_\mathrm{{N,{polyorder}}}$'),
             mlines.Line2D([], [], color='C4', ls = '-', label=r'$\mathcal{Q}_\mathrm{N}$'),
             ]
    plt.legend(handles=lines)
    #cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,32)
    plt.ylim(0.,1.)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$\mathcal{E}$')
    plt.savefig(saveloc+'flux_error.eps')


def plot_doL_scan():
    doL = [0.01, 0.05, 0.1, 0.2, 0.33, 0.5]
    coll_ref = 16.7
    what_arr = [0.05, 0.1, 0.5, 1.]
    what_lab = [r'$1/20$', r'$1/10$',r'$1/2$',r'$1$',]

    plt.figure('delta scan')
    for what,lab in zip(what_arr,what_lab):
        coll = np.zeros(len(doL))
        for i in range(len(doL)):
            Data = load_data(what,doL[i])
            if doL[i]==0.33:
                ind = np.argmin(np.abs(Data['coll']-coll_ref))
            else: 
                ind = 0
            coll[i] = (Data['ndN']/Data['TdN']**2)[ind]
        plt.loglog(doL,coll, 'o-',label=r'$\hat{\omega}=$'+lab)
    #plt.loglog(doL, doL, 'k:',label=r'$\hat{\delta}$')
    plt.legend()
    plt.xlabel(r'$\hat{\delta}$')
    plt.ylabel(r'$\nu_\mathrm{d,N}/\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'delta_sensitivity_nu_dN.eps')

if __name__=="__main__":
    what_vals = [0.1, 1.]
    lhat_vals = [1., 2.]
    doL_vals = [0.1,0.2]
    P = Plot(lhat_vals=lhat_vals, doL_vals=doL_vals)
    P.find_what_uF_indices(what_vals=what_vals)
    P.plot_good_params_what_uF()
    #P.plot_good_params(truncated=False)
    #P.plot_good_params(truncated=True)
    P.plot_near_SOL(what_uF=False)
    P.plot_near_SOL(what_uF=True)
    P.plot_far_SOL(what_uF=False)
    P.plot_far_SOL(what_uF=True)
    #P.plot_coll()
    #P.compare_what_ref_uF_near()

    if False:
        plot_near_SOL(what_arr,doL=0.1)
        plot_far_SOL(what_arr,doL=0.1)
        plot_TdF_vs_TuN(what_arr,doL=0.1)
        plot_coll(what_arr,doL=0.1)
        plot_nvt(what_arr,doL=0.1)
        plot_pressure(what_arr,doL=0.1)
        plot_wall_flux(what_arr,doL=0.1)
        plot_heat_fluxes(what_arr,doL=0.1)
    #plot_doL_scan()
