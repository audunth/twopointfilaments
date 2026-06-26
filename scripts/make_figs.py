import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cosmoplots
plt.style.use(["cosmoplots.default"])
from twopointfilaments import xi_factor
from scipy.integrate import quad

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

# These values are assumed for all plots
GHAT = 3.5
XI_FACTOR = xi_factor(1,'deuterium')

# collmax[doL][what]
# There is no theoretical limit at what=0. Above 1, the limit is very low. Below 0.5, the limit is very large.
collmax = {0.33: {1.0:29.9,
                  0.5:57.8,
                  0.1:60.,
                  0.05:60.,
                  0.01:60.,
                  0.0: 60.},
           0.2: {1.0:17.1,
                 0.5:30.6,
                 0.1:60.,
                 0.05:60.,
                 0.01:60.,
                 0.0: 60.},
           }

def load_data(what,doL):
    if what == 0.:
        file = "TwoPoint.npz"
    else:
        file = f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz"
    return np.load(dataloc+file)

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
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$S_F(0)/\hat{\omega}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$2 Q_F(0)/(5\hat{\omega})$')]
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
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$Q_F(0)$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$Q_{\parallel\mathrm{, tot}}(1)$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$Q_{\parallel\mathrm{, conv}}(1)$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$Q_{\parallel\mathrm{, cond}}(1)$'),
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
        
        ax.plot(Data['coll'], (doL*Data['nuF']*(1-np.exp(-1/doL)))/Data['SFint'], 'C0'+ls)
        ax.plot(Data['coll'], (doL*Data['nuF']*Data['TuF']*(1-np.exp(-1/doL)))/Data['QFint'], 'C1'+ls)

        SN_approx = (doL*(1-np.exp(-1/doL)))
        SN_linear = doL*(doL*(Data['ndN']-1)+1+np.exp(-1/doL)*(doL-(1+doL)*Data['ndN']))
        SN_polynomial = np.zeros(Data['coll'].size)
        for i in range(Data['coll'].size):
            def integrand(x):
                return np.exp(-x/doL)*(1+(Data['ndN'][i]-1)*x**polyorder)
            SN_polynomial[i],_ = quad(integrand,0,1)
        QN_approx = (doL*Data['TuN']*(1-np.exp(-1/doL)))
        QN_linear = doL*(doL*(Data['ndN']*Data['TdN']-Data['TuN'])+Data['TuN']+np.exp(-1/doL)*(doL*Data['TuN']-(1+doL)*Data['ndN']*Data['TdN']))


        ax.plot(Data['coll'], SN_approx/SN_linear, 'C2'+ls)
        ax.plot(Data['coll'], SN_approx/SN_polynomial, 'C3'+ls)
        ax.plot(Data['coll'], QN_approx/QN_linear, 'C4'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathcal{S}_\mathrm{F}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathcal{Q}_\mathrm{F}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathcal{S}_\mathrm{N,1}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=rf'$\mathcal{{S}}_\mathrm{{N,{polyorder}}}$'),
             mlines.Line2D([], [], color='C4', ls = '-', label=r'$\mathcal{Q}_\mathrm{N}$'),
             ]
    plt.legend(handles=lines)
    #cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,32)
    plt.ylim(0.,1.2)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$\mathcal{E}$')
    plt.savefig(saveloc+'flux_error.eps')


def plot_solvability():
    doL=0.33
    nu = np.linspace(15,55,100)
    what_log = np.linspace(-3,1.1,100)

    Nu, What_log = np.meshgrid(nu,what_log,indexing='ij')
    Xi = XI_FACTOR*Nu
    
    def cond(doL):
        return Xi**3.5*GHAT**2.5*(GHAT*Xi-3)*10**(What_log) - (7/4)*2.5**5.2/(doL*np.cosh(1/doL)**2)
    plt.figure('condition')
    plt.axvline(16.8,color='grey',ls=':')
    plt.contourf(Nu, What_log, cond(doL),levels=[0.,1e8],cmap="bone")
    plt.contour(Nu,What_log,cond(doL),levels=[0.,],colors='k')
    plt.contour(Nu,What_log,cond(0.2),levels=[0.,],colors='C0')
    plt.contour(Nu,What_log,cond(0.5),levels=[0.,],colors='C1')
    
    what_num_log = np.zeros(7)
    nu_num = np.zeros(7)

    for i,w in enumerate(collmax[doL].keys()):
        if w>0:
            what_num_log[i] = np.log10(w)
            nu_num[i] = collmax[doL][w]
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\hat{\delta}=1/5$'),
             mlines.Line2D([], [], color='k', ls = '-', label=r'$\hat{\delta}=1/3$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\hat{\delta}=1/2$'),
             ]
    plt.legend(handles=lines)
    plt.scatter(nu_num,what_num_log, s=6,color='w', edgecolor='C2')
    plt.yticks([-3,-2,-1,0,1],labels=[r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$1$',r'$10$'])
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$\hat{\omega}$')
    plt.savefig(saveloc+'condition.eps')

def plot_delta_condition():
    doL = 10.**np.linspace(-3,0,100)
    cond = (7/4)*2.5**5.2/(doL*np.cosh(1/doL)**2)
    plt.figure('delta condition')
    plt.semilogx(doL,cond)
    plt.xlabel(r'$\hat{\delta}$')
    plt.ylabel(r'$(7/4)(5/2)^{5/2} \mathrm{sech}(1/\hat{\delta})^2/\hat{\delta}$')
    plt.savefig(saveloc+'delta_condition.eps')


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
    what_arr = [0.,0.05,0.5]
    plot_near_SOL(what_arr,doL=0.2)
    plot_far_SOL(what_arr,doL=0.2)
    plot_TdF_vs_TuN(what_arr,doL=0.2)
    plot_wall_flux(what_arr,doL=0.2)
    plot_coll(what_arr,doL=0.2)
    plot_nvt(what_arr,doL=0.2)
    plot_pressure(what_arr,doL=0.2)
    plot_heat_fluxes(what_arr,doL=0.2)
    plot_flux_error(what_arr,doL=0.33)
    #plot_solvability()
    #plot_delta_condition()
    #plot_doL_scan()
