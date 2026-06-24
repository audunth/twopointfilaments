import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cosmoplots
plt.style.use(["cosmoplots.default"])

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

def load_data(what,doL):
    file = dataloc+f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz"
    return np.load(file)

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
    plt.xlim(1,26)
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
    plt.xlim(1,26)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'far.eps')

def plot_TdF_vs_TuN(what_arr, ghat=3.5, xi_factor=0.0711, doL=0.33):
    # ghat and xi_factor are for tau=1 and deuterium
    fig=plt.figure('TdF vs TuN')
    ax = fig.gca()
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.loglog(Data['coll'], Data['TdF']/Data['TuN'], 'C0'+ls)
        #plt.loglog(Data['coll'], Data['TdN']/Data['TuN'], 'C1'+ls)
        plt.loglog(Data['coll'], Data['TdF'], 'C1'+ls)
    plt.loglog(Data['coll'],2.5/(ghat*xi_factor*Data['coll']),'k-.')
    #plt.loglog(Data['coll'][Data['coll']<7],3*Data['coll'][Data['coll']<7]**(-0.7),c='grey',ls='-.')
    #ax.text(3,0.7,r'$\nu_\mathrm{ref}^{-0.7}$',c='grey')
    cosmoplots.change_log_axis_base(ax, base=10)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$T_\mathrm{d,F}/T_\mathrm{ref}$'),
             mlines.Line2D([], [], color='k', ls = '-.', label=r'$5 /(2 \hat{\gamma}\xi)$')]
    plt.legend(handles=lines)
    plt.xlim(1,26)
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
        ax.plot(Data['coll'], Data['nuF']*Data['TuF'], 'C1'+ls)

    cosmoplots.change_log_axis_base(ax, base=10)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$S_F(0)/\hat{\omega}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$2 Q_F(0)/(5\hat{\omega})$')]
    plt.legend(handles=lines)
    plt.xlim(1,26)
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
        plt.plot(Data['coll'], Data['nuF']/Data['TuF']**2, 'C2'+ls)
        plt.plot(Data['coll'], Data['ndF']/Data['TdF']**2, 'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$'),
             mlines.Line2D([], [], color='k', ls = '-.', label=r'$\mathrm{ref}$')]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    plt.xlim(1,26)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$\nu/\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'coll.eps')

def plot_nvt(what_arr, ghat=3.5, xi_factor=0.0711, doL=0.33):
    # ghat and xi_factor are for tau=1 and deuterium

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
    plt.xlim(2e-3,6e2)
    plt.ylim(3e-2,5e1)
    plt.xlabel(r'$n \nu_\mathrm{ref}/n_\mathrm{u,N}$')
    plt.ylabel(r'$T/T_\mathrm{ref}$')
    plt.savefig(saveloc+'nvt.eps')

def plot_nvtT(what_arr, ghat=3.5, xi_factor=0.0711, doL=0.33):
    # ghat and xi_factor are for tau=1 and deuterium

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
        plt.plot(Data['nuF'],Data['TuF']*Data['coll']**(-0.5),  'C2'+ls)
        plt.plot(Data['ndF'],Data['TdF']*Data['coll']**(-0.5),  'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
             mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$')]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,26)
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
        plt.plot(Data['coll'], Data['nuF']*Data['TuF'], 'C2'+ls)
        #plt.plot(Data['coll'], Data['ndF']*Data['TdF'], 'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$\mathrm{u,N}$'),
             #mlines.Line2D([], [], color='C1', ls = '-', label=r'$\mathrm{d,N}$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$\mathrm{u,F}$'),
             #mlines.Line2D([], [], color='C3', ls = '-', label=r'$\mathrm{d,F}$'),
             #mlines.Line2D([], [], color='k', ls = '-.', label=r'$\mathrm{ref}$')]
             ]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    plt.xlim(1,26)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$n T / n_\mathrm{u,N} T_\mathrm{ref}$')
    plt.savefig(saveloc+'pressure.eps')

def plot_heat_fluxes(what_arr, doL=0.33):
    fig=plt.figure('heat_fluxes')
    ax = fig.gca()
    for i,ls in zip([1,2],['--',':']):
        Data = load_data(what_arr[i],doL)
        ax.semilogy(Data['coll'], 2.5*what_arr[i]*Data['nuF']*Data['TuF'], 'C0'+ls)
        ax.plot(Data['coll'], Data['Qpar'], 'C1'+ls)
        ax.plot(Data['coll'], 3*Data['ndF']*Data['TdF']**1.5, 'C2'+ls)
        #ax.plot(Data['coll'], np.abs(Data['Qpar']-3*Data['ndF']*Data['TdF']**1.5), 'C3'+ls)
    lines = [mlines.Line2D([], [], color='C0', ls = '-', label=r'$Q_F(0)$'),
             mlines.Line2D([], [], color='C1', ls = '-', label=r'$Q_{\parallel\mathrm{, tot}}(1)$'),
             mlines.Line2D([], [], color='C2', ls = '-', label=r'$Q_{\parallel\mathrm{, conv}}(1)$'),
             #mlines.Line2D([], [], color='C3', ls = '-', label=r'$\left| Q_{\parallel\mathrm{, cond}}\right|$'),
             ]
    plt.legend(handles=lines)
    cosmoplots.change_log_axis_base(ax, base=10)
    #plt.xlim(1,26)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$Q A_{q \parallel}/P_\mathrm{SOL}$')
    plt.savefig(saveloc+'heat_fluxes.eps')

if __name__=="__main__":
    what_arr = [0.,0.05,0.5]
    #plot_near_SOL(what_arr)
    #plot_far_SOL(what_arr)
    plot_TdF_vs_TuN(what_arr)
    #plot_wall_flux(what_arr)
    #plot_coll(what_arr)
    #plot_nvt(what_arr)
    #plot_pressure(what_arr)
    plot_heat_fluxes(what_arr)
