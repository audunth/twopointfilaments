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
    plt.xlim(0,30)
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
    #plt.xlim(0,30)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'far.eps')

def plot_TdF_vs_TuN(what_arr, ghat=3.5, xi_factor=0.0711, doL=0.33):
    # ghat and xi_factor are for tau=1 and deuterium
    plt.figure('TdF vs TuN')
    for i,ls in enumerate(['-','--',':']):
        Data = load_data(what_arr[i],doL)
        plt.loglog(Data['coll'], Data['TdF']/Data['TuN'], 'C0'+ls)
    plt.loglog(Data['coll'],2.5/(ghat*xi_factor*Data['coll']),'k',label=r'$5 /(2 \hat{\gamma}\xi)$')
    plt.legend()
    #plt.xlim(0,30)
    #plt.ylim(0,4)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel(r'$T_\mathrm{d,F}/T_\mathrm{u,N}$')
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
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.savefig(saveloc+'wall_flux.eps')

def plot_data(what,doL):
    Data = load_data(what,doL)
    
    plt.figure('wall fluxes')
    plt.title('wall fluxes')
    plt.plot(Data['coll'], Data['nuF'], label='Gamma/(v/lam)')
    plt.plot(Data['coll'], Data['nuF']*Data['TuF'],label='q/(5/2 v/lam)')
    plt.legend()
    plt.xlabel('nu*')
    plt.savefig(saveloc+'wall.png')

    plt.figure('far nd vs td')
    plt.title('far nd td')
    plt.loglog(Data['coll'], Data['ndF']*Data['TdF']**0.5)
    plt.xlabel('ndf')
    plt.ylabel('tdf')
    plt.savefig(saveloc+'nvt.png')

if __name__=="__main__":
    what_arr = [0.,0.05,0.5]
    plot_near_SOL(what_arr)
    plot_far_SOL(what_arr)
    plot_TdF_vs_TuN(what_arr)
    plot_wall_flux(what_arr)
