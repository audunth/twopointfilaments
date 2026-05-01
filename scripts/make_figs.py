import numpy as np
import matplotlib.pyplot as plt
import cosmoplots

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

def load_data(what,doL):
    file = dataloc+f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz"
    return np.load(file)


def plot_data(what,doL):
    Data = load_data(what,doL)
    plt.figure('near')
    plt.title('near')
    plt.plot(Data['coll'], Data['ndN'], label='ndN/nuN')
    plt.plot(Data['coll'], Data['TdN']/Data['TuN'], label='TdN/TuN')
    plt.plot(Data['coll'], Data['TuN'], label='TuN')
    plt.legend()
    plt.xlabel('nu*')
    plt.savefig(saveloc+'near.png')

    plt.figure('far')
    plt.title('far')
    plt.plot(Data['coll'], Data['ndF']/Data['nuF'], label='ndF/nuF')
    plt.plot(Data['coll'], Data['TdF']/Data['TuF'], label='TdF/TuF')
    plt.legend()
    plt.xlabel('nu*')
    plt.savefig(saveloc+'far.png')
    
    plt.figure('T')
    plt.title('T')
    plt.loglog(Data['coll'], Data['coll']*(Data['TdF']/Data['TuN']))
    #plt.plot(Data['coll'],8/Data['coll'],label='8/nu*')
    plt.legend()
    plt.xlabel('nu*')
    plt.ylabel('TdF/TuN')
    plt.savefig(saveloc+'T.png')
    
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

plot_data(0.05,0.33)
