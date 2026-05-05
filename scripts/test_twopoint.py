import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import cosmoplots
plt.style.use(["cosmoplots.default"])

# Here, we check that the different calculations of the near-SOL in the 
# classic Two-Point model and the Two-Box extension give the same result for what=0.

dataloc='/home/ath019/Documents/manuscripts/two-point/code/data/'
saveloc='/home/ath019/Documents/manuscripts/two-point/code/figures/'

def load_data(what=0.0,doL=0.33):
    TBfile = dataloc+f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz"
    TPfile = dataloc+"TwoPoint.npz"
    return np.load(TPfile), np.load(TBfile)

def plot_near_SOL(what=0.0, doL=0.33):
    plt.figure('near sol')
    TwoPoint, TB = load_data(what,doL)
    plt.plot(TwoPoint['coll'], TwoPoint['ndN']/TB['ndN'], 'C0', label=r'ndN')
    plt.plot(TwoPoint['coll'], TwoPoint['TdN']/TB['TdN'], 'C1--', label='TdN')
    plt.plot(TwoPoint['coll'], TwoPoint['TuN']/TB['TuN'], 'C2:', label='TuN')
    plt.legend()
    plt.ylim(0,2)
    plt.xlabel(r'$\nu_\mathrm{ref}$')
    plt.ylabel('Classic Two-Point / Two-Box')
    plt.savefig(saveloc+'near_test.eps')


if __name__=="__main__":
    plot_near_SOL()
