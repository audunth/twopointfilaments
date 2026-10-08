import numpy as np
from twopointfilaments import TBTPW,TwoPoint

"""
Here, we run the TBTPW model for the largest possible range of collisionality
and several choices of the what parameter,
"""

saveloc='/home/ath019/Documents/manuscripts/two-point/code/data/'

def save_scan(coll, what, lhat, doL):
    P = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    ndN = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    TuN = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    TdN = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    nuF = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    ndF = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    TuF = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan
    TdF = np.ones([len(coll),len(what),len(lhat),len(doL)])*np.nan

    for ci in range(len(coll)):
        print(f'nuref={coll[ci]}',flush=True)
        for wk in range(len(what)):
            for Lj in range(len(lhat)):
                for dl in range(len(doL)):
                    Z = TBTPW(coll[ci], what[wk], lhat=lhat[Lj], doL=doL[dl])
                    P[ci,wk,Lj,dl] = Z.status
                    # Both OK
                    if Z.status == 0:
                        ndN[ci,wk,Lj,dl]=Z.ndN
                        TuN[ci,wk,Lj,dl]=Z.TuN
                        TdN[ci,wk,Lj,dl]=Z.TdN
                        nuF[ci,wk,Lj,dl]=Z.nuF
                        ndF[ci,wk,Lj,dl]=Z.ndF
                        TuF[ci,wk,Lj,dl]=Z.TuF
                        TdF[ci,wk,Lj,dl]=Z.TdF
                    # Only Near-SOL OK
                    elif Z.status < -3:
                        ndN[ci,wk,Lj,dl]=Z.ndN
                        TuN[ci,wk,Lj,dl]=Z.TuN
                        TdN[ci,wk,Lj,dl]=Z.TdN
                    # Far-SOL

    np.savez(saveloc + "TBTPW_scan.npz",
             coll=coll, what=what, lhat=lhat, doL=doL,
             P=P,
             ndN=ndN, TuN=TuN, TdN=TdN,
             nuF=nuF, ndF=ndF, TuF=TuF, TdF=TdF)


def save_data(coll, what,doL=0.33, lhat=5):
    D = np.ones([11,coll.size])*np.nan

    for i in range(coll.size):
        S = TBTPW(coll[i],what,doL=doL, lhat=lhat)
        #x, n, M, T = S.get_far_sol_profiles(xpoints=1000)
        if S.status == 0:
            D[0,i] = S.ndN
            D[1,i] = S.TuN
            D[2,i] = S.TdN
            D[3,i] = S.nuF
            D[4,i] = S.ndF
            D[5,i] = S.TuF
            D[6,i] = S.TdF
            D[7,i] = S.ghat*S.xi*S.ndF*S.TdF**1.5 # Parallel heat flux to target
            D[8,i] = S.xi*(S.ghat-3)*S.ndF*S.TdF**1.5 # Parallel conductive heat flux to target
        #D[9,i] = np.trapezoid((np.exp(-x/doL)*n)[::-1],x[::-1]) # Integrated perpendicular particle flux (except constants)
        #D[10,i] = np.trapezoid((np.exp(-x/doL)*n*T)[::-1],x[::-1]) # Integrated perpendicular heat flux (except constants)

    np.savez(saveloc + f"TBTPW_what{what:.2f}_doL{doL:.2f}_lhat{lhat:.2f}.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:],
             nuF=D[3,:], ndF=D[4,:], TuF=D[5,:], TdF=D[6,:], 
             Qpar=D[7,:], Qparcond=D[8,:])#, SFint=D[9,:], QFint=D[10,:])

def save_data_twopoint(coll, what=0, doL=0.33, lhat=5):
    D = np.zeros([3,coll.size])

    for i in range(coll.size):
        S = TwoPoint(coll[i])
        D[0,i] = S.ndN
        D[1,i] = S.TuN
        D[2,i] = S.TdN

    np.savez(saveloc + "TwoPoint.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:])

if __name__=="__main__":
    coll = 10**np.linspace(-1,2,2**5)
    what = 10**np.linspace(-3,2,2**5)
    lhat = np.array([1.,2.])#,5.,10.])
    doL = np.array([1e-1, 2e-1])#np.array([2e-2,5e-2,1e-1,2e-1])
    
    save_scan(coll, what, lhat, doL)
    save_data_twopoint(coll)

    #for what in what_arr:
    #    for doL in doL_arr:
    #        save_data(coll, what=what, doL=doL, lhat=lhat)
