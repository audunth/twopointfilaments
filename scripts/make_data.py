import numpy as np
from twopointfilaments import TBTPW,TwoPoint

"""
Here, we run the TBTPW model for the largest possible range of collisionality
and several choices of the what parameter,
"""

saveloc='/home/ath019/Documents/manuscripts/two-point/code/data/'

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
    coll = np.linspace(1,60,100)
    doL_arr = [5e-2, 1e-1, 2e-1]
    lhat = 5
    what_arr = [1e-1, 1, 1e1, 1e2]
    
    save_data_twopoint(coll)

    for what in what_arr:
        for doL in doL_arr:
            save_data(coll, what=what, doL=doL, lhat=lhat)
