import numpy as np
from twopointfilaments import TBTPW,TwoPoint
"""
Here, we run the TBTPW model for the largest possible range of collisionality
and several choices of the what parameter,
"""

saveloc='/home/ath019/Documents/manuscripts/two-point/code/data/'

# collmax[doL][what]
# There is no theoretical limit at what=0. Above 10, the limit is just ghat*xi=3 -> coll = 16.8
collmax = {0.33: {10.0:16.8,
                  5.0:17.1,
                  1.0:21.2,
                  0.5:24.0,
                  0.1:32.0,
                  0.05:36.1,
                  0.01:49.8,
                  0.0: 50}}

def save_data(what,doL=0.33):
    coll = np.linspace(1.,collmax[doL][what],100)

    D = np.zeros([8,coll.size])

    for i in range(coll.size):
        S = TBTPW(coll[i],what,doL=doL)
        D[0,i] = S.ndN
        D[1,i] = S.TuN
        D[2,i] = S.TdN
        D[3,i] = S.nuF
        D[4,i] = S.ndF
        D[5,i] = S.TuF
        D[6,i] = S.TdF
        D[7,i] = S.ghat*S.xi*S.ndF*S.TdF**1.5 # Parallel heat flux to target

    np.savez(saveloc + f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:],
             nuF=D[3,:], ndF=D[4,:], TuF=D[5,:], TdF=D[6,:], Qpar=D[7,:])

def save_data_twopoint(what=0, doL=0.33):
    coll = np.linspace(1.,collmax[doL][what],100)

    D = np.zeros([3,coll.size])

    for i in range(coll.size):
        S = TwoPoint(coll[i])
        D[0,i] = S.ndN
        D[1,i] = S.TuN
        D[2,i] = S.TdN

    np.savez(saveloc + "TwoPoint.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:])

if __name__=="__main__":
    save_data_twopoint()
    what_arr = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    for what in what_arr:
        save_data(what)
