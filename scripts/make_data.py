import numpy as np
from twopointfilaments import TBTPW
"""
Here, we run the TBTPW model for the largest possible range of collisionality
and several choices of the what parameter,
"""

# collmax[doL][what]
# There is no theoretical limit at what=0.
collmax = {0.33: {0.5:17.2,
                  0.1:22.9,
                  0.05:26.1,
                  0.01:35.8,
                  0.0: 40}}

def save_data(what,doL=0.33,saveloc='/home/ath019/Documents/manuscripts/two-point/code/data/'):
    coll = np.linspace(1.,collmax[doL][what],100)

    D = np.zeros([7,coll.size])

    for i in range(coll.size):
        S = TBTPW(coll[i],what,doL=doL)
        D[0,i] = S.ndN
        D[1,i] = S.TuN
        D[2,i] = S.TdN
        D[3,i] = S.nuF
        D[4,i] = S.ndF
        D[5,i] = S.TuF
        D[6,i] = S.TdF

    np.savez(saveloc + f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:],
             nuF=D[3,:], ndF=D[4,:], TuF=D[5,:], TdF=D[6,:])


if __name__=="__main__":
    what_arr = [0.0, 0.05, 0.5]
    for what in what_arr:
        save_data(what)
