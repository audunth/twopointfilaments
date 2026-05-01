import numpy as np
from twopointfilaments.twobox import TBTPW
"""
Here, we run the TBTPW model for the largest possible range of collisionality
and several choices of the what parameter,
"""

saveloc = "../data/"

doL = 0.33
what = 0.05

# collmax[doL][what]
collmax = {0.33: {0.5:17.2,
                  0.1:22.9,
                  0.05:26.1,
                  0.01:35.8}}

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
         ndN=D[0,:], TuN=D[1,:], TdN=D[2,:],
         nuF=D[3,:], ndF=D[4,:], TuF=D[5,:], TdF=D[6,:])
