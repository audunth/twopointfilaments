import numpy as np
from twopointfilaments import TBTPW,TwoPoint
"""
Here, we run the TBTPW model for the largest possible range of collisionality
and several choices of the what parameter,
"""

saveloc='/home/ath019/Documents/manuscripts/two-point/code/data/'

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

def save_data(coll, what,doL=0.33):
    D = np.zeros([11,coll.size])

    for i in range(coll.size):
        S = TBTPW(coll[i],what,doL=doL)
        x, n, M, T = S.get_far_sol_profiles(xpoints=1000)

        D[0,i] = S.ndN
        D[1,i] = S.TuN
        D[2,i] = S.TdN
        D[3,i] = S.nuF
        D[4,i] = S.ndF
        D[5,i] = S.TuF
        D[6,i] = S.TdF
        D[7,i] = S.ghat*S.xi*S.ndF*S.TdF**1.5 # Parallel heat flux to target
        D[8,i] = (-3.5*S.TdF**2.5*np.diff(T[:2])/np.diff(x[:2]))[0] # Parallel conductive heat flux to target
        D[9,i] = np.trapezoid((np.exp(-x/doL)*n)[::-1],x[::-1]) # Integrated perpendicular particle flux (except constants)
        D[10,i] = np.trapezoid((np.exp(-x/doL)*n*T)[::-1],x[::-1]) # Integrated perpendicular heat flux (except constants)

    np.savez(saveloc + f"TBTPW_what{what:.2f}_doL{doL:.2f}.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:],
             nuF=D[3,:], ndF=D[4,:], TuF=D[5,:], TdF=D[6,:], 
             Qpar=D[7,:], Qparcond=D[8,:], SFint=D[9,:], QFint=D[10,:])

def save_data_twopoint(coll, what=0, doL=0.33):
    D = np.zeros([3,coll.size])

    for i in range(coll.size):
        S = TwoPoint(coll[i])
        D[0,i] = S.ndN
        D[1,i] = S.TuN
        D[2,i] = S.TdN

    np.savez(saveloc + "TwoPoint.npz",
             coll=coll, ndN=D[0,:], TuN=D[1,:], TdN=D[2,:])

if __name__=="__main__":
    if False:
        # doL scan
        doL = 0.5
        what_arr = [0.05, 0.1, 0.5, 1.]
        coll = np.array([16.7,])
        for what in what_arr:
            save_data(coll, what, doL)

    

    if True:
        # What - coll scan
        doL = 0.2
        save_data_twopoint(np.linspace(1.,collmax[doL][0.],100))

        what_arr = [0.01, 0.05, 0.1, 0.5, 1.]
        for what in what_arr:
            coll = np.linspace(1.,collmax[doL][what],100)
            save_data(coll, what, doL)
