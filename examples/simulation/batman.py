import numpy as np

def wfn(x):
    return 3*np.sqrt(1-(x/7)**2)
def nwfn(x):
    return -3*np.sqrt(1-(x/7)**2)
def hfn(x):
    return (1/2)*(3*(np.abs(x - 1/2) + np.abs(x + 1/2) + 6) - \
                  11*(np.abs(x - 3/4) + np.abs(x + 3/4)))
def lfn(x):
    return (6/7)*np.sqrt(10) + (3 + x)/2 - (3/7)*np.sqrt(10)* np.sqrt(4 - (x + 1)**2)
def rfn(x):
    return (6/7)*np.sqrt(10) + (3 - x)/2 - (3/7)*np.sqrt(10)* np.sqrt(4 - (x - 1)**2)
def upper_f(x):
    return np.piecewise(x, [(x > -3) & (x < -1),
                            (x >= -1) & (x < 1), (x >= 1) & (x < 3),
                            (x <= -3) | (x >= 3)], [lfn, hfn, rfn, wfn])
def cfn(x):
    return .5*( abs(.5*x) + np.sqrt(1-(abs(abs(x)-2)-1)**2) - 1/112*(3*np.sqrt(33)-7)*x**2 \
               + 3*np.sqrt(1-(1/7*x)**2) - 3 ) * \
           (np.sign(x+4)-np.sign(x-4)) - 3*np.sqrt(1-(1/7*x)**2)
def lower_g(x):
    return np.piecewise(x, [(x<-4) | (x>4), (x>=-4) & (x<=4)], [nwfn, cfn] )

def batman(x, y, scale = (1, 1), shift = (0, 0)):
    z = np.zeros(x.shape, dtype=bool)
    indx = ((x-shift[0])/scale[0] < 7) & ((x-shift[0])/scale[0] > -7)
    z[indx] = (y[indx]-shift[1] < scale[1] * upper_f((x[indx]-shift[0])/scale[0])) & \
              (y[indx]-shift[1] > scale[1] * lower_g((x[indx]-shift[0])/scale[0]))
    return z
