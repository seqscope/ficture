import numpy as np

def pixel_to_hex(pts, size, offset_x = 0, offset_y = 0):
    n,d = pts.shape
    assert d == 2
    mtx = np.array([[np.sqrt(3)/3,-1/3],[0,2/3]])
    hex_frac = mtx @ pts.transpose()
    hex_frac /= size
    hex_frac[0,] += offset_x
    hex_frac[1,] += offset_y
    rx = np.round(hex_frac[0,])
    ry = np.round(hex_frac[1,])
    hex_frac[0,] -= rx
    hex_frac[1,] -= ry
    indx = np.abs(hex_frac[0,]) < np.abs(hex_frac[1,])
    x = rx+np.round(hex_frac[0,]+0.5*hex_frac[1,])
    y = ry
    x[indx] = rx[indx]
    y[indx] = ry[indx]+np.round(hex_frac[1,indx]+0.5*hex_frac[0,indx])
    return x.astype(int), y.astype(int)

def hex_to_pixel(x,y,size,offset_x=0,offset_y=0):
    n = len(x)
    assert len(y) == n
    ptx = size * (np.sqrt(3) * (x-offset_x) + np.sqrt(3)/2 * (y-offset_y))
    pty = size * 3/2 * (y-offset_y)
    return ptx, pty
