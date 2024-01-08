import numpy as np
import pandas as pd

### Transform between cartesian and hexagon coordinates
### "pointy-top" orientation & axial coordinate
### (https://www.redblobgames.com/grids/hexagons/)

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
    if hasattr(x, "__len__"):
        assert hasattr(y, "__len__")
        x = np.array(x)
        y = np.array(y)
        assert len(y) == len(x)
    ptx = size * (np.sqrt(3) * (x-offset_x) + np.sqrt(3)/2 * (y-offset_y))
    pty = size * 3/2 * (y-offset_y)
    return ptx, pty

def collapse_to_hex(df, hex_width = -1, radius = -1, n_move = 1, key = "Count", ):
    if hex_width > 0:
        radius = hex_width / np.sqrt(3)
    assert radius > 0
    df.rename(columns={"x":"X","y":"Y"}, inplace=True)
    assert "X" in df.columns and "Y" in df.columns
    brc = pd.DataFrame()
    for i in range(n_move):
        for j in range(n_move):
            cnt = pd.DataFrame()
            cnt["hex_x"], cnt["hex_y"] = pixel_to_hex(df.loc[:, ['X','Y']].values, radius, i/n_move, j/n_move)
            cnt[key] = df[key].values
            cnt = cnt.groupby(by = ['hex_x','hex_y']).agg({key:sum}).reset_index()
            cnt["offset"] = f"{i}_{j}"
            cnt["ID"] = list(zip(cnt.hex_x, cnt.hex_y, cnt.offset))
            cnt["x"], cnt["y"] = hex_to_pixel(cnt.hex_x, cnt.hex_y, radius, i/n_move, j/n_move)
            cnt.drop(columns=["hex_x","hex_y","offset"], inplace=True)
            brc = pd.concat([brc, cnt.reset_index(drop=True)])
    return brc
