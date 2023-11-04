import sys, os, warnings, copy
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture

from ficture.utils.hexagon_fn import pixel_to_hex, hex_to_pixel


def filter_by_density_mixture(df, key, radius, n_move, args):
    pt = pd.DataFrame()
    m0v=[]
    m1v=[]
    hex_area = radius**2 * np.sqrt(3) * 3 / 2
    for i in range(n_move):
        for j in range(n_move):
            cnt = pd.DataFrame()
            cnt["hex_x"], cnt["hex_y"] = pixel_to_hex(np.asarray(df.loc[:, ['X','Y']]), radius, i/n_move, j/n_move)
            cnt[key] = copy.copy(df[key].values)
            cnt = cnt.groupby(by = ['hex_x','hex_y']).agg({key:sum}).reset_index()
            cnt = cnt[cnt[key] > hex_area * args.min_abs_mol_density_squm]
            if cnt.shape[0] < 10:
                continue
            if args.hard_threshold > 0:
                cnt['det'] = cnt[key] > hex_area * args.hard_threshold
            else:
                v = np.log10(cnt[key].values).reshape(-1, 1)
                print(f"{len(v)}")
                if len(v) > args.max_npts_to_fit_model:
                    indx = np.random.choice(len(v), int(args.max_npts_to_fit_model), replace=False)
                    gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v[indx])
                else:
                    gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v)
                lab_keep = np.argmax(gm.means_.squeeze())
                cnt['det'] = gm.predict(v) == lab_keep
                m0=(10**gm.means_.squeeze()[lab_keep])/hex_area
                m1=(10**gm.means_.squeeze()[1-lab_keep])/hex_area
                if m1 > m0 * 0.5 or m0 < args.min_abs_mol_density_squm:
                    v = cnt[key].values.reshape(-1, 1)
                    if len(v) > args.max_npts_to_fit_model:
                        indx = np.random.choice(len(v), int(args.max_npts_to_fit_model), replace=False)
                        gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v[indx])
                    else:
                        gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v)
                    lab_keep = np.argmax(gm.means_.squeeze())
                    cnt['det'] = gm.predict(v) == lab_keep
                    m0=gm.means_.squeeze()[lab_keep]/hex_area
                    m1=gm.means_.squeeze()[1-lab_keep]/hex_area
                if m0 < args.min_abs_mol_density_squm:
                    continue
                if m1 > m0 * .7:
                    cnt['det'] = 1
                m0v.append(m0)
                m1v.append(m1)
            if cnt.det.eq(True).sum() < 2:
                continue
            m0 = cnt.loc[cnt.det.eq(True), key].median()/hex_area
            m1 = cnt.loc[cnt.det.eq(False), key].median()/hex_area
            m0v.append(m0)
            m1v.append(m1)
            anchor_x, anchor_y = hex_to_pixel(cnt.loc[cnt.det.eq(True), 'hex_x'].values, cnt.loc[cnt.det.eq(True), 'hex_y'].values, radius,i/n_move,j/n_move)
            pt = pd.concat([pt,\
                    pd.DataFrame({'x':anchor_x, 'y':anchor_y})])
    return pt, np.mean(m0v), np.mean(m1v)
