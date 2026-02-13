import sys, os, warnings, copy
import numpy as np
import pandas as pd
import sklearn.neighbors
import sklearn.mixture

from ficture.utils.hexagon_fn import pixel_to_hex, hex_to_pixel


def filter_by_density_mixture(df, key, radius, n_move, args):
    '''
    df: dataframe with columns X, Y, and key, (X, Y) are in um
    Return a dataframe with kept anchor point coordinates x, y
    '''
    pt = pd.DataFrame()
    m0v=[]
    m1v=[]
    hex_area = radius**2 * np.sqrt(3) * 3 / 2
    for i in range(n_move):
        for j in range(n_move):
            cnt = pd.DataFrame()
            cnt["hex_x"], cnt["hex_y"] = pixel_to_hex(np.asarray(df.loc[:, ['X','Y']]), radius, i/n_move, j/n_move)
            cnt[key] = copy.copy(df[key].values)
            cnt = cnt.groupby(by = ['hex_x','hex_y']).agg({key:"sum"}).reset_index()
            cnt = cnt[cnt[key] > hex_area * args.min_abs_mol_density_squm]
            if cnt.shape[0] < 10:
                continue
            if args.hard_threshold > 0:
                cnt['det'] = cnt[key] > hex_area * args.hard_threshold
            else:
                vorg = cnt[key].values
                if len(vorg) > args.max_npts_to_fit_model:
                    vorg = np.random.choice(vorg, int(args.max_npts_to_fit_model), replace=False)
                print(f"[{i}, {j}], {len(vorg)} units")
                # 1st try: fit a 2-component mixture model to log transformed density
                v = np.log10(vorg).reshape(-1, 1)
                gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v)
                lab_keep = np.argmax(gm.means_.squeeze())
                cnt['det'] = gm.predict(v) == lab_keep
                m0=(10**gm.means_.squeeze()[lab_keep])/hex_area
                m1=(10**gm.means_.squeeze()[1-lab_keep])/hex_area
                kept_min = cnt.loc[cnt.det.eq(True), key].min() / hex_area
                print(f"1st: log, 2 component. {m0:.3f} v.s. {m1:.3f}, kept min density {kept_min:.3f}")
                # If it does not seem right
                # 2nd try: fit a 3-component mixture model to log transformed density
                if m1 > m0 * 0.5 or m0 < args.min_abs_mol_density_squm_dense:
                    v = np.log10(vorg)
                    gm = sklearn.mixture.GaussianMixture(n_components=3, random_state=0).fit(v)
                    lab_rank = np.argsort(gm.means_.squeeze())
                    lab_keep = lab_rank[-1]
                    cnt['det'] = gm.predict(v) == lab_keep
                    m0=(10**gm.means_.squeeze()[lab_keep])/hex_area
                    m1=(10**gm.means_.squeeze()[lab_rank[1]])/hex_area
                    m2=(10**gm.means_.squeeze()[lab_rank[0]])/hex_area
                    kept_min = cnt.loc[cnt.det.eq(True), key].min() / hex_area
                    print(f"2nd: log, 3 component. {m0:.3f} v.s. {m1:.3f} & {m2:.3f}, kept min density {kept_min:.3f}")
                # # 3rd try: fit a 2-component mixture model to density of original scale
                # if m1 > m0 * 0.5 or m0 < args.min_abs_mol_density_squm_dense:
                #     v = vorg.reshape(-1, 1)
                #     gm = sklearn.mixture.GaussianMixture(n_components=2, random_state=0).fit(v)
                #     lab_keep = np.argmax(gm.means_.squeeze())
                #     cnt['det'] = gm.predict(v) == lab_keep
                #     m0=gm.means_.squeeze()[lab_keep]/hex_area
                #     m1=gm.means_.squeeze()[1-lab_keep]/hex_area
                #     kept_min = cnt.loc[cnt.det.eq(True), key].min() / hex_area
                #     print(f"2nd: 2 component. {m0:.3f} v.s. {m1:.3f}, kept min density {kept_min:.3f}")
                if m0 < args.min_abs_mol_density_squm_dense:
                    cnt['det'] = cnt[key] > hex_area * args.min_abs_mol_density_squm_dense
                elif m1 > m0 * .7:
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
