'''
Read pixels from one or more consecutive regions
and group them into hexagonal bins
Assume the reader reads a file containing columns gene, X, Y
If input contains a column provided by region_id, each region is processed separately, regions do not have to be consecutive or be in the same coordinate system
'''
import sys, os
import numpy as np
import pandas as pd
from scipy.sparse import coo_array, vstack

from ficture.utils.hexagon_fn import *

class PixelToUnit:

    def __init__(self, reader, ft_dict, key, radius, scale=1, region_id=None, min_ct_per_unit=1, sliding_step=1, major_axis=None, xy_lattice=False) -> None:
        self.reader = reader
        self.ft_dict = ft_dict
        self.M = len(self.ft_dict)
        self.key = key
        self.radius = radius
        self.file_is_open = True
        self.min_ct_per_unit = min_ct_per_unit
        self.scale = scale
        self.region_id = region_id
        self.n_move = sliding_step
        self.Y = major_axis
        self.mj = self.Y if self.Y is not None else 'X'
        self.mi = 'X' if self.mj == 'Y' else 'Y'
        self.df = pd.DataFrame()
        self.brc = None
        self.mtx = None
        self.lattice = xy_lattice
        self.n_batch = 0
        self.bound = self.radius * np.sqrt(3)/2

    def read_chunk(self, min_size = 200):
        if not self.file_is_open:
            return 0
        if self.region_id is None:
            return self._read_chunk_consecutive(min_size)
        else:
            return self._read_chunk_region(min_size)

    def _read_chunk_region(self, min_size = 200):
        # todo: test this function
        if not self.file_is_open:
            return -1
        mj_range = [-np.inf, np.inf]
        mi_range = [-np.inf, np.inf]
        mj_size = 0
        mi_size = 0
        region_list = []
        prev_region = None
        if len(self.df) > 0:
            mj_range = [self.df[self.mj].max(), self.df[self.mj].min()]
            mi_range = [self.df[self.mi].max(), self.df[self.mi].min()]
            mj_size = mj_range[0] - mj_range[1]
            mi_size = mi_range[0] - mi_range[1]
            region_list = list(self.df[self.region_id].unique())
            prev_region = self.df[self.region_id].iloc[-1]
        while (mj_size < min_size or mi_size < min_size) and len(region_list) < 2:
            try:
                chunk = next(self.reader)
            except StopIteration:
                self.file_is_open = False
                break
            chunk = chunk[chunk.gene.isin(self.ft_dict)]
            chunk.X = chunk.X.astype(float) * self.scale
            chunk.Y = chunk.Y.astype(float) * self.scale
            last_region = chunk[self.region_id].iloc[-1]
            indx = chunk[self.region_id].eq(last_region)
            if last_region == prev_region:
                mj_range[0] = max(mj_range[0], chunk.loc[indx, self.mj].max())
                mj_range[1] = min(mj_range[1], chunk.loc[indx, self.mj].min())
                mi_range[0] = max(mi_range[0], chunk.loc[indx, self.mi].max())
                mi_range[1] = min(mi_range[1], chunk.loc[indx, self.mi].min())
            else:
                mj_range = [chunk.loc[indx, self.mj].max(), chunk.loc[indx, self.mj].min()]
                mi_range = [chunk.loc[indx, self.mi].max(), chunk.loc[indx, self.mi].min()]
            mj_size = mj_range[0] - mj_range[1]
            mi_size = mi_range[0] - mi_range[1]
            prev_region = last_region
            region_list += [x for x in chunk[self.region_id].unique() if x not in region_list]
            self.df = pd.concat([self.df, chunk])
        left = pd.DataFrame()
        if len(region_list) > 1 and (mj_size < min_size or mi_size < min_size):
            left = self.df[self.df[self.region_id].eq(region_list[-1])]
            region_list = region_list[:-1]
        elif self.Y is not None:
            left = self.df.loc[self.df[self.region_id].eq(region_list[-1]) & (self.df[self.mj] > mj_range[1])]
        self.df['hex_id'] = ''
        self.brc = pd.DataFrame()
        self.mtx = coo_array(([], ([], [])), shape = (0, self.M))
        for reg in region_list:
            indx = self.df[self.region_id].eq(reg)
            for offs_x in range(self.n_move):
                for offs_y in range(self.n_move):
                    offs_iden = str(offs_x) + '_' + str(offs_y)
                    x, y = pixel_to_hex(self.df.loc[indx, ['X', 'Y']].values, self.radius, offs_x/self.n_move, offs_y/self.n_move)
                    self.df.loc[indx, "hex_id"] = list(zip(x, y))
                    ct = self.df[indx].groupby('hex_id').agg({self.key: "sum"}).reset_index()
                    ct.drop(index = ct.index[ct[self.key] < self.min_ct_per_unit], inplace=True)
                    if len(ct) < 1:
                        continue
                    bt_dict = {x:i for i,x in enumerate(ct.hex_id.values)}
                    N = len(bt_dict)
                    brc = self.df[indx & self.df.hex_id.isin(bt_dict)].groupby(by = ['hex_id', 'gene']).agg({self.key: "sum"}).reset_index()
                    self.mtx = vstack([self.mtx, coo_array((brc[self.key].values, (brc.hex_id.map(bt_dict).values, brc.gene.map(self.ft_dict).values) ), shape = (N, self.M))])
                    if self.lattice:
                        ct['x'], ct['y'] = \
                            hex_to_pixel([v[0] for v in ct.hex_id.values],\
                                         [v[1] for v in ct.hex_id.values],\
                            self.radius, offs_x/self.n_move, offs_y/self.n_move)
                    else:
                        ct = brc.groupby(by = ['hex_id']).agg({self.key: sum,\
                            'X': np.median, 'Y':np.median}).reset_index()
                        ct.rename(columns = {'X':'x', 'Y':'y'}, inplace=True)
                    ct.index = ct.hex_id.map(bt_dict)
                    ct.sort_index(inplace=True)
                    ct['hex_id'] = ct.hex_id.map(lambda x : '_'.join([str(u) for u in x]))
                    ct.hex_id = offs_iden + '_' + ct.hex_id.values
                    ct[self.region_id] = reg
                    self.brc = pd.concat([self.brc, ct])
        self.mtx = self.mtx.tocsr()
        self.df = left
        self.brc.index = range(self.brc.shape[0])
        return self.brc.shape[0]


    def _read_chunk_consecutive(self, min_size = 200):
        if not self.file_is_open:
            return -1
        mj_range = [-np.inf, np.inf]
        mi_range = [-np.inf, np.inf]
        mj_size = 0
        mi_size = 0
        if len(self.df) > 0:
            mj_range = [self.df[self.mj].iloc[-1], self.df[self.mj].iloc[0]]
            mi_range = [self.df[self.mi].max(), self.df[self.mi].min()]
            mj_size = mj_range[0] - mj_range[1]
            mi_size = mi_range[0] - mi_range[1]
        while mj_size < min_size or mi_size < min_size:
            try:
                chunk = next(self.reader)
            except StopIteration:
                self.file_is_open = False
                break
            chunk = chunk[chunk.gene.isin(self.ft_dict)]
            chunk.X = chunk.X.astype(float) * self.scale
            chunk.Y = chunk.Y.astype(float) * self.scale
            mj_range[0] = chunk[self.mj].iloc[-1]
            mj_range[1] = min(mj_range[1], chunk[self.mj].iloc[0])
            mi_range[0] = max(mi_range[0], chunk[self.mi].max())
            mi_range[1] = min(mi_range[1], chunk[self.mi].min())
            mj_size = mj_range[0] - mj_range[1]
            mi_size = mi_range[0] - mi_range[1]
            self.df = pd.concat([self.df, chunk])
        self.df['hex_id'] = ''
        self.brc = pd.DataFrame()
        self.mtx = coo_array(([], ([], [])), shape = (0, self.M))
        for offs_x in range(self.n_move):
            for offs_y in range(self.n_move):
                offs_iden = str(offs_x) + '_' + str(offs_y)
                x, y = pixel_to_hex(self.df[['X', 'Y']].values, self.radius, offs_x/self.n_move, offs_y/self.n_move)
                self.df.hex_id = list(zip(x, y))
                ct = self.df.groupby('hex_id').agg({self.key: "sum"}).reset_index()
                ct['X'], ct['Y'] = \
                        hex_to_pixel([v[0] for v in ct.hex_id.values],\
                                     [v[1] for v in ct.hex_id.values],\
                        self.radius, offs_x/self.n_move, offs_y/self.n_move)
                # Drop hexagons with centers within sqrt(3)*raidus/2 from each boundary
                ct.drop(index = ct.index[\
                    (ct[self.mj].lt(mj_range[1] + self.bound) ) | \
                    (ct[self.mj].gt(mj_range[0] - self.bound) ) ], inplace=True)
                ct.drop(index = ct.index[ct[self.key] < self.min_ct_per_unit], inplace=True)
                if len(ct) < 1:
                    continue
                bt_dict = {x:i for i,x in enumerate(ct.hex_id.values)}
                N = len(bt_dict)
                brc = self.df[self.df.hex_id.isin(bt_dict)].groupby(by = ['hex_id', 'gene']).agg({self.key: "sum"}).reset_index()
                self.mtx = vstack([self.mtx, coo_array((brc[self.key].values, (brc.hex_id.map(bt_dict).values, brc.gene.map(self.ft_dict).values) ), shape = (N, self.M))])
                if not self.lattice:
                    ct = brc.groupby(by = ['hex_id']).agg({self.key: sum, 'X': np.median, 'Y':np.median}).reset_index()
                ct.rename(columns = {'X':'x', 'Y':'y'}, inplace=True)
                ct.index = ct.hex_id.map(bt_dict)
                ct.sort_index(inplace=True)
                ct['hex_id'] = ct.hex_id.map(lambda x : '_'.join([str(u) for u in x]))
                ct.hex_id = offs_iden + '_' + ct.hex_id.values
                self.brc = pd.concat([self.brc, ct])
        self.df = self.df[self.df[self.mj] >= mj_range[0] - 2 * self.radius]
        self.df.drop(columns = ['hex_id'], inplace = True)
        self.mtx = self.mtx.tocsr()
        self.brc.index = range(self.brc.shape[0])
        self.n_batch += 1
        return self.brc.shape[0]
