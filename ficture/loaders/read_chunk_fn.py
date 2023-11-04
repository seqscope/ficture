import sys, os, copy, gzip, logging
import numpy as np
import pandas as pd
from scipy.sparse import *
from dataclasses import dataclass
from sklearn.neighbors import BallTree

from ficture.loaders.data_loader import factor_space_stream

@dataclass
class BlockProfile:
    weight: float
    xst: float
    yst: float
    xed: float
    yed: float
    profile: np.ndarray # Unnormalized

class SlidingPosteriorCount:

    def __init__(self, pixel_reader, index_axis, key, factor_file, ft_dict,\
                 size_um, slide_step, mu_scale,\
                 precision = 1, radius = 3, init_pos = 0, debug = 0) -> None:
        self.pixel_reader = pixel_reader
        self.index_axis = index_axis
        self.key = key
        self.ft_dict = ft_dict
        self.size_um = int(size_um / precision) * precision
        self.slide_step = slide_step
        self.mu_scale = mu_scale
        self.precision = precision
        self.radius = radius
        self.block_size = self.size_um / self.slide_step
        self.left_over = pd.DataFrame()
        self.ymin = 0
        self.ymax = 0
        self.file_is_open = True
        self.block_ct_min = 100
        self.blocks = {}
        self.windows = {}
        self.window_centers = []
        self.ref = None
        self.factor_map = factor_space_stream(file=factor_file, index_axis=index_axis, init_pos = init_pos)
        self.K = self.factor_map.K
        self.M = len(self.ft_dict)
        self.debug = debug

    def update_reference(self, st, ed):
        """
        Update the reference to cover from st to ed
        """
        ymin = st - self.size_um
        ymax = ed + self.size_um
        self.add_block(ymin, ymax)
        self.remove_block(ymin)
        self.block_to_windows()
        self.ref = BallTree(np.array([x[:2] for x in self.window_centers]))

    def query(self, x, y):
        """
        Query the nearest window to (x, y)
        Return the distance and the window profile
        """
        pt = np.array([[x, y]])
        dist, indx = self.ref.query(pt, 1)
        k = self.window_centers[indx[0][0]][2]
        return dist[0][0], self.windows[k].profile

    def block_to_windows(self):
        """
        Sum over blocks in each window
        """
        self.window_centers = []
        for k, v in self.blocks.items():
            weight = 0
            beta = np.zeros((self.K, self.M))
            n_block = 0
            for i in range(self.slide_step):
                for j in range(self.slide_step):
                    u = self.blocks.get((k[0] + i, k[1] + j))
                    if u is None:
                        continue
                    beta += u.profile
                    weight += u.weight
                    n_block += 1
            xst = k[0] * self.block_size
            yst = k[1] * self.block_size
            xed = xst + self.size_um
            yed = yst + self.size_um
            xct = (xst + xed) / 2
            yct = (yst + yed) / 2
            self.window_centers.append([xct, yct, k])
            self.windows[k] = BlockProfile(weight = weight, xst = xst, yst = yst, xed = xed, yed = yed, profile = beta)
            if self.debug > 1:
                print(f"Created window {k} with {n_block} blocks, total weight {weight}")

    def add_block(self, ymin, ymax):
        """
        Read data until beyond ymax
        """
        df = copy.copy(self.left_over)
        ymin = ymin / self.mu_scale
        self.block_indx_max = int(ymax / self.block_size) + 1
        ymax = self.block_indx_max * self.block_size
        while self.ymax < ymax:
            try:
                chunk = next(self.pixel_reader)
                chunk = chunk.loc[(chunk[self.index_axis] > ymin) & (chunk[self.key] > 0), :]
                if chunk.shape[0] == 0:
                    continue
                chunk.X *= self.mu_scale
                chunk.Y *= self.mu_scale
                df = pd.concat([df, chunk])
                self.ymax = df[self.index_axis].max()
                if self.debug:
                    print(f"Current cursor: {self.ymax}")
            except StopIteration:
                self.file_is_open = False
                break
        df = df.loc[df.gene.isin(self.ft_dict), :]
        # only process blocks completely within ymax
        left = df[self.index_axis] >= ymax
        self.left_over = copy.copy(df.loc[left, :])
        if self.debug:
            print(f"Read data entries {df.shape[0]}, leftover {self.left_over.shape[0]}")
        df = df.loc[~left, :]
        if df.shape[0] == 0:
            return
        df.X = (df.X / self.precision).astype(int)
        df.Y = (df.Y / self.precision).astype(int)
        df['j'] = list(zip(df.X, df.Y))
        df = df.groupby(by = ['j', 'gene']).agg({self.key: sum}).reset_index()
        brc = df.groupby(by = ['j']).agg({self.key: sum}).reset_index()
        brc['X'] = brc.j.map(lambda x: x[0]) * self.precision
        brc['Y'] = brc.j.map(lambda x: x[1]) * self.precision
        x = (brc.X / self.block_size).astype(int).values
        y = (brc.Y / self.block_size).astype(int).values
        brc['w_id'] = list(zip(x, y))

        # count total reads per block
        w_ct = brc.groupby(by = ['w_id']).agg({self.key: sum}).reset_index()
        # keep only blocks with more than 100 reads
        w_id = set(w_ct.loc[w_ct[self.key] > self.block_ct_min, 'w_id'].values)
        w_id = {x : np.array([x[0],x[0]+1,x[1],x[1]+1]) * self.block_size for x in w_id}
        if self.debug:
            print(f"Total blocks: {brc.w_id.nunique()}, kept blocks: {len(w_id)}")
        brc = brc.loc[brc.w_id.isin(w_id), :]

        # Make DGE
        df = df.loc[df.j.isin(brc.j.values), :]
        barcode_kept = list(brc.j.values)
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [bc_dict[x] for x in df['j']]
        indx_col = [self.ft_dict[x] for x in df['gene']]
        mtx = coo_array((df[self.key].values, (indx_row, indx_col)), shape=(len(barcode_kept), self.M)).tocsr() # (N, M)
        if self.debug:
            print(f"Made DGE: {mtx.shape}")

        # find factor loading of pixels
        theta = self.factor_map.impute_factor_loading(pos = brc.loc[:, ["X", "Y"]].values, k = 1, radius = self.radius, include_self = True) # (N, K)
        # factor specific DGE by block
        for w, v in w_id.items():
            indx = brc.loc[brc.w_id == w, 'j'].map(bc_dict).values
            weight = brc.loc[brc.w_id == w, self.key].sum()
            beta = theta[indx, :].T @ mtx[indx, :] # (K, M)
            self.blocks[w] = BlockProfile(weight = weight, xst = v[0], yst = v[2], xed = v[1], yed = v[3], profile = beta)
            if self.debug > 1:
                print(f"Add block {w} at {v[0]}, {v[2]}, with weight {weight}")

    def remove_block(self, ymin):
        """
        Remove blocks that are below ymin
        """
        for k, v in self.blocks.items():
            if v.yed < ymin/self.block_size:
                self.blocks.pop(k, None)
