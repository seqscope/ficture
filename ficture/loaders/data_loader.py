import sys, os, copy, gzip, logging, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_array, csr_array, vstack
import sklearn.neighbors
from sklearn.preprocessing import normalize

from ficture.utils import utilt

class factor_space_stream:
    """
    Load spatial factor decoding results by chunk
    Assuming both input and queries are sorted by index_axis
    Answer queries of spatial locations by averaging over the nearest neighbors
    """
    def __init__(self, file, index_axis, debug =0, init_pos = 0,\
                 sparsify = 1e-4, buffer = 30, chunksize = 100000,\
                 x = "X", y = "Y", factor_header = None) -> None:
        assert os.path.exists(file), "File does not exist: {}".format(file)
        assert index_axis in [x, y], "Index axis must be the same as either the provided x (X) or y (Y)[]"
        self.debug = debug
        self.file = file
        self.index_col = index_axis
        self.index_axis = 0 if index_axis == x else 1
        self.sparsify = sparsify
        self.buffer = buffer
        self.factor_header = factor_header
        if self.factor_header is None:
            with gzip.open(self.file, 'rt') as rf:
                input_header = rf.readline().strip().split('\t')
            self.factor_header = utilt.get_string_with_integer_suff(input_header)
        self.K = len(self.factor_header)
        self.reader = pd.read_csv(gzip.open(self.file, 'rt'), sep='\t', \
                                  usecols = [x,y] + self.factor_header, \
                                  chunksize=chunksize)
        self.recolumn = {x:"X", y:"Y"}
        self.recolumn.update({x:str(k) for k,x in enumerate(self.factor_header)})
        self.factor_header = [str(k) for k in range(self.K)]
        self.pts = np.empty((0, 2))
        self.factor_loading = csr_array((0, self.K))
        self.current_ymax = -1
        self.current_ymin = -1
        self.file_is_open = True
        self.update_reference(init_pos, init_pos)
        if self.debug:
            print(f"Current view is from {self.current_ymin} to {self.current_ymax}")

    def read_chunk(self, ymin):
        try:
            chunk = next(self.reader)
            chunk.drop_duplicates(subset = ['X','Y'], inplace = True)
        except StopIteration:
            return 0
        self.current_ymax = chunk[self.index_col].iloc[-1]
        chunk = chunk.loc[chunk[self.index_col] > ymin, :]
        if chunk.shape[0] == 0:
            return 1
        chunk.rename(columns=self.recolumn, inplace=True)
        self.pts = np.vstack((self.pts, chunk.loc[:, ["X","Y"]].values))
        chunk = np.array(chunk.loc[:, self.factor_header].values)
        chunk[chunk < self.sparsify] = 0
        chunk = csr_array(chunk)
        chunk.eliminate_zeros()
        self.factor_loading = vstack((self.factor_loading, chunk))
        self.current_ymax = self.pts[-1, self.index_axis]
        return 1

    def update_reference(self, ymax, ymin = None):
        if ymin is None:
            ymin = self.current_ymax
        ymin -= self.buffer
        ymax += self.buffer
        indx_buffer = self.pts[:, self.index_axis] > ymin
        self.pts  = self.pts[indx_buffer, :]
        self.factor_loading = self.factor_loading[indx_buffer, :]
        while self.current_ymax < ymax and self.file_is_open:
            if self.debug > 1:
                print(f"Current cursor: {self.current_ymax}, target: {ymax}")
            self.file_is_open = self.read_chunk(ymin)
        self.ref = sklearn.neighbors.BallTree(self.pts)
        self.current_ymax = self.pts[:, self.index_axis].max()
        self.current_ymin = self.pts[:, self.index_axis].min()

    def impute_factor_loading(self, pos, k = 1, radius = 5, halflife = .7, include_self = False):
        ymin = pos[:, self.index_axis].min()
        ymax = pos[:, self.index_axis].max()
        if ymax > self.current_ymax:
            self.update_reference(ymax, ymin)
        nu = np.log(.5) / np.log(halflife)
        dist, indx = self.ref.query(pos, k = k, return_distance = True, sort_results = True)
        min_dist = 0
        if include_self:
            min_dist = -1
        mask = (dist > min_dist) & (dist < radius)
        indx = [[x for k,x in enumerate(v) if mask[i,k]] for i,v in enumerate(indx) ]
        dist = 1. - (dist / radius) ** nu
        dist[~mask] = -1
        r = [i for i,y in enumerate(indx) for x in range(len(y))]
        c = [x for y in indx for x in y]
        w = [x for y in dist for x in y if x >= 0]
        mtx = coo_array((w, (r, c)), shape = (pos.shape[0], self.pts.shape[0])).tocsr()
        mtx = normalize(mtx, norm='l1', axis=1, copy=False)
        mtx.eliminate_zeros()
        return mtx @ self.factor_loading





class factor_space:
    """
    Load spatial factor decoding results
    Answer queries of spatial locations by averaging over the nearest neighbors
    """
    def __init__(self, file, debug = 0, sparsify = 1e-4,\
                 x = "x", y = "y", factor_header = None) -> None:
        assert os.path.exists(file), "File does not exist: {}".format(file)
        self.file = file
        self.debug = debug
        self.sparsify = sparsify
        self.factor_header = factor_header
        with gzip.open(self.file, 'rt') as rf:
            input_header = rf.readline().strip().split('\t')
        if self.factor_header is None:
            self.factor_header = utilt.get_string_with_integer_suff(input_header)
        self.K = len(self.factor_header)
        self.reader = pd.read_csv(gzip.open(self.file, 'rt'), sep='\t', \
                                  chunksize=100000)
        self.recolumn = {x.lower():"X", y.lower():"Y"}
        self.recolumn.update({x.lower():str(k) for k,x in enumerate(self.factor_header)})
        self.factor_header = [str(k) for k in range(self.K)]
        self.pts = np.empty((0, 2))
        self.factor_loading = csr_array((0, self.K))
        self.update_reference()

    def read_chunk(self):
        try:
            chunk = next(self.reader)
            chunk.columns = [x.lower() for x in chunk.columns]
        except StopIteration:
            return 0
        chunk.rename(columns=self.recolumn, inplace=True)
        self.pts = np.vstack((self.pts, chunk.loc[:, ["X","Y"]].values))
        chunk = np.array(chunk.loc[:, self.factor_header].values)
        chunk[chunk < self.sparsify] = 0
        chunk = csr_array(chunk)
        chunk.eliminate_zeros()
        self.factor_loading = vstack((self.factor_loading, chunk))
        if self.debug > 1:
            print(f"Read chunk {chunk.shape}, {self.factor_loading.shape}")
        return 1

    def update_reference(self):
        while True:
            if self.read_chunk() == 0:
                break
        self.ref = sklearn.neighbors.BallTree(self.pts)
        if self.debug:
            print(f"Build reference with {self.pts.shape[0]} points")

    def impute_factor_loading(self, pos, k = 1, radius = 5, halflife = .7, include_self = False):
        nu = np.log(.5) / np.log(halflife)
        dist, indx = self.ref.query(pos, k = k, return_distance = True, sort_results = True)
        min_dist = 0
        if include_self:
            min_dist = -1
        if self.debug > 1:
            v = np.min(dist,axis=1)
            print(radius, sum(v > radius), len(v))
        mask = (dist > min_dist) & (dist < radius)
        indx = [[x for k,x in enumerate(v) if mask[i,k]] for i,v in enumerate(indx) ]
        dist = 1. - (dist / radius) ** nu
        dist[~mask] = -1
        r = [i for i,y in enumerate(indx) for x in range(len(y))]
        c = [x for y in indx for x in y]
        w = [x for y in dist for x in y if x >= 0]
        mtx = coo_array((w, (r, c)), shape = (pos.shape[0], self.pts.shape[0])).tocsr()
        mtx = normalize(mtx, norm='l1', axis=1, copy=False)
        mtx.eliminate_zeros()
        return mtx @ self.factor_loading





class StreamUnit:
    """
    Read a long format DGE by chunk
    Output matrices sequentially
    """
    def __init__(self, reader, unit_id, key, ft_dict, feature_id="gene",
                 min_batch_size = 256, min_ct_per_unit = 2):
        self.reader = reader
        self._unit_id = unit_id
        self._feature_id = feature_id
        self._key = key
        self._min_n = min_batch_size
        self._min_c = min_ct_per_unit
        self.ft_dict = copy.copy(ft_dict)
        self.M = len(self.ft_dict)
        self.df = pd.DataFrame()

    def get_matrix(self):
        for chunk in self.reader:
            chunk = chunk[chunk.gene.isin(self.ft_dict)]
            chunk.rename(inplace=True, \
                         columns = {self._unit_id:'j', self._feature_id:'gene'})
            # Incomplete left over
            last_indx = chunk.j.iloc[-1]
            left = copy.copy(chunk[chunk.j.eq(last_indx)])
            chunk = chunk.loc[~chunk.j.eq(last_indx), :]

            ct = chunk.groupby(by = ['j']).agg({self._key: "sum"}).reset_index()
            kept_unit = set(ct.loc[ct[self._key] > self._min_c, "j"].values)
            self.df = pd.concat([self.df, chunk[chunk.j.isin(kept_unit)]])
            if len(self.df.j.unique()) < self._min_n: # Wait until more data
                self.df = pd.concat([self.df, left])
                continue

            # Total molecule count per unit
            brc = self.df.groupby(by = ['j']).agg({self._key: "sum"}).reset_index()
            brc = brc[brc[self._key] > self._min_c]
            brc.index = range(brc.shape[0])
            self.df = self.df[self.df.j.isin(brc.j.values)]
            # Make DGE
            barcode_kept = list(brc.j.values)
            bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
            indx_row = [ bc_dict[x] for x in self.df['j']]
            indx_col = [ self.ft_dict[x] for x in self.df['gene']]
            N = len(barcode_kept)
            mtx = coo_array((self.df[self._key].values, (indx_row, indx_col)), shape=(N, self.M)).tocsr()
            yield mtx
            self.df = copy.copy(left)

        if len(self.df.j.unique()) > 1:
            brc = self.df.groupby(by = ['j']).agg({self._key: "sum"}).reset_index()
            brc = brc[brc[self._key] > self._min_c]
            brc.index = range(brc.shape[0])
            self.df = self.df[self.df.j.isin(brc.j.values)]
            # Make DGE
            barcode_kept = list(brc.j.values)
            bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
            indx_row = [ bc_dict[x] for x in self.df['j']]
            indx_col = [ self.ft_dict[x] for x in self.df['gene']]
            N = len(barcode_kept)
            mtx = coo_array((self.df[self._key].values, (indx_row, indx_col)), shape=(N, self.M)).tocsr()
            yield mtx
