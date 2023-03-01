import sys, os, copy, gzip, logging
import numpy as np
import pandas as pd
from scipy.sparse import *

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

            ct = chunk.groupby(by = ['j']).agg({self._key: sum}).reset_index()
            kept_unit = set(ct.loc[ct[self._key] > self._min_c, "j"].values)
            self.df = pd.concat([self.df, chunk[chunk.j.isin(kept_unit)]])
            if len(self.df.j.unique()) < self._min_n: # Wait until more data
                self.df = pd.concat([self.df, left])
                continue

            # Total mulecule count per unit
            brc = self.df.groupby(by = ['j']).agg({self._key: sum}).reset_index()
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
            brc = self.df.groupby(by = ['j']).agg({self._key: sum}).reset_index()
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
