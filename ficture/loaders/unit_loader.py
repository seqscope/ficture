### Read (randomized) hexagons from file, construct minibatch
import sys, os, gzip, copy, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from ficture.models.lda_minibatch import PairedMinibatch, Minibatch

class UnitLoaderAugmented:

    def __init__(self, reader, ft_dict, key, bkey, batch_id_prefix=0, min_ct_per_unit=1, unit_attr=['x','y']) -> None:
        self.reader = reader
        self.ft_dict = ft_dict
        self.key = key
        self.bkey= bkey
        self.df = pd.DataFrame()
        self.file_is_open = True
        self.batch = None
        self.brc = None
        self.prefix = batch_id_prefix
        self.min_ct_per_unit = min_ct_per_unit
        self.unit_attr = list(unit_attr)
        self.M = max(self.ft_dict.values()) + 1
        self.batch_id_list = set()

    def update_batch(self, bsize):
        n_unit = 0
        if len(self.df) > 0:
            n_unit = len(self.df.unit.unique())
        if not self.file_is_open:
            return 0
        while n_unit <= bsize:
            try:
                chunk = next(self.reader)
            except StopIteration:
                self.file_is_open = False
                break
            self.df = pd.concat([self.df, chunk])
            n_unit = len(self.df.unit.unique())
        left = pd.DataFrame()
        if self.file_is_open:
            last_indx = self.df.unit.iloc[-1]
            left = self.df[self.df.unit == last_indx]
            self.df = self.df[self.df.unit != last_indx]
        self.brc = self.df[['unit']+self.unit_attr].drop_duplicates(subset=['unit'])
        self.brc = self.brc.merge(right = self.df.groupby(by='unit').agg({self.key:sum, self.bkey:sum}).reset_index(), on = 'unit', how = 'inner' )
        self.brc = self.brc[self.brc[self.key] >= self.min_ct_per_unit]
        buffer_weight = self.brc[self.bkey].values / (self.brc[self.bkey].values + self.brc[self.key].values)
        barcode_kept=list(self.brc.unit)
        bt_dict={x:i for i,x in enumerate(barcode_kept)}
        N = len(bt_dict)
        self.df = self.df[self.df.unit.isin(bt_dict) & self.df.gene.isin(self.ft_dict)]
        if self.prefix > 0:
            self.batch_id_list.update(set(self.df.unit.map(lambda x : x[:self.prefix]).values))
        self.batch = PairedMinibatch(\
            mtx_focal = coo_array((self.df[self.key].values, \
                            (self.df.unit.map(bt_dict).values, \
                             self.df.gene.map(self.ft_dict).values) ), \
                            shape=(N, self.M)).tocsr(),
            mtx_buffer = coo_array((self.df[self.bkey].values, \
                            (self.df.unit.map(bt_dict).values, \
                             self.df.gene.map(self.ft_dict).values) ), \
                            shape=(N, self.M)).tocsr(),
            buffer_weight = buffer_weight)
        self.df = copy.copy(left)
        return self.batch.n



class UnitLoader:

    def __init__(self, reader, ft_dict, key, batch_id_prefix=0, min_ct_per_unit=1, unit_id='unit', unit_attr=['x','y'], debug=False) -> None:
        self.reader = reader
        self.ft_dict = ft_dict
        self.key = key
        self.df = pd.DataFrame()
        self.file_is_open = True
        self.mtx = None
        self.brc = None
        self.prefix = batch_id_prefix
        self.min_ct_per_unit = min_ct_per_unit
        self.unit_id = unit_id
        self.unit_attr = list(unit_attr)
        self.M = max(self.ft_dict.values()) + 1
        self.debug = debug
        self.batch_id_list = set()

    def update_batch(self, bsize):
        n_unit = 0
        if len(self.df) > 0:
            n_unit = len(self.df.unit.unique())
        if not self.file_is_open:
            return 0
        while n_unit <= bsize:
            try:
                chunk = next(self.reader)
                chunk.rename(columns={self.unit_id:'unit'}, inplace=True)
                chunk = chunk[chunk.gene.isin(self.ft_dict)]
                if self.debug:
                    print(f"Read {chunk.shape[0]} lines from file")
            except StopIteration:
                self.file_is_open = False
                break
            self.df = pd.concat([self.df, chunk])
            n_unit = len(self.df.unit.unique())
        left = pd.DataFrame()
        if self.file_is_open:
            last_indx = self.df.unit.iloc[-1]
            left = self.df[self.df.unit == last_indx]
            self.df = self.df[self.df.unit != last_indx]
        self.brc = self.df[['unit']+self.unit_attr].drop_duplicates(subset=['unit'])
        self.brc = self.brc.merge(right = self.df.groupby(by='unit').agg({self.key:sum}).reset_index(), on = 'unit', how = 'inner' )
        self.brc = self.brc[self.brc[self.key] >= self.min_ct_per_unit]
        barcode_kept=list(self.brc.unit)
        bt_dict={x:i for i,x in enumerate(barcode_kept)}
        N = len(bt_dict)
        self.df = self.df[self.df.unit.isin(bt_dict)]
        if self.prefix > 0:
            self.batch_id_list.update(set(self.df.unit.map(lambda x : x[:self.prefix]).values))
        self.mtx = coo_array((self.df[self.key].values, \
                            (self.df.unit.map(bt_dict).values, \
                             self.df.gene.map(self.ft_dict).values) ), \
                            shape=(N, self.M)).tocsr()
        self.df = copy.copy(left)
        return N
