### Read (randomized) hexagons from file, construct minibatch
import sys, os, gzip, copy, re
import numpy as np
import pandas as pd
from scipy.sparse import coo_array

from ficture.models.lda_minibatch import PairedMinibatch

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
        self.brc = self.brc.merge(right = self.df.groupby(by='unit').agg({self.key:"sum", self.bkey:"sum"}).reset_index(), on = 'unit', how = 'inner' )
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

    def __init__(self, reader, ft_dict, key, batch_id_prefix=0, min_ct_per_unit=1, unit_id='unit', unit_attr=['x','y'], train_key=None, epoch = 2**15, skip_epoch=[], debug=False) -> None:
        self.reader = reader
        self.ft_dict = ft_dict
        self.key = key
        self.df = pd.DataFrame()
        self.file_is_open = True
        self.mtx = None
        self.brc = None
        self.prefix = batch_id_prefix
        self.epoch = epoch
        self.min_ct_per_unit = min_ct_per_unit
        self.unit_id = unit_id
        self.unit_attr = list(unit_attr)
        self.M = max(self.ft_dict.values()) + 1
        self.debug = debug
        self.batch_id_list = []
        self.skip_epoch = set(skip_epoch)
        self.train_key = key if train_key is None else train_key
        self.test_mtx = None

    def _make_matrix(self):
        self.brc = self.df[['unit']+self.unit_attr].drop_duplicates(subset=['unit'])
        if self.prefix > 0:
            lab = self.brc.unit.str[:self.prefix].unique()
            self.batch_id_list += [x for x in lab if x not in self.batch_id_list]
        self.brc = self.brc.merge(right = self.df.groupby(by='unit').agg({self.train_key:"sum"}).reset_index(), on = 'unit', how = 'inner' )
        if self.key != self.train_key:
            self.brc = self.brc.merge(right = self.df.groupby(by='unit').agg({self.key:"sum"}).reset_index(), on = 'unit', how = 'inner' )
        self.brc = self.brc[self.brc[self.key] >= self.min_ct_per_unit]
        barcode_kept=list(self.brc.unit)
        bt_dict={x:i for i,x in enumerate(barcode_kept)}
        N = len(bt_dict)
        self.df = self.df[self.df.unit.isin(bt_dict)]
        self.mtx = coo_array((self.df[self.train_key].values, \
                            (self.df.unit.map(bt_dict).values, \
                             self.df.gene.map(self.ft_dict).values) ), \
                            shape=(N, self.M)).tocsr()
        if self.key != self.train_key:
            self.test_mtx = coo_array(( \
                (self.df[self.key] - self.df[self.train_key]).values, \
                            (self.df.unit.map(bt_dict).values, \
                             self.df.gene.map(self.ft_dict).values) ), \
                            shape=(N, self.M)).tocsr()
            self.test_mtx.eliminate_zeros()
        return N

    def update_batch(self, bsize):
        n_unit = 0
        if len(self.df) > 0:
            n_unit = len(self.df.unit.unique())
        if len(self.batch_id_list) > self.epoch or not self.file_is_open:
            return 0
        while n_unit <= bsize:
            try:
                chunk = next(self.reader)
            except StopIteration:
                self.file_is_open = False
                break
            chunk.rename(columns={self.unit_id:'unit'}, inplace=True)
            chunk = chunk[chunk.gene.isin(self.ft_dict)]
            if len(self.skip_epoch) > 0:
                lab = chunk.unit.str[:self.prefix]
                chunk = chunk[~lab.isin(self.skip_epoch)]
            if len(chunk) == 0:
                continue
            if self.debug:
                print(f"Read {chunk.shape[0]} lines from file")
            self.df = pd.concat([self.df, chunk])
            n_unit = len(self.df.unit.unique())
        left = pd.DataFrame()
        if self.file_is_open:
            last_indx = self.df.unit.iloc[-1]
            left = self.df[self.df.unit == last_indx]
            self.df = self.df[self.df.unit != last_indx]
        N = self._make_matrix()
        self.df = copy.copy(left)
        return N


    def read_one_epoch(self):
        if not self.file_is_open:
            return 0
        if self.prefix <= 0:
            print(f"UnitLoader::read_one_epoch Will read the whole file")
        left = pd.DataFrame()
        local_epoch_list = []
        if len(self.df > 0) and self.prefix > 0:
            local_epoch_list = list(np.unique([x[:self.prefix] for x in self.df.unit.unique()]) )
        while len(local_epoch_list) < 2:
            try:
                chunk = next(self.reader)
            except StopIteration:
                self.file_is_open = False
                break
            chunk.rename(columns={self.unit_id:'unit'}, inplace=True)
            chunk = chunk[chunk.gene.isin(self.ft_dict)]
            if len(chunk) == 0:
                continue
            if self.prefix > 0:
                lab = np.unique([x[:self.prefix] for x in chunk.unit.unique()])
                local_epoch_list += [x for x in lab if x not in local_epoch_list]
                if len(local_epoch_list) > 1:
                    left = chunk[~chunk.unit.str[:self.prefix].eq(local_epoch_list[0])]
                    chunk = chunk[chunk.unit.str[:self.prefix].eq(local_epoch_list[0])]
            self.df = pd.concat([self.df, chunk])
            if self.debug:
                print(f"Read {chunk.shape[0]} lines from file")
        N = self._make_matrix()
        self.df = copy.copy(left)
        return N
