### Read a batched pixel file, construct slda minibatch

import sys, os, gzip, copy, re, logging
import numpy as np
import pandas as pd

from scipy.sparse import coo_array
import sklearn.neighbors
import sklearn.preprocessing
from joblib.parallel import Parallel, delayed
from sklearn.decomposition._online_lda_fast import _dirichlet_expectation_2d

from ficture.utils import utilt
from ficture.models.slda_minibatch import minibatch

class PixelMinibatch:

    def __init__(self, reader, ft_dict, batch_id, key, mu_scale, radius, halflife, adj_penal=-1, precision=0.1, thread=1, verbose=0) -> None:
        self.df_full = pd.DataFrame()
        self.pixel_reader = reader
        self.batch_id = batch_id
        self.mu_scale = mu_scale
        self.file_is_open = True
        self.ft_dict = ft_dict
        self.M = len(self.ft_dict)
        self.key = key
        self.precision = precision
        self.radius = radius
        self.adj_penal = adj_penal
        self.nu = np.log(.5) / np.log(halflife)
        self.out_buff = self.radius * halflife
        self.thread = thread
        self.verbose = verbose

    def load_anchor(self, anchor_file, anchor_in_um = True):
        self.grid_info = pd.read_csv(anchor_file,sep='\t')
        header_map = {"X":"x", "Y":"y"}
        self.factor_header = []
        for x in self.grid_info.columns:
            y = re.match('^[A-Za-z]*_*(\d+)$', x)
            if y:
                header_map[y.group(0)] = y.group(1)
                self.factor_header.append(int(y.group(1)) )
        self.grid_info.rename(columns = header_map, inplace=True)
        if not anchor_in_um:
            self.grid_info.x *= self.mu_scale
            self.grid_info.y *= self.mu_scale
        self.n = self.grid_info.shape[0]
        self.grid_info.index = range(self.n)
        self.ref = sklearn.neighbors.BallTree(np.array(self.grid_info.loc[:, ['x','y']]))
        self.factor_header.sort()
        self.factor_header = [str(x) for x in self.factor_header]
        self.K = len(self.factor_header)
        self.adj_mtx = None

    def _anchor_adj(self):
        if self.adj_penal < 0:
            self.adj_penal = self.radius * 2
        self.adj_mtx = sklearn.neighbors.radius_neighbors_graph(self.ref, self.adj_penal, mode='connectivity', include_self=True) # Do we want to include diagonal?

    def read_chunk(self, nbatch):
        batch_ids = set()
        while len(batch_ids) <= nbatch:
            try:
                chunk = next(self.pixel_reader)
            except StopIteration:
                self.file_is_open = False
                break
            chunk = chunk.loc[chunk.gene.isin(self.ft_dict) & \
                              (chunk[self.key] > 0), :]
            chunk.X *= self.mu_scale
            chunk.Y *= self.mu_scale
            if self.precision > 0:
                chunk.X = (chunk.X / self.precision).astype(int)
                chunk.Y = (chunk.Y / self.precision).astype(int)
                chunk = chunk.groupby(by=[self.batch_id,"gene","X","Y"]).agg({self.key: "sum"}).reset_index()
                chunk.X *= self.precision
                chunk.Y *= self.precision
            random_pref = chunk[self.batch_id].map(lambda x : x[-5:]).values
            chunk['j'] = random_pref + '_' + (chunk.X*100).astype(int).astype(str) + '_' + (chunk.Y*100).astype(int).astype(str)
            # Keep pixels close enough to at least one anchor
            pts = chunk[["j", "X", "Y"]].drop_duplicates(subset="j")
            dist, indx = self.ref.query(X = np.array(pts[['X','Y']]), k = 1, return_distance = True)
            dist = dist.squeeze()

            kept_pixel = dist < self.radius
            if np.sum(kept_pixel) == 0:
                continue
            chunk = chunk.loc[chunk.j.isin(pts.loc[kept_pixel, 'j'].values), :]
            batch_ids.update(set(chunk[self.batch_id].unique()))
            self.df_full = pd.concat([self.df_full, chunk], axis=0)
            # logging.info(f"Read {len(batch_ids)} minibatches")

        left = pd.DataFrame()
        if self.file_is_open:
            last_indx = self.df_full[self.batch_id].iloc[-1]
            left = copy.copy(self.df_full.loc[self.df_full[self.batch_id].eq(last_indx), :])
            self.df_full = self.df_full.loc[~self.df_full[self.batch_id].eq(last_indx), :]

        ### Process chunk of data
        self.batch_index = list(self.df_full[self.batch_id].unique() )
        self.brc = self.df_full[[self.batch_id,"j","X","Y"]].drop_duplicates(subset=["j"])
        self.N0 = self.brc.shape[0]
        self.brc.index = range(self.N0)
        self.df_full = self.df_full.groupby(by = [self.batch_id, "j", "gene"]).agg({self.key:"sum"}).reset_index()
        logging.info(f"Read {self.N0} pixels, forming {len(self.batch_index)} batches.")
        self.bt = sklearn.neighbors.BallTree(np.asarray(self.brc[['X','Y']]))
        # Make DGE
        barcode_kept = list(self.brc['j'])
        bc_dict = {x:i for i,x in enumerate( barcode_kept ) }
        indx_row = [ bc_dict[x] for x in self.df_full['j']]
        indx_col = [ self.ft_dict[x] for x in self.df_full['gene']]
        self.dge_mtx = coo_array((self.df_full[self.key], (indx_row, indx_col)), shape=(self.N0, self.M)).tocsr()
        self.df_full = left
        return len(self.batch_index)

    def _prepare_batch(self, b, init_bound):

        indx = self.brc[self.batch_id].eq(b)
        x_min, x_max = self.brc.loc[indx, 'X'].min(), self.brc.loc[indx, 'X'].max()
        y_min, y_max = self.brc.loc[indx, 'Y'].min(), self.brc.loc[indx, 'Y'].max()
        grid_indx = (self.grid_info.x >= x_min - self.radius) &\
                    (self.grid_info.x <= x_max + self.radius) &\
                    (self.grid_info.y >= y_min - self.radius) &\
                    (self.grid_info.y <= y_max + self.radius)
        grid_pt = self.grid_info.loc[grid_indx, ["x","y"]]
        if grid_pt.shape[0] < 10:
            return None, None, None, None

        # Initilize anchor
        theta = np.array(self.grid_info.loc[grid_indx, self.factor_header])
        theta = sklearn.preprocessing.normalize(np.clip(theta, init_bound, 1.-init_bound), norm='l1', axis=1)
        n = theta.shape[0]

        # Pixel to anchor weight
        indx, dist = self.bt.query_radius(X = np.array(grid_pt), r = self.radius, return_distance = True)
        r_indx = [i for i,x in enumerate(indx) for y in range(len(x))] # anchor
        c_indx = [x for y in indx for x in y] # pixel
        wij = np.array([x for y in dist for x in y])
        wij = 1-(wij / self.radius)**self.nu
        wij = coo_array((wij, (r_indx,c_indx)),shape=(n,self.N0)).tocsc().T
        wij.eliminate_zeros()
        nchoice=(wij != 0).sum(axis = 1)
        b_indx = np.arange(self.N0)[nchoice > 0]
        wij = wij[b_indx, :]
        wij.data = np.clip(wij.data, .05, .95)
        return b_indx, grid_pt, wij, theta

    def one_batch(self, batch_index, slda, init_bound):

        pixel_result = pd.DataFrame()
        anchor_result = pd.DataFrame()
        post_count = np.zeros((self.K, self.M))
        for b in batch_index:
            b_indx, grid_pt, wij, theta = self._prepare_batch(b, init_bound)
            if b_indx is None:
                continue
            N = len(b_indx)
            grid_pt = np.array(grid_pt)
            x_min, y_min = grid_pt.min(axis = 0)
            x_max, y_max = grid_pt.max(axis = 0)
            psi_org = sklearn.preprocessing.normalize(wij, norm='l1', axis=1)
            batch = minibatch()
            batch.init_from_matrix(self.dge_mtx[b_indx, :], grid_pt, wij, psi = psi_org, m_gamma = theta)
            _ = slda.do_e_step(batch)

            tmp = copy.copy(self.brc.loc[b_indx, ['j','X','Y']] )
            tmp.index = range(tmp.shape[0])
            tmp = pd.concat([tmp, pd.DataFrame(batch.phi, \
                             columns = self.factor_header)], axis = 1)
            if self.file_is_open:
                v = np.arange(N)[(tmp.X > x_min+self.out_buff) &\
                                (tmp.X < x_max-self.out_buff) &\
                                (tmp.Y > y_min+self.out_buff) &\
                                (tmp.Y < y_max-self.out_buff)]
                post_count += batch.phi[v, :].T @ batch.mtx[v, :]
                tmp = tmp.iloc[v, :]
            else:
                post_count += batch.phi.T @ batch.mtx
            pixel_result = pd.concat([pixel_result, tmp], axis = 0)

            expElog_theta = np.exp(_dirichlet_expectation_2d(batch.gamma))
            expElog_theta/= expElog_theta.sum(axis = 1).reshape((-1, 1))
            tmp = pd.DataFrame({'minibatch':b,'X':grid_pt[:,0],'Y':grid_pt[:,1]})
            asum = batch.psi.T @ batch.mtx.sum(axis = 1).reshape((-1, 1))
            tmp['avg_size'] = np.array(asum).reshape(-1)
            for v in range(self.K):
                tmp[str(v)] = expElog_theta[:, v]
            if self.file_is_open:
                tmp = tmp.loc[(tmp.X > x_min+self.out_buff) & \
                            (tmp.X < x_max-self.out_buff) & \
                            (tmp.Y > y_min+self.out_buff) & \
                            (tmp.Y < y_max-self.out_buff), :]
            anchor_result = pd.concat([anchor_result, tmp], axis = 0)
        return post_count, pixel_result, anchor_result

    def run_chunk(self, slda, init_bound):
        if self.thread > 1:
            idx_slices = [[ self.batch_index[x] for x in y ] for y in utilt.gen_even_slices(len(self.batch_index), self.thread)]
            with Parallel( n_jobs=self.thread, backend='threading', verbose=self.verbose) as parallel:
                result_list = parallel(delayed(self.one_batch)(idx, slda, init_bound) for idx in idx_slices)
            pixel_result = pd.DataFrame()
            anchor_result = pd.DataFrame()
            post_count = np.zeros((self.K, self.M))
            for obj in result_list:
                post_count += obj[0]
                pixel_result = pd.concat([pixel_result, obj[1]], axis = 0)
                anchor_result = pd.concat([anchor_result, obj[2]], axis = 0)
            return post_count, pixel_result, anchor_result
        else:
            return self.one_batch(self.batch_index, slda, init_bound)



    def run_chunk_penalized(self, slda, init_bound):
        assert slda._zeta > 0 and slda._zeta < 1, "To run slda with penalized likelihood, please set slda.zeta within (0,1)"
        if self.adj_mtx is None:
            self._anchor_adj()
        pixel_result = pd.DataFrame()
        anchor_result = pd.DataFrame()
        post_count = np.zeros((self.K, self.M))
        for b in self.batch_index:
            b_indx, grid_pt, wij, theta = self._prepare_batch(b, init_bound)
            if b_indx is None:
                continue
            N = len(b_indx)
            anchor_index = grid_pt.index.to_list()
            grid_pt = np.array(grid_pt)
            psi_org = sklearn.preprocessing.normalize(wij, norm='l1', axis=1)
            batch = minibatch()
            batch.init_from_matrix(self.dge_mtx[b_indx, :], grid_pt, wij, psi = psi_org, m_gamma = theta, anchor_adj = self.adj_mtx[anchor_index, :][:, anchor_index])

            _ = slda.update_lambda_penalized(batch)

            tmp = self.brc.iloc[b_indx, :]
            x_min, x_max = tmp.X.min(), tmp.X.max()
            y_min, y_max = tmp.Y.min(), tmp.Y.max()
            v = np.arange(N)[(tmp.X > x_min+self.out_buff) &\
                             (tmp.X < x_max-self.out_buff) &\
                             (tmp.Y > y_min+self.out_buff) &\
                             (tmp.Y < y_max-self.out_buff)]
            post_count += batch.phi[v, :].T @ batch.mtx[v, :]

            tmp = copy.copy(self.brc.loc[b_indx, ['j','X','Y']] )
            tmp.index = range(tmp.shape[0])
            tmp = pd.concat([tmp, pd.DataFrame(batch.phi, \
                             columns = self.factor_header)], axis = 1)
            tmp = tmp.iloc[v, :]
            pixel_result = pd.concat([pixel_result, tmp], axis = 0)

            expElog_theta = np.exp(_dirichlet_expectation_2d(batch.gamma))
            expElog_theta/= expElog_theta.sum(axis = 1).reshape((-1, 1))
            tmp = pd.DataFrame({'minibatch':b,'X':grid_pt[:,0],'Y':grid_pt[:,1]})
            asum = batch.psi.T @ batch.mtx.sum(axis = 1).reshape((-1, 1))
            tmp['avg_size'] = np.array(asum).reshape(-1)
            for v in range(self.K):
                tmp[str(v)] = expElog_theta[:, v]
            tmp = tmp.loc[(tmp.X > x_min+self.out_buff) & \
                          (tmp.X < x_max-self.out_buff) & \
                          (tmp.Y > y_min+self.out_buff) & \
                          (tmp.Y < y_max-self.out_buff), :]
            anchor_result = pd.concat([anchor_result, tmp], axis = 0)
        return post_count, pixel_result, anchor_result
