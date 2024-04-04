import copy
import numpy as np
import pandas as pd
import sklearn.neighbors
from scipy.sparse import *
import matplotlib as mpl

#########################################################
############# Whole data in memory, write image by row
#########################################################
class ImgRowIterator:
    def __init__(self, pts, mtx, radius, verbose=500, chunksize=500):
        self.pts = pts
        self.N = self.pts.shape[0]
        self.width, self.height = self.pts.max(axis = 0) + 1
        self.current = -1
        self.mtx = mtx
        self.dt = mtx.dtype
        self.verbose = verbose
        self.radius = radius
        self.chunksize = chunksize
        print(f"Image size (w x h): {self.width} x {self.height}")
        self.buffer_lower = self.pts[:, 1].min()
        self.buffer_upper = self.buffer_lower + self.chunksize
        indx = (self.pts[:, 1] >= self.buffer_lower) & (self.pts[:, 1] <= self.buffer_upper)
        self.buffer_index = np.arange(self.N)[indx]
        while len(self.buffer_index) < 2 and self.buffer_upper < self.height:
            self.buffer_upper += self.chunksize
            indx = (self.pts[:, 1] >= self.buffer_lower) & (self.pts[:, 1] <= self.buffer_upper)
            self.buffer_index = np.arange(self.N)[indx]
        assert len(self.buffer_index) > 1, "Input coordinates are ouside input range"
        if self.radius >= 1:
            self.ref = sklearn.neighbors.BallTree(self.pts[self.buffer_index, :])
        print(f"Initialized buffer for block {self.buffer_lower} - {self.buffer_upper}")
        return

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current % self.verbose == 0:
            print(f"{self.current}/{self.height}")
        if self.current >= self.height:
            raise StopIteration
        self.update_buffer()
        nodes = np.array([[i, self.current] for i in range(self.width)])
        out = np.zeros(self.width * 3, dtype = self.dt)
        if self.radius < 1:
            iv = np.arange(self.N)[self.pts[:, 1] == self.current]
            iu = self.pts[iv, 0]
        else:
            dv, iv = self.ref.query(nodes, k = 1)
            indx = dv[:, 0] < self.radius
            iv = self.buffer_index[iv[indx, 0]]
            iu = np.arange(self.width)[indx]
        if len(iv) == 0:
            return out
        for c in range(3):
            out[iu*3+c] = self.mtx[iv, c]
        return out

    def update_buffer(self):
        if self.radius < 1:
            return
        if self.current < self.buffer_upper - self.radius - 1:
            return
        if self.buffer_upper >= self.height:
            return
        self.buffer_lower = self.current - self.radius - 1
        self.buffer_upper = self.buffer_lower + self.chunksize
        indx = (self.pts[:, 1] >= self.buffer_lower) & (self.pts[:, 1] <= self.buffer_upper)
        self.buffer_index = np.arange(self.N)[indx]
        while len(self.buffer_index) < 2 and self.buffer_upper < self.height:
            self.buffer_upper += self.chunksize
            indx = (self.pts[:, 1] >= self.buffer_lower) & (self.pts[:, 1] <= self.buffer_upper)
            self.buffer_index = np.arange(self.N)[indx]
        if len(self.buffer_index) > 0:
            self.ref = sklearn.neighbors.BallTree(self.pts[self.buffer_index, :])
            return





#########################################################
############# Stream in data, write image by row
#########################################################
class ImgRowIterator_stream:
    def __init__(self, reader, w, h, cmtx,\
                 xmin = 0, ymin = 0, pixel_size = 1, \
                 verbose=500, dtype=np.uint8, plot_top=0):
        self.reader = reader
        self.cmtx = cmtx
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + w
        self.ymax = ymin + h
        self.width  = int((self.xmax - self.xmin) / pixel_size) + 1
        self.height = int((self.ymax - self.ymin) / pixel_size) + 1
        self.pixel_size = pixel_size
        self.K = cmtx.shape[0]
        self.buffer_y = -1
        self.current = -1
        self.dtype = dtype
        self.verbose = verbose
        self.plot_top = plot_top
        self.file_is_open = True
        self.feature_header = [str(k) for k in range(self.K)]
        self.data_header = ["x", "y"] + self.feature_header
        self.pts = pd.DataFrame([], columns = ["x", "y"])
        self.mtx = np.zeros((0, 3))
        self.leftover = pd.DataFrame([], columns = self.data_header)
        while self.buffer_y < 0 and self.file_is_open:
            self.file_is_open = self.read_chunk()
        if self.buffer_y < 0:
            print("Input does not contain pixels in range")
            return
        print(f"Image size (w x h): {self.width} x {self.height}")
        y0, y1 = self.pts.y.min(), self.pts.y.max()
        N = self.pts.shape[0]
        print(f"Current buffer: ({y0}, {y1}), {N}")
        return

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current % self.verbose == 0:
            print(f"{self.current}/{self.height}")
        if self.current >= self.height:
            raise StopIteration
        while self.buffer_y < self.current and self.file_is_open:
            # Read more data
            self.file_is_open = self.read_chunk()
        iv = np.arange(self.pts.shape[0])[self.pts.y.eq(self.current)]
        out = np.zeros(self.width * 3, dtype = self.dtype)
        if len(iv) == 0:
            return out
        for c in range(3):
            for i in iv:
                out[self.pts.x.iloc[i]*3+c] = self.mtx[i, c]
        return out

    def read_chunk(self):
        try:
            chunk = next(self.reader)
        except StopIteration:
            print(f"Reach the end of file")
            return 0
        if chunk.y.min() > self.ymax:
            print(f"Read all pixels in range")
            return 0
        chunk = chunk[(chunk.x > self.xmin) & (chunk.x < self.xmax) &\
                      (chunk.y > self.ymin) & (chunk.y < self.ymax)]
        if chunk.shape[0] == 0:
            return 1
        chunk.x -= self.xmin
        chunk.y -= self.ymin
        # Translate into image pixel coordinates
        chunk['x'] = np.round(chunk.x.values / self.pixel_size, 0).astype(int)
        chunk['y'] = np.round(chunk.y.values / self.pixel_size, 0).astype(int)
        # Concatenate with leftover
        chunk = pd.concat([self.leftover.loc[:, self.data_header],\
                           chunk.loc[:, self.data_header]])
        # Save the last row (incomplete) for later
        indx = chunk.y.eq(chunk.y.max())
        self.leftover = copy.copy(chunk.loc[indx, self.data_header])
        if np.sum(indx) == chunk.shape[0]: # Need to read more
            return 1
        chunk = chunk.loc[~indx, :]
        # Collapse
        chunk = chunk.groupby(by = ["x", "y"]).agg({\
                      x:np.mean for x in self.feature_header }).reset_index()
        self.pts = chunk.loc[:, ["x", "y"]]
        self.pts.x = np.clip(self.pts.x, 0, self.width-1)
        self.pts.y = np.clip(self.pts.y, 0, self.height-1)
        self.buffer_y = self.pts.y.max()
        if self.plot_top:
            N = self.pts.shape[0]
            self.mtx = coo_array((np.ones(N,dtype=self.dtype), (range(N),\
                np.array(chunk.loc[:, self.feature_header]).argmax(axis = 1))),\
                shape=(N, self.K)).toarray()
            self.mtx = np.clip(np.around(self.mtx @ self.cmtx * 255),0,255).astype(self.dtype)
        else:
            self.mtx = np.clip(np.around(np.array(\
                    chunk.loc[:, self.feature_header]) @ self.cmtx * 255),\
                    0, 255).astype(self.dtype)
        return 1








#########################################################
############# Stream in data, write image by row
#########################################################
class ImgRowIterator_stream_singlechannel:
    def __init__(self, reader, w, h, key, \
                cmap='plasma', xmin = 0, ymin = 0, cutoff = .05,\
                pixel_size = 1, verbose = 500, dtype = np.uint8):
        self.reader = reader
        self.cmap = cmap
        self.key = key
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmin + w
        self.ymax = ymin + h
        self.width  = int((self.xmax - self.xmin) / pixel_size) + 1
        self.height = int((self.ymax - self.ymin) / pixel_size) + 1
        self.pixel_size = pixel_size
        self.buffer_y = -1
        self.current = -1
        self.dtype = dtype
        self.cutoff = cutoff
        self.verbose = verbose
        self.file_is_open = True
        self.data_header = ["x", "y", self.key]
        self.pts = pd.DataFrame([], columns = ["x", "y"])
        self.mtx = np.zeros((0, 3))
        self.leftover = pd.DataFrame([], columns = self.data_header)
        while self.buffer_y < 0 and self.file_is_open:
            self.file_is_open = self.read_chunk()
        if self.buffer_y < 0:
            print("Input does not contain pixels in range")
            return
        print(f"Image size (w x h): {self.width} x {self.height}")
        if self.cmap not in mpl.colormaps():
            self.cmap = "plasma"
        y0, y1 = self.pts.y.min(), self.pts.y.max()
        N = self.pts.shape[0]
        print(f"Current buffer: ({y0}, {y1}), {N}")
        return

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current % self.verbose == 0:
            print(f"{self.current}/{self.height}")
        if self.current >= self.height:
            raise StopIteration
        while self.buffer_y < self.current and self.file_is_open:
            # Read more data
            self.file_is_open = self.read_chunk()
        iv = np.arange(self.pts.shape[0])[self.pts.y.eq(self.current)]
        out = np.zeros(self.width * 3, dtype = self.dtype)
        if len(iv) == 0:
            return out
        for c in range(3):
            for i in iv:
                out[self.pts.x.iloc[i]*3+c] = self.mtx[i, c]
        return out

    def read_chunk(self):
        try:
            chunk = next(self.reader)
        except StopIteration:
            print(f"Reach the end of file")
            return 0
        if chunk.y.min() > self.ymax:
            print(f"Read all pixels in range")
            return 0
        # Crop
        chunk = chunk[(chunk.x > self.xmin) & (chunk.x < self.xmax) &\
                      (chunk.y > self.ymin) & (chunk.y < self.ymax) &\
                      (chunk[self.key] > self.cutoff)]
        if chunk.shape[0] == 0:
            return 1
        chunk.x -= self.xmin
        chunk.y -= self.ymin
        # Translate into image pixel coordinates
        chunk['x'] = np.round(chunk.x.values / self.pixel_size, 0).astype(int)
        chunk['y'] = np.round(chunk.y.values / self.pixel_size, 0).astype(int)
        # Concatenate with leftover
        chunk = pd.concat([self.leftover.loc[:, self.data_header],\
                           chunk.loc[:, self.data_header]])
        # Save the last row (incomplete) for later
        indx = chunk.y.eq(chunk.y.max())
        self.leftover = copy.copy(chunk.loc[indx, self.data_header])
        if np.sum(indx) == chunk.shape[0]: # Need to read more
            return 1
        # Collapse
        chunk = chunk.loc[~indx, :]
        chunk = chunk.groupby(by = ["x", "y"]).agg({\
                              self.key : np.mean }).reset_index()
        self.pts = chunk.loc[:, ["x", "y"]]
        self.pts.x = np.clip(self.pts.x, 0, self.width-1)
        self.pts.y = np.clip(self.pts.y, 0, self.height-1)
        self.buffer_y = self.pts.y.max()
        v = np.clip(chunk[self.key].values,0,1)
        self.mtx = np.clip(mpl.colormaps[self.cmap](v)[:,:3] * 255,\
                           0, 255).astype(self.dtype)
        return 1
