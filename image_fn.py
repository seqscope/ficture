import copy
import numpy as np
import pandas as pd
import sklearn.neighbors

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
        dv, iv = self.ref.query(nodes, k = 1)
        indx = dv[:, 0] < self.radius
        iv = self.buffer_index[iv[indx, 0]]
        iu = np.arange(self.width)[indx]
        if sum(indx) == 0:
            return np.zeros(self.width * 3, dtype = self.dt)
        out = np.zeros(self.width * 3, dtype = self.dt)
        for c in range(3):
            out[iu*3+c] = self.mtx[iv, c]
        return out

    def update_buffer(self):
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

class ImgRowIterator_stream:
    def __init__(self, reader, w, h, cmtx,\
                 xmin = 0, ymin = 0, pixel_size = 1, \
                 verbose=500, dtype=np.uint8):
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
        self.file_is_open = True
        self.feature_header = [str(k) for k in range(self.K)]
        self.data_header = ["x", "y"] + self.feature_header
        self.pts = np.zeros((0, 2))
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
        if iv.sum() == 0:
            out = np.zeros(self.width * 3, dtype = self.dtype)
            return out
        out = np.zeros(self.width * 3, dtype = self.dtype)
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
        # Crop
        chunk = chunk[(chunk.x > self.xmin) & (chunk.x < self.xmax) &\
                      (chunk.y > self.ymin) & (chunk.y < self.ymax)]
        if chunk.shape[0] == 0:
            return 1
        chunk.x -= self.xmin
        chunk.y -= self.ymin
        # Concatenate with leftover
        chunk = pd.concat([self.leftover.loc[:, self.data_header],\
                           chunk.loc[:, self.data_header]])
        # Translate into image pixel coordinates
        chunk['x'] = np.round(chunk.x.values / self.pixel_size, 0).astype(int)
        chunk['y'] = np.round(chunk.y.values / self.pixel_size, 0).astype(int)
        # Save the last row (incomplete) for later
        indx = chunk.y.eq(chunk.y.iloc[-1])
        self.leftover = copy.copy(chunk.loc[indx, self.data_header])
        self.leftover.x *= self.pixel_size # Back to um
        self.leftover.y *= self.pixel_size
        if chunk.y.iloc[0] == chunk.y.iloc[-1]: # Need to read more
            return 1
        # Collapse
        chunk = chunk.loc[~indx, :]
        chunk = chunk.groupby(by = ["x", "y"]).agg({\
                      x:np.mean for x in self.feature_header }).reset_index()
        self.pts = chunk.loc[:, ["x", "y"]]
        self.buffer_y = self.pts.y.max()
        self.mtx = np.clip(np.around(np.array(\
                   chunk.loc[:, self.feature_header]) @ self.cmtx * 255),\
                   0, 255).astype(self.dtype)
        return 1
