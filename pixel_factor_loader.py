import sys, os, gzip, copy, re, logging, warnings
import numpy as np
import pandas as pd
import subprocess as sp

class BlockIndexedLoader:

    def __init__(self, input, xmin = -np.inf, xmax = np.inf, ymin = -np.inf, ymax = np.inf, full = False) -> None:
        self.meta = {}
        nheader = 0
        with gzip.open(input, 'rt') as rf:
            for line in rf:
                if line[0] != "#":
                    break
                nheader += 1
                if line[:2] == "##":
                    wd = line[(line.rfind("#")+1):].strip().split(';')
                    wd = [[y.strip() for y in x.strip().split("=")] for x in wd]
                    for v in wd:
                        if v[1].lstrip('-+').isdigit():
                            self.meta[v[0]] = int(v[1])
                        elif v[1].replace('.','',1).lstrip('-+').isdigit():
                            self.meta[v[0]] = float(v[1])
                        else:
                            self.meta[v[0]] = v[1]
                else:
                    header = line[(line.rfind("#")+1):].strip().split('\t')
        logging.info("Read header %s", self.meta)

        # Input reader
        self.kheader = ['K'+str(k+1) for k in range(self.meta['TOPK'])]
        self.pheader = ['P'+str(k+1) for k in range(self.meta['TOPK'])]
        dty={'BLOCK':str, 'X':int, 'Y':int}
        dty.update({x : str for x in self.kheader})
        dty.update({x : np.float16 for x in self.pheader})
        self.file_is_open = True
        if full or np.isfinite(xmin) or np.isfinite(xmax):
            self.reader = pd.read_csv(input,sep='\t',skiprows=nheader,chunksize=1000000,names=header, dtype=dty)
        else:
            # Translate target region to index
            block = [int(x / self.meta['BLOCK_SIZE']) for x in [xmin, xmax - 1] ]
            pos_range = [int((x - self.meta['OFFSET_Y'])*self.meta['SCALE']) for x in [ymin, ymax]]
            if self.meta['BLOCK_AXIS'] == "Y":
                block = [int(x / self.meta['BLOCK_SIZE']) for x in [ymin, ymax - 1] ]
                pos_range = [int((x - self.meta['OFFSET_X'])*self.meta['SCALE']) for x in [xmin, xmax]]
            block = np.arange(block[0], block[1]+1) * self.meta['BLOCK_SIZE']
            query = []
            pos_range = '-'.join([str(x) for x in pos_range])
            for i,b in enumerate(block):
                query.append( str(b)+':'+pos_range )

            cmd = ["tabix", input]+query
            process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
            self.reader = pd.read_csv(process.stdout,sep='\t',chunksize=1000000,names=header, dtype=dty)
            logging.info(" ".join(cmd))

        self.xmin = max(xmin, self.meta['OFFSET_X'])
        self.xmax = min(xmax, self.meta['OFFSET_X'] + self.meta["SIZE_X"])
        self.ymin = max(ymin, self.meta['OFFSET_Y'])
        self.ymax = min(ymax, self.meta['OFFSET_Y'] + self.meta["SIZE_Y"])

    def __iter__(self):
        return self

    def __next__(self):
        if not self.file_is_open:
            raise StopIteration
        try:
            chunk = next(self.reader)
        except StopIteration:
            self.file_is_open = False
            raise StopIteration
        chunk['X']=chunk.X/self.meta['SCALE']+self.meta['OFFSET_X']
        chunk['Y']=chunk.Y/self.meta['SCALE']+self.meta['OFFSET_Y']
        drop_index = chunk.index[(chunk.X<self.xmin)|(chunk.X>self.xmax)|\
                                 (chunk.Y<self.ymin)|(chunk.Y>self.ymax)]
        return chunk.drop(index=drop_index)
