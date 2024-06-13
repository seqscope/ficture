import sys, os, gzip, copy, re, logging, warnings
import numpy as np
import pandas as pd
import subprocess as sp

class BlockIndexedLoader:

    def __init__(self, input, xmin = -np.inf, xmax = np.inf, ymin = -np.inf, ymax = np.inf, full = False, offseted = True, filter_cmd = "", idtype={}, chunksize=1000000) -> None:
        self.meta = {}
        self.header = []
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
                    self.header = line[(line.rfind("#")+1):].strip().split('\t')
        logging.basicConfig(level= getattr(logging, "INFO", None), format='%(asctime)s %(message)s', datefmt='%I:%M:%S %p')
        logging.info("Read header %s", self.meta)

        if np.isinf(xmin) and np.isinf(xmax) and np.isinf(ymin) and np.isinf(ymax):
            full = True
        self.xmin = xmin if offseted else xmin - self.meta['OFFSET_X']
        self.xmax = xmax if offseted else xmax - self.meta['OFFSET_X']
        self.ymin = ymin if offseted else ymin - self.meta['OFFSET_Y']
        self.ymax = ymax if offseted else ymax - self.meta['OFFSET_Y']
        # Input reader
        dty={'BLOCK':str, 'X':int, 'Y':int}
        if 'TOPK' in self.meta:
            dty.update({f"K{k+1}" : str for k in range(self.meta['TOPK']) })
            dty.update({f"P{k+1}" : float for k in range(self.meta['TOPK']) })
        if len(idtype) > 0:
            dty.update(idtype)
        self.file_is_open = True
        self.xmin = max(xmin, 0)
        self.xmax = min(xmax, self.meta["SIZE_X"])
        self.ymin = max(ymin, 0)
        self.ymax = min(ymax, self.meta["SIZE_Y"])
        if full:
            self.reader = pd.read_csv(input,sep='\t',skiprows=nheader,chunksize=chunksize,names=self.header, dtype=dty)
        else:
            # Translate target region to index
            if self.meta['BLOCK_AXIS'] == "Y":
                block = [int(x / self.meta['BLOCK_SIZE']) for x in [self.ymin, self.ymax - 1] ]
                pos_range = [int(x*self.meta['SCALE']) for x in [self.xmin, self.xmax]]
            else:
                block = [int(x / self.meta['BLOCK_SIZE']) for x in [self.xmin, self.xmax - 1] ]
                pos_range = [int(x*self.meta['SCALE']) for x in [self.ymin, self.ymax]]
            block = np.arange(block[0], block[1]+1) * self.meta['BLOCK_SIZE']
            query = []
            pos_range = '-'.join([str(x) for x in pos_range])
            for i,b in enumerate(block):
                query.append( str(b)+':'+pos_range )

            cmd = " ".join( ["tabix", input] + query )
            if filter_cmd != "":
                cmd = cmd + " | " + filter_cmd
            logging.info(cmd)
            process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, shell=True)
            self.reader = pd.read_csv(process.stdout,sep='\t',chunksize=chunksize,names=self.header, dtype=dty)

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
        chunk['X']=chunk.X/self.meta['SCALE']
        chunk['Y']=chunk.Y/self.meta['SCALE']
        drop_index = chunk.index[(chunk.X<self.xmin)|(chunk.X>self.xmax)|\
                                 (chunk.Y<self.ymin)|(chunk.Y>self.ymax)]
        return chunk.drop(index=drop_index)
