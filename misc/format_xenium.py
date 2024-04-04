import sys, os, re, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd

def format_xenium():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input transcript file from Xenium output, likely named transcripts.csv.gz')
    parser.add_argument('--output', type=str, help='Output file (tsv)')
    parser.add_argument('--feature', type=str, default='', help='Output gene list')
    parser.add_argument('--min_phred_score', type=float, default=13, help='Quality score cutoff')
    parser.add_argument('--dummy_genes', type=str, default='', help='A single name or a regex describing the names of negative control probes')
    args = parser.parse_args()

    unit_info=['X','Y','gene','cell_id','overlaps_nucleus']
    oheader = unit_info + ['Count']

    feature=pd.DataFrame()
    xmin=sys.maxsize
    xmax=0
    ymin=sys.maxsize
    ymax=0

    with open(args.output, 'w') as wf:
        _ = wf.write('\t'.join(oheader)+'\n')
    for chunk in pd.read_csv(args.input,header=0,chunksize=500000):
        chunk = chunk.loc[(chunk.qv > args.min_phred_score)]
        chunk.rename(columns = {'feature_name':'gene'}, inplace=True)
        if args.dummy_genes != '':
            chunk = chunk[~chunk.gene.str.contains(args.dummy_genes, flags=re.IGNORECASE, regex=True)]
        chunk.rename(columns = {'x_location':'X', 'y_location':'Y'}, inplace=True)
        chunk['Count'] = 1
        chunk[oheader].to_csv(args.output,sep='\t',mode='a',index=False,header=False,float_format="%.2f")
        logging.info(f"{chunk.shape[0]}")
        feature = pd.concat([feature, chunk.groupby(by='gene').agg({'Count':"sum"}).reset_index()])
        x0 = chunk.X.min()
        x1 = chunk.X.max()
        y0 = chunk.Y.min()
        y1 = chunk.Y.max()
        xmin = min(xmin, x0)
        xmax = max(xmax, x1)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    if os.path.exists(os.path.dirname(args.feature)):
        feature = feature.groupby(by='gene').agg({'Count':"sum"}).reset_index()
        feature.to_csv(args.feature,sep='\t',index=False)

    f = os.path.join( os.path.dirname(args.output), "coordinate_minmax.tsv" )
    with open(f, 'w') as wf:
        wf.write(f"xmin\t{xmin:.2f}\n")
        wf.write(f"xmax\t{xmax:.2f}\n")
        wf.write(f"ymin\t{ymin:.2f}\n")
        wf.write(f"ymax\t{ymax:.2f}\n")

if __name__ == '__main__':
    format_xenium()
