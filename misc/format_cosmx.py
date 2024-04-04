import sys, os, re, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd

def format_cosmx():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input transcript file from CosMx SMI raw output')
    parser.add_argument('--px_to_um', type=float, help='Convert pixel unit as used in x_local_px and x_global_px to micrometer, this number should be found in the README of your SMI output, it is likely 0.12~0.18')
    parser.add_argument('--output', type=str, help='Full path of the output tsv file')
    parser.add_argument('--feature', type=str, default="", help='Full path to store an output gene list')
    parser.add_argument('--gcol', type=str, default="target", help='The column name used as the gene/probe identifier')
    parser.add_argument('--annotation', type=str, nargs='*', default=["cell_ID","CellComp"], help='Additional information to carry over in the output file')
    parser.add_argument('--dummy_genes', type=str, default='NegPrb', help='A single name or a regex describing the names of negative control probes')
    args = parser.parse_args()

    xcol="x_global_px"
    ycol="y_global_px"
    gcol=args.gcol

    unit_info=['X','Y','gene']+args.annotation
    oheader = unit_info + ['Count']

    feature=pd.DataFrame()
    xmin=sys.maxsize
    xmax=0
    ymin=sys.maxsize
    ymax=0

    with open(args.output, 'w') as wf:
        _ = wf.write('\t'.join(oheader)+'\n')
    for chunk in pd.read_csv(args.input,header=0,chunksize=500000):
        chunk.rename(columns = {gcol:'gene'}, inplace=True)
        if args.dummy_genes != '':
            chunk = chunk[~chunk.gene.str.contains(args.dummy_genes, flags=re.IGNORECASE, regex=True)]
        chunk['X'] = chunk[xcol] * args.px_to_um
        chunk['Y'] = chunk[ycol] * args.px_to_um
        chunk['Count'] = 1
        chunk[oheader].to_csv(args.output,sep='\t',mode='a',index=False,header=False,float_format="%.2f")
        logging.info(f"{chunk.shape[0]}")
        feature = pd.concat([feature, chunk.groupby(by='gene').agg({'Count':sum}).reset_index()])
        x0 = chunk.X.min()
        x1 = chunk.X.max()
        y0 = chunk.Y.min()
        y1 = chunk.Y.max()
        xmin = min(xmin, x0)
        xmax = max(xmax, x1)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    if os.path.exists(os.path.dirname(args.feature)):
        feature = feature.groupby(by='gene').agg({'Count':sum}).reset_index()
        feature.to_csv(args.feature,sep='\t',index=False)

    f = os.path.join( os.path.dirname(args.output), "coordinate_minmax.tsv" )
    with open(f, 'w') as wf:
        wf.write(f"xmin\t{xmin:.2f}\n")
        wf.write(f"xmax\t{xmax:.2f}\n")
        wf.write(f"ymin\t{ymin:.2f}\n")
        wf.write(f"ymax\t{ymax:.2f}\n")

if __name__ == '__main__':
    format_cosmx()
