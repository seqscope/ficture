import sys, os, re, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd

def format_vizgen():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input transcript file from Vizgen MERSCOPE, likely named like detected_transcripts.csv.gz')
    parser.add_argument('--output', type=str, help='Output transcript file')
    parser.add_argument('--feature', type=str, default='', help='Output gene list')
    parser.add_argument('--coor_minmax', type=str, default='', help='Record coordinate ranges to a file')
    parser.add_argument('--dummy_genes', type=str, default='', help='A single name or a regex describing the names of negative control probes')
    parser.add_argument('--precision', type=int, default=2, help='Number of digits to store the transcript coordinates in micrometer')
    parser.add_argument('--debug', action="store_true", help='')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    unit_info=['X','Y','gene','MoleculeID']
    oheader = unit_info+['Count']
    float_format = f"%.{args.precision}f"

    feature=pd.DataFrame()

    if args.output.endswith('.gz'):
        with gzip.open(args.output, 'wt') as wf:
            _ = wf.write('\t'.join(oheader)+'\n')
    else:
        with open(args.output, 'w') as wf:
            _ = wf.write('\t'.join(oheader)+'\n')

    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf

    for chunk in pd.read_csv(args.input,header=0,chunksize=500000,index_col=0):
        if args.dummy_genes != '':
            chunk = chunk[~chunk.gene.str.contains(args.dummy_genes, flags=re.IGNORECASE, regex=True)]
        chunk.rename(columns={'global_x':'X','global_y':'Y'},inplace=True)
        chunk['Count'] = 1
        chunk['MoleculeID'] = chunk.index.values
        x,y = chunk[['X','Y']].values.min(axis = 0)
        xmin = min(xmin,x)
        ymin = min(ymin,y)
        x,y = chunk[['X','Y']].values.max(axis = 0)
        xmax = max(xmax,x)
        ymax = max(ymax,y)
        chunk[oheader].to_csv(args.output,sep='\t',mode='a',index=False,header=False,float_format=float_format)
        logging.info(f"{chunk.shape[0]}")
        feature = pd.concat([feature, chunk.groupby(by=['gene','transcript_id']).agg({'Count':sum}).reset_index()])
        if args.debug:
            break

    if os.path.exists(os.path.dirname(args.feature)):
        feature = feature.groupby(by=['gene','transcript_id']).agg({'Count':sum}).reset_index()
        feature.to_csv(args.feature,sep='\t',index=False)

    if os.path.exists(os.path.dirname(args.coor_minmax)):
        line = f"xmin\t{xmin:.{args.precision}f}\nxmax\t{xmax:.{args.precision}f}\nymin\t{ymin:.{args.precision}f}\nymax\t{ymax:.{args.precision}f}\n"
        with open(args.coor_minmax, 'w') as wf:
            _ = wf.write(line)

if __name__ == '__main__':
    format_vizgen()
