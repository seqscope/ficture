import sys, os, re, copy, gzip, time, logging, pickle, argparse
import numpy as np
import pandas as pd


def get_xy_max_from_img(transcripts_path):
    """
    Get the maximum X and Y coordinates in micrometers from the morphology OME-TIFF image file.
    Expects the transcripts_path to be in the original Xenium output directory structure.
    Expects a morphology_focus/morphology_focus_0000.ome.tif file to be present.
    If not found, defaults to returning None, None.
    """
    import os
    import tifffile
    import xml.etree.ElementTree as ET

    #from transcripts parquet, get parent and img directory
    parent_dir = os.path.dirname(transcripts_path)
    img_path = os.path.join(parent_dir, "morphology_focus", "morphology_focus_0000.ome.tif")
    #if image exists, proceed
    if not os.path.exists(img_path):
        print("Using transcripts file to set image boundaries")
        return None, None
    else: 
        print(f"Using {img_path} to set image boundaries")
        #get image metadata to access shape
        img = tifffile.TiffFile(img_path)
        root = ET.fromstring(img.ome_metadata)
        #iterate through xml to find Pixels tag and get PhysicalSizeX and PhysicalSizeY
        for elem in root.iter():
            if 'Pixels' in elem.tag:
                x_res, y_res = float(elem.attrib['PhysicalSizeX']), float(elem.attrib['PhysicalSizeY'])
        
        #get image shape in pixels
        y_px, x_px = img.pages[0].shape
        #calculate image shape in um
        y_um, x_um = round(y_px * y_res), round(x_px * x_res)
        return y_um, x_um

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

    with open(args.output, 'w') as wf:
        _ = wf.write('\t'.join(oheader)+'\n')

    #create a generator based on file type
    if args.input.endswith('.parquet'):
        import pyarrow.parquet as pq
        import pyarrow
        parq =  pq.ParquetFile(args.input)
        generator = parq.iter_batches(batch_size=500000)
    elif args.input.endswith('.csv.gz'):
        generator = pd.read_csv(args.input,header=0,chunksize=500000)

    feature=pd.DataFrame()
    #xmin=sys.maxsize

    #xmax=0
    #ymin=sys.maxsize
    #ymax=0
    #get x and y max from tif image

    ymax, xmax = get_xy_max_from_img(args.input)
    if ymax is None and xmax is None:
        print("Image file not found, calculating min/max from transcript coordinates")
        xmax = 0
        ymax = 0
        ymin = sys.maxsize
        xmin = sys.maxsize
    else:
        xmin = 0
        ymin = 0

    for chunk in generator:
        #if chunk is of typ py.lib.RecordBatch, convert to pandas dataframe
        if type(chunk) is pyarrow.lib.RecordBatch:
            chunk = chunk.to_pandas()
        chunk = chunk.loc[(chunk.qv > args.min_phred_score)]
        chunk.rename(columns = {'feature_name':'gene'}, inplace=True)
        if args.dummy_genes != '':
            chunk = chunk[~chunk.gene.str.contains(args.dummy_genes, flags=re.IGNORECASE, regex=True)]
        chunk.rename(columns = {'x_location':'X', 'y_location':'Y'}, inplace=True)
        chunk['Count'] = 1
        chunk[oheader].to_csv(args.output,sep='\t',mode='a',index=False,header=False,float_format="%.2f")
        logging.info(f"{chunk.shape[0]}")
        feature = pd.concat([feature, chunk.groupby(by='gene').agg({'Count':"sum"}).reset_index()])
        if ymax is None and xmax is None:
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

    print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

    f = os.path.join( os.path.dirname(args.output), "coordinate_minmax.tsv" )
    with open(f, 'w') as wf:
        wf.write(f"xmin\t{xmin:.2f}\n")
        wf.write(f"xmax\t{xmax:.2f}\n")
        wf.write(f"ymin\t{ymin:.2f}\n")
        wf.write(f"ymax\t{ymax:.2f}\n")

if __name__ == '__main__':
    format_xenium()
