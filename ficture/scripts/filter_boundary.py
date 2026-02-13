### Filtering pixeld based on input boundaries (in the form of polygons)

import sys, os, gzip, argparse, warnings, logging
import numpy as np
import pandas as pd
from matplotlib.path import Path
from geopandas import GeoSeries
import shapely, geojson
from shapely.geometry import Point, Polygon

from ficture.utils.utilt import extract_polygons_from_json, svg_parse_list

def filter_by_boundary(_args):

    parser = argparse.ArgumentParser(prog = "filter_by_boundary")
    parser.add_argument('--input', type=str, help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--boundary', type=str, help='Boundary info (currently only support geojson or tsv with a path column in svg format)')
    parser.add_argument('--boundary_unit_in_um', type=float, default=1, help='')
    parser.add_argument('--feature', type=str, default='', help='')
    parser.add_argument('--mu_scale', type=float, default=26.67, help='Coordinate to um translate')
    parser.add_argument('--offset_x', type=float, default=0, help='In um')
    parser.add_argument('--offset_y', type=float, default=0, help='In um')
    parser.add_argument('--transpose', action='store_true')

    args = parser.parse_args(_args)
    if len(_args) == 0:
        parser.print_help()
        return

    if os.path.exists(args.output):
        warnings.warn("Output file already exists")
    if not os.path.exists(args.input):
        sys.exit("Cannot find input file")
    if not os.path.exists(args.boundary):
        sys.exit("Cannot find input file")

    logging.basicConfig(level= getattr(logging, "INFO", None))

    if args.boundary.endswith('.geojson'):
        vertices = extract_polygons_from_json(args.boundary)
        vertices = [(x * args.boundary_unit_in_um + np.array([args.offset_x, args.offset_y])) for x in vertices]
        n_poly = len(vertices)
        poly_list = [shapely.polygons(x) for x in vertices]
        poly_list = [x if x.is_valid else x.buffer(0) for x in poly_list]
        poly = shapely.unary_union(poly_list)
        # poly = shapely.unary_union([shapely.polygons(x) for x in vertices])
    elif args.boundary.endswith('.tsv'):
        poly_df = pd.read_csv(args.boundary, sep='\t')
        paths = poly_df.path.tolist()
        poly_list = []
        for i in range(len(paths)):
            path = eval(paths[i] )
            codes, verts = svg_parse_list(path)
            verts = np.array(verts)
            # Translate into pixel coordinate
            verts *= args.boundary_unit_in_um
            verts[:, 0] += args.offset_x
            verts[:, 1] += args.offset_y
            mpl_path = Path(verts, codes)
            poly_list.append(Polygon(mpl_path.to_polygons()[0]))
            print(verts.min(axis=0), verts.max(axis=0))
        n_poly = len(poly_list)
        poly_list = [x if x.is_valid else x.buffer(0) for x in poly_list]
        poly = shapely.unary_union(poly_list)
    else:
        sys.exit("Unknown boundary format")

    logging.info(f"Read boundary info with {n_poly} polygons, total area {poly.area:.1f} um^2")

    gene_kept = set()
    if os.path.exists(args.feature):
        feature = pd.read_csv(args.feature, sep='\t', header=0)
        gene_kept = set(feature.gene.values)
        logging.info(f"Read feature info with {len(gene_kept)} genes")

    with gzip.open(args.input, 'rt') as rf:
        header = rf.readline()
    if args.output.endswith('.gz'):
        with gzip.open(args.output, 'wt') as wf:
            _=wf.write(header)
    else:
        with open(args.output, 'w') as wf:
            _=wf.write(header)

    chunk_size = 500000
    for chunk in pd.read_csv(gzip.open(args.input, 'rb'),\
        sep='\t', header=0, chunksize=chunk_size):
        chunk.rename(columns={'x': 'X', 'y': 'Y'}, inplace=True)
        if len(gene_kept) != 0:
            chunk = chunk.loc[chunk.gene.isin(gene_kept)]
            if chunk.shape[0] == 0:
                continue
        print(chunk.loc[:, ['X', 'Y']].max(axis = 0).values / args.mu_scale)
        print(chunk.loc[:, ['X', 'Y']].min(axis = 0).values / args.mu_scale)
        if args.transpose:
            pts = GeoSeries(map(Point, zip(chunk.Y.values / args.mu_scale,\
                                        chunk.X.values / args.mu_scale)))
        else:
            pts = GeoSeries(map(Point, zip(chunk.X.values / args.mu_scale,\
                                        chunk.Y.values / args.mu_scale)))
        chunk = chunk.loc[pts.within(poly).values, :]
        logging.info(f"Output {chunk.shape[0]} rows ...")
        if chunk.shape[0] == 0:
            continue
        chunk.to_csv(args.output, mode='a', sep='\t', index=False, header=False)

if __name__ == "__main__":
    filter_by_boundary(sys.argv[1:])
