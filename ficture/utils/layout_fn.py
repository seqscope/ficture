import pandas as pd

def layout_map(meta_data, layout, lane = None):

    ### Range / offset
    # Lane & tile ids are treated as string
    # Assume manifest file has an id column with format lane_tile
    mani=pd.read_csv(meta_data, sep='\t')
    mani["lane"] = mani["id"].map(lambda x : x.split('_')[0])
    mani["tile"] = mani["id"].map(lambda x : x.split('_')[1])
    mani["ybin"] = mani.ymax - mani.ymin
    mani["xbin"] = mani.xmax - mani.xmin

    ### Layout
    layout = pd.read_csv(layout, sep='\t', dtype=str)
    for key in ['row', 'col']:
        layout[key] = layout[key].astype(int)

    if lane is not None:
        try:
            lane = [str(lane)]
        except:
            lane = [str(x) for x in list(lane)]
        mani = mani[mani.lane.isin(lane)]
        layout = layout[layout.lane.isin(lane)]

    df = layout.merge(right = mani, on = ["lane", "tile"], how = "left")
    df.sort_values(by = ['lane', 'row', 'col'], inplace=True)
    df.row = df.row - df.row.min()
    df.col = df.col - df.col.min()
    nrows = df.row.max() + 1
    ncols = df.col.max() + 1
    lanes = []
    tiles = []
    for i in range(nrows):
        lanes.append( [None] * ncols )
        tiles.append( [None] * ncols )
    for it, row in df.iterrows():
        i = row['row']
        j = row['col']
        lanes[i][j] = row['lane']
        tiles[i][j] = row['tile']
    df.set_index(["lane", "tile"], inplace=True)

    return df, lanes, tiles
