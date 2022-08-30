import pandas as pd

def layout_map(meta_data, layout, lane = None):

    ### Range / offset
    mani=pd.read_csv(meta_data, sep='\t')
    mani["lane"] = mani["id"].map(lambda x : x.split('_')[0]).astype(int)
    mani["tile"] = mani["id"].map(lambda x : x.split('_')[1]).astype(int)

    ### Layout
    layout = pd.read_csv(layout, sep='\t', dtype=int)

    if lane is not None:
        try: 
            lane = int(lane)
            lane = [lane]
        except:
            lane = [int(x) for x in list(lane)]
        mani = mani[mani.lane.isin(lane)]
        layout = layout[layout.lane.isin(lane)]

    layout.sort_values(by = ['lane', 'row', 'col'], inplace=True)
    df = layout.merge(right = mani[["lane", "tile", 'xmin', 'xmax', 'ymin', 'ymax']],
                      on = ["lane", "tile"], how = "left")
    df.row = df.row - df.row.min()
    df.col = df.col - df.col.min()
    nrows = df.row.max() + 1
    ncols = df.col.max() + 1
    lanes = []
    tiles = []
    for i in range(nrows):
        lanes.append( [None] * ncols )
        tiles.append( [None] * ncols )
    for index, row in df.iterrows():
        i = int(row['row'])
        j = int(row['col'])
        lanes[i][j] = str(row['lane'])
        tiles[i][j] = str(row['tile'])

    return df, lanes, tiles