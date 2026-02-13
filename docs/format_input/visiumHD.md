# Process Visium HD raw data

## Key points
1) Look at the json file, find 'microns_per_pixel' and later set `mu_scale=1/microns_per_pixel` for your analysis.
2) Combine information from multiple raw files and create a single input file *sorted* by one axis, either X or Y.
3) The coordinates in the input files are in "pixel" unit, but please keep track of the min and max coordinate values of X and Y axis in *micrometer*, they would be needed to create the final pixel level visualization.
4) Since Visium HD has resolution $2\mu m$, set `--plot_um_per_pixel 2` when visualizing the final pixel output (in `ficture plot_pixel_full`).

## Alternative: Using `spatula convert-sge` command

The [spatula sge-convert](https://seqscope.github.io/spatula/tools/convert_sge/) tool offers a convenient way to convert Visium HD raw data to FICTURE input format. You may want to use the tool to convert the raw Visium HD data to FICTURE input format instead of following the manual steps below.

## Details

Visium HD output contains a sparse count matrix and a separate parquet file defining pixels' spatial locations.

Locate Visium HD outputs
```bash linenums="1"
brc_parq=/path/to/spatial/tissue_positions.parquet
mtx_path=/path/to/filtered_feature_bc_matrix # or raw_feature_bc_matrix
opath=/output/directory
```

The barcode (`tissue_positions.parquet`) file looks like
```plaintext linenums="1"
barcode 	in_tissue 	array_row 	array_col 	pxl_row_in_fullres 	pxl_col_in_fullres
s_002um_00000_00000-1 	0 	0 	0 	44.189527 	21030.743288
s_002um_00000_00001-1 	0 	0 	1 	44.302100 	21023.440349
```
(but stored in the parquet format)

`pxl_row_in_fullres` and `pxl_col_in_fullres` are in the unit of "pixel", we need to look at the `scalefactors_json.json` file (should be in the same folder as `tissue_positions.parquet`) to get its ratio to micrometer. In our example the json file looks like

```json linenums="1"
{
    "spot_diameter_fullres": 7.303953797779634,
    "bin_size_um": 2.0,
    "microns_per_pixel": 0.2738242950835738,
    "regist_target_img_scalef": 0.2505533,
    "tissue_lowres_scalef": 0.02505533,
    "fiducial_diameter_fullres": 1205.1523766336395,
    "tissue_hires_scalef": 0.2505533
}
```
Record the number `microns_per_pixel` and later set `mu_scale=1/microns_per_pixel` for your analysis.



The matrix directory looks like
```bash linenums="1"
ls ${mtx_path}
```

```bash linenums="1"
barcodes.tsv.gz  features.tsv.gz  matrix.mtx.gz
```

We need to annotate the sparse matrix `matrix.mtx.gz` with barcode locations from `tissue_positions.parquet` by getting the barcode ID from `barcodes.tsv.gz` then lookuping its spatial coordinate. Unfortunately barcodes in `tissue_positions.parquet` and in `barcodes.tsv.gz` are stored in different orders and the parquet file contains a single row group (why??) based on the few public datasets we've inspected, making this process uglier. Although one could read all four files fully in memory and match them, the following is a slower alternative.

The requirements for the merged file are

1) Containing the following columns: X, Y, gene, Count.

2) Is sorted according to one axis. The output from the following commands is sorted first along the Y-axis, so later you would set `major_axis=Y`.

Given the current data formats, we first match barcodes' integer indices in the matrix with their spatial locations, then annotate the spatial locations and gene IDs to the sparse count matrix.

You may need to install a tool to read parquet file, one option is `pip install parquet-tools`. The following command takes ~8.5 min for the public [Visium HD mouse brain](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) dataset.

```bash linenums="1"
bfile=${mtx_path}/barcodes.tsv.gz
mfile=${mtx_path}/matrix.mtx.gz
ffile=${mtx_path}/features.tsv.gz
output=${opath}/transcripts.tsv.gz

# Extract the spatial locations
brc_raw=${opath}/tissue_positions.raw.csv
parquet-tools csv ${brc_parq} > ${brc_raw}

# Check coordinate range (for future record)
microns_per_pixel=0.2738242950835738 # read from json
coor=${opath}/coordinate_minmax.tsv
IFS=' ' read -r xmin xmax ymin ymax <<<$(cut -d',' -f 5-6 ${brc_raw} | tail -n +2 | awk -v mu=$microns_per_pixel -v FS=',' 'NR == 1 { xmin = $1; xmax = $1; ymin = $2; ymax = $2 } {\
if ($1 > xmax) { xmax = $1 }\
if ($1 < xmin) { xmin = $1 }\
if ($2 > ymax) { ymax = $2 }\
if ($2 < ymin) { ymin = $2 }\
} END { print xmin*mu, xmax*mu, ymin*mu, ymax*mu }')
echo -e "xmin\t${xmin}\nxmax\t${xmax}\nymin\t${ymin}\nymax\t${ymax}" > ${coor}
```

Coordinate range in `${opath}/coordinate_minmax.tsv` is in micrometer, for the mouse brain data it looks like
```plaintext linenums="1"
xmin    0
xmax    6806.13
ymin    0
ymax    5858.36
```

Merge coordinate and gene count informations
```bash linenums="1"
# First match barcode index (in the matrix file) with their spatial coordinates (in the tissue_positions file)

awk -v FS=',' -v OFS='\t' 'NR==FNR{bcd[$1]=NR; next} ($1 in bcd){ printf "%d\t%.2f\t%.2f\n", bcd[$1], $2, $3 } ' <(zcat $bfile) <(cut -d',' -f 1,5,6 ${brc_raw}) | sort -k1,1n > ${opath}/barcodes.tsv

# Then annotate the coordinates and gene ID to the matrix file (assume matrix.mtx.gz is sorted by the pixel indices, which seems to be always true)

awk 'BEGIN{FS=OFS="\t"} NR==FNR{ft[NR]=$1 FS $2; next} ($4 in ft) {print $1, $2, $3, ft[$4], $5 }' \
<(zcat $ffile) \
<(\
join -t $'\t' -1 1 -2 2 ${opath}/barcodes.tsv <(zcat $mfile | tail -n +4 | sed 's/ /\t/g') \
) | sort -k3,3n -k2,2n | sed '1 s/^/#barcode_idx\tX\tY\tgene_id\tgene\tCount\n/' | gzip -c > ${output}
```

(You might want to either delte or zip the intermediate files `tissue_positions.raw.csv` and `barcodes.tsv `)

The sorting by coordinates part can take some time for large data, you could check intermediate results first to see if it makes sense.

Output looks like
```bash linenums="1"
zcat $output | head
```

```plaintext linenums="1"
#barcode_idx    X       Y       gene_id gene    Count
4116872 12859.72        318.87  ENSMUSG00000002980      Bcam    1
4116872 12859.72        318.87  ENSMUSG00000022426      Josd1   1
4116872 12859.72        318.87  ENSMUSG00000023885      Thbs2   1
4116872 12859.72        318.87  ENSMUSG00000041120      Nbl1    1
4116872 12859.72        318.87  ENSMUSG00000072235      Tuba1a  1
2159571 12867.01        318.98  ENSMUSG00000015090      Ptgds   1
2159571 12867.01        318.98  ENSMUSG00000028478      Clta    1
```
(The first column is just to retain the original barcode index in `matrix.mtx.gz` (the row number in `barcodes.tsv.gz`), it will be ignored in analysis.)
