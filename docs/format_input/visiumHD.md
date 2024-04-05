# Process Visium HD raw data

Visium HD output contains a sparse count matrix and a separate parquet file defining pixels' spatial locations. The following is how to combine these two pieces of information before analysis. The following is test on

Visium HD outputs
```bash
brc_parq=/path/to/spatial/tissue_positions.parquet
mtx_path=/path/to/filtered_feature_bc_matrix # or raw_feature_bc_matrix
opath=/output/directory
```

The barcode (`tissue_positions.parquet`) file looks like
```
barcode 	in_tissue 	array_row 	array_col 	pxl_row_in_fullres 	pxl_col_in_fullres
s_002um_00000_00000-1 	0 	0 	0 	44.189527 	21030.743288
s_002um_00000_00001-1 	0 	0 	1 	44.302100 	21023.440349
```
(but stored in the parquet format)


The matrix directory looks like
```bash
ls ${mtx_path}
```

```
barcodes.tsv.gz  features.tsv.gz  matrix.mtx.gz
```

We need to annotate the sparse matrix `matrix.mtx.gz` with barcode locations from `tissue_positions.parquet` by getting the barcode ID from `barcodes.tsv.gz` then lookuping its spatial coordinate. Unfortunately barcodes in `tissue_positions.parquet` and in `barcodes.tsv.gz` are stored in different orders and the parquet file contains a single row group (why??) based on the few public datasets we've inspected, making this process uglier. Although one could read all four files fully in memory and match them, the following is a slower alternative.


Given the current data formats, we first match barcodes integer indices in the matrix and their spatial locations.

You may need to install a tool to read parquet file, one option is `pip install parquet-tools`. The following command takes ~8.5 min for the public [Visium HD mouse brain](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-brain-he) dataset.

```bash
bfile=${mtx_path}/barcodes.tsv.gz
mfile=${mtx_path}/matrix.mtx.gz
ffile=${mtx_path}/features.tsv.gz
output=${opath}/transcripts.tsv.gz

awk 'BEGIN{FS=OFS="\t"} NR==FNR{ft[NR]=$1 FS $2; next} ($4 in ft) {print $1, $2, $3, ft[$4], $5 }' \
<(zcat $ffile) \
<(\
  join -t $'\t' -1 1 -2 2 <(awk -v FS=',' -v OFS='\t' 'NR==FNR{bcd[$1]=NR; next} ($1 in bcd){ printf "%d\t%.2f\t%.2f\n", bcd[$1], $2, $3 } ' <(zcat $bfile) <(parquet-tools csv ${brc_parq} | cut -d',' -f 1,5,6) | sort -k1,1n) <(zcat $mfile | tail -n +4 | sed 's/ /\t/g')
) | sort -k2,2n -k3,3n | sed '1 s/^/#barcode_idx\tX\tY\tgene_id\tgene\tCount\n/' | gzip -c > ${output}
```

The sorting by coordinates part can take some time for large data, you could check intermediate results first to see if it makes sense.

Output looks like
```bash
zcat $output | head
```

```
#barcode_idx    X       Y       gene_id gene    Count
5680891 329.31  15798.08        ENSMUSG00000019790      Stxbp5  1
4885554 329.65  15776.17        ENSMUSG00000037343      Taf2    1
665018  329.76  15768.87        ENSMUSG00000032870      Smap2   1
665018  329.76  15768.87        ENSMUSG00000034832      Tet3    1
665018  329.76  15768.87        ENSMUSG00000064354      mt-Co2  1
```
(The first column is just to retain the original barcode index in `matrix.mtx.gz` (the row number in `barcodes.tsv.gz`), it will be ignored in analysis)
