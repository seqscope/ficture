# Process Visium HD raw data

Visium HD output contains a sparse count matrix and a separate file defining pixels' spatial locations. The following is how to combine these two pieces of information with command lines.

Visium HD outputs
```
brc_pos=/path/to/tissue_positions.csv
mtx_path=/path/to/count/matrix/filtered_feature_bc_matrix
opath=/output/directory
```

The `brc_pos`, (`tissue_positions.csv`) file looks like
```
barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres
CGAGGATATTCAGAGC-1,0,0,0,45641,6872
TCTGGTACTAATGCGG-1,0,0,2,45641,7238
```

The matrix directory looks like
```
ls ${mtx_path}
```

```
barcodes.tsv.gz  features.tsv.gz  matrix.mtx.gz
```

The following command does the following steps
1 Match the barcode indices in the sparse matrix (`matrix.mtx.gz`) with their global spatial locations, here would be the 5th and 6th columns, `pxl_row_in_fullres,pxl_col_in_fullres`, in `tissue_positions.csv`
2 Match gene ids with gene indices (this is trivial)
3 Combine all the information into one file, sort by coordinate on one axis (here is the X-axis).

```
bfile=${mtx_path}/barcodes.tsv.gz
mfile=${mtx_path}/matrix.mtx.gz
ffile=${mtx_path}/features.tsv.gz

output=${opath}/transcripts.tsv.gz

awk 'BEGIN{FS=OFS="\t"} NR==FNR{ft[NR]=$1 FS $2; next} ($5 in ft) {print $1 FS $3 FS $4 FS ft[$5] FS $6 }' <(zcat $ffile) <(join -t $'\t' -1 4 -2 2 <(join -t $'\t' -1 1 -2 1 <(cut -d',' -f 1,5,6 ${brc_pos} | sed 's/,/\t/g' | sort -k1,1 ) <(zcat ${mtx_path}/barcodes.tsv.gz | cat -n | tr -s ' ' | awk ' {print $2 "\t" $1} ' ) ) <(zcat $mfile | tail -n +4 | sed 's/ /\t/g') ) | sort -S 1G -k2,2n -k3,3n | sed '1 s/^/#barcode_idx\tX\tY\tgene_id\tgene\tCount\n/' | gzip -c > ${output}

```
(The sorting by coordinates part can take some time for large data, you could check the unsorted output from the above command up to right before `sort -S 1G -k2,2n -k3,3n`  first to see if it makes sense)

Output looks like
```
zcat $output | head
```

```
#barcode_idx    X       Y       gene_id gene    Count
1       13089   15433   ENSMUSG00000000001      Gnai3   13
1       13089   15433   ENSMUSG00000000028      Cdc45   8
1       13089   15433   ENSMUSG00000000056      Narf    12
1       13089   15433   ENSMUSG00000000058      Cav2    13
```
(The first column is just to retain the original barcode index in `matrix.mtx.gz`, it will be ignored in analysis)
