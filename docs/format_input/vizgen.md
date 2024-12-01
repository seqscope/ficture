# Process Vizgen MERSCOPE data

Locate the transcript file from your Vizgen MERSCOPE output, most likely named as `detected_transcripts.csv.gz`.

The first few lines may look like (from the public Human lung cancer 1 FFPE dataset)
```bash linenums="1"
,barcode_id,global_x,global_y,global_z,x,y,fov,gene,transcript_id
747,0,2123.7725,156.28409,1.0,349.0,1891.4949,0,PDK4,ENST00000005178
846,0,2239.8335,-1.2225803,2.0,1423.6388,433.0998,0,PDK4,ENST00000005178
2133,0,2133.042,8.58588,5.0,434.82895,523.9189,0,PDK4,ENST00000005178
```

We will collapse pixels from all z-planes to 2D, essentially using only `global_x`, `global_y`, and `transcript_id`.



The following command assume your input is in `inpath` and you you want to store output to `path`. In some data the negative control proves are names "Blank-*", `--dummy_genes Blank` will regard any `transcript_id` containing the substring "Blank" as control probes.

The script `foramt_Vizgen.py` can be found in `misc/` in the FICTURE repository.

```bash
iden=mouselung # Set how you want to call your dataset
input=${inpath}/detected_transcripts.csv.gz
output=${path}/filtered.matrix.${iden}.tsv
feature=${path}/feature.clean.${iden}.tsv.gz
coor=${path}/coordinate_minmax.tsv

python foramt_Vizgen.py --input ${input} --output ${output} --feature ${feature} --coor_minmax ${coor} --precision 2 --dummy_genes Blank

sort -S 4G -k1,1g ${output} | gzip -c > ${output}.gz
rm ${output}
```
