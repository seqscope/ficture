# Process 10X xenium data

Locate the transcript file from your Xenium output, most likely named as `transcripts.csv.gz`.

The first few lines may look like (from the public human breast cancer dataset)
```plaintext linenums="1"
"transcript_id","cell_id","overlaps_nucleus","feature_name","x_location","y_location","z_location","qv"
281474976710656,565,0,"SEC11C",4.395842,328.66647,12.019493,18.66248
281474976710657,540,0,"NegControlCodeword_0502",5.074415,236.96484,7.6085105,18.634956
```

We will collapse pixels from all z-planes to 2D, essentially using only `x_location`, `y_location`, and `transcript_id`. You may want to keep only transcript with quality score `qv` above certain threshold.


The following command assume your input is in `inpath` and you you want to store output to `path`. In some data the negative control proves are names "Blank-*", `--dummy_genes BLANK\|NegCon` will regard any `transcript_id` containing the substring "BLANK" or "NegCon" as control probes. You could provide a regex according to your data.

The script `format_xenium.py` can be found in `misc/` in the FICTURE repository.

```bash linenums="1"
input=${inpath}/transcripts.csv.gz
output=${path}/filtered.matrix.${iden}.tsv
feature=${path}/feature.clean.${iden}.tsv.gz

python format_xenium.py --input ${input} --output ${output} --feature ${feature} --min_phred_score 15 --dummy_genes BLANK\|NegCon

sort -k2,2g ${output} | gzip -c > ${output}.gz
rm ${output}
```
