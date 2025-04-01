# Process CosMx SMI raw data

Locate the transcript file from your SMI output, most likely it is named `*_tx_file.csv.gz` with the following columns

```plaintext linenums="1"
"fov","cell_ID","x_global_px","y_global_px","x_local_px","y_local_px","z","target","CellComp"
1,0,298943.990047619,19493.2809095238,896.371,4433.7571,0,"Snap25","None"
1,0,298685.619047619,19489.3238095238,638,4429.8,0,"Fth1","None"
1,0,298688.648047619,19487.8095095238,641.029,4428.2857,0,"Dnm1","None"
1,0,298943.890047619,19478.7667095238,896.271,4419.2429,0,"Pbx1","Nuclear"
```

What we need are `x_global_px`, `y_global_px`, and `target`. Optionally, we could carry over some pixel level annotations to the output file by specifying `--annotation`, see the following example.

We may also like to translate the pixel unit into micrometer, the ratio can be found in a SMI Data File ReadMe come with your raw data. For example, for the public mouse brain dataset the README says

> - x_local_px
    - The x position of this transcript within the FOV, measured in pixels. To convert to microns multiply the pixel value by 0.168 um per pixel.

So in the following commands we set `px_to_um=0.168`, then for all later analysis we would use `mu_scale=1`
The output is sorted first along the Y-axis, so later you would set `major_axis=Y`.

(Alternatively, we can preferve the integer pixel coordinates for now by not setting `--px_to_um` (or set it to 1) then specify `mu_scale=5.95` (1/0.168) when we later process the data and run FICTURE.)

The python script can be found in `ficture/misc`. use `python format_cosmx.py -h` to see the full options.

```bash linenums="1"
input=/path/to/input/Tissue5_tx_file.csv.gz # Change it to your transcript file
path=/path/to/output
iden=brain # how you identify your files
dummy="Negativ|System" # Name or regex that match the negative control probs
px_to_um=0.168 # convert the pixel unit in the input to micrometer

output=${path}/filtered.matrix.${iden}.tsv
feature=${path}/feature.clean.${iden}.tsv.gz

python misc/format_cosmx.py --input ${input} --output ${output} --feature ${feature} --dummy_genes ${dummy} --px_to_um ${px_to_um} --annotation cell_ID --precision 2
sort -k2,2g -k1,1g ${output} | gzip -c > ${output}.gz
rm ${output}
```

If we would like to merge pixels with (almost?) identical coordinates, replace the last two lines by
```bash linenums="1"
sort -k2,2g -k1,1g ${output} |
awk 'BEGIN { OFS="\t"; print "X", "Y", "gene", "cell_ID", "Count" }
     NR > 1 {
       if ($1 == prevX && $2 == prevY && $3 == prevGene) {
         sumCount += $5;
       } else {
         if (NR > 2) {
           print prevX, prevY, prevGene, firstCellID, sumCount;
         }
         prevX = $1; prevY = $2; prevGene = $3; firstCellID = $4; sumCount = $5;
       }
     }
     END { print prevX, prevY, prevGene, firstCellID, sumCount; }' | gzip -c > ${output}.gz

rm ${output}
```
You might need to asjust the column numbers depending on what annotation columns you have chosen to retain in the output.

(If your data is very dense it may be nicer to collapse, but it would not affect the analysis much.)
