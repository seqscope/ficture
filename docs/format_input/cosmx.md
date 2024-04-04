# Process CosMx SMI raw data

Locate the transcript file from your SMI output, most likely it is named `*_tx_file.csv.gz` with the following columns

```
"fov","cell_ID","x_global_px","y_global_px","x_local_px","y_local_px","z","target","CellComp"
1,0,298943.990047619,19493.2809095238,896.371,4433.7571,0,"Snap25","None"
1,0,298685.619047619,19489.3238095238,638,4429.8,0,"Fth1","None"
1,0,298688.648047619,19487.8095095238,641.029,4428.2857,0,"Dnm1","None"
1,0,298943.890047619,19478.7667095238,896.271,4419.2429,0,"Pbx1","Nuclear"
```

What we need are `x_global_px`, `y_global_px`, and `target`.

We would also like to translate the pixel unit into micrometer, the ratio can be found in a SMI Data File ReadMe come with your raw data. For example, for the public mouse brain dataset the README says

> - x_local_px
    - The x position of this transcript within the FOV, measured in pixels. To convert to microns multiply the pixel value by 0.168 um per pixel.

So in the following commands we set `px_to_um=0.168`.

The python script can be found in `ficture/misc`.

```bash
input=/path/to/input/Tissue5_tx_file.csv.gz # Change it to your transcript file
path=/path/to/output
iden=brain # how you identify your files
dummy=NegPrb # Name of the negative controls
px_to_um=0.168 # convert the pixel unit in the input to micrometer

output=${path}/filtered.matrix.${iden}.tsv
feature=${path}/feature.clean.${iden}.tsv.gz

python format_cosmx.py --input ${input} --output ${output} --feature ${feature} --dummy_genes ${dummy} --px_to_um ${px_to_um}
sort -k2,2g -k1,1g ${output} | gzip -c > ${output}.gz
rm ${output}
```
