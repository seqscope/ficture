# Real data process in a local linux machine

We take a small sub-region of Vizgen MERSCOPE mouse liver data as an example.

## Setup

### Input

```
examples/data/transcripts.tsv.gz
examples/data/feature.clean.tsv.gz
```
(All filese are tab-delimited text files unless specified otherwise.)

See the following explanation for each file. If you have trouble, try [Format input](format_input.md) to see some examples of formating raw output from different platforms.

**Transcripts**

One file contains the molecular or pixel level information, the required columns are `X`, `Y`, `gene`, and `Count`. (There could be other columns in the file which would be ignored.)

The coordinates `(X, Y)` can be float or integer numbers in arbitrary units, but if it is not in the unit of $\mu m$ we would need to specify the translation ratio later.

The file has to be **sorted** by one of the coordinates. (Usually it is the longer axis, but it does not matter if the tissue area is not super asymmetric.)

`Count` (could be any other name) is the number of transcripts for the specified `gene` observed at the coordinate. For imaging based technologies where each molecule has its unique coordinates, `Count` could be always 1.

**Gene list**

Another file contains the (unique) names of genes that should be used in analysis. The required columns is just `gene` (including the header), the naming of genes should match the `gene` column in the transcript file. If your data contain negative control probes or if you would like to remove certain genes this is where you can specify. (If you would like to use all genes present in your input transcript file the gene list is not necessary, but you would need to modify the command in `generic_III.sh` to remove the argument `--feature` )

**Meta data**

We also prefer to keep a file listing the min and max of the coordinates (this is primarily for visuaizing very big tissue region where we do not read all data at once but would want to know the image dimension). The unit of the coordinates is micrometer.
```
examples/data/coordinate_minmax.tsv
```

### Prepare environment

Suppose you have installed dependencies following [Install](install.md)

Activate your virtual environmnet if needed
```
source venv/with/requirements/installed/bin/activate
```


## Process

Specify the base directory that contains the input data
```bash
path=examples/data
```

Data specific setup:

`mu_scale` is the ratio between $\mu m$ and the unit used in the transcript coordinates. For example, if the coordinates are sotred in `nm` this number should be `1000`.

`key` is the column name in the transcripts file corresponding to the gene counts (`Count` in our example). `major_axis` specify which axis the transcript file is sorted by.

```bash
mu_scale=1 # If your data's coordinates are already in micrometer
key=Count
major_axis=Y # If your data is sorted by the Y-axis
gitpath=path/to/ficture # path to this repository
```


### Preprocessing
Create pixel minibatches (`${path}/batched.matrix.tsv.gz`)
```bash
batch_size=500
batch_buff=30
input=${path}/transcripts.tsv.gz
output=${path}/batched.matrix.tsv.gz
batch=${path}/batched.matrix.tsv

python ${gitpath}/script/make_spatial_minibatch.py --input ${input} --output ${batch} --mu_scale ${mu_scale} --batch_size ${batch_size} --batch_buff ${batch_buff} --major_axis ${major_axis}

sort -S 4G -k2,2n -k1,1g ${batch} | gzip -c > ${batch}.gz
rm ${batch}
```


Prepare training minibatches, only need to run once if you plan to fit multiple models (say with different number of factors)
```bash
# Prepare training minibatches, only need to run once if you plan to fit multiple models (say with different number of factors)
train_width=12 # \sqrt{3} x the side length of the hexagon (um)
min_ct_per_unit=50
input=${path}/transcripts.tsv.gz
out=${path}/hexagon.d_${train_width}.tsv
python ${gitpath}/script/make_dge_univ.py --key ${key} --count_header ${key} --input ${input} --output ${out} --hex_width ${train_width} --n_move 2 --min_ct_per_unit ${min_ct_per_unit} --mu_scale ${mu_scale} --precision 2 --major_axis ${major_axis}

sort -S 4G -k1,1n ${out} | gzip -c > ${out}.gz # Shuffle hexagons
rm ${out}
```


### Model training
Parameters for initializing the model
```bash
nFactor=12 # Number of factors
sliding_step=2
train_nEpoch=3
# train_width=12 # should be the same as used in the above step
model_id=nF${nFactor}.d_${train_width} # An identifier kept in output file names
min_ct_per_feature=20 # Ignore genes with total count \< 20
R=10 # We use R random initializations and pick one to fit the full model
thread=4 # Number of threads to use
```

Initialize the model
```bash
# parameters
min_ct_per_unit_fit=20
cmap_name="turbo"
fit_nmove=$((fit_width/anchor_res))

# output identifiers
model_id=nF${nFactor}.d_${train_width}
output_id=${model_id}
anchor_info=prj_${fit_width}.r_${anchor_res}
output_path=${path}/analysis/${model_id}
figure_path=${output_path}/figure
if [ ! -d "${figure_path}/sub" ]; then
    mkdir -p ${figure_path}/sub
fi


# input files
hexagon=${path}/hexagon.d_${train_width}.tsv.gz
pixel=${path}/transcripts.tsv.gz
feature=${path}/feature.clean.tsv.gz
# output
output=${output_path}/${output_id}
model=${output}.model.p

# Fit model
python ${gitpath}/script/init_model_selection.py --input ${hexagon} --output ${output} --feature ${feature} --nFactor ${nFactor} --epoch ${train_nEpoch} --epoch_id_length 2 --unit_attr X Y --key ${key} --min_ct_per_feature ${min_ct_per_feature} --test_split 0.5 --R ${R} --thread ${thread}

# Choose color
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}
cmap=${figure_path}/${output_id}.rgb.tsv
python ${gitpath}/script/choose_color.py --input ${input} --output ${output} --cmap_name ${cmap_name}

# Coarse plot for inspection
cmap=${figure_path}/${output_id}.rgb.tsv
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}.coarse
fillr=$((fit_width/2+1))
python ${gitpath}/script/plot_base.py --input ${input} --output ${output} --fill_range ${fillr} --color_table ${cmap} --plot_um_per_pixel 1 --plot_discretized
```


### Pixel level decoding

Parameters for pixel level decoding
```bash
fit_width=12 # Often equal or smaller than train_width (um)
anchor_res=4 # Distance between adjacent anchor points (um)
radius=$(($anchor_res+1))
anchor_info=prj_${fit_width}.r_${anchor_res} # An identifier
coor=${path}/coordinate_minmax.tsv
cmap=${figure_path}/${output_id}.rgb.tsv
```

```bash
# Transform
output=${output_path}/${output_id}.${anchor_info}
python ${gitpath}/script/transform_univ.py --input ${pixel} --output_pref ${output} --model ${model} --key ${key} --major_axis ${major_axis} --hex_width ${fit_width} --n_move ${fit_nmove} --min_ct_per_unit ${min_ct_per_unit_fit} --mu_scale ${mu_scale} --thread ${thread} --precision 2

# Pixel level decoding & visualization
prefix=${output_id}.decode.${anchor_info}_${radius}
input=${path}/batched.matrix.tsv.gz
anchor=${output_path}/${output_id}.${anchor_info}.fit_result.tsv.gz
output=${output_path}/${prefix}
# Output only a few top factors per pixel
topk=3 # Fix for now
python ${gitpath}/script/slda_decode.py --input ${input} --output ${output} --model ${model} --anchor ${anchor} --anchor_in_um --neighbor_radius ${radius} --mu_scale ${mu_scale} --key ${key} --precision 0.1 --lite_topk_output_pixel ${topk} --lite_topk_output_anchor ${topk} --thread ${thread}
```

### Optional post-processing

The following is not strictly necessary but it generates summary statistics and helps visualization.


Sort the pixel level output, this is for visualize large images with limited memory usage.
```bash
input=${output_path}/${prefix}.pixel.tsv.gz # j, X, Y, K1, ..., KJ, P1, ..., PJ, J=topk
output=${output_path}/${prefix}.pixel.sorted.tsv.gz

K=$( echo $model_id | sed 's/nF\([0-9]\{1,\}\)\..*/\1/' )
while IFS=$'\t' read -r r_key r_val; do
    export "${r_key}"="${r_val}"
done < ${coor}
echo -e "${xmin}, ${xmax}; ${ymin}, ${ymax}"

offsetx=${xmin}
offsety=${ymin}
rangex=$( echo "(${xmax} - ${xmin} + 0.5)/1+1" | bc )
rangey=$( echo "(${ymax} - ${ymin} + 0.5)/1+1" | bc )
bsize=2000
scale=100
header="##K=${K};TOPK=3\n##BLOCK_SIZE=${bsize};BLOCK_AXIS=X;INDEX_AXIS=Y\n##OFFSET_X=${offsetx};OFFSET_Y=${offsety};SIZE_X=${rangex};SIZE_Y=${rangey};SCALE=${scale}\n#BLOCK\tX\tY\tK1\tK2\tK3\tP1\tP2\tP3"

(echo -e "${header}" && zcat ${input} | tail -n +2 | perl -slane '$F[0]=int(($F[1]-$offx)/$bsize) * $bsize; $F[1]=int(($F[1]-$offx)*$scale); $F[1]=($F[1]>=0)?$F[1]:0; $F[2]=int(($F[2]-$offy)*$scale); $F[2]=($F[2]>=0)?$F[2]:0; print join("\t", @F);' -- -bsize=${bsize} -scale=${scale} -offx=${offsetx} -offy=${offsety} | sort -S 4G -k1,1g -k3,3g ) | bgzip -c > ${output}

tabix -f -s1 -b3 -e3 ${output}
rm ${input}

```

Report differentially expressed genes. This is a naive pseudo-bulk chi-squared test, please view the results with caution.
```bash
# DE
max_pval_output=1e-3
min_fold_output=1.5
input=${output_path}/${prefix}.posterior.count.tsv.gz
output=${output_path}/${prefix}.bulk_chisq.tsv
python ${gitpath}/script/de_bulk.py --input ${input} --output ${output} --min_ct_per_feature ${min_ct_per_feature} --max_pval_output ${max_pval_output} --min_fold_output ${min_fold_output} --thread ${thread}


# Report (color table and top DE genes)
cmap=${output_path}/figure/${output_id}.rgb.tsv
output=${output_path}/${prefix}.factor.info.html
python ${gitpath}/script/factor_report.py --path ${output_path} --pref ${prefix} --color_table ${cmap}

```

Generalize pixel level images representing the factorization result
```bash
# Make pixel level figures
cmap=${output_path}/figure/${output_id}.rgb.tsv
input=${output_path}/${prefix}.pixel.sorted.tsv.gz
output=${figure_path}/${prefix}.pixel.png
python ${gitpath}/script/plot_pixel_full.py --input ${input} --color_table ${cmap} --output ${output} --plot_um_per_pixel 0.5 --full
```

Generate heatmaps for individual factors. If the data is very large, making all individual factor maps may take some time.

Generate everything in one run
```bash
# Make single factor heatmaps, plot_subbatch balances speed and memory ...
# batch size 8 should be safe for 7 or 14G in most cases
output=${figure_path}/sub/${prefix}.pixel
python ${gitpath}/script/plot_pixel_single.py --input ${input} --output ${output} --plot_um_per_pixel 0.5 --full --all
```

Alternatively, you can generate by batch
```bash
plot_subbatch=8
st=0
ed=$((plot_subbatch+st-1))
while [ ${st} -lt ${K} ]; do
    if [ ${ed} -gt ${K} ]; then
        ed=$((K-1))
    fi
    id_list=$( seq ${st} ${ed} )
    echo $id_list

    python ${gitpath}/script/plot_pixel_single.py --input ${input} --output ${output} --id_list ${id_list} --plot_um_per_pixel 0.5 --full
    st=$((ed+1))
    ed=$((plot_subbatch+st-1))
done
```




## Output
In the above example the analysis outputs are stored in
```
${path}/analysis/${model_id} # examples/data/analysis/nF12.d_12
```

There is an html file reporting the color code and top genes of the inferred factors
```
nF12.d_12.decode.prj_12.r_4_5.factor.info.html
```

Pixel level visualizating
```
figure/nF12.d_12.decode.prj_12.r_4_5.pixel.png
```

Pixel level output is
```
nF12.d_12.decode.prj_12.r_4_5.pixel.sorted.tsv.gz
```

We store the top 3 factors and their corresponding posterior probabilities for each pixel in tab delimted text files.
As a temporary hack for accessing specific regions in large dataset faster, we divided the data along one axis (X or Y), sorted within each block by the other axis.
The first 3 lines of the file, starting with `##`, are metadata, the 4th line, starting with `#`, contains columns names.
To use the file as plain text, you can ignore this complication and read the file from the 4th line.

The first few lines of the file are as follows:

```
##K=12;TOPK=3
##BLOCK_SIZE=2000;BLOCK_AXIS=X;INDEX_AXIS=Y
##OFFSET_X=6690;OFFSET_Y=6772;SIZE_X=676;SIZE_Y=676;SCALE=100
#BLOCK  X       Y       K1      K2      K3      P1      P2      P3
0       10400   360     2       1       8       9.07e-01        9.27e-02        2.61e-13
0       10669   360     2       1       8       9.36e-01        6.37e-02        4.20e-08
0       10730   360     2       1       8       8.85e-01        1.15e-01        1.83e-05
```

The 4th line contains the column names. From the 5th line on, each line contains the information for one pixel with coordinates `(X, Y)`, the top 3 factors indicated by `K1, K2, K3` and their corresponding posterior probabilities `P1, P2, P3`. Factors are 0-indexed.

The 1st line indicates that the data is from a model with 12 factors (`K=12`) and we store the top 3 factors for each pixel (`TOPK=3`).

The 2nd line indicates that the data is separated into blocks by the X axis (`BLOCK_AXIS=X`) with block size 2000$\mu m$ (`BLOCK_SIZE=2000`), then within each block the data is sorted by the Y axis (`INDEX_AXIS=Y`).
The block IDs (first column in the file) are integer multiples of the block size (in $\mu m$), i.e. the 1st block, with $X \in [0, 2000)$ have block ID 0, the 2nd block, with $X \in [2000, 4000)$ have block ID 2000, etc.


The 3rd line describes the translation between the stored cooredinates and the physical coordinates in $\mu m$.
Take `(X, Y)` as a pixel coordinates read from the file, the physical coordinates in $\mu m$ is `(X / SCALE + OFFSET_X, Y / SCALE + OFFSET_Y)`.
In this above example, the raw data from Vizgen MERSCOPE mouse liver data contains negative coordinates, but for convineince we shifted all coordinates to positive. `SIZE_X` and `SIZE_Y` record the size of the raw data in $\mu m$.
