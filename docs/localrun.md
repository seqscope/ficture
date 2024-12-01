# Running FICTURE in a local machine

## Overview 

This document provides detailed instructions on how to run FICTURE on a local machine with real data.
This instruction is intended for Ubuntu OS, but it should also work for Mac OS X and other Unix-like systems.
If you rather want to run all steps together 
with `run_togetehr` command, please refer to [Quick start](quickstart.md) for details.

## Setup

### Input Data

A small sub-region of Vizgen MERSCOPE mouse liver data is provided as an example in the GitHub repository

```bash linenums="1"
## main transcript file
examples/data/transcripts.tsv.gz
## gene list file (optional)
examples/data/feature.clean.tsv.gz
## bounding box file (optional)
examples/data/coordinate_minmax.tsv 
```
(All filese are tab-delimited text files unless specified otherwise.)

See the following explanation for each file. If you have trouble, try [Format input](format_input.md) to see some examples of formatting raw output from different platforms.

#### Transcript file

One file contains the molecular or pixel level information, the required columns are `X`, `Y`, `gene`, and `Count`. (There could be other columns in the file which would be ignored.)

The coordinates `(X, Y)` can be float or integer numbers in arbitrary units, but if it is not in the unit of $\mu m$ we would need to specify the translation ratio later.

The file has to be **sorted** by one of the coordinates. (Usually it is the longer axis, but it does not matter if the tissue area is not super asymmetric.)

`Count` (could be any other name) is the number of transcripts for the specified `gene` observed at the coordinate. For imaging based technologies where each molecule has its unique coordinates, `Count` could be always 1.

#### Gene list file

Another file contains the (unique) names of genes that should be used in analysis. The required columns is just `gene` (including the header), the naming of genes should match the `gene` column in the transcript file. If your data contain negative control probes or if you would like to remove certain genes this is where you can specify. (If you would like to use all genes present in your input transcript file the gene list is not necessary, but you would need to modify the command in `examples/script/generic_III.sh` to remove the argument `--feature` )

#### Bounding box of spatial coordinates

We also prefer to keep a file listing the min and max of the coordinates (this is primarily for visualizing very big tissue region where we do not read all data at once but would want to know the image dimension). The unit of the coordinates is micrometer.

```bash linenums="1"
examples/data/coordinate_minmax.tsv
```

Note that, when `run_together` command is used, the gene list file and bounding box files will be automatically generated. 

#### Prepare environment

Activate your virtual environment if needed:

```bash linenums="1"
VENV=/path/to/venv/name   ## replace /path/to/venv/name with your virtual environment path
source ${VENV}/bin/activate
```

Suppose you have installed FICTURE and dependencies following [Install](install.md) in this environment. Verify FICTURE is successfully installed with command `ficture`.


## Analysis with FICTURE

### Key parameters

First, specify the base directory that contains the input data

```bash linenums="1"
path=examples/data
```

The following data-specific setup may be required:

* `mu_scale` is the ratio between $\mu m$ and the unit used in the transcript coordinates. For example, if the coordinates are stored in `nm` this number should be `1000`.
* `key` is the column name in the transcripts file corresponding to the gene counts (`Count` in our example). 
* `major_axis` specify which axis the transcript file is sorted by. (either `X` or `Y`)

```bash
mu_scale=1   # If your data's coordinates are already in micrometer
key=Count    # If you data has 'Count' as the column name for gene counts
major_axis=Y # If your data is sorted by the Y-axis
```


### Preprocessing


#### Anchor-level minibatch

Create pixel minibatches (`${path}/batched.matrix.tsv.gz`) that will be used for anchor-level analysis using the following command: 

```bash linenums="1"
batch_size=500
batch_buff=30
input=${path}/transcripts.tsv.gz
output=${path}/batched.matrix.tsv.gz
batch=${path}/batched.matrix.tsv

ficture make_spatial_minibatch --input ${input} --output ${batch} --mu_scale ${mu_scale} --batch_size ${batch_size} --batch_buff ${batch_buff} --major_axis ${major_axis}

sort -S 4G -k2,2n -k1,1g ${batch} | gzip -c > ${batch}.gz
rm ${batch}
```

#### Training hexagons

Prepare training hexagons. Even if you need to fit multiple models with different number of factors, you only need to run once for each training width. The training width is the flat-to-flat width of the hexagon in $\mu m$.

```bash linenums="1"
## set up the parameters
train_width=12      # flat-to-flat width = \sqrt{3} x the side length of the hexagon (um)
min_ct_per_unit=50  # filter out hexagons with total count < 50
input=${path}/transcripts.tsv.gz
out=${path}/hexagon.d_${train_width}.tsv

## create hexagons
ficture make_dge --key ${key} --count_header ${key} --input ${input} --output ${out} --hex_width ${train_width} --n_move 2 --min_ct_per_unit ${min_ct_per_unit} --mu_scale ${mu_scale} --precision 2 --major_axis ${major_axis}

## shuffle the hexagons based on random index
sort -S 4G -k1,1n ${out} | gzip -c > ${out}.gz 
rm ${out}
```


### LDA Model training

To run FICTURE in a fully unsupervised manner, you need to initialize the model with LDA based on the hexagons created in the previous step. 

#### Parameters for initializing the model

```bash linenums="1"
nFactor=12 # Number of factors
sliding_step=2
train_nEpoch=3
# train_width=12 # should be the same as used in the above step
model_id=nF${nFactor}.d_${train_width} # An identifier kept in output file names
min_ct_per_feature=20 # Ignore genes with total count \< 20
R=10 # We use R random initializations and pick one to fit the full model
thread=4 # Number of threads to use
```

#### Setting the input and output paths 

```bash linenums="1"
# parameters
min_ct_per_unit_fit=20
cmap_name="turbo"

# output identifiers
model_id=nF${nFactor}.d_${train_width}
output_id=${model_id}
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
```

#### Initialize the model with LDA

```bash linenums="1"
# Fit model with unsupervised LDA 
ficture fit_model --input ${hexagon} --output ${output} --feature ${feature} --nFactor ${nFactor} --epoch ${train_nEpoch} --epoch_id_length 2 --unit_attr X Y --key ${key} --min_ct_per_feature ${min_ct_per_feature} --test_split 0.5 --R ${R} --thread ${thread}
```

#### (Optional) Initializing LDA model from pseudo-bulk data

Instead of initializing the model using LDA as shown above, if you want to initialize the model using pseudo-bulk data, you can 
prepare the pseudo-bulk data as a model matrix in the following TSV format in a `tsv.gz` file:

```plaintext linenums="1" 
gene    celltype1   celltype2   ...
gene1   10          20          ...
gene2   5           15          ...
...
```

This model matrix can be directly used for pixel-level decoding step described below. 
However, if the gene list do not match between the pseudo-bulk data and the raw data, you may need to use the following command to initialize the model from the pseudo-bulk data.

```bash linenums="1"
# Fit model from pseudo-bulk data
ficture init_model_from_pseudobulk --input ${hexagon} --output ${output} --feature ${feature} --epoch 0 --scale_model_rel -1 --reorder-factors --key ${key} --min_ct_per_feature ${min_ct_per_feature}--thread ${thread}

# create a model matrix from the posterior count
cp ${output}.posterior.count.tsv.gz ${output}.model_matrix.tsv.gz
```

After running the following command, the model will be initialized using the pseudo-bulk data.


#### Visualizing the model

The results from the initial model fitting can be visualized using the following commands:

```bash linenums="1"
# Choose color
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}
cmap=${figure_path}/${output_id}.rgb.tsv
ficture choose_color --input ${input} --output ${output} --cmap_name ${cmap_name}

# Coarse plot for inspection
cmap=${figure_path}/${output_id}.rgb.tsv
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}.coarse
fillr=$((train_width/2+1))
ficture plot_base --input ${input} --output ${output} --fill_range ${fillr} --color_table ${cmap} --plot_um_per_pixel 1 --plot_discretized
```


### Pixel level decoding

#### Parameters for pixel level decoding

After fitting the model, FICTURE performs pixel level decoding to infer the factors for each pixel. 
The pixel-level decoding consists of two steps:
* Perform anchor-level projection based on the fitted model
* Perform pixel-level decoding based on anchor-level projection

The following parameters can be used for pixel level decoding steps.

```bash linenums="1"
fit_width=12 # Often equal or smaller than train_width (um)
anchor_res=4 # Distance between adjacent anchor points (um)
fit_nmove=$((fit_width/anchor_res))
anchor_info=prj_${fit_width}.r_${anchor_res}
radius=$(($anchor_res+1))
anchor_info=prj_${fit_width}.r_${anchor_res} # An identifier
coor=${path}/coordinate_minmax.tsv
cmap=${figure_path}/${output_id}.rgb.tsv
```

#### Produce anchor-level projection

Anchor-level projection can be performed using the following command:

```bash linenums="1"
# Output prefix
output=${output_path}/${output_id}.${anchor_info}

# Perform anchor-level projection
ficture transform --input ${pixel} --output_pref ${output} --model ${model} --key ${key} --major_axis ${major_axis} --hex_width ${fit_width} --n_move ${fit_nmove} --min_ct_per_unit ${min_ct_per_unit_fit} --mu_scale ${mu_scale} --thread ${thread} --precision 2
```

#### Perform pixel-level decoding

Pixel-level decoding can be performed using the following command:

```bash linenums="1"
# Input/output parameters for pixel-level decoding
prefix=${output_id}.decode.${anchor_info}_${radius}
input=${path}/batched.matrix.tsv.gz
anchor=${output_path}/${output_id}.${anchor_info}.fit_result.tsv.gz
output=${output_path}/${prefix}
topk=3 # Output only a few top factors per pixel

# Perform pixel-level decoding
ficture slda_decode --input ${input} --output ${output} --model ${model} --anchor ${anchor} --anchor_in_um --neighbor_radius ${radius} --mu_scale ${mu_scale} --key ${key} --precision 0.1 --lite_topk_output_pixel ${topk} --lite_topk_output_anchor ${topk} --thread ${thread}
```

#### Optional post-processing

Although not required, after performing pixel-level decoding, it is useful to 
generates summary statistics and visualize the results.


First step is to sort the pixel level output. This is 
primarily for visualizing large images with limited memory usage.

```bash linenums="1"
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
# rm ${input} # Make sure the sorted file makes sense before you remove the unsorted file
```

Next, we can identify differentially expressed genes for each factor. This is a naive pseudo-bulk chi-squared test, please view the results with caution.

```bash linenums="1"
# DE
max_pval_output=1e-3
min_fold_output=1.5
input=${output_path}/${prefix}.posterior.count.tsv.gz
output=${output_path}/${prefix}.bulk_chisq.tsv
ficture de_bulk --input ${input} --output ${output} --min_ct_per_feature ${min_ct_per_feature} --max_pval_output ${max_pval_output} --min_fold_output ${min_fold_output} --thread ${thread}


# Report (color table and top DE genes)
cmap=${output_path}/figure/${output_id}.rgb.tsv
output=${output_path}/${prefix}.factor.info.html

# generate a report for each factor
ficture factor_report --path ${output_path} --pref ${prefix} --color_table ${cmap}
```

Next, generalize pixel level images representing the factorization result

```bash linenums="1"
# Make pixel level figures
cmap=${output_path}/figure/${output_id}.rgb.tsv
input=${output_path}/${prefix}.pixel.sorted.tsv.gz
output=${figure_path}/${prefix}.pixel.png

# plot pixel level images
ficture plot_pixel_full --input ${input} --color_table ${cmap} --output ${output} --plot_um_per_pixel 0.5 --full
```

You may also want to generate heatmaps for individual factors. If the data is very large, making all individual factor maps may take some time.

Generate everything in one run
```bash linenums="1"
# Make single factor heatmaps, plot_subbatch balances speed and memory ...
# batch size 8 should be safe for 7 or 14G in most cases
output=${figure_path}/sub/${prefix}.pixel
ficture plot_pixel_single --input ${input} --output ${output} --plot_um_per_pixel 0.5 --full --all
```

Alternatively, you can generate by batch
```bash  linenums="1"
plot_subbatch=8
st=0
ed=$((plot_subbatch+st-1))
while [ ${st} -lt ${K} ]; do
    if [ ${ed} -gt ${K} ]; then
        ed=$((K-1))
    fi
    id_list=$( seq ${st} ${ed} )
    echo $id_list

    ficture plot_pixel_single --input ${input} --output ${output} --id_list ${id_list} --plot_um_per_pixel 0.5 --full
    st=$((ed+1))
    ed=$((plot_subbatch+st-1))
done
```


## Output

In the above example the analysis outputs are stored in

```bash linenums="1"
${path}/analysis/${model_id} # examples/data/analysis/nF12.d_12
```

There is an html file reporting the color code and top genes of the inferred factors
```bash linenums="1"
nF12.d_12.decode.prj_12.r_4_5.factor.info.html
```

Pixel level visualization is stored in
```bash linenums="1"
figure/nF12.d_12.decode.prj_12.r_4_5.pixel.png
```

Pixel level output is stored in

```bash linenums="1"
nF12.d_12.decode.prj_12.r_4_5.pixel.sorted.tsv.gz
```

We store the top 3 factors and their corresponding posterior probabilities for each pixel in tab delimted text files.
As a temporary hack for accessing specific regions in large dataset faster, we divided the data along one axis (X or Y), sorted within each block by the other axis.
The first 3 lines of the file, starting with `##`, are metadata, the 4th line, starting with `#`, contains columns names.
To use the file as plain text, you can ignore this complication and read the file from the 4th line.

The first few lines of the file are as follows:

```plaintext linenums="1"
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
