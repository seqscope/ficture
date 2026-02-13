# Processing large real datasets

This document describes how to run large real datasets with FICTURE. Most of the contents here are similar to the [small run](localrun.md) document, but we provide more details on how to submit jobs to a SLURM cluster, using prepared scripts in the `examples/script/` directory.

In this example, we will use a small sub-region of Vizgen MERSCOPE mouse liver data as an example. (The same region we showed in supplementary figure X)

This document assume you have intalled FICTURE. See [installing FICTURE](install.md) for more details.

### Input

```bash linenums="1"
examples/data/transcripts.tsv.gz
```
(All filese are tab-delimited text files unless specified otherwise.)

**Transcripts**

One file contains the molecular or pixel level information, the required columns are `X`, `Y`, `gene`, and `Count`. (There could be other columns in the file which would be ignored.)

The coordinates `(X, Y)` can be float or integer numbers in arbitrary units, but if it is not in the unit of $\mu m$ we would need to specify the translation ratio later.

The file has to be **sorted** by one of the coordinates. (Usually it is the longer axis, but it does not matter if the tissue area is not super asymmetric.)

`Count` (could be any other name) is the number of transcripts for the specified `gene` observed at the coordinate. For imaging based technologies where each molecule has its unique coordinates, `Count` could be always 1.

<!-- **Gene list**

Another file contains the (unique) names of genes that should be used in analysis. The required columns is just `gene` (including the header), the naming of genes should match the `gene` column in the transcript file. If your data contain negative control probes or if you would like to remove certain genes this is where you can specify. (If you would like to use all genes present in your input transcript file the gene list is not necessary, but you would need to modify the command in `generic_III.sh` to remove the argument `--feature` ) -->

**Bounding box of spatial coordinates**

We also prefer to keep a file listing the min and max of the coordinates (this is primarily for visualizing very big tissue region where we do not read all data at once but would want to know the image dimension). The unit of the coordinates is micrometer.
```
examples/data/coordinate_minmax.tsv
```


### Process

Specify the base directory that contains the input data
```bash linenums="1"
path=examples/data
```

Data specific setup:

`mu_scale` is the ratio between $\mu m$ and the unit used in the transcript coordinates. For example, if the coordinates are sotred in `nm` this number should be `1000`.

`key` is the column name in the transcripts file corresponding to the gene counts (`Count` in our example). `MJ` specify which axis the transcript file is sorted by.


```bash linenums="1"
mu_scale=1 # If your data's coordinates are already in micrometer
key=Count
MJ=Y # If your data is sorted by the Y-axis
env=venv/with/ficture/installed/bin/activate

# Uncomment and modify the following line if you are using SLURM
#SLURM_ACCOUNT= # For submitting jobs to slurm
```

Example bash scripts are in `examples/script/`, you will need to modify them to work on your system.

Create pixel minibatches (`${path}/batched.matrix.tsv.gz`)

```bash linenums="1"
input=${path}/transcripts.tsv.gz
output=${path}/batched.matrix.tsv.gz
rec=$(sbatch --job-name=vz1 --account=${SLURM_ACCOUNT} --partition=standard --cpus-per-task=1 examples/script/generic_I.sh input=${input} output=${output} MJ=${MJ} env=${env} )
IFS=' ' read -ra ADDR <<< "$rec"
jobid1=${ADDR[3]}
```


Set up parameters for initializing the model.
```bash linenums="1"
nFactor=12 # Number of factors
sliding_step=2
train_nEpoch=3
train_width=12 # \sqrt{3} x the side length of the hexagon (um)
model_id=nF${nFactor}.d_${train_width} # An identifier kept in output file names
min_ct_per_feature=20 # Ignore genes with total count \< 20
R=10 # We use R random initializations and pick one to fit the full model
thread=4 # Number of threads to use
```

Parameters for pixel level decoding
```bash linenums="1"
fit_width=12 # Often equal or smaller than train_width (um)
anchor_res=4 # Distance between adjacent anchor points (um)
radius=$(($anchor_res+1))
anchor_info=prj_${fit_width}.r_${anchor_res} # An identifier
coor=${path}/coordinate_minmax.tsv
```

Perform model fitting and pixel-level decoding

```bash linenums="1"
# Prepare training minibatches, only need to run once if you plan to fit multiple models (say with different number of factors)
input=${path}/transcripts.tsv.gz
hexagon=${path}/hexagon.d_${train_width}.tsv.gz
rec=$(sbatch --job-name=vz2 --account=${SLURM_ACCOUNT} --partition=standard --cpus-per-task=1 examples/script/generic_II.sh env=${env} key=${key} mu_scale=${mu_scale} major_axis=${MJ} path=${path} input=${input} output=${hexagon} width=${train_width} sliding_step=${sliding_step})
IFS=' ' read -ra ADDR <<< "$rec"
jobid2=${ADDR[3]}

# Model training
rec=$(sbatch --job-name=vz3 --account=${SLURM_ACCOUNT} --partition=standard --cpus-per-task=${thread} --dependency=afterok:${jobid2} examples/script/generic_III.sh env=${env} key=${key} mu_scale=${mu_scale} major_axis=${MJ} path=${path} pixel=${input} hexagon=${hexagon} model_id=${model_id} train_width=${train_width} nFactor=${nFactor} R=${R} train_nEpoch=${train_nEpoch} fit_width=${fit_width} anchor_res=${anchor_res} min_ct_per_feature=${min_ct_per_feature} thread=${thread})
IFS=' ' read -ra ADDR <<< "$rec"
jobid3=${ADDR[3]}

# Pixel level decoding & visualization
rec=$(sbatch --job-name=vz4 --account=${SLURM_ACCOUNT} --partition=standard --cpus-per-task=${thread} --dependency=afterok:${jobid3},${jobid1} examples/script/generic_V.sh env=${env} key=${key} mu_scale=${mu_scale} path=${path} model_id=${model_id} anchor_info=${anchor_info} radius=${radius} coor=${coor} thread=${thread})
IFS=' ' read -ra ADDR <<< "$rec"
jobid4=${ADDR[3]}
```

### Output

In the above example the analysis outputs are stored in

```bash linenums="1"
${path}/analysis/${model_id} # examples/data/analysis/nF12.d_12
```

There is an html file reporting the color code and top genes of the inferred factors

```bash linenums="1"
nF12.d_12.decode.prj_12.r_4_5.factor.info.html
```

Pixel level visualizating

```bash linenums="1"
figure/nF12.d_12.decode.prj_12.r_4_5.pixel.png
```

Pixel level output is

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
