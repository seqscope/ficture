# Quickstart

The following is a quickstart guide to get you started with FICTURE.

## Install FICTURE

The simplest way to FICTURE using PyPI. Please see [installing FICTURE](install.md) for other options.

```bash linenums="1"
## Install FICTURE from PyPI
pip install ficture
```

<!-- Please make sure that `bgzip` and `tabix` is available in your system. Otherwise, please visit [htslib](https://www.htslib.org/download/) website to install htslib. -->

## Clone the repository

To access the example data, let's clone the FICTURE repository using the following command:

```bash linenums="1"
## Clone the FICTURE repository to access the example data
git clone https://github.com/seqscope/ficture
```

## Run example datasets

Using example datasets, you can run FICTURE with the following command:

```bash linenums="1"
## Run all steps together with the example datasets
ficture run_together --in-tsv examples/data/transcripts.tsv.gz \
    --in-minmax examples/data/coordinate_minmax.tsv \
    --in-feature examples/data/feature.clean.tsv.gz \
    --major-axis Y \
    --out-dir output1 --seed 1 --all
```

This command will create a GNU makefile that runs the FICTURE local pipeline. When executed (`make -j 4 -f output1/Makefile`), it will run FICTURE on the example datasets and save the results to the `output1` directory.

## Running FICTURE with multiple parameter settings

You can change the parameter settings, such as the width of training parameters, or the number of concurrent jobs to run.

```bash linenums="1"
## Specify multiple LDA training widths and number of factors
ficture run_together --in-tsv examples/data/transcripts.tsv.gz \
    --in-minmax examples/data/coordinate_minmax.tsv \
    --in-feature examples/data/feature.clean.tsv.gz \
    --major-axis Y \
    --out-dir output2 \
    --train-width 12,18 \
    --n-factor 6,12 \
    --n-jobs 4 \
    --plot-each-factor \
    --all
```

## Running FICTURE with your own model matrix and color scheme
Caution: your input model and color scheme file must be in the same format as those created by the above commands.
```bash linenums="1"
ficture run_together --in-tsv ../examples/data/transcripts.tsv.gz \
    --in-minmax ../examples/data/coordinate_minmax.tsv \
    --in-feature ../examples/data/feature.clean.tsv.gz \
    --out-dir ./out --major-axis Y --threads 4 \
    --decode-from-external-model \
    --fit-width 12 \
    --external-model YOUR_MODEL_FILE \
    --external-cmap YOUR_COLOR_SCHEME_FILE
```

## More information

If you want to see more options with the local pipeline, please run the following command:

```bash
ficture run_together --help
```

Note that the local pipeline is a wrapper for individual commands, and provided for convenience.
For large datasets, you may need to run individual commands to mitigate memory or CPU constraints.

If you want to run FICTURE with individual commands rather than using the local pipeline, please refer to other sections in this documentation.
