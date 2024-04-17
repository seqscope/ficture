# Quickstart

The following is a quickstart guide to get you started with FICTURE.

## Install FICTURE

We recommend installing FICTURE using `pip`

```bash
pip install ficture
```

Please make sure that `bgzip` and `tabix` is available in your system. Otherwise, please visit [htslib](https://www.htslib.org/download/) website to install htslib.

## Clone the repository

To access the example data, let's clone the FICTURE repository using the following command:

```bash
git clone https://github.com/seqscope/ficture
```

## Run example datasets

Using example datasets, you can run FICTURE with the following command:

```bash
ficture run_together --in-tsv examples/data/transcripts.tsv.gz \
    --in-minmax examples/data/coordinate_minmax.tsv \
    --in-feature examples/data/feature.clean.tsv.gz \
    --out-dir output1 --all
```

This command will create a GNU makefile and run the FICTURE local pipeline.

This will run FICTURE on the example datasets and save the results in the `output1` directory.

## Running FICTURE with multiple parameter settings

You can change the parameter settings, such as the width of training parameters, or the number of concurrent jobs to run.

```bash
ficture run_together --in-tsv examples/data/transcripts.tsv.gz \
    --in-minmax examples/data/coordinate_minmax.tsv \
    --in-feature examples/data/feature.clean.tsv.gz \
    --out-dir output2 \
    --train-width 12,18 \
    --n-factor 6,12 \
    --n-jobs 4 \
    --plot-each-factor \
    --all
```

## More information

If you want to see more options with the local pipeline, please run the following command:

```bash
ficture run_together --help
```

Note that the local pipeline is a wrapper for individual commands, and provided for convenience. 
For large datasets, you may need to run individual commands to mitigate memory or CPU constraints.

If you want to run FICTURE with individual commands rather than using the local pipeline, please refer to other sections in this documentation.