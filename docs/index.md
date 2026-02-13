# FICTURE documentation

## What is FICTURE?

FICTURE is a software tool that performs segmentation-free analysis of submicron-resolution analysis of
spatial transcriptomics data.

In brief, FICTURE...

* identifies cell types and disease-specific tissue microenvironments from high-resolution (<1µm) spatial transcriptomics (ST) data without compromising its original resolution.
* is compatible with almost all major high-resolution ST platforms including Seq-Scope, Stereo-seq, 10x Xenium, 10x Visium HD, Vizgen MERSCOPE, CosMx SMI, Pixel-seq, OpenST, and Nova-ST. 
* does NOT require externally provided cell segmentation or histology. 
* works in all-in-one mode (unsupervised clustering + pixel-level decoding) or projection mode (pixel-level decoding from celltypes/clusters identified by external tools or datasets).
* successfully identified fine-scale tissue architectures for challenging tissues where existing methods have struggled, including vascular, fibrotic, muscular and lipid-laden areas.

For more details, please refer to the following publication:

Si, Y., Lee, C., Hwang, Y. et al. FICTURE: scalable segmentation-free analysis of submicron-resolution spatial transcriptomics. *Nat Methods* 21, 1843–1854 (2024). [https://doi.org/10.1038/s41592-024-02415-2](https://doi.org/10.1038/s41592-024-02415-2)


## Documentation Overview

This documentation provides several instructions on how to install and run FICTURE on your data in various details.

* [Quick start](quickstart.md) gives a quick walkthrough of installing and running FICTURE on an example data.
* [Small run](localrun.md) provides detailed examples of executing individual commands on a local machine.
* [Run](run.md) describes how to run FICTURE for a large dataset in a linux environment, including instructions of  submitting jobs via SLURM.
* [Format input](format_input.md) describes how to prepare the input files from different platforms' raw output.

<!-- [Legacy small run](localrun_legacy.md): run a small example on a local machine assuming you are on the (legacy) `stable` branch of FICTURE. -->

## Important Notes in Preparing Input Data

FICTURE is compatible with various high-resolution spatial transcriptomics platforms as detailed in the [Format input](format_input.md) section, but in essence, a TSV file that contains the spatial coordinates and gene expression values is what we need. Here are some key considerations when preparing the input data:

1. **Coordinate unit in micrometer**. FICTURE recommends using micrometer unit for spatial coordinates (X and Y) in your input file. Depending on your data generation platform, the spatial coordinates in your raw input file may not be in micrometer units. If your input data is not a micrometer unit, you would need to either (1) convert your input data into micrometer scale, or (2) set the translation ratio in a parameter `mu_scale`. See [Format input](format_input.md) for more details.
2. **Sort your input by coordinates**. To scale to really large datasets, we assume the input data is sorted according to one axis, either X or Y, and you would need to let the software know by setting `major_axis`.
3. **Bounding box of spatial coordinates**. To visualize the final high resolution pixel level result, we would need to tell the software the minimum and maximum values of the coordinates, in *micrometer*. See [Format input](format_input.md) for more details.
