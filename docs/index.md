# FICTURE documentation

Documentation overview

[Quick start](quickstart.md): install and run FICTURE on an example data.

[Small run](localrun.md): execute individual commands on a local machine.

[Format input](format_input.md): how to prepare the input files from different platforms' raw output.

[Run](run.md): how to run FICTURE for a large dataset in a linux environment including how to submit jobs to SLURM.

<!-- [Legacy small run](localrun_legacy.md): run a small example on a local machine assuming you are on the (legacy) `stable` branch of FICTURE. -->

**Common issues**

1) Coordinate units. Depending on your data generation platform, the spatial coordinates in your raw input file may be in units other than micrometer. You would need to set the translation ratio in a parameter `mu_scale`, see [Format input](format_input.md) for more details.

2) Sort your input by coordinates. To scale to really large datasets, we assume the input data is sorted according to one axis, either X or Y, and you would need to let the software know by setting `major_axis`.

3) Visualization. To visualize the final high resolution pixel level result, we would need to tell the software the min and max values of the coordinates, in *micrometer*.
