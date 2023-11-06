# Simulation

Let's simulate a small dataset similar to what we have shown in Figure 3.

(Assuming FICTURE is sucessfully installed in the current environment.)

### Simulation

Set directory
```bash
path=/path/to/store/simulated/data # where to store the simulated dataset
gitpath=/path/to/ficture # location of ficture
```

Simulation parameters
```bash
xmax=500 # dimension of the simulated data in um
ymax=500
avg_umi_per_pixel=1 # each pixel is exactly one molecule, mimicking data from imaging-based technologies
pixel_density=4 # average number of pixels per 1 um^2
background=0.2 # extracellular region has lower density
cdist=15 # average distance between cell centers
f_scatter=0.2 # fraction of cells that are "scattered"
# save the size of the simulated region
coor=${path}/coordinate_minmax.tsv
echo -e "xmin\t0\nxmax\t${xmax}\nymin\t0\nymax\t${ymax}" > ${coor}
```

We simulate gene expression from 500 genes and 10 cell types selected from the tabula muris scRNA-seq reference.
```bash
model=${gitpath}/examples/simulation/model.tsv.gz
spike="Kupffer_cell granulocyte" # cell types that are scattered across the region
block="fibroblast epithelial_cell_of_proximal_tubule endothelial_cell keratinocyte hepatocyte cardiac_muscle_cell immature_NK_T_cell cell_of_skeletal_muscle" # cell types that are localized
```

Run simulation. This script simulates 3 cell shapes, default to have 40% round, 30% rod, and 30% diamond (rhombus) cells.
```bash
python ${gitpath}/examples/simulation/simu.py --path ${path} --model ${model} --spike ${spike} --block ${block} --block_x ${xmax} --block_y ${ymax} --avg_umi_per_pixel ${avg_umi_per_pixel} --pixel_density ${pixel_density} --f_rod 0.3 --f_diamond 0.3 --background ${background} --f_scatter ${f_scatter} --avg_cdist ${cdist} --seed 1984
```

Your `$path` should now contain the following files:

Pixel level input to FICTURE `${path}/matrix.tsv.gz` contains the required columns X, Y, gene, and Count. It also contains the ID and shape of the cell that each pixel is generated from in the simulation, but these annotations are ignored by FICTURE and only used for evaluation. This file is sorted by the X axis.

A list of genes and their total counts in the data `feature.tsv.gz`.

Pixel level cell type annotation `pixel_label.uniq.tsv.gz` for visualization.

Simulated model `model.true.tsv.gz` and a color map `model.rgb.tsv` for visualization.


Visualize the simulated data
```bash
cmap=${path}/model.rgb.tsv
input=${path}/pixel_label.uniq.tsv.gz
output=${path}/truth.pixel
ficture plot_base --input ${input} --output ${output} --color_table ${cmap} --xmin 0 --xmax ${xmax} --ymin 0 --ymax ${ymax} --plot_um_per_pixel 0.5 --category_column cell_label --color_table_category_name cell_label
```
The above command generates `${path}/truth.pixel.png` that visualizes simulated pixels at resolution 0.5 um (so the png is of size 1000 X 1000) colored by simulated cell types.

### Run FICTURE

Commands for preparing pixel and hexagon minibatches are in `./examples/simulation/cmd_prepare.sh`
```bash
./examples/simulation/cmd_prepare.sh path=${path} width=12 sliding_step=2
```
(Submit this script to a cluster if you have a large dataset.)


Commands for running FICTURE are in `./examples/simulation/cmd_ficture.sh`

```bash
./examples/simulation/cmd_ficture.sh path=${path} nFactor=10 thread=1 train_width=12 train_nEpoch=3
```
(Submit this script to a cluster and consider using multiple threads if you have a large dataset.)

Final output will be in `${path}/analysis/nF10.d_12`

### Evaluation

Match factors with the simulated cell types and identify wrongly assigned pixels
```bash
output_path=${path}/analysis/nF10.d_12
prefix=nF10.d_12.decode.prj_9.r_3_4
query=${output_path}/${prefix}.pixel.sorted.tsv.gz
output=${output_path}/${prefix}
python ${gitpath}/examples/simulation/simu.eval.py --path ${path} --query ${query} --output ${output} --K1 ${nFactor} --query_scale 0.01
```

Plot error (colored by the true generating cell type)
```bash
input=${output_path}/${prefix}.bad_pixel.tsv.gz
cmap=${path}/model.rgb.tsv
output=${figure_path}/${prefix}.bad_pixel
ficture plot_base --input ${input} --output ${output} --color_table ${cmap} --xmin 0 --xmax $xmax --ymin 0 --ymax $ymax --plot_um_per_pixel 0.5 --category_column cell_label --color_table_category_name cell_label
```

Plot the pixel level result again, using the color code defined during simulation
```bash
input=${output_path}/${prefix}.pixel.sorted.tsv.gz
cmap=${output_path}/${prefix}.matched.rgb.tsv
output=${figure_path}/${prefix}.repaint.pixel.png
ficture plot_pixel_full --input ${input} --color_table ${cmap} --output ${output} --plot_um_per_pixel ${pixel_resolution} --full
```
