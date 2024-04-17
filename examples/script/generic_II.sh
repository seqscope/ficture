#!/bin/bash

#SBATCH --output=/home/%u/out/%x-%j.log
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=7g

### Pixel to hexagon
# Minimal required input: path, iden, input, width
# Alternative: env, path, input, output, width

key=Count
major_axis=Y
sliding_step=2
mu_scale=1
min_ct_per_unit=50
overwrite=0

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

source ${env}
set -xe
set -o pipefail

out=$(echo $output | sed 's/\.gz$//g')

ficture make_dge --key ${key} --count_header ${key} --input ${input} --output ${out} --hex_width ${width} --n_move ${sliding_step} --min_ct_per_unit ${min_ct_per_unit} --mu_scale ${mu_scale} --precision 2 --major_axis ${major_axis}

sort -S 4G -k1,1n ${out} | gzip -c > ${output} # Shuffle hexagons
rm ${out}
