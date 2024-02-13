#!/bin/bash

# Minimal required input: path, width

# width=12 # \sqrt{3} x Hexagon side length
sliding_step=2
MJ=X
env=-

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"
   export "$KEY"="$VALUE"
done

if [ ${env} != "-" ]; then
    source ${env}
fi

set -xe
set -o pipefail

### Prepare pixel minibatches
input=${path}/matrix.tsv.gz
batch=${path}/batched.matrix.tsv
ficture make_spatial_minibatch --input ${input} --output ${batch} --mu_scale 1 --batch_size 200 --batch_buff 50 --major_axis ${MJ}
gzip -f ${batch}


### Prepare hexagons
input=${path}/matrix.tsv.gz
out=${path}/hexagon.d_${width}.s_${sliding_step}.tsv
# Generate sliding hexagons
ficture make_dge --count_header Count --input ${input} --output ${out} --hex_width ${width} --n_move ${sliding_step} --min_ct_per_unit 50 --precision 2 --major_axis ${MJ}
# Shuffle hexagons
sort -S 4G -k1,1n ${out} | gzip -c > ${out}.gz
rm ${out}
