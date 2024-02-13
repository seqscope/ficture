#!/bin/bash

#SBATCH --output=/home/%u/out/%x-%j.log
#SBATCH --time=80:00:00

#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=7g

### Pixel to hexagon
# Minimal required input: input output MJ env gitpath

mu_scale=1
batch_size=500
batch_buff=30
skipshuffle=0

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

# pixel minibatch
batch=$(echo $output | sed 's/\.gz$//g')
python ${gitpath}/script/make_spatial_minibatch.py --input ${input} --output ${batch} --mu_scale ${mu_scale} --batch_size ${batch_size} --batch_buff ${batch_buff} --major_axis ${MJ}

if [ "${skipshuffle}" == "0" ]; then
   # shuffle minibatches
    sort -S 4G -k2,2n -k1,1g ${batch} | gzip -c > ${batch}.gz
    rm ${batch}
else
   gzip -f ${batch}
fi
