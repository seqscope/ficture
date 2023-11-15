#!/bin/bash

#SBATCH --output=/home/%u/out/%x-%j.log
#SBATCH --time=80:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=7g

### Model fitting
# Minimal required input: path, feature, nFactor
# Alternative - path, nFactor, hexagon, pixel, model_id, output_id

key=Count
input_s=2
major_axis=Y
mu_scale=1
min_ct_per_unit=50
min_ct_per_unit_fit=20
min_ct_per_feature=50

thread=1
train_nEpoch=1
train_width=24
fit_width=24
anchor_res=4

cmap_name="turbo"

# For DE output
min_ct_per_feature=50
max_pval_output=1e-3
min_fold_output=1.5

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

fit_nmove=${fit_nmove:-$((fit_width/anchor_res))}
model_id=${model_id:-nF${nFactor}.d_${train_width}}
output_id=${output_id:-${model_id}}
hexagon=${hexagon:-${path}/hexagon.d_${train_width}.s_${input_s}.tsv.gz}
pixel=${pixel:-${path}/filtered.matrix.tsv.gz}

anchor_info=prj_${fit_width}.r_${anchor_res}

output_path=${path}/analysis/${model_id}
figure_path=${output_path}/figure

if [ ! -d "${figure_path}/sub" ]; then
    mkdir -p ${figure_path}/sub
fi

model=${output_path}/${output_id}.model.p
# Fit model
# command time -v ficture lda --epoch ${train_nEpoch} --epoch_id_length 2 --unit_attr X Y --feature ${feature} --key ${key} --input ${hexagon} --output ${output_path}/${output_id} --nFactor ${nFactor} --min_ct_per_unit ${min_ct_per_unit} --min_ct_per_feature ${min_ct_per_feature} --thread ${thread} --overwrite

# Choose color
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}
cmap=${figure_path}/${output_id}.rgb.tsv
# command time -v ficture choose_color  --input ${input} --output ${output} --cmap_name ${cmap_name}

# Transform
output=${output_path}/${output_id}.${anchor_info}
anchor=${output}.fit_result.tsv.gz
postcount=${output}.posterior.count.tsv.gz
command time -v ficture transform --input ${pixel} --output_pref ${output} --model ${model} --key ${key} --major_axis Y --hex_width ${fit_width} --n_move ${fit_nmove} --min_ct_per_unit ${min_ct_per_unit_fit} --mu_scale ${mu_scale} --thread ${thread} --precision 2
