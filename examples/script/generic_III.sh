#!/bin/bash

#SBATCH --output=/home/%u/out/%x-%j.log
#SBATCH --time=80:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=7g

### Model fitting
# Minimal required input: path, nFactor
# Alternative - path, nFactor, hexagon, pixel, model_id, output_id

key=Count
major_axis=Y
mu_scale=1
min_ct_per_unit=50
min_ct_per_unit_fit=20
min_ct_per_feature=50

thread=1
R=10
train_nEpoch=1
train_width=24
fit_width=24
anchor_res=4

cmap_name="turbo"

# For DE output
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
hexagon=${hexagon:-${path}/hexagon.d_${train_width}.tsv.gz}
pixel=${pixel:-${path}/filtered.matrix.tsv.gz}

anchor_info=prj_${fit_width}.r_${anchor_res}

output_path=${path}/analysis/${model_id}
figure_path=${output_path}/figure

if [ ! -d "${figure_path}/sub" ]; then
    mkdir -p ${figure_path}/sub
fi

output=${output_path}/${output_id}
model=${output}.model.p
# Fit model
ficture fit_model --input ${hexagon} --output ${output} --nFactor ${nFactor} --epoch ${train_nEpoch} --epoch_id_length 2 --unit_attr X Y --key ${key} --min_ct_per_unit ${min_ct_per_unit}  --min_ct_per_feature ${min_ct_per_feature} --test_split 0.5 --R ${R} --thread ${thread}

# Choose color
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}
cmap=${figure_path}/${output_id}.rgb.tsv
ficture choose_color --input ${input} --output ${output} --cmap_name ${cmap_name}

# Coarse plot for inspection
cmap=${figure_path}/${output_id}.rgb.tsv
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}.coarse
fillr=$((fit_width/2+1))
ficture plot_base --input ${input} --output ${output} --fill_range ${fillr} --color_table ${cmap} --plot_um_per_pixel 1 --plot_discretized

# Transform
output=${output_path}/${output_id}.${anchor_info}
ficture transform --input ${pixel} --output_pref ${output} --model ${model} --key ${key} --major_axis ${major_axis} --hex_width ${fit_width} --n_move ${fit_nmove} --min_ct_per_unit ${min_ct_per_unit_fit} --mu_scale ${mu_scale} --thread ${thread} --precision 2
