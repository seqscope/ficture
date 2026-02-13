#!/bin/bash

### Model fitting, pixel level inference, and evaluation
# Minimal required input: path, nFactor, train_width

thread=1
key=Count
mu_scale=1
major_axis=X

min_ct_per_unit=50
min_ct_per_unit_fit=20
min_ct_per_feature=50

input_s=2
train_nEpoch=3
train_width=12
fit_width=9
anchor_res=3

cmap_name=turbo
pixel_resolution=0.5

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

model_id=nF${nFactor}.d_${train_width}
output_id=${output_id:-${model_id}}
hexagon=${path}/hexagon.d_${train_width}.s_${input_s}.tsv.gz

radius=${radius:-$(($anchor_res+1))}
fit_nmove=${fit_nmove:-$((fit_width/anchor_res))}
anchor_info=prj_${fit_width}.r_${anchor_res}

output_path=${path}/analysis/${model_id}
figure_path=${output_path}/figure
if [ ! -d "${figure_path}/sub" ]; then
    mkdir -p ${figure_path}/sub
fi

cmap=${figure_path}/${output_id}.rgb.tsv
coor=${coor:-${path}/coordinate_minmax.tsv}
while IFS=$'\t' read -r r_key r_val; do
    export "${r_key}"="${r_val}"
done < ${coor}
echo -e "${xmin}, ${xmax}; ${ymin}, ${ymax}"


# Fit model
feature=${feature:-${path}/feature.tsv.gz}
model=${output_path}/${output_id}.model.p

command time -v ficture lda --epoch ${train_nEpoch} --epoch_id_length 2 --unit_attr X Y --feature ${feature} --key ${key} --input ${hexagon} --output ${output_path}/${output_id} --nFactor ${nFactor} --min_ct_per_unit ${min_ct_per_unit} --min_ct_per_feature ${min_ct_per_feature} --thread ${thread} --overwrite



# Transform
pixel=${pixel:-${path}/matrix.tsv.gz}
model=${output_path}/${output_id}.model.p
output=${output_path}/${output_id}.${anchor_info}

command time -v ficture transform --key ${key} --model ${model} --input ${pixel} --output_pref ${output} --hex_width ${fit_width} --n_move ${fit_nmove} --min_ct_per_unit ${min_ct_per_unit_fit} --mu_scale ${mu_scale} --thread ${thread} --precision 2 --major_axis ${major_axis}



# Projection without adaptation
pixel=${path}/batched.matrix.tsv.gz
anchor=${output_path}/${output_id}.${anchor_info}.fit_result.tsv.gz
prefix=${output_id}.decode.${anchor_info}_${radius}
output=${output_path}/${prefix}
topk=3

command time -v ficture slda_decode --input ${pixel} --output ${output} --model ${model} --anchor ${anchor} --anchor_in_um --neighbor_radius ${radius} --mu_scale ${mu_scale} --key ${key} --precision 0.25 --lite_topk_output_pixel ${topk} --lite_topk_output_pixel ${topk} --thread ${thread}

# Sort and index pixel level result
bsize=1000
scale=100
input=${output_path}/${prefix}.pixel.tsv.gz # j, X, Y, K1, ..., KJ, P1, ..., PJ, J=topk
output=${output_path}/${prefix}.pixel.sorted.tsv.gz

offsetx=${xmin}
offsety=${ymin}
rangex=$( echo "(${xmax} - ${xmin} + 0.5)/1+1" | bc )
rangey=$( echo "(${ymax} - ${ymin} + 0.5)/1+1" | bc )

header="##K=${K};TOPK=3\n##BLOCK_SIZE=${bsize};BLOCK_AXIS=X;INDEX_AXIS=Y\n##OFFSET_X=${offsetx};OFFSET_Y=${offsety};SIZE_X=${rangex};SIZE_Y=${rangey};SCALE=${scale}\n#BLOCK\tX\tY\tK1\tK2\tK3\tP1\tP2\tP3"

(echo -e "${header}" && zcat ${input} | tail -n +2 | perl -slane '$F[0]=int(($F[1]-$offx)/$bsize) * $bsize; $F[1]=int(($F[1]-$offx)*$scale); $F[1]=($F[1]>=0)?$F[1]:0; $F[2]=int(($F[2]-$offy)*$scale); $F[2]=($F[2]>=0)?$F[2]:0; print join("\t", @F);' -- -bsize=${bsize} -scale=${scale} -offx=${offsetx} -offy=${offsety} | sort -S 4G -k1,1g -k3,3g ) | bgzip -c > ${output}

tabix -f -s1 -b3 -e3 ${output}
rm ${input}




### Visualization

# Choose color
input=${output_path}/${output_id}.fit_result.tsv.gz
output=${figure_path}/${output_id}
ficture choose_color --input ${input} --output ${output} --cmap_name ${cmap_name}

# Make hexagon level figure
anchor=${output_path}/${output_id}.${anchor_info}.fit_result.tsv.gz
output=${figure_path}/${output_id}
fillr=$((anchor_res/2+1))
ficture plot_base --input ${anchor} --output ${output} --fill_range ${fillr} --color_table ${cmap} --plot_um_per_pixel 1 --xmin $xmin --xmax $xmax --ymin $ymin --ymax $ymax --plot_discretized


# Make pixel level figures
input=${output_path}/${prefix}.pixel.sorted.tsv.gz
output=${figure_path}/${prefix}.pixel.png
ficture plot_pixel_full --input ${input} --color_table ${cmap} --output ${output} --plot_um_per_pixel ${pixel_resolution} --full


# # Evaluation
# query=${output_path}/${prefix}.pixel.sorted.tsv.gz
# output=${output_path}/${prefix}
# if [ "${overwrite}" == "1" ] || [ ! -f "${output}.bad_pixel.tsv.gz" ]; then
#     python /nfs/turbo/sph-hmkang/ycsi/script/simu.eval_cluster.v2.py --path ${path} --query ${query} --output ${output} --K1 ${nFactor} --query_scale 0.01
# fi


# # Plot error
# input=${output_path}/${prefix}.bad_pixel.tsv.gz
# # cmap2=${output_path}/${prefix}.matched.rgb.tsv
# cmap2=${path}/model.rgb.tsv
# output=${figure_path}/${prefix}.bad_pixel
# if [ "${overwrite}" == "1" ] || [ ! -f "${output}.png" ]; then
#     python ${gitpath}/script/plot_base.py --input ${input} --output ${output} --color_table ${cmap2} --xmin $xmin --xmax $xmax --ymin $ymin --ymax $ymax --plot_um_per_pixel ${pixel_resolution} --category_column cell_label --color_table_category_name cell_type
# fi

# output=${figure_path}/${prefix}.bad_pixel.res1um
# python ${gitpath}/script/plot_base.py --input ${input} --output ${output} --color_table ${cmap2} --xmin $xmin --xmax $xmax --ymin $ymin --ymax $ymax --plot_um_per_pixel 1 --category_column cell_label --color_table_category_name cell_type


# # Plot with matched color
# input=${output_path}/${prefix}.pixel.sorted.tsv.gz
# cmap2=${output_path}/${prefix}.matched.rgb.tsv
# output=${figure_path}/${prefix}.repaint.pixel.png
# if [ "${overwrite}" == "1" ] || [ ! -f "${output}" ]; then
#     python ${gitpath}/script/plot_pixel_full.py --input ${input} --color_table ${cmap2} --output ${output} --plot_um_per_pixel ${pixel_resolution} --full
# fi
