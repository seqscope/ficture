#!/bin/bash

#SBATCH --output=/home/%u/out/%x-%j.log
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=7g

### Pixel level decoding
# Input: env, path, model_id, anchor_info, radius, coor, pixel

key=Count
mu_scale=1
thread=1
pixel_resolution=0.5    # um, for visualization
analysis_resolution=0.1 # um

# For DE output
min_ct_per_feature=50
max_pval_output=1e-3
min_fold_output=1.5

# For indexing pixel level output
bsize=2000
scale=100

plot_individual_factor=0

plot_subbatch=8

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

coor=${coor:-${path}/coordinate_minmax.tsv}
pixel=${pixel:-${path}/batched.matrix.tsv.gz}

output_path=${path}/analysis/${model_id}
figure_path=${output_path}/figure
output_id=${model_id}

feature=${feature:-${path}/feature.clean.tsv.gz}
model=${model:-${output_path}/${output_id}.model_matrix.tsv.gz}

anchor=${output_path}/${output_id}.${anchor_info}.fit_result.tsv.gz
cmap=${figure_path}/${output_id}.rgb.tsv
K=$( echo $model_id | sed 's/nF\([0-9]\{1,\}\)\..*/\1/' )

while IFS=$'\t' read -r r_key r_val; do
    export "${r_key}"="${r_val}"
done < ${coor}
echo -e "${xmin}, ${xmax}; ${ymin}, ${ymax}"



# Projection without adaptation
prefix=${output_id}.decode.${anchor_info}_${radius}
output=${output_path}/${prefix}
# Output only a few top factors per pixel
topk=3 # Fix for now
python ${gitpath}/script/slda_decode.py --input ${pixel} --output ${output} --model ${model} --anchor ${anchor} --anchor_in_um --neighbor_radius ${radius} --mu_scale ${mu_scale} --key ${key} --precision ${analysis_resolution} --lite_topk_output_pixel ${topk} --lite_topk_output_anchor ${topk} --thread ${thread}


# Sort and index pixel level result
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





# DE
input=${output_path}/${prefix}.posterior.count.tsv.gz
output=${output_path}/DE/${prefix}.bulk_chisq.tsv
python ${gitpath}/script/de_bulk.py --input ${input} --output ${output} --min_ct_per_feature ${min_ct_per_feature} --max_pval_output ${max_pval_output} --min_fold_output ${min_fold_output} --thread ${thread}


# Report (color table and top DE genes)
cmap=${output_path}/figure/${output_id}.rgb.tsv
output=${output_path}/${prefix}.factor.info.html
python ${gitpath}/script/factor_report.py --path ${output_path} --pref ${prefix} --color_table ${cmap}


# Make pixel level figures
cmap=${output_path}/figure/${output_id}.rgb.tsv
input=${output_path}/${prefix}.pixel.sorted.tsv.gz
output=${figure_path}/${prefix}.pixel.png
python ${gitpath}/script/plot_pixel_full.py --input ${input} --color_table ${cmap} --output ${output} --plot_um_per_pixel ${pixel_resolution} --full


if [ "${plot_individual_factor}" -eq 0 ]; then
    exit 0
fi

# If the data is very large, making all individual factor maps may take some time
# Make single factor heatmaps, plot_subbatch balances speed and memory ...
# batch size 8 should be safe for 7 or 14G in most cases
output=${figure_path}/sub/${prefix}.pixel
if [ "${plot_subbatch}" -lt 1 ] || [ "${plot_subbatch}" -gt ${K} ]; then
    python ${gitpath}/script/plot_pixel_single.py --input ${input} --output ${output} --plot_um_per_pixel ${pixel_resolution} --full --all
else
    st=0
    ed=$((plot_subbatch+st-1))
    while [ ${st} -lt ${K} ]; do
        if [ ${ed} -gt ${K} ]; then
            ed=$((K-1))
        fi
        id_list=$( seq ${st} ${ed} )
        echo $id_list

        python ${gitpath}/script/plot_pixel_single.py --input ${input} --output ${output} --id_list ${id_list} --plot_um_per_pixel ${pixel_resolution} --full

        st=$((ed+1))
        ed=$((plot_subbatch+st-1))
    done
fi
