import sys, os, gzip, argparse, logging, warnings, shutil

from ficture.utils.minimake import minimake

def run_together(_args):
    """Run all functions in FICTURE by using GNU Makefile
    This function is meant to be used in a local environment that has sufficient resources to run all functions in FICTURE at once.
    This function performs the following tasks:
    (1) Take the input parameters relevant to the FICTURE runs
    (2) Identify the sequence of commands to run FICTURE
    (3) Create a GNU makefile to run the commands in parallel
    (4) Run the GNU makefile
    """

    parser = argparse.ArgumentParser()

    cmd_params = parser.add_argument_group("Commands", "FICTURE commands to run together")
    cmd_params.add_argument('--all', action='store_true', default=False, help='Run all FICTURE commands (preprocess, segment, lda, decode)')
    cmd_params.add_argument('--preprocess', action='store_true', default=False, help='Perform preprocess step')
    cmd_params.add_argument('--segment', action='store_true', default=False, help='Perform hexagonal segmentation')
    cmd_params.add_argument('--sge', action='store_true', default=False, help='Create hexagonal SGEs')
    cmd_params.add_argument('--lda', action='store_true', default=False, help='Perform LDA model training')
    cmd_params.add_argument('--decode', action='store_true', default=False, help='Perform pixel-level decoding')

    run_params = parser.add_argument_group("Run Options", "Run options for FICTURE commands")
    run_params.add_argument('--dry-run', action='store_true', default=False, help='Dry run. Generate only the Makefile without running it')
    run_params.add_argument('--restart', action='store_true', default=False, help='Restart the run. Ignore all intermediate files and start from the beginning')
    run_params.add_argument('--threads', type=int, default=1, help='Maximum number of threads to use in each process')
    run_params.add_argument('--n-jobs', type=int, default=1, help='Number of jobs (processes) to run in parallel')

    key_params = parser.add_argument_group("Key Parameters", "Key parameters that requires user's attention")
    key_params.add_argument('--in-tsv', required=True, type=str, help='Input TSV file (e.g. transcript.tsv.gz)')
    key_params.add_argument('--out-dir', required= True, type=str, help='Output directory')
    key_params.add_argument('--in-minmax', type=str, help='Input coordinate minmax TSV file (e.g. coordinate_minmax.tsv). If absent, it will be generated')
    key_params.add_argument('--in-feature', type=str, help='Input TSV file (e.g. feature.clean.tsv.gz) that specify which genes to use as input. If absent, it will be use all genes')
    key_params.add_argument('--major-axis', type=str, default='Y', help='Axis where transcripts.tsv.gz are sorted')
    key_params.add_argument('--mu-scale', type=float, default=1.0, help='Scale factor for mu (pixels per um)')
    key_params.add_argument('--train-width', type=str, default="12", help='Hexagon flat-to-flat width (in um) during training. Use comma to specify multiple values')
    key_params.add_argument('--n-factor', type=str, default="12", help='Number of factors to train. Use comma to specify multiple values')
    key_params.add_argument('--anchor-res', type=float, default=4, help='Anchor resolution for decoding')
    key_params.add_argument('--plot-each-factor', action='store_true', default=False, help='Plot individual factors in pixel-level decoding')

    aux_params = parser.add_argument_group("Auxiliary Parameters", "Auxiliary parameters (using default is recommended)")
    aux_params.add_argument('--train-epoch', type=int, default=3, help='Training epoch for LDA model')
    aux_params.add_argument('--train-epoch-id-len', type=int, default=2, help='Training epoch ID length')
    aux_params.add_argument('--train-n-move', type=int, default=2, help='Level of hexagonal sliding during training')
    aux_params.add_argument('--sge-n-move', type=int, default=1, help='Level of hexagonal sliding during SGE generation')
    aux_params.add_argument('--fit-width', type=float, help='Hexagon flat-to-flat width (in um) during model fitting (default: same to train-width)')
    aux_params.add_argument('--key-col', type=str, default="Count", help='Columns from the input file to be used as key')
    aux_params.add_argument('--minibatch-size', type=int, default=500, help='Batch size used in minibatch processing')
    aux_params.add_argument('--minibatch-buffer', type=int, default=30, help='Batch buffer used in minibatch processing')
    aux_params.add_argument('--min-ct-unit-dge', type=int, default=50, help='Minimum count per hexagon in DGE generation')
    aux_params.add_argument('--min-ct-unit-sge', type=int, default=1, help='Minimum count per hexagon in SGE generation')
    aux_params.add_argument('--min-ct-feature', type=int, default=20, help='Minimum count per feature during LDA training')
    aux_params.add_argument('--min-ct-unit-fit', type=int, default=20, help='Minimum count per hexagon unit during model fitting')
    aux_params.add_argument('--lda-rand-init', type=int, default=10, help='Number of random initialization during model training')
    aux_params.add_argument('--decode-top-k', type=int, default=3, help='Top K columns to output in pixel-level decoding results')
    aux_params.add_argument('--de-max-pval', type=float, default=1e-3, help='p-value cutoff for differential expression')
    aux_params.add_argument('--de-min-fold', type=float, default=1.5, help='Fold-change cutoff for differential expression')
    aux_params.add_argument('--decode-block-size', type=int, default=100, help='Block size for pixel decoding output')
    aux_params.add_argument('--decode-scale', type=int, default=100, help='Scale parameters for pixel decoding output')
    aux_params.add_argument('--cmap-name', type=str, default="turbo", help='Name of color map')
    aux_params.add_argument('--dge-precision', type=float, default=2, help='Output precision of hexagon coordinates')
    aux_params.add_argument('--fit-precision', type=float, default=2, help='Output precision of model fitting')
    aux_params.add_argument('--decode-precision', type=float, default=0.1, help='Precision of pixel level decoding')
    aux_params.add_argument('--lda-plot-um-per-pixel', type=float, default=1, help='Image resolution for LDA plot')
    aux_params.add_argument('--decode-plot-um-per-pixel', type=float, default=0.5, help='Image resolution for pixel decoding plot')
    aux_params.add_argument('--decode-sub-um-per-pixel', type=float, default=1, help='Image resolution for individual subplots')
    aux_params.add_argument('--bgzip', type=str, default="bgzip", help='Path to bgzip binary. For faster processing, use "bgzip -@ 4')
    aux_params.add_argument('--tabix', type=str, default="tabix", help='Path to tabix binary')
    aux_params.add_argument('--gzip', type=str, default="gzip", help='Path to gzip binary. For faster processing, use "pigz -p 4"')
    aux_params.add_argument('--sort', type=str, default="sort", help='Path to sort binary. For faster processing, you may add arguments like "sort -T /path/to/new/tmpdir --parallel=20 -S 10G"')

    args = parser.parse_args(_args)

    if len(_args) == 0:
        parser.print_help()
        return

    if args.all:
        args.preprocess = True
        args.segment = True
        args.sge = False
        args.lda = True
        args.decode = True

    ## parse input parameters
    train_widths = [int(x) for x in args.train_width.split(",")]
    n_factors = [int(x) for x in args.n_factor.split(",")]

    batch_tsv = f"{args.out_dir}/batched.matrix.tsv"
    batch_out = f"{args.out_dir}/batched.matrix.tsv.gz"
    minmax_out = args.in_minmax if args.in_minmax is not None else f"{args.out_dir}/coordinate_minmax.tsv"

    ## create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    mm = minimake()

    ## gzip and sort have to exist across everywhere..
    if not shutil.which(args.gzip.split(" ")[0]):
        logging.error(f"Cannot find {args.gzip}. Please make sure that the path to --gzip is correct")
        sys.exit(1)

    if not shutil.which(args.sort.split(" ")[0]):
        logging.error(f"Cannot find {args.sort}. Please make sure that the path to --sort is correct")
        sys.exit(1)

    if args.preprocess:
        ## create output directory if needed
        cmds = []
        cmds.append(rf"$(info --------------------------------------------------------------)")
        cmds.append(rf"$(info Creating minibatch from {args.in_tsv}...)")
        cmds.append(rf"$(info --------------------------------------------------------------)")
        ## create minibatch
        cmds.append(f"ficture make_spatial_minibatch --input {args.in_tsv} --output {batch_tsv} --mu_scale {args.mu_scale} --batch_size {args.minibatch_size} --batch_buff {args.minibatch_buffer} --major_axis {args.major_axis}")
        cmds.append(f"{args.sort} -k 2,2n -k 1,1g {batch_tsv} | {args.gzip} -c > {batch_out}")
        cmds.append(f"rm {batch_tsv}")
        mm.add_target(batch_out, [args.in_tsv], cmds)

        if args.in_minmax is None:
            script_path = f"{args.out_dir}/write_minmax.sh"
            with open(script_path, "w") as f:
                f.write(r"""#!/bin/bash
input=$1
output=$2
mu_scale=$3
gzip -cd ${input} | awk 'BEGIN{FS=OFS="\t"} NR==1{for(i=1;i<=NF;i++){if($i=="X")x=i;if($i=="Y")y=i}print $x,$y;next}{print $x,$y}' | perl -slane 'print join("\t",$F[0]/${mu_scale},$F[1]/${mu_scale})' -- -mu_scale="${mu_scale}" | awk -F'\t' ' BEGIN { min1 = "undef"; max1 = "undef"; min2 = "undef"; max2 = "undef"; } { if (NR == 2 || $1 < min1) min1 = $1; if (NR == 2 || $1 > max1) max1 = $1; if (NR == 2 || $2 < min2) min2 = $2; if (NR == 2 || $2 > max2) max2 = $2; } END { print "xmin\t", min1; print "xmax\t", max1; print "ymin\t", min2; print "ymax\t", max2; }' > ${output}
""")
            cmds = []
            cmds.append(rf"$(info --------------------------------------------------------------)")
            cmds.append(rf"$(info Obtaining boundary coordinates to {minmax_out}...)")
            cmds.append(rf"$(info --------------------------------------------------------------)")
            cmds.append(f"bash {script_path} {args.in_tsv} {minmax_out} {args.mu_scale}")
            mm.add_target(minmax_out, [args.in_tsv], cmds)

    if args.segment:
        for train_width in train_widths:
            dge_out = f"{args.out_dir}/hexagon.d_{train_width}.tsv"
            cmds = []
            cmds.append(rf"$(info --------------------------------------------------------------)")
            cmds.append(rf"$(info Creating DGE for {train_width}um...)")
            cmds.append(rf"$(info --------------------------------------------------------------)")
            cmds.append(f"ficture make_dge --key {args.key_col} --input {args.in_tsv} --output {dge_out} --hex_width {train_width} --n_move {args.train_n_move} --min_ct_per_unit {args.min_ct_unit_dge} --mu_scale {args.mu_scale} --precision {args.dge_precision} --major_axis {args.major_axis}")
            cmds.append(f"{args.sort} -k 1,1n {dge_out} | {args.gzip} -c > {dge_out}.gz")
            cmds.append(f"rm {dge_out}")
            mm.add_target(f"{dge_out}.gz", [args.in_tsv], cmds)

    if args.sge:
        for train_width in train_widths:
            sge_out_dir = f"{args.out_dir}/segment/sge.d_{train_width}"
            cmds = []
            cmds.append(rf"$(info --------------------------------------------------------------)")
            cmds.append(rf"$(info Creating SGE for {train_width}um...)")
            cmds.append(rf"$(info --------------------------------------------------------------)")
            cmds.append(f"ficture make_sge_by_hexagon --key {args.key_col} --input {args.in_tsv} --feature {args.in_feature} --output_path {sge_out_dir} --hex_width {train_width} --n_move {args.sge_n_move} --min_ct_per_unit {args.min_ct_unit_sge} --mu_scale {args.mu_scale} --precision {args.dge_precision} --major_axis {args.major_axis} --transfer_gene_prefix")
            mm.add_target(f"{sge_out_dir}/barcodes.tsv.gz", [args.in_tsv], cmds)

    if args.lda:
        for train_width in train_widths:
            for n_factor in n_factors:
                model_id=f"nF{n_factor}.d_{train_width}"
                model_path=f"{args.out_dir}/analysis/{model_id}"
                figure_path=f"{model_path}/figure"
                hexagon = f"{args.out_dir}/hexagon.d_{train_width}.tsv.gz"
                model_prefix=f"{model_path}/{model_id}"
                model=f"{model_prefix}.model.p"
                feature_arg = f"--feature {args.in_feature}" if args.in_feature is not None else ""

                cmds = []
                cmds.append(rf"$(info --------------------------------------------------------------)")
                cmds.append(rf"$(info Creating LDA for {train_width}um and {n_factor} factors...)")
                cmds.append(rf"$(info --------------------------------------------------------------)")
                cmds.append(f"mkdir -p {model_path}/figure")
                cmds.append(f"ficture fit_model --input {hexagon} --output {model_prefix} {feature_arg} --nFactor {n_factor} --epoch {args.train_epoch} --epoch_id_length {args.train_epoch_id_len} --unit_attr X Y --key {args.key_col} --min_ct_per_feature {args.min_ct_feature} --test_split 0.5 --R {args.lda_rand_init} --thread {args.threads}")

                fit_tsv=f"{model_path}/{model_id}.fit_result.tsv.gz"
                fig_prefix=f"{figure_path}/{model_id}"
                cmap=f"{figure_path}/{model_id}.rgb.tsv"
                cmds.append(f"ficture choose_color --input {fit_tsv} --output {fig_prefix} --cmap_name {args.cmap_name}")

                fillr = (train_width / 2 + 1)
                cmds.append(f"ficture plot_base --input {fit_tsv} --output {fig_prefix}.coarse --fill_range {fillr} --color_table {cmap} --plot_um_per_pixel {args.lda_plot_um_per_pixel} --plot_discretized")
                cmds.append(f"touch {model_prefix}.done")

                mm.add_target(f"{model_prefix}.done", [args.in_tsv, hexagon], cmds)

    if args.decode:
        if not shutil.which(args.bgzip.split(" ")[0]):
            logging.error(f"Cannot find {args.bgzip}. Please make sure that the path to --bgzip is correct")
            sys.exit(1)

        if not shutil.which(args.tabix.split(" ")[0]):
            logging.error(f"Cannot find {args.tabix}. Please make sure that the path to --tabix is correct")
            sys.exit(1)


        script_path = f"{args.out_dir}/sort_decode.sh"
        with open(script_path, "w") as f:
            f.write(r"""#!/bin/bash
input=$1
output=$2
coor=$3
model_id=$4
bsize=$5
scale=$6
topk=$7
bgzip=$8
tabix=$9

K=$( echo $model_id | sed 's/nF\([0-9]\{1,\}\)\..*/\1/' )
while IFS=$'\t' read -r r_key r_val; do
    export "${r_key}"="${r_val}"
done < ${coor}
echo -e "${xmin}, ${xmax}; ${ymin}, ${ymax}"

offsetx=${xmin}
offsety=${ymin}
rangex=$( echo "(${xmax} - ${xmin} + 0.5)/1+1" | bc )
rangey=$( echo "(${ymax} - ${ymin} + 0.5)/1+1" | bc )
bsize=2000
scale=100
header="##K=${K};TOPK=${topk}\n##BLOCK_SIZE=${bsize};BLOCK_AXIS=X;INDEX_AXIS=Y\n##OFFSET_X=${offsetx};OFFSET_Y=${offsety};SIZE_X=${rangex};SIZE_Y=${rangey};SCALE=${scale}\n#BLOCK\tX\tY\tK1\tK2\tK3\tP1\tP2\tP3"

(echo -e "${header}" && gzip -cd "${input}" | tail -n +2 | perl -slane '$F[0]=int(($F[1]-$offx)/$bsize) * $bsize; $F[1]=int(($F[1]-$offx)*$scale); $F[1]=($F[1]>=0)?$F[1]:0; $F[2]=int(($F[2]-$offy)*$scale); $F[2]=($F[2]>=0)?$F[2]:0; print join("\t", @F);' -- -bsize="${bsize}" -scale="${scale}" -offx="${offsetx}" -offy="${offsety}" | sort -S 1G -k1,1g -k3,3g ) | ${bgzip} -c > ${output}

${tabix} -f -s1 -b3 -e3 ${output}
rm ${input}
""")
        for train_width in train_widths:
            for n_factor in n_factors:
                batch_in = f"{args.out_dir}/batched.matrix.tsv.gz"
                model_id=f"nF{n_factor}.d_{train_width}"
                model_path=f"{args.out_dir}/analysis/{model_id}"
                figure_path=f"{model_path}/figure"
                model_prefix=f"{model_path}/{model_id}"
                cmap=f"{figure_path}/{model_id}.rgb.tsv"
                model=f"{args.out_dir}/analysis/{model_id}/{model_id}.model.p"

                if args.fit_width is None:
                    fit_widths = [train_width]
                else:
                    fit_widths = [float(x) for x in args.fit_width.split(",")]
                for fit_width in fit_widths:
                    cmds = []

                    fit_nmove = int(fit_width / args.anchor_res)
                    anchor_info=f"prj_{fit_width}.r_{args.anchor_res}"
                    radius = args.anchor_res + 1

                    prj_prefix = f"{model_path}/{model_id}.{anchor_info}"
                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(rf"$(info Creating projection for {train_width}um and {n_factor} factors, at {fit_width}um)")
                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(f"ficture transform --input {args.in_tsv} --output_pref {prj_prefix} --model {model} --key {args.key_col} --major_axis {args.major_axis} --hex_width {fit_width} --n_move {fit_nmove} --min_ct_per_unit {args.min_ct_unit_fit} --mu_scale {args.mu_scale} --thread {args.threads} --precision {args.fit_precision}")

                    batch_input=f"{args.out_dir}/batched.matrix.tsv.gz"
                    anchor=f"{prj_prefix}.fit_result.tsv.gz"
                    decode_basename=f"{model_id}.decode.{anchor_info}_{radius}"
                    decode_prefix=f"{model_path}/{decode_basename}"

                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(rf"$(info Performing pixel-level decoding..)")
                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(f"ficture slda_decode --input {batch_in} --output {decode_prefix} --model {model} --anchor {anchor} --anchor_in_um --neighbor_radius {radius} --mu_scale {args.mu_scale} --key {args.key_col} --precision {args.decode_precision} --lite_topk_output_pixel {args.decode_top_k} --lite_topk_output_anchor {args.decode_top_k} --thread {args.threads}")

                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(rf"$(info Sorting and reformatting the pixel-level output..)")
                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(f"bash {script_path} {decode_prefix}.pixel.tsv.gz {decode_prefix}.pixel.sorted.tsv.gz {minmax_out} {model_id} {args.decode_block_size} {args.decode_scale} {args.decode_top_k} {args.bgzip} {args.tabix}")

                    de_input=f"{decode_prefix}.posterior.count.tsv.gz"
                    de_output=f"{decode_prefix}.bulk_chisq.tsv"

                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(rf"$(info Performing pseudo-bulk differential expression analysis..)")
                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(f"ficture de_bulk --input {de_input} --output {de_output} --min_ct_per_feature {args.min_ct_feature} --max_pval_output {args.de_max_pval} --min_fold_output {args.de_min_fold} --thread {args.threads}")

                    cmap=f"{figure_path}/{model_id}.rgb.tsv"
                    cmds.append(f"ficture factor_report --path {model_path} --pref {decode_basename} --color_table {cmap}")

                    decode_tsv=f"{decode_prefix}.pixel.sorted.tsv.gz"
                    decode_png=f"{model_path}/figure/{decode_basename}.pixel.png"

                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(rf"$(info Drawing pixel-level output image...)")
                    cmds.append(rf"$(info --------------------------------------------------------------)")
                    cmds.append(f"ficture plot_pixel_full --input {decode_tsv} --color_table {cmap} --output {decode_png} --plot_um_per_pixel {args.decode_plot_um_per_pixel} --full")

                    if args.plot_each_factor:
                        sub_prefix=f"{model_path}/figure/sub/{decode_basename}.pixel"
                        cmds.append(f"mkdir -p {model_path}/figure/sub")
                        cmds.append(f"ficture plot_pixel_single --input {decode_tsv} --output {sub_prefix} --plot_um_per_pixel {args.decode_sub_um_per_pixel} --full --all")

                    cmds.append(f"touch {decode_prefix}.done")
                    mm.add_target(f"{decode_prefix}.done", [batch_in, hexagon,f"{model_prefix}.done"], cmds)


    if len(mm.targets) == 0:
        logging.error("There is no target to run. Please make sure that at least on run option was turned on")
        sys.exit(1)

    ## write makefile
    mm.write_makefile(f"{args.out_dir}/Makefile")

    if args.dry_run:
        ## run makefile
        os.system(f"make -f {args.out_dir}/Makefile -n")
        print(f"To execute the pipeline, run the following command:\nmake -f {args.out_dir}/Makefile -j {args.n_jobs}")
    else:
        os.system(f"make -f {args.out_dir}/Makefile -j {args.n_jobs}")

if __name__ == "__main__":
    run_together(sys.argv[1:])
