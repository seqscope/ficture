Use Latent Dirichlet Allocation on sliding hexagons for clustering and differentially expressed gene identification.

**Scripts**
Pre-process multiple tiles with layout info
```/home/ycsi/git/factor_analysis/script/preprocess_multitile.py
```
Run LDA with total counts (does not work for velo output)
```/home/ycsi/git/factor_analysis/script/lda_base_multitile.py
```

The above scripts include an ad hoc step to exclude regions with few reads, purely based on relative density. The filtering might not always be proper, the filtered data is stored (see below) for reference.

Some tiles are processed, results are in NGST-output/PROJECT_NAME/lda. Currently processed example data is listed here
```
/nfs/turbo/sph-hmkang/ycsi/info/high_priority_dataset.tsv
```

**Example usage**
1 For mouse muscle data with spliced and unspliced reads separated

```
iden=HD31-HMKYV-AG-comb
lane=1
tile="2107,2108,2109,2207,2208,2209"
outpath="/scratch/hmkang_root/hmkang0/shared_data/LDA/HD31-HMKYV-AG-comb"

inpath="/scratch/hmkang_root/hmkang0/shared_data/NGST-sDGE/${iden}"
gene_info="/home/ycsi/data/ref/gencode.human.mouse.combined.gene.types.tsv"
layout="/home/ycsi/git/STtools/doc/hiseq.layout.tsv"

# Key words that exists in the gene type ID to identify genes to keep
# Not sure if this is an useful choice of labels, just for example
kept_gene_type="protein,IG,TR,lnc"

gitpath="/nfs/turbo/sph-hmkang/ycsi/git/factor_analysis"
script="${gitpath}/script/preprocess_multitile.py"

command time -v python ${script} --layout ${layout} --input_path ${inpath} --output_path ${outpath} --identifier ${iden} --lane ${lane} --tile ${tile} --gene_type_info ${gene_info} --gene_type_keyword ${kept_gene_type} --auto_rm_background_by_density

width_train=18
width_test=12
sliding_step=6

for nFactor in 4 6 8 10 12 15 18; do

    echo ${nFactor}
    command time -v python ${script} --experiment_id LDA_hexagon --input_path ${inpath} --output_path ${outpath} --identifier ${iden} --lane ${lane} --tile ${tile} --nFactor ${nFactor} --hex_width ${width_train} --hex_width_fit ${width_test} --n_move_hex_tile ${sliding_step} --gene_type_info ${gene_info} --gene_type_keyword ${kept_gene_type}

done
```

**Output**
In the above example, outputs are saved here
```
/scratch/hmkang_root/hmkang0/shared_data/LDA/HD31-HMKYV-AG-comb
```
/analysis includes
1. Assigning pixels to one best fitted factor \*.assign_pixel.tsv.gz
    Columns: unique pixel ID, X and Y in the combined rectangle field, top factor assigned, probability allocated to the top factor.
2. Differentially expressed genes (test one factor v.s. the rest) \*.DEgene.tsv.gz
    Naive Chi-sq test statistics (p-value cutoff at $1e-3$ then sorted by fold change)
3. Model fitting results (by hexagons) \*.fit_result.tsv.gz
    Columns:
    Topic_x: probabilities for each factor
    Hex_center_x(y): hexagon center cooredinate, in um
    offs_x(y) and hex_x(y) are hexagon center in axial coordinates

/{lane} stores the merged/filtered intermediate data used for analysis
