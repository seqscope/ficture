# Format raw output from different platforms

FICTURE only needs a file with 2D coordinates, gene/transcript ID, and the corresponding transcript count.

We have tested FICTURE on Seq-scope, Stereo-seq, CosMx SMI, Xenium, MERSCOPE, and Visium HD data. Here we document how to format the raw data from each platform to the required input files like the following example data.

```bash linenums="1"
examples/data/transcripts.tsv.gz
examples/data/feature.clean.tsv.gz
```

[CosMx SMI](format_input/cosmx.md)

[10X Xenium](format_input/xenium.md)

[10X Visium HD](format_input/visiumHD.md)

[Vizgen MERSCOPE](format_input/vizgen.md)
