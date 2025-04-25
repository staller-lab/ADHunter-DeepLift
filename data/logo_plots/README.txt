Logo plots of DeepLIFT importance scores for strongest activators in 
Gcn4 ortholog screen AND aligned Gcn4 CADs

SINGLE TILES:
These are the importance scores of each residue x position across the 
~43 most active sequences in the Gcn4 screen. I wanted to make plots for 
the 20 strongest, but there were a lot of sequences with the same high 
activation score. DeepLIFT was run using an ADHunter model trained on 
the Gcn4 dataset with 80,000 as the activation cutoff, and I used two 
different baselines: 
- null_baseline: a 1x40x20 tensor of all 0s 
- neg_baseline: the average one-hot-encoded value across all tiles with 
activation <50,000

Since many of these tiles were present in multiple species, I chose not 
to include species names in these plots.

COMBINED ALIGNED TILES:
I ran DeepLIFT using a null baseline on the 70aa sequences contained in
AllSeqs_IntegralAround_WxxLF_-50_+20_top138.fasta, corresponding to aligned
Gcn4 cADs aligned on the W in the WxxLF motifs
