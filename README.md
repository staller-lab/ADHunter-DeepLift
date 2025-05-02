# staller-rotation

## notebooks 
- `250304_ADHunter_explicit1he.ipynb`: Notebook where I made the `adhunter_1he.pt` model that I used for most subsequent analysis. This involved slightly modifying the ADhunter architecture by replacing the initial `nn.Embedding` layer with manual one-hot encoding and a `nn.Linear` layer to enable interpretation via DeepLIFT of the contributions from the one-hot input. `NOTE:` The model tweaks are all implemented in `../adhunter/actpred/actpred/models.py` and thus in my install of ADhunter, so subsequent notebooks just import the installed ADhunter. 
- `250306_interpret_model.ipynb`: Contains some of my initial attempts at using DeepLIFT to interpret the model trained in the previous notebook. 
- `250310_tada_classification.ipynb`: I tried to play around with classifying the Gcn4 ADs according to the kmeans clusters assigned in the PADI paper, and a bit of reclassifying the PADI ADs. This didn't really go anywhere because the way they classified in the PADI paper is kinda nonsensical (they use the 2D tSNE embeddings to do the clustering rather than clustering based on PCs/other features and then *projecting* those values on the tSNE).
- `250311_regressions.ipynb`: Comparisons between ADhunter and linear regressions, showing that ADhunter outperforms regressions, especially in cases where e.g. composition alone isn't enough information.
- `250318_deeplift.ipynb`: A very polished notebook walking through how DeepLIFT works and generating my finalized DeepLIFT scores using Gcn4 ADs. This notebook was done entirely using a "negative baseline", or a baseline defined as the averaged one-hot encoded values for all non-ADs in the Gcn4 data (which I defined as tiles with measured activations below 50,000, in contrast to ADs which were defined as those with scores $\geq$ 80,000)
<!-- - `250319_adhunter_other_datasets.ipynb` -->
- `250320_train_ADHunter_synthetics.ipynb`: Notebook where I trained an ADhunter model on Marissa's synthetics data. 
- `250324_trebl_model.ipynb`: I trained an ADhunter model using the TREBL-seq data to predict both the maximum activation ($v_{max}$) and the time it takes for that AD to get to half of that ($t_{1/2}$) for ADs in the TREBL-seq data. I tried using the weights of the `adhunter_1he.pt` model to facilitate training, but overall this wasn't successful. This could potentially be improved by freezing the weights of the convolutional layers, which I have not tried.  
- `250324_trebl_model_t_half.ipynb`: Similar to `250324_trebl_model.ipynb`, I trained a model to predict *ONLY* $t_{1/2}$ for the TREBL-seq data. 
<!-- - `250325_trebl_explore.ipynb` -->
- `250325_trebl_model_vmax.ipynb`: Similar to `250324_trebl_model.ipynb`, I trained a model to predict *ONLY* $v_{max}$ for the TREBL-seq data, and this worked alright. The improvement in performance is likely because both ADs and non-ADs have sensible $v_{max}$ values, so I had more data to train on.  
- `250326_trebl_regr.ipynb`: I trained linear regressions to predict $t_{1/2}$ for the TREBL-seq data and compared these results to those of ADhunter. Each model was trained using different train-test splits and I saw that performance varied a lot based on this, indicating that we don't have enough TREBL-seq data to predict $t_{1/2}$ yet. 
- `250327_trebl_model_t_half_clean.ipynb`: A cleaner version of `250324_trebl_model_t_half.ipynb`
  - `NOTE`: figure this out 
<!-- - `250331_logos.ipynb` -->
- `250331_train_ADHunter_synthetics_gcn4.ipynb`: Notebook where I trained an ADhunter model on Marissa's synthetics data harmonized with the Gcn4 data. 
- `250401_simulations.ipynb`: I did some simulations of what determines activation and trained ADhunter on those simulated outputs to see if it DeepLIFT for these models would show that acidic residues are important. This was motivated by consistent confusion resulting from the DeepLIFT outputs using the average non-activating baseline (see `250318_deeplift.ipynb`), where DeepLIFT assigns negative contribution scores to acidic residues. 
- `250403_deeplift_0baseline.ipynb`: Very similar to `250318_deeplift.ipynb`, but instead I used a baseline of all 0's (which I refer to as a null baseline; it's like a one-hot encoded sequence with no 1's). This led to more sensible DeepLIFT results and **contains the most up-to-date DeepLIFT interpretation of `adhunter_1he.pt`**
<!-- - `250403_padi.ipynb` -->
- `250404_sog1_explore.ipynb`: Some exploratory plots of sog1 shuffle data, 
including some kmer enrichment analysis
- `250407_adhunter_sog1.ipynb`: Training an ADhunter model on the sog1 shuffle library.
It didn't really work. 
<!-- - `250408_starling.ipynb` -->
<!-- - `250417_conservation.ipynb` -->
- `250417_mutagenesis.ipynb`: Some development for the *in silico* mutagenesis approach for optimizing AD sequences, as finalized in `scripts/mutagenesis.py`.
<!-- - `250425_ADHunter_PADI.ipynb` -->

## scripts
- `aligned_logos.py`: Script to make logo plots for 70aa sequences corresponding to 
gcn4 orthologs aligned to the W in the WxxLF motif. 
- `logo_plots.py`: Makes logo plots for the most activating Gcn4 tiles using 
the averaged non-activating baseline. 
- `logo_plots_null_baseline.py`: Makes logo plots for the most activating Gcn4 tiles using the null baseline (same shape as one-hot encoded, but no 1's). 
- `mutagenesis.py`: Script to generate *in silico* mutagenesis-derived optimal 
sequences starting from a random sequence or the top activating sequences in 
Marissa's synthetics screen using a given ADhunter model.

## src
- `plotting.py`: Contains some of my favorite helper functions for plotting.
- `src.py`: Contains some functions to do some of the common tasks for this 
project. I didn't end up using these that much in my actual notebooks.

## data
Contains the csvs for the datasets I used as well as outputs from various
analyses. 
### Files
- `EC_TREBLactivities_20250321_v2.csv`: TREBL-seq data
- `Gcn4Array_Design.csv`
- `HarmonizedGcn4SynADs.csv`: Harmonized Gcn4 AD screen + Marissa synthetics screen
- `OrthologTilingDFwActivities_20240930.csv`: Gcn4 AD screen
- `OrthologTilingDFwActivities_20240930_train_test_val.csv`: \<SCRATCH>
- `PADI.csv`: PADI dataset from [Morffy et al.](https://www.nature.com/articles/s41586-024-07707-3)
- `SynAD_Glu_Filtered_ActivityCtrlStd_replicateactivities.csv`: Marissa synthetic ADs dataset
- `SynAD_Glu_Filtered_ActivityCtrlStd_replicateactivities_simpleave.csv`: Marissa synthetic ADs dataset (reprocessed)

### Directories
- `alignments`: Gcn4 orthologs central activation domain alignments
- `logo_plots`: Plots of DeepLIFT importance scores from ADHunter model (trained on Gcn4 data) on various 40aa tiles against various baselines. See enclosed README
- `mutagenesis`: AD sequence optimization via *in silico* mutagenesis. The name of the model used is contained in the file. 
- `sog1`: Sog1 AD shuffle screen data

## adhunter
`actpred`: Contains the source code for ADhunter. Mostly the same as Hunter's ~original
version, but with a few tweaks with respect to how the inputs were handled
to allow interpretation with respect to the one-hot encoded version of the 
inputs rather than using a `nn.Embedding` layer to both encode and embed. There
are instructions in the README contained here for installing ADhunter via pip.
- `actpred/models.py`: contains the code for the ADhunter model
- `actpred/utils.py`: contains some utilities, like the code for the 
train/test/val split. 
  - I did change the splitting function a little bit to try to output 
the actual indices chosen for each split, but there's a bug there so don't
pay attention to that. 
- `Train_And_Use_ADHunter.ipynb`
<!-- - `actcnn_model.onnx` -->
<!-- - `actpred` -->
- `adhunter.pt`: A model trained on the Gcn4 data using Hunter's original version
of the model.
- `adhunter_1he.pt`: A model trained on Gcn4 data using my updated code. This is 
the main model I used for my analyses.
- `adhunter_1he_harmonized.pt`: A model trained on the Gcn4 data harmonized with
Marissa's synthetics data. 
- `adhunter_1he_syn.pt`: A model trained on the Marissa's synthetics data. 
- `adhunter_pretrained.pt`: A version of ADhunter that I fetched from savio
but didn't really use for anything.
- `data`: Contains a version of the Gcn4 screen and some other stuff. 
<!-- - `logs`
- `models` -->
