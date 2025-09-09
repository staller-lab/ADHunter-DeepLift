# ADHunter DeepLift
This contains all the scripts, notebooks, and data used/made by Jack Demaray to use DeepLIFT to interpret ADhunter, a model to predict if a given 40aa protein sequence can function as a transcriptional activation domain. Analysis showed that generally ADhunter considers the amino acids we expect to be important to be important (namely: bulky aromatics/hydrophobics in general and acidics).


## notebooks 
- `250304_ADHunter_explicit1he.ipynb`: Notebook where I made the `adhunter_1he.pt` model that I used for most subsequent analysis. This involved slightly modifying the ADhunter architecture by replacing the initial `nn.Embedding` layer with manual one-hot encoding and a `nn.Linear` layer to enable interpretation via DeepLIFT of the contributions from the one-hot input. `NOTE:` The model tweaks are all implemented in `../adhunter/actpred/actpred/models.py` and thus in my install of ADhunter, so subsequent notebooks just import the installed ADhunter. 
- `250311_regressions.ipynb`: Comparisons between ADhunter and linear regressions, showing that ADhunter outperforms regressions, especially in cases where e.g. composition alone isn't enough information.
- `250318_deeplift.ipynb`: A very polished notebook walking through how DeepLIFT works and generating DeepLIFT scores using Gcn4 ADs. This notebook was done entirely using a "negative baseline", or a baseline defined as the averaged one-hot encoded values for all non-ADs in the Gcn4 data (which I defined as tiles with measured activations below 50,000, in contrast to ADs which were defined as those with scores $\geq$ 80,000)
<!-- - `250319_adhunter_other_datasets.ipynb` -->
- `250403_deeplift_0baseline.ipynb`: Very similar to `250318_deeplift.ipynb`, but instead I used a baseline of all 0's (which I refer to as a null baseline; it's like a one-hot encoded sequence with no 1's). This led to more sensible DeepLIFT results and **contains the most up-to-date DeepLIFT interpretation of `adhunter_1he.pt`**

## scripts
- `aligned_logos.py`: Script to make logo plots for 70aa sequences corresponding to 
gcn4 orthologs aligned to the W in the WxxLF motif. 
- `logo_plots.py`: Makes logo plots for the most activating Gcn4 tiles using 
the averaged non-activating baseline. 
- `logo_plots_null_baseline.py`: Makes logo plots for the most activating Gcn4 tiles using the null baseline (same shape as one-hot encoded, but no 1's). 

## src
- `plotting.py`: Contains some of my favorite helper functions for plotting.
- `src.py`: Contains some functions to do some of the common tasks for this 
project. I didn't end up using these that much in my actual notebooks.

## data
Contains the csvs for the datasets I used as well as outputs from various
analyses. 
### Files
- `Gcn4Array_Design.csv`
- `OrthologTilingDFwActivities_20240930.csv`: Gcn4 AD screen
- `OrthologTilingDFwActivities_20240930_train_test_val.csv`: \<SCRATCH>

### Directories
- `alignments`: Gcn4 orthologs central activation domain alignments
- `logo_plots`: Plots of DeepLIFT importance scores from ADHunter model (trained on Gcn4 data) on various 40aa tiles against various baselines. See enclosed README
- `mutagenesis`: AD sequence optimization via *in silico* mutagenesis. The name of the model used is contained in the file. 

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
