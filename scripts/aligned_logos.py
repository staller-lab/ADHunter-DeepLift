import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from actpred.models import ActCNNSystem
from captum.attr import LayerDeepLift
import logomaker
import sys
from Bio import SeqIO
sys.path.append(".")
from logo_plots import get_df, logo_plot

# Load the fasta file and get 40mers
fn = "../data/alignments/AllSeqs_IntegralAround_WxxLF_-50_+20_top138.fasta"
records = [[record.id, str(record.seq)] for record in SeqIO.parse(fn, "fasta")]
adf = pd.DataFrame(records, columns=["id", "seq"])
adf = adf.loc[adf["seq"].str.len() == 70]
adf.loc[~adf["seq"].str.contains(r"\."), "seq"]
split_df = pd.concat(
    [adf["id"], adf["seq"].apply(
        # lambda seq: pd.Series([seq[:40], seq[-40:]]))], 
        lambda seq: pd.Series([seq[:40], seq[30:]]))], 
    axis=1)
# load in model and encode sequences
kernel_size = 5
dilation = 3 
hidden = 64
model = ActCNNSystem(hidden, kernel_size, dilation, num_res_blocks=3)
model.load_state_dict(torch.load("../adhunter/adhunter_1he.pt"))
model.eval()
X0 = torch.stack(split_df[0].apply(model.model.encode).tolist()).squeeze()
X1 = torch.stack(split_df[1].apply(model.model.encode).tolist()).squeeze()
# Define negative baseline as all 0 tensor
baseline_null = torch.zeros(1, 40, 20)
# Run DeepLIFT
dl = LayerDeepLift(model, layer=model.model.emb, multiply_by_inputs=True,)
X0_attr = dl.attribute(X0, baseline_null, attribute_to_layer_input=True).detach().numpy()
X1_attr = dl.attribute(X1, baseline_null, attribute_to_layer_input=True).detach().numpy()
# Average contributions for overlapping parts of 40mers
X0_part = X0_attr[:, :30, :]            # (125, 30, 20)
avg_part = (X0_attr[:, 30:, :] + X1_attr[:, :10, :]) / 2  # (125, 10, 20)
X1_part = X1_attr[:, 10:, :]            # (125, 30, 20)
attr = np.concatenate([X0_part, avg_part, X1_part], axis=1)
# Plot logos
alphabet="ACDEFGHIKLMNPQRSTVWY"
aa_to_i = {aa:i for i, aa in enumerate(alphabet)}
# Make a (colorblind-friendly) colormap
basics_color, acidics_color, hydrophobics_color, aro_color = [
    sns.color_palette("colorblind")[0], 
    # sns.color_palette("colorblind")[3], 
    "#e41a1c",
    sns.color_palette("colorblind")[-2], 
    sns.color_palette("colorblind")[2]]
other_color = 'grey'

color_scheme = {
        'A': other_color,
        'C': other_color,
        'D': acidics_color,
        'E': acidics_color,
        'F': aro_color,
        'G': other_color,
        'H': basics_color,
        'I': hydrophobics_color,
        'K': basics_color,
        'L': hydrophobics_color,
        'M': hydrophobics_color,
        'N': other_color,
        'P': other_color,
        'Q': other_color,
        'R': basics_color,
        'S': other_color,
        'T': other_color,
        'V': hydrophobics_color,
        'W': aro_color,
        'Y': aro_color
    }
for i in range(len(adf)):
    seq_name = adf["id"].values[i]
    # Make a df for plotting the actual sequence up top
    seq_i = adf["seq"].tolist()[i]
    seq_i = np.array([aa_to_i[j] for j in seq_i])
    seq_arr_i = np.zeros((70, 20))
    seq_arr_i[np.arange(70), seq_i] += 1
    seq_df_i = get_df(seq_arr_i)
    # Make and save the plot
    fig, axs = logo_plot(
        get_df(attr[i]).T, 
        seq_df_i, 
        title=seq_name, 
        ylabel="Contribution",
        color_scheme=color_scheme)
    out_name_i = f"../data/logo_plots/aligned/{seq_name}.png"
    fig.savefig(out_name_i, transparent=False)
    plt.close()