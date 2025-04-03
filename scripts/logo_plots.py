import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from actpred.models import ActCNNSystem
from captum.attr import LayerDeepLift
import logomaker


def get_attr(dl, X, baseline, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    attr = dl.attribute(X, baselines=baseline, attribute_to_layer_input=True).detach().numpy()
    attr_df = pd.DataFrame(attr.mean(axis=0)).T
    attr_df.index = list(alphabet)
    return attr_df


def get_df(arr, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    df = pd.DataFrame(arr).T
    df.index = list(alphabet)
    return df


def logo_plot(attr_df:pd.DataFrame, seq_df:pd.DataFrame, ylabel:str="", title:str=""):
    """Creates a logo plot of importance scores with an additional logo plot on top
    corresponding to the actual sequence present. 

    Args:
        attr_df (pd.DataFrame): DataFrame containing DeepLIFT attribution scores
        seq_df (pd.DataFrame): DataFrame containing one-hot-encoded sequence
        ylabel (str, optional): ylabel for score logo plot. Defaults to "".
        title (str, optional): Suptitle for plot. Defaults to "".

    Returns:
        _type_: _description_
    """    
    fig, axs = plt.subplots(2, 1, height_ratios=[1, 13], dpi=300)
    actual = logomaker.Logo(seq_df.T,
                            shade_below=.5,
                            fade_below=.5,
                            font_name='Arial Rounded MT Bold',
                            color_scheme="dmslogo_funcgroup",
                            ax=axs[0])
    actual.style_spines(visible=False)
    actual.ax.axis("off")
    logo = logomaker.Logo(attr_df,
                            shade_below=.3,
                            fade_below=.1,
                            font_name='Arial Rounded MT Bold',
                            color_scheme="dmslogo_funcgroup",
                            ax=axs[1])
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left', 'bottom'], visible=True)
    logo.ax.axhline(0, c='r', linewidth=1)
    logo.ax.set_ylabel(ylabel)
    logo.ax.set_xlabel("Position")
    fig.suptitle(title, y=.95)
    return fig, axs


def main():
    alphabet="ACDEFGHIKLMNPQRSTVWY"
    aa_to_i = {aa:i for i, aa in enumerate(alphabet)}
    i_to_aa = {i:aa for i, aa in enumerate(alphabet)}

    # load in AD data
    fn = "../data/OrthologTilingDFwActivities_20240930.csv"
    df = pd.read_csv(fn)
    seqs = df.Seq.to_numpy()
    activity = df.Activity.to_numpy()
    thresh = 80_000
    thresh_neg = 50_000
    df_ad = df.iloc[np.where(df["Activity"] >= thresh)[0]].copy()
    # scale data to have mean 0 and variance 1
    y_cont = activity.reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(y_cont)
    y_cont = scaler.transform(y_cont)
    X = np.asarray([[aa_to_i[aa] for aa in x] for x in seqs])
    ad_indices = np.where(df["Activity"] >= thresh)[0]
    X_ad = torch.tensor(X[ad_indices])
    # Load model
    kernel_size = 5
    dilation = 3 
    hidden = 64
    model = ActCNNSystem(hidden, kernel_size, dilation, num_res_blocks=3)
    model.load_state_dict(torch.load("../adhunter/adhunter_1he.pt"))
    model.eval()
    # Define negative baseline
    X_neg = torch.tensor(X[np.where(df["Activity"] < thresh_neg)[0]])
    baseline_neg = model.model.encode(X_neg).mean(dim=0).unsqueeze(0)
    # Get top ADs
    top_ads_idx = df["Activity"].rank(ascending=False, method="min") <= 20
    X_top_ad = model.model.encode(torch.tensor(X[top_ads_idx]))
    top_ad_species = df.loc[top_ads_idx, "SpeciesNames"].values
    top_ad_seqs = df.loc[top_ads_idx, "Seq"].values
    # Get difference from baseline for each of the top ADs
    dx = (X_top_ad - baseline_neg).detach().numpy()
    # Calculate DeepLIFT attribution scores
    dl = LayerDeepLift(model, layer=model.model.emb, multiply_by_inputs=False,)
    attr = dl.attribute(
        X_top_ad, baseline_neg, attribute_to_layer_input=True
        ).detach().numpy()
    attr_scaled = attr * np.abs(dx)
    attr_dfs = [get_df(i) for i in attr_scaled]
    # Make logo plots!
    ylabel="$m_{i}|\Delta{x_i}|$"
    for i in range(len(attr_dfs)):
        species_i=top_ad_species[i]
        seq_i = top_ad_seqs[i]
        seq_df_i= get_df(
            model.model.encode(seq_i).squeeze().detach().numpy())
        attr_i = attr_dfs[i].T
        fig, axs = logo_plot(attr_df=attr_i, seq_df=seq_df_i, ylabel=ylabel, title=species_i)
        fig.savefig(f"../data/logo_plots/{seq_i}_{species_i}.png", transparent=False)
        plt.close()


if __name__ == "__main__":
    main()