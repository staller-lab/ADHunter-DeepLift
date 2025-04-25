"""
Script to use ADHunter + in silico mutagenesis to create ~optimal ADs
Example usage:
    python mutagenesis.py \
        -o ../data/mutagenesis/random \
        -m ../adhunter/adhunter_1he.pt \
        -n 20 \
        --mode random
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from actpred.models import ActCNNSystem
import copy
import argparse
import os
os.chdir("../adhunter")
import sys
sys.path.append("../src/")
import src
from plotting import legend_kwargs

def mut_step(seq):
    """For a given sequence, randomly replace an AA at each 
    position with another AA, then run ADHunter on all of 
    those sequences + the original and select the one with 
    the highest score

    Args:
        seq (torch.tensor): Starting sequence. Must be a 
        tensor of integers corresponding to AAs

    Returns:
        best_seq: highest scoring sequence
        best_score: highest score
    """    
    rand_seqs = [seq]
    for i in range(len(seq)):
        seq_i = copy.deepcopy(seq)
        seq_i[i] = np.random.randint(0, 20)
        rand_seqs.append(seq_i)

    rand_seqs = torch.tensor(np.vstack(rand_seqs))
    with torch.no_grad():
        rand_scores = model(rand_seqs)

    idxmax = torch.argmax(rand_scores)
    best_seq, best_score = rand_seqs[idxmax], rand_scores[idxmax]
    return best_seq, best_score

def run_mutagenesis(model, seq0=None, n_iters=100):
    """Starting with a certain sequence, randomly mutagenize each residue and 
    pick the most activating one given and ADHunter model.

    Args:
        seq0 (_type_, optional): _description_. Defaults to None.
        n_iters (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """    
    if type(seq0) == str:
        # alphabet=src.get_alphabet()
        aa_to_i, _ = src.get_encoding()
        seq0 = torch.tensor([aa_to_i[i] for i in seq0])
    if seq0 is None:
        seq0 = torch.tensor(np.random.randint(0, 20, 40))
    with torch.no_grad():
        seqs, scores = [seq0], [model(seq0.unsqueeze(0))]
    seqs, scores
    for i in range(n_iters):
        seq_i, score_i = mut_step(seqs[-1])
        seqs.append(seq_i)
        scores.append(score_i)
    seqs = torch.vstack(seqs)
    scores = torch.concat(scores)
    return seqs, scores


if __name__ == "__main__":
    # ARGUMENTS
    parser = argparse.ArgumentParser(  
        description='Script to use ADHunter + in silico mutagenesis to create ' \
        '~optimal ADs')  
    parser.add_argument(
        '-o', '--output', type=str, required=True, 
        help="Output directory"
    )
    parser.add_argument(
        '-m', '--model', type=str, required=True, 
        help="Path to ADHunter model"
    )
    parser.add_argument(
        '-n', '--n_iter', type=int, required=True, default=100,
        help="Number of iterations for mutagenesis"
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help="Random seed for mutagenesis"
    )
    parser.add_argument(
        '--mode', type=str, default="random",
        help="Mode for starting sequences for mutagenesis. Can use 'random' to " \
        "start from random sequences or 'synthetics' to start with top activators in " \
        "synthetics screen"
    )
    args = parser.parse_args()


    np.random.seed(args.seed)
    alphabet = src.get_alphabet()
    aa_to_i, i_to_aa = src.get_encoding()
    # Load in model
    model = src.load_model(args.model)
    model_name = args.model.split("/")[-1].split(".pt")[0]
    model.eval()


    translator = lambda seq: "".join([i_to_aa[i] for i in seq.numpy()])


    if args.mode == "random":
        n_rand = 30
        seed_seqs = torch.tensor(np.random.randint(0, 20, (n_rand, 40)))
    if args.mode == "synthetics":
        sdf = pd.read_csv(
            "../data/SynAD_Glu_Filtered_ActivityCtrlStd_replicateactivities.csv")
        sdf.sort_values(by="Activity", ascending=False, inplace=True)
        amax = sdf["Activity"].max()
        seed_seqs = (torch.tensor(
            np.vstack(
                sdf.loc[sdf["Activity"] == amax, "AAseq"]
                .apply(lambda seq: np.array([aa_to_i[i] for i in seq]))
                .values)))

    initial_activations = model(seed_seqs).detach().squeeze()


    seqs_initial, scores, seqs_final = [], [], []
    plt.figure(figsize=(15, 8))
    for seq in seed_seqs:
        seed_seq = torch.tensor(np.random.randint(0, 20, 40))
        seqs_i, scores_i = run_mutagenesis(model, seed_seq, args.n_iter)
        seq_final = translator(seqs_i[-1])
        # Store stuff
        seqs_initial.append(translator(seed_seq))
        seqs_final.append(seq_final)
        scores.append(scores_i[-1].item())
        # Plot 
        plt.plot(range(len(scores_i)), scores_i, label=seq_final)
    plt.legend(title="Final Sequence", **legend_kwargs())
    plt.ylabel("Activity")
    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output, 
                     f"{model_name}_iter{args.n_iter}_mutagenesis.png"))
    
    plt.close()

    result_df = pd.DataFrame(
        np.vstack([seqs_initial, initial_activations, seqs_final, scores]).T)
    result_df.columns = ["initial_seq", "initial_activity",
                         "final_seq", "activity"]
    result_df.to_csv(
        os.path.join(args.output, 
                     f"{model_name}_iter{args.n_iter}_mutagenesis.csv"))
