import torch
from actpred.models import ActCNNSystem
"""Helper functions for ADHunter
"""

def get_alphabet():
    return "ACDEFGHIKLMNPQRSTVWY"


def get_encoding():
    """Returns dictionaries to encode amino acid letters as
    ints and vice versa.
    """    
    alphabet = get_alphabet()
    aa_to_i = {aa:i for i, aa in enumerate(alphabet)}
    i_to_aa = {i:aa for i, aa in enumerate(alphabet)}
    return aa_to_i, i_to_aa


def load_model(model_path):
    """Loads an ADHunter model

    Args:
        model_path (str): Path to model.pt

    Returns:
        ActCNNSystem object corresponding to ADHunter model
    """    
    kernel_size = 5
    dilation = 3 
    hidden = 64
    model = ActCNNSystem(hidden, kernel_size, dilation, num_res_blocks=3)
    model.load_state_dict(torch.load(model_path))
    # model.eval()
    return model