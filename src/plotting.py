import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def legend_kwargs():
    # Commonly used kwargs for figure legends
    kwargs = {
        "frameon" : False, 
        "bbox_to_anchor" : (1, 0.5), 
        "loc" : "center left"}
    return kwargs

def initialize_multiplot(n_plots, n_cols=3, **kwargs):
    '''Given a certain number of plots, initialize subplots that have that many
    plots divided into n_cols columns. 
    '''    
    n_rows = n_plots // n_cols+1 if n_plots % n_cols != 0 else n_plots // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, **kwargs)
    # Remove any plots that won't be used
    lastrow = n_rows - 1
    first_unused = (n_plots ) % n_cols
    if first_unused != 0:
        for col in range(first_unused, n_cols):
            axs[lastrow, col].remove()
    return fig, axs, get_axes(n_plots, n_cols)

def get_axes(n_plots, n_cols):
    '''Generator to use with the above initialize_multiplots function that gets 
    the correct axis indices as you iterate your plotting function

    USAGE: 
    fig, axs, which_ax = initialize_multiplot(n_cols, n_plots)
    for ...:
        row, col = next(which_ax)
        plot_fxn(*args, ax=axs[row, col])
    '''
    idx = 0
    n_rows = n_plots//n_cols+1 if n_plots%n_cols!=0 else n_plots//n_cols
    while idx <= n_plots-1:
        row = idx // n_cols
        col = idx % n_cols
        yield row, col
        idx+=1

