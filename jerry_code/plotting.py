"""
Functions for plotting results
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_log_reco_err(pu_errors, hs_errors, hs_errs_2 = None):
    log_pu_err = np.log(pu_errors + 1e-9)
    log_hs_err = np.log(hs_errors + 1e-9)

    # Create figure and axes
    fig, ax1 = plt.subplots()

    # Plot histogram for PU Error Data
    ax1.hist(log_pu_err, bins=50, alpha=0.5, color='blue', label='Log PU Errors')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency (PU Errors)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a twin Axes sharing the x-axis
    ax2 = ax1.twinx()

    # Plot histogram for HS Error Data
    ax2.hist(log_hs_err, bins=50, alpha=0.5, color='red', label='Log HS Errors')
    if hs_errs_2 is not None:
        ax2.hist(np.log(hs_errs_2 + 1e-9), bins=50, alpha=0.5, color='yellow', label='Log VBF HS Errors')
    ax2.set_ylabel('Frequency (HS Errors)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Histogram of Two Datasets')
    plt.show()


def plot_err_vs_pt2(err_lst, pt_lst, labels=None):
    color_lst = ["blue", "red", "green"]
    N_colors = len(color_lst)
    if len(err_lst) != len(pt_lst):
        raise ValueError("Insufficient sets of label data to complement every set of pt data")
    for i in range(len(err_lst)):
        data_label = f"Dataset {i}" if labels is None else labels[i]
        plt.scatter(pt_lst[i], err_lst[i], color=color_lst[i % N_colors], marker='o', s=2, label=data_label, alpha=0.5)
    plt.plot([1, 1000], [1, 1000], color='yellow', alpha=0.5)
    plt.xlabel("Sum pt2", fontsize=18)
    plt.ylabel("Reconstruction error", fontsize=18)
    plt.yscale('log')
    plt.xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=18)
    plt.show()


def line_plot(x_data_lst, y_data_lst, y_err_lst=None, labels=None, title="", xlabel="", ylabel="", ylim=None):
    colors = ['blue', 'green', 'red', 'orange']

    if len(x_data_lst) != len(y_data_lst):
        raise ValueError("Length of x / y inputs don't match")
    if len(x_data_lst) > len(colors):
        raise RuntimeError("More datasets than there are colors! This plot is too busy.")
    if labels is not None and len(x_data_lst) != len(labels):
        labels = None
        raise RuntimeWarning("Number of labels not the same as number of datasets to plot. Setting labels to NONE.")
    if y_err_lst is not None and len(x_data_lst) != len(y_err_lst):
        y_err_lst = None
        raise RuntimeWarning("Number of error-value sets not equal to number of datasets. Setting error list to NONE.")

    for i in range(len(x_data_lst)):
        data_label = f"Dataset {i}" if labels is None else labels[i]
        y_err = [0 for i in range(len(y_data_lst[i]))] if y_err_lst is None else y_err_lst[i]
        plt.plot(x_data_lst[i], y_data_lst[i], color=colors[i], marker='o', markersize=2, linestyle='-', alpha=0.5, label=data_label)
        # Create scatter plot with error bars
        # plt.errorbar(x_data_lst[i], y_data_lst[i], yerr=y_err, fmt='o', color=colors[i], label=data_label)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title(title, fontsize=24)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.show()
