"""
Generation of figures
"""

from typing import List, Tuple

from matplotlib import ticker
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import xarray as xr

from iqm.benchmarks.benchmark_definition import BenchmarkObservationIdentifier
from mGST.reporting.reporting import compute_matched_ideal_hamiltonian_params, generate_basis_labels


SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

cmap = plt.colormaps.get_cmap("coolwarm")
norm = Normalize(vmin=-1, vmax=1)


def set_size(w, h, ax=None):
    """Forcing a figure to a specified size

    Parameters
    ----------
    w : floar
        width in inches
    h : float
        height in inches
    ax : matplotlib axes
        The optional axes
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_objf(res_list, title, delta = None):
    """Plots the objective function over iterations in the algorithm

    Parameters:
    res_list (List[float]): The residual values
    delta (float): The success threshold
    title (str): The plot title

    Returns:
    """
    if delta is not None:
        plt.semilogy(res_list)
        plt.axhline(delta, color="green", label="conv. threshold")
        plt.legend()
    else:
        plt.plot(res_list)
    plt.ylabel(f"Objective function")
    plt.xlabel(f"Iterations")
    plt.title(title)
    plt.show()


def generate_spam_err_pdf(filename, E, rho, E2, rho2, title=None, spam2_content="ideal"):
    """Generate pdf plots of two sets of POVM + state side by side in vector shape - Pauli basis
    The input sets can be either POVM/state directly or a difference different SPAM parametrizations to
    visualize errors.

    Parameters
    ----------
    filename : str
        The name under which the figures are saved in format "folder/name"
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    E2 : numpy array
        POVM #2
    rho2 : numpy array
        Initial state #2
    title : str
        The Figure title
    basis_labels : list[str]
        A list of labels for the basis elements. For the Pauli basis ["I", "X", "Y", "Z"] or the multi-qubit version.
    spam2_content : str
        Label of the right SPAM plot to indicate whether it is the ideal SPAM parametrization or for instance
        the error between the reconstructed and target SPAM

    Returns
    -------
    """
    r = rho.shape[0]
    pdim = int(np.sqrt(r))
    n_povm = E.shape[0]
    fig, axes = plt.subplots(ncols=2, nrows=n_povm + 1, sharex=True)
    plt.rc("image", cmap="RdBu")

    ax = axes[0, 0]
    ax.imshow(rho, vmin=-1, vmax=1)  # change_basis(S_true_maps[0],"std","pp")
    ax.set_xticks(np.arange(r))
    ax.set_title(r"rho")
    ax.yaxis.set_major_locator(ticker.NullLocator())

    ax = axes[0, 1]
    im0 = ax.imshow(rho2, vmin=-1, vmax=1)  # change_basis(S_true_maps[0],"std","pp")
    ax.set_xticks(np.arange(r))
    ax.set_title(r"rho - " + spam2_content)
    ax.yaxis.set_major_locator(ticker.NullLocator())

    for i in range(n_povm):
        ax = axes[1 + i, 0]
        ax.imshow(E[i], vmin=-1, vmax=1)  # change_basis(S_true_maps[0],"std","pp")
        ax.set_xticks(np.arange(pdim))
        ax.set_xticklabels(np.arange(pdim) + 1)
        ax.set_title(f"E%i" % (i + 1))
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())

        ax = axes[1 + i, 1]
        ax.imshow(E2[i], vmin=-1, vmax=1)  # change_basis(S_true_maps[0],"std","pp")
        ax.set_xticks(np.arange(pdim))
        ax.set_xticklabels(np.arange(pdim) + 1)
        ax.set_title(f"E%i - " % (i + 1) + spam2_content)
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), pad=0.1)
    cbar.ax.set_ylabel(r"Pauli basis coefficient", labelpad=5, rotation=90)

    if title:
        fig.suptitle(title)
    if r > 16:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0, shrink=0.6)
        fig.subplots_adjust(left=0, right=0.7, top=0.90, bottom=0.05, wspace=-0.6, hspace=0.4)
    else:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0)
        fig.subplots_adjust(left=0, right=0.7, top=0.90, bottom=0.05, wspace=-0.6, hspace=0.8)

    set_size(3, 2)
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()


def generate_spam_err_std_pdf(filename, E, rho, E2, rho2, basis_labels=False, title=None, return_fig=False):
    """Generate pdf plots of two sets of POVM + state side by side in matrix shape - standard basis
    The input sets can be either POVM/state directly or a difference different SPAM parametrizations to
    visualize errors.

    Parameters
    ----------
    filename : str
        The name under which the figures are saved in format "folder/name"
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    E2 : numpy array
        POVM #2
    rho2 : numpy array
        Initial state #2
    title : str
        The Figure title
    basis_labels : list[str]
        A list of labels for the basis elements. For the standard basis this could be ["00", "01",...]
    return_fig : bool
        If set to True, a figure object is returned by the function, otherwise the plot is saved as <filename>
    Returns
    -------
    """
    dim = rho.shape[0]
    pdim = int(np.sqrt(dim))
    n_povm = E.shape[0]

    fig, axes = plt.subplots(ncols=3, nrows=n_povm + 1, gridspec_kw={"width_ratios": [1, 1, 1]}, sharex=True)
    plt.rc("image", cmap="RdBu")

    for i in range(n_povm + 1):
        if i == 0:
            plot_matrices = [np.real(rho), np.real(rho2), np.real(rho - rho2)]
            axes[i, 0].set_ylabel(f"rho", rotation=90, fontsize="large")
            axes[i, 0].set_title(f"Estimate", fontsize="large")
            axes[i, 1].set_title(f"Reference", fontsize="large")
            axes[i, 2].set_title(f"Deviation", fontsize="large")
        else:
            plot_matrices = [np.real(E[i - 1]), np.real(E2[i - 1]), np.real(E[i - 1] - E2[i - 1])]
            axes[i, 0].set_ylabel(f"E_%i" % (i - 1), rotation=90, fontsize="large")

        for j in range(3):
            ax = axes[i, j]
            ax.patch.set_facecolor("whitesmoke")
            ax.set_aspect("equal")
            max_entry = np.abs(plot_matrices[j]).max()
            max_weight = 2 ** np.ceil(np.log2(max_entry, where=max_entry > 0))
            for (x, y), w in np.ndenumerate(plot_matrices[j].reshape(pdim, pdim)):
                if j == 2:
                    size = np.sqrt(np.abs(w) / max_weight)
                else:
                    size = np.sqrt(np.abs(w))
                rect = plt.Rectangle(
                    [x + (1 - size) / 2, y + (1 - size) / 2],
                    size,
                    size,
                    facecolor="#d62728" if w < 0 else "#1f77b4",
                    edgecolor="#d62728" if w < 0 else "#1f77b4",
                )
                ax.add_patch(rect)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(pdim + 1), labels=[])

            ax.set_yticks(np.arange(pdim + 1), labels=[])
            ax.tick_params(which="major", length=0)  # Turn dummy ticks invisible
            ax.tick_params(which="minor", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=1)

            if pdim > 4:
                ax.grid(visible="True", alpha=0.4, lw=0.1)
                ax.set_xticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, rotation=45, fontsize=2)
                ax.set_yticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, fontsize=2)
            else:
                ax.grid(visible="True", alpha=0.4)
                ax.set_xticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, rotation=45, fontsize=6)
                ax.set_yticks(np.arange(pdim) + 0.5, minor=True, labels=basis_labels, fontsize=6)
    if title:
        fig.suptitle(title)

    if dim >= 16:
        fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.05, wspace=0.2, hspace=0.2)
        set_size(np.sqrt(dim), 2 * np.sqrt(dim))
    else:
        fig.subplots_adjust(left=0, right=1, top=0.76, bottom=0.05, wspace=-0.8, hspace=0.4)
        set_size(3 * np.sqrt(dim), 1.2 * np.sqrt(dim))

    if return_fig:
        return fig

    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()
    return None


def generate_gate_err_pdf(
    filename, gates1, gates2, basis_labels=False, gate_labels=False, magnification=5, return_fig=False
):
    """Main routine to generate plots of reconstructed gates, ideal gates and the noise channels
    of the reconstructed gates. The matrices are shown as Hinton diagrams, where the size of each square represents
    the magnitude of the matrix element and the color represents its sign as well as the magnitude.
    The basis is arbitrary but using gates in the Pauli basis is recommended.

    Parameters
    ----------
    filename : str
        The name under which the figures are saved in format "folder/name"
    gates1 : numpy array
        A gate set in the same format as the "X"-tensor. These gates are assumed to be the GST estimates.
    gates1 : numpy array
        A gate set in the same format as the "X"-tensor. These are assumed to be the target gates.
    basis_labels : list[str]
        A list of labels for the basis elements. For the standard basis this could be ["00", "01",...]
        and for the Pauli basis ["I", "X", "Y", "Z"] or the multi-qubit version.
    gate_labels : list[str]
        A list of names for the gates
    magnification : float
        A factor to be applied to magnify errors in the rightmost plot.
    return_fig : bool
        If set to True, a figure object is returned by the function, otherwise the plots are saved as <filename>
    """
    d = gates1.shape[0]
    dim = gates1[0].shape[0]
    if not basis_labels:
        basis_labels = np.arange(dim)
    if not gate_labels:
        gate_labels = [f"G%i" % k for k in range(d)]
    plot3_title = r"id - G U^{-1}"

    figures = []
    for i in range(d):
        if dim > 16:
            fig, axes = plt.subplots(ncols=1, nrows=3, gridspec_kw={"height_ratios": [1, 1, 1]}, sharex=True)
        else:
            fig, axes = plt.subplots(ncols=3, nrows=1, gridspec_kw={"width_ratios": [1, 1, 1]}, sharex=True)
        dim = gates1[0].shape[0]
        plot_matrices = [
            np.real(gates1[i]),
            np.real(gates2[i]),
            np.eye(dim) - np.real(gates1[i] @ la.inv(gates2[i])),
        ]

        for j in range(3):
            ax = axes[j]
            ax.patch.set_facecolor("whitesmoke")
            ax.set_aspect("equal")
            max_weight = 2 ** np.ceil(np.log2(np.abs(plot_matrices[j]).max()))
            for (x, y), w in np.ndenumerate(plot_matrices[j].T):
                if j == 2:
                    size = np.sqrt(np.abs(w) / max_weight)
                else:
                    size = np.sqrt(np.abs(w))
                rect = plt.Rectangle(
                    [x + (1 - size) / 2, y + (1 - size) / 2],
                    size,
                    size,
                    facecolor="#d62728" if w < 0 else "#1f77b4",
                    edgecolor="#d62728" if w < 0 else "#1f77b4",
                )
                ax.add_patch(rect)
            ax.invert_yaxis()
            ax.set_xticks(np.arange(dim + 1), labels=[])
            ax.set_yticks(np.arange(dim + 1), labels=[])

            if dim > 16:
                ax.grid(visible="True", alpha=0.4, lw=0.1)
                ax.set_xticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels, fontsize=2, rotation=45)
            else:
                ax.grid(visible="True", alpha=0.4)
                ax.set_xticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels, rotation=45)
            ax.set_yticks(np.arange(dim) + 0.5, minor=True, labels=basis_labels)
            ax.tick_params(which="major", length=0)  # Turn dummy ticks invisible
            ax.tick_params(which="minor", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)

        axes[0].set_title(r"G (estimate)", fontsize="large")
        axes[0].set_ylabel(gate_labels[i], rotation=90, fontsize="large")
        axes[1].set_title(r"U (ideal gate)", fontsize="large")
        axes[2].set_title(plot3_title + "\n(renormalized)", fontsize="large")
        fig.suptitle(f"Process matrices in the Pauli basis\n(red:<0; blue:>0)")

        if dim > 16:
            fig.tight_layout(rect=[0, 0, 1, 0.90])  # Leave space for suptitle
            set_size(0.5 * np.sqrt(dim), 1.3 * np.sqrt(dim))
        else:
            fig.tight_layout(rect=[0, 0, 1, 0.85])  # Leave space for suptitle
            set_size(2 * np.sqrt(dim), 0.8 * np.sqrt(dim))
        figures.append(fig)
        if not return_fig:
            plt.savefig(filename + f"G%i.pdf" % i, dpi=150, transparent=True, bbox_inches="tight")
    return figures


def generate_hamiltonian_visualizations(
    dataset: xr.Dataset, n_errs: int = 4, threshold: float = 0.01
) -> Tuple[List[List[Figure]], List[List[Figure]]]:
    """
    Plots the coherent errors as the difference of the entries of the Hamiltonian in the Pauli basis to its ideal values.
    1. Matrix plots showing all parameters with color coding
    2. Bar plots showing the largest coherent errors

    Args:
        dataset (xarray.Dataset): A dataset containing counts from the experiment, results, and configurations.
        n_errs (int): Number of largest errors to plot in bar charts. Default is 4.
        threshold (float, optional): Minimum threshold for errors to display in bar charts. The threshold can lead to
            more than n_errs errors being displayed.

    Returns:
        Tuple[List[List[matplotlib.figure.Figure]], List[List[matplotlib.figure.Figure]]]:
            A tuple containing two nested lists of figures for each qubit layout and gate:
            - Matrix plots
            - Bar plots
    """
    # Get the Hamiltonian parameters and their ideal values
    hamiltonian_params, hamiltonian_params_ideal = compute_matched_ideal_hamiltonian_params(dataset)
    param_labels = generate_basis_labels(dataset.attrs["pdim"], basis="Pauli")
    gate_labels = list(dataset.attrs["gate_labels"].values())

    # Collect upper and lower end of the confidence intervals
    qubit_layouts = dataset.attrs["qubit_layouts"]
    has_uncertainties = all(
        dataset.attrs[f"results_layout_{BenchmarkObservationIdentifier(layout).string_identifier}"][
            "hamiltonian_params"
        ].get("uncertainties")
        is not None
        for layout in qubit_layouts
    )

    if has_uncertainties:
        hamiltonian_params_low = np.array(
            [
                dataset.attrs[f"results_layout_{BenchmarkObservationIdentifier(layout).string_identifier}"][
                    "hamiltonian_params"
                ]["uncertainties"][0]
                for layout in qubit_layouts
            ]
        )
        hamiltonian_params_high = np.array(
            [
                dataset.attrs[f"results_layout_{BenchmarkObservationIdentifier(layout).string_identifier}"][
                    "hamiltonian_params"
                ]["uncertainties"][1]
                for layout in qubit_layouts
            ]
        )
    else:
        # Create dummy arrays if no uncertainties are available
        hamiltonian_params_low = hamiltonian_params.copy()
        hamiltonian_params_high = hamiltonian_params.copy()

    plots = {}

    for l_idx, layout in enumerate(qubit_layouts):
        # Iterate through each layout's parameters
        for params, params_ideal, params_low, params_high, gate_label in zip(
            hamiltonian_params[l_idx],
            hamiltonian_params_ideal[l_idx],
            hamiltonian_params_low[l_idx],
            hamiltonian_params_high[l_idx],
            gate_labels[l_idx].values(),
        ):

            # Generate figures for each gate in the layout
            # Calculate error vectors (difference between measured and ideal)
            param_delta = params - params_ideal
            param_delta_low = params_low - params_ideal if has_uncertainties else param_delta
            param_delta_high = params_high - params_ideal if has_uncertainties else param_delta

            # Get uncertainties as difference vectors for error bars
            yerr_low = np.abs(param_delta - param_delta_low) if has_uncertainties else None
            yerr_high = np.abs(param_delta_high - param_delta) if has_uncertainties else None

            # 1. Generate matrix plot
            params_reshaped = params.reshape(-1, 1)
            params_low_reshaped = params_low.reshape(-1, 1)
            params_high_reshaped = params_high.reshape(-1, 1)
            shape = params_reshaped.shape
            max_size = 16
            num_splits = int(np.ceil(shape[0] / max_size))

            # Split param_vector into a smaller array to plot
            param_splits = np.array_split(params_reshaped, num_splits, axis=0)
            param_splits_low = np.array_split(params_low_reshaped, num_splits, axis=0)
            param_splits_high = np.array_split(params_high_reshaped, num_splits, axis=0)

            fig_matrix, axes = plt.subplots(len(param_splits), 1, figsize=(10, len(param_splits)))
            if len(param_splits) == 1:
                axes = [axes]

            for idx, (param_split, param_split_low, param_split_high, ax) in enumerate(
                zip(param_splits, param_splits_low, param_splits_high, axes)
            ):
                split_size = param_split.shape[0]
                im = ax.matshow(param_split.T, cmap="coolwarm", vmin=-0.05, vmax=0.05)
                ax.set_yticks(np.arange(1), labels=[], rotation=0)
                ax.set_xticks(
                    np.arange(split_size), labels=list(param_labels)[idx * split_size : (idx + 1) * split_size]
                )

                for i in range(param_split.shape[0]):
                    for j in range(param_split.shape[1]):
                        param_value = f"{param_split[i, j]:.3e}"

                        if has_uncertainties:
                            upper_bound = f"{param_split_high[i, j]:.3e}"
                            lower_bound = f"{param_split_low[i, j]:.3e}"

                            # Display upper bound above the parameter value
                            ax.text(i, j - 0.2, upper_bound, va="center", ha="center", color="black", fontsize=6)
                            # Display parameter value in the middle
                            ax.text(i, j, param_value, va="center", ha="center", color="black", fontweight="bold")
                            # Display lower bound below the parameter value
                            ax.text(i, j + 0.2, lower_bound, va="center", ha="center", color="black", fontsize=6)
                        else:
                            # Just display the parameter value if no uncertainties
                            ax.text(i, j, param_value, va="center", ha="center", color="black")

            plt.title(f"Hamiltonian parameters for {gate_label}", fontsize=10)

            fig_matrix.colorbar(im, ax=axes, fraction=0.005, pad=0.04)
            plots.update({f"layout_{layout}_{gate_label}_Hamiltonian": fig_matrix})
            plt.close(fig_matrix)

            # 2. Generate bar plot for largest errors
            fig_bar, ax = plt.subplots(figsize=(6, 4))

            # Sort indices by absolute magnitude
            sorting_indices = np.argsort(np.abs(param_delta))[::-1]
            param_delta_sorted = param_delta[sorting_indices]

            # Apply a threshold if provided to show more than n_errs errors
            if threshold is not None:
                mask = np.abs(param_delta_sorted) >= threshold
                n_errs = max(n_errs, np.sum(mask))

            # Truncate to determined number of errors
            param_delta_sorted = param_delta_sorted[:n_errs]
            sorting_indices = sorting_indices[:n_errs]

            # Create bars
            bars = ax.bar(
                range(n_errs),
                param_delta_sorted,
                color="#1f77b4",
                alpha=0.7,
            )

            # Add error bars if available
            if has_uncertainties:
                error_low = yerr_low[sorting_indices]
                error_high = yerr_high[sorting_indices]
                error_positions = np.arange(n_errs)

                ax.errorbar(
                    error_positions,
                    param_delta_sorted,
                    yerr=[error_low, error_high],
                    fmt="none",
                    ecolor=[0.2, 0.2, 0.2, 0.5],
                    capsize=3,
                )
                error_extend = np.max([np.max(np.abs(param_delta_high)), np.max(np.abs(param_delta_low))])
                ax.set_ylim(-error_extend * 1.1, error_extend * 1.1)
            else:
                param_range = np.max(param_delta_sorted) - np.min(param_delta_sorted)
                ax.set_ylim(
                    np.min(param_delta_sorted) - param_range / 10, np.max(param_delta_sorted) + param_range / 10
                )

            # Configure axis labels and title
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(n_errs))
            ax.set_xticklabels(np.array(list(param_labels))[sorting_indices])
            ax.set_xlabel("Pauli labels")
            ax.set_ylabel("Deviation from target")
            ax.set_title(f"Largest coherent errors for {gate_label}", fontsize=10)

            # Add values on top of each bar
            for i, bar in enumerate(bars):
                value = param_delta[sorting_indices[i]]
                height = bar.get_height()
                ax.annotate(
                    f"{value:.2e}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2) if height > 0 else (0, -11),  # vertical offset above or below the bar
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plots.update({f"layout_{layout}_{gate_label}_largest_coherent_errs": fig_bar})
            plt.close(fig_bar)

    return plots
