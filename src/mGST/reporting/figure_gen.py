"""
Generation of figures
"""

from matplotlib import ticker
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


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

cmap = plt.colormaps.get_cmap("RdBu")
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


def plot_objf(res_list, delta, title):
    """Plots the objective function over iterations in the algorithm

    Parameters:
    res_list (List[float]): The residual values
    delta (float): The success threshold
    title (str): The plot title

    Returns:
    """
    plt.semilogy(res_list)
    plt.ylabel(f"Objective function")
    plt.xlabel(f"Iterations")
    plt.axhline(delta, color="green", label="conv. threshold")
    plt.title(title)
    plt.legend()
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


def generate_spam_err_std_pdf(
    filename, E, rho, E2, rho2, basis_labels=False, title=None, magnification=10, return_fig=False
):
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
    magnification : float
        A factor to be applied to magnify errors in the rightmost plot.
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
        else:
            plot_matrices = [np.real(E[i - 1]), np.real(E2[i - 1]), np.real(E[i - 1] - E2[i - 1]) * magnification]
            axes[i, 0].set_ylabel(f"E_%i" % (i - 1), rotation=90, fontsize="large")

        for j in range(3):
            ax = axes[i, j]
            ax.patch.set_facecolor("whitesmoke")
            ax.set_aspect("equal")
            for (x, y), w in np.ndenumerate(plot_matrices[j].reshape(pdim, pdim)):
                size = np.sqrt(np.abs(w))
                rect = plt.Rectangle(
                    [x + (1 - size) / 2, y + (1 - size) / 2],
                    size,
                    size,
                    facecolor=cmap((w + 1) / 2),
                    edgecolor=cmap((w + 1) / 2),
                )
                # print(cmap(size))
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

    if dim > 16:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0, shrink=0.6)
        fig.subplots_adjust(left=0, right=0.7, top=0.90, bottom=0.05, wspace=-0.6, hspace=0.4)
    else:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0)
        fig.subplots_adjust(left=0, right=0.7, top=0.90, bottom=0.05, wspace=-0.6, hspace=0.8)

    if return_fig:
        return fig

    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()
    return None


def generate_gate_err_pdf(
    filename, gates1, gates2, basis_labels=False, gate_labels=False, magnification=5, return_fig=False
):
    """Main routine to generate plots of reconstructed gates, ideal gates and the noise channels
    of the reconstructed gates.
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
            magnification * (np.eye(dim) - np.real(gates1[i] @ la.inv(gates2[i]))),
        ]

        for j in range(3):
            ax = axes[j]
            ax.patch.set_facecolor("whitesmoke")
            ax.set_aspect("equal")
            for (x, y), w in np.ndenumerate(plot_matrices[j].T):
                size = np.sqrt(np.abs(w))
                rect = plt.Rectangle(
                    [x + (1 - size) / 2, y + (1 - size) / 2],
                    size,
                    size,
                    facecolor=cmap((w + 1) / 2),
                    edgecolor=cmap((w + 1) / 2),
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
        axes[2].set_title(plot3_title, fontsize="large")
        fig.suptitle(f"Process matrices in the Pauli basis", va="bottom")

        if dim > 16:
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), pad=0, shrink=0.6)
            fig.subplots_adjust(left=0.1, right=0.76, top=0.85, bottom=0.03)
            set_size(0.5 * np.sqrt(dim), 1.3 * np.sqrt(dim))
        else:
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), pad=0)
            fig.subplots_adjust(left=0.1, right=0.76, top=0.85, bottom=0.03, hspace=0.2)
            set_size(2 * np.sqrt(dim), 0.8 * np.sqrt(dim))
        figures.append(fig)
        if not return_fig:
            plt.savefig(filename + f"G%i.pdf" % i, dpi=150, transparent=True, bbox_inches="tight")
    return figures
