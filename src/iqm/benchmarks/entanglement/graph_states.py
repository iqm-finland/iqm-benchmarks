# Copyright 2025 IQM Benchmarks developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-many-lines

"""
Graph states benchmark
"""
from collections import defaultdict
import itertools
from time import strftime
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Type, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
import xarray as xr

from iqm.benchmarks import Benchmark, BenchmarkCircuit, BenchmarkRunResult, CircuitGroup, Circuits
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import (
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    add_counts_to_dataset,
)
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.randomized_benchmarking_common import import_native_gate_cliffords
from iqm.benchmarks.utils import (  # marginal_distribution, perform_backend_transpilation,
    bootstrap_counts,
    generate_state_tomography_circuits,
    get_neighbors_of_edges,
    get_Pauli_expectation,
    get_tomography_matrix,
    median_with_uncertainty,
    remove_directed_duplicates_to_list,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    set_coupling_map,
    split_sequence_in_chunks,
    submit_execute,
    timeit,
    xrvariable_to_counts,
)
from iqm.benchmarks.utils_plots import GraphPositions, rx_to_nx_graph
from iqm.benchmarks.utils_shadows import get_local_shadow, get_negativity, local_shadow_tomography
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def find_edges_with_disjoint_neighbors(
    graph: Sequence[Sequence[int]],
) -> List[List[Sequence[int]]]:
    """Finds sets of edges with non-overlapping neighboring nodes.

    Args:
        graph (Sequence[Sequence[int]]): The input graph specified as a sequence of edges (Sequence[int]).
    Returns:
        List[List[Tuple[int]]]: A list of lists of edges (Tuple[int]) from the original graph with non-overlapping neighboring nodes.
    """
    # Build adjacency list representation of the graph
    adjacency = defaultdict(set)
    for u, v in graph:
        adjacency[u].add(v)
        adjacency[v].add(u)

    # Function to get neighboring nodes of an edge
    def get_edge_neighbors(edge):
        u, v = edge
        return (adjacency[u] | adjacency[v]) - {u, v}

    remaining_edges = set(graph)  # Keep track of remaining edges
    iterations = []  # Store the edges chosen in each iteration

    while remaining_edges:
        current_iteration = set()  # Edges chosen in this iteration
        used_nodes = set()  # Nodes already used in this iteration

        for edge in list(remaining_edges):
            u, v = edge
            # Check if the edge is disconnected from already chosen edges
            if u in used_nodes or v in used_nodes:
                continue

            # Get neighboring nodes of this edge
            edge_neighbors = get_edge_neighbors(edge)

            # Check if any neighbor belongs to an edge already in this iteration
            if any(neighbor in used_nodes for neighbor in edge_neighbors):
                continue

            # Add the edge to the current iteration
            current_iteration.add(edge)
            used_nodes.update([u, v])

        # Add the chosen edges to the result
        iterations.append(list(current_iteration))
        remaining_edges -= current_iteration  # Remove chosen edges from the remaining edges

    return iterations


def generate_minimal_edge_layers(cp_map: CouplingMap) -> Dict[int, List[List[int]]]:
    """Sorts the edges of a coupling map, arranging them in a dictionary with values being subsets of the coupling map with no overlapping nodes.
    Each item will correspond to a layer of pairs of qubits in which parallel 2Q gates can be applied.

    Args:
        cp_map (CouplingMap): A list of lists of pairs of integers, representing a coupling map.
    Returns:
        Dict[int, List[List[int]]]: A dictionary with values being subsets of the coupling map with no overlapping nodes.
    """
    # Build a conflict graph - Treat the input list as a graph
    # where each sublist is a node, and an edge exists between nodes if they share any integers
    undirect_cp_map_list = remove_directed_duplicates_to_list(cp_map)

    n = len(undirect_cp_map_list)
    graph: Dict[int, Set] = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if set(undirect_cp_map_list[i]) & set(undirect_cp_map_list[j]):  # Check for shared integers
                graph[i].add(j)
                graph[j].add(i)

    # Reduce to a graph coloring problem;
    # each color represents a group in the dictionary
    colors: Dict[int, int] = {}
    for node in range(n):
        # Find all used colors among neighbors
        neighbor_colors = {colors[neighbor] for neighbor in graph[node] if neighbor in colors}
        # Assign the smallest unused color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color

    # Group by colors - minimize the number of groups
    groups: Dict[int, List[List[int]]] = {}
    for idx, color in colors.items():
        if color not in groups:
            groups[color] = []
        groups[color].append(undirect_cp_map_list[idx])

    return groups


def generate_graph_state(qubits: Sequence[int], backend: IQMBackendBase | str) -> QuantumCircuit:
    """Generates a circuit with minimal depth preparing a native graph state for a given backend using given qubits.

    Args:
        qubits (Sequence[int]): A list of integers representing the qubits.
        backend (IQMBackendBase): The backend to target the graph state generating circuit.
    Returns:
        QuantumCircuit: The circuit generating a graph state in the target backend.
    """
    num_qubits = len(qubits)
    qc = QuantumCircuit(num_qubits)
    coupling_map = set_coupling_map(qubits, backend, physical_layout="fixed")
    layers = generate_minimal_edge_layers(coupling_map)
    # Add all H
    for q in range(num_qubits):
        qc.r(np.pi / 2, np.pi / 2, q)
    # Add all CZ
    for layer in layers.values():
        for edge in layer:
            qc.cz(edge[0], edge[1])
    # Transpile
    qc_t = transpile(qc, backend=backend, initial_layout=qubits, optimization_level=3)
    return qc_t


def plot_density_matrix(
    matrix: np.ndarray,
    qubit_pair: Sequence[int],
    projection: str,
    negativity: Dict[str, float],
    backend_name: str,
    timestamp: str,
    tomography: Literal["state_tomography", "shadow_tomography"],
    num_RM_samples: Optional[int] = None,
    num_MoMs_samples: Optional[int] = None,
) -> Tuple[str, Figure]:
    """Plots a density matrix for corresponding qubit pairs, neighbor qubit projections, and negativities.

    Args:
        matrix (np.ndarray): The matrix to plot.
        qubit_pair (Sequence[int]): The corresponding qubit pair.
        projection (str): The projection corresponding to the matrix to plot.
        negativity (Dict[str, float]): A dictionary with keys "value" and "uncertainty" and values being respective negativities.
        backend_name (str): The name of the backend for the corresponding experiment.
        timestamp (str): The timestamp for the corresponding experiment.
        tomography (Literal["state_tomography", "shadow_tomography"]): The type of tomography used to gather the data of the matrix to plot.
        num_RM_samples (Optional[int] = None): The number of randomized measurement samples if tomography is shadow_tomography.
            * Default is None if tomography is state_tomography.
        num_MoMs_samples (Optional[int] = None): The number of Median of Means used per randomized measurement if tomography is shadow_tomography.
            * Default is None if tomography is state_tomography.
    Returns:
        Tuple[str, Figure]: The figure label and the density matrix plot figure.
    """

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 6))
    cmap = "winter_r"
    fig_name = str(qubit_pair)

    ax[0].matshow(matrix.real, interpolation="nearest", vmin=-np.max(matrix.real), vmax=np.max(matrix.real), cmap=cmap)
    ax[0].set_title(r"$\mathrm{Re}(\hat{\rho})$")
    for (i, j), z in np.ndenumerate(matrix.real):
        ax[0].text(
            j,
            i,
            f"{z:0.2f}",
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.3"},
        )

    im1 = ax[1].matshow(
        matrix.imag, interpolation="nearest", vmin=-np.max(matrix.real), vmax=np.max(matrix.real), cmap=cmap
    )
    ax[1].set_title(r"$\mathrm{Im}(\hat{\rho})$")
    for (i, j), z in np.ndenumerate(matrix.imag):
        ax[1].text(
            j,
            i,
            f"{z:0.2f}",
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.3"},
        )

    if tomography == "shadow_tomography":
        fig.suptitle(
            f"Average shadow for qubits {qubit_pair} ({num_RM_samples} local RM samples x {num_MoMs_samples} Median of Means samples)\n"
            f"Projection: {projection}\nNegativity: {negativity['value']:.4f} +/- {negativity['uncertainty']:.4f}\n"
            f"{backend_name} --- {timestamp}"
        )
    else:
        fig.suptitle(
            f"Tomographically reconstructed density matrix for qubits {qubit_pair}\n"
            f"Projection: {projection}\nNegativity: {negativity['value']:.4f} +/- {negativity['uncertainty']:.4f}\n"
            f"{backend_name} --- {timestamp}"
        )
    fig.colorbar(im1, shrink=0.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1.25))

    plt.close()

    return fig_name, fig


def plot_max_negativities(
    negativities: Dict[str, Dict[str, str | float]],
    backend_name: str,
    qubit_names: Dict[int, str],
    timestamp: str,
    tomography: Literal["shadow_tomography", "state_tomography"],
    num_shots: int,
    num_bootstraps: Optional[int] = None,
    num_RM_samples: Optional[int] = None,
    num_MoMs_samples: Optional[int] = None,
) -> Tuple[str, Figure]:
    """Plots the maximum negativity for each corresponding pair of qubits.

    Args:
        negativities (Dict[str, Dict[str, str | float]]): A dictionary (str qubit keys) of dictionaries (keys "value"/"uncertainty") of negativities (float) to plot.
        backend_name (str): The name of the backend for the corresponding experiment.
        qubit_names (Dict[int, str]): A dictionary of qubit names corresponding to qubit indices.
        timestamp (str): The timestamp of the corresponding experiment.
        tomography (Literal["shadow_tomography", "state_tomography"]): The type of tomography that was used.
        num_shots (int): The number of shots used in the corresponding experiment.
        num_bootstraps (Optional[int]): The number of bootstraps used if tomography corresponds to state tomography.
            * Defaults to None if the tomography type is "shadow_tomography".
        num_RM_samples (Optional[int]): The number of randomized measurement samples used if tomography corresponds to shadow tomography.
            * Defaults to None if the tomography type is "state_tomography".
        num_MoMs_samples (Optional[int]): The number of Median of Means samples per randomized measurement used if tomography corresponds to shadow tomography.
            * Defaults to None if the tomography type is "shadow_tomography".

    Returns:
        Tuple[str, Figure]: The figure label and the max negativities plot figure.
    """
    fig_name = f"max_negativities_{backend_name}_{timestamp}".replace(" ", "_")
    # Sort the negativities by value
    sorted_negativities = dict(sorted(negativities.items(), key=lambda item: item[1]["value"]))

    x = [x.replace("(", "").replace(")", "").replace(", ", "-") for x in list(sorted_negativities.keys())]
    x_updated = [
        f"{cast(str, qubit_names[int(a)])[2:]}-{cast(str, qubit_names[int(b)])[2:]}"
        for edge in x
        for a, b in [edge.split("-")]
    ]  ## reindexes the edges label as in the QPU graph.

    y = [a["value"] for a in sorted_negativities.values()]
    yerr = [a["uncertainty"] for a in sorted_negativities.values()]

    cmap = plt.get_cmap("winter")

    fig = plt.figure()
    ax = plt.axes()

    if tomography == "shadow_tomography":
        errorbar_labels = rf"$1 \mathrm{{SEM}}$ (N={cast(int, num_RM_samples)*cast(int,num_MoMs_samples)} RMs)"
    else:
        errorbar_labels = rf"$1 \sigma$ ({cast(int, num_bootstraps)} bootstraps)"

    plt.errorbar(
        x_updated,
        y,
        yerr=yerr,
        capsize=2,
        color=cmap(0.15),
        fmt="o",
        alpha=1,
        mec="black",
        markersize=3,
        label=errorbar_labels,
    )
    plt.axhline(0.5, color=cmap(1.0), linestyle="dashed")

    ax.set_xlabel("Qubit pair")
    ax.set_ylabel("Negativity")

    # Major y-ticks every 0.1, minor ticks every 0.05
    major_ticks = np.arange(0, 0.5, 0.1)
    minor_ticks = np.arange(-0.05, 0.55, 0.05)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which="both")

    lower_y = np.min(y) - 1.75 * float(yerr[0]) - 0.02 if np.min(y) - float(yerr[0]) < 0 else -0.01
    upper_y = np.max(y) + 1.75 * float(yerr[-1]) + 0.02 if np.max(y) + float(yerr[-1]) > 0.5 else 0.51
    ax.set_ylim(
        (
            lower_y,
            upper_y,
        )
    )

    plt.xticks(rotation=90)
    if tomography == "shadow_tomography":
        plt.title(
            f"Max entanglement negativities for qubit pairs in {backend_name}\n{num_RM_samples} local RM samples x {num_MoMs_samples} Median of Means samples\n{timestamp}"
        )
    else:
        plt.title(
            f"Max entanglement negativities for qubit pairs in {backend_name}\nShots per tomography sample: {num_shots}; Bootstraps: {num_bootstraps}\n{timestamp}"
        )
    plt.legend(fontsize=8)

    ax.margins(tight=True)

    if len(x) <= 40:
        ax.set_aspect((2 / 3) * len(x))
        ax.autoscale(enable=True, axis="x")
    else:
        ####################################################################################
        # Solution to fix tick spacings taken from:
        # https://stackoverflow.com/questions/44863375/how-to-change-spacing-between-ticks
        plt.gca().margins(x=0.01)
        plt.gcf().canvas.draw()
        tl = plt.gca().get_xticklabels()
        maxsize = max(t.get_window_extent().width for t in tl)
        m = 0.2  # inch margin
        s = maxsize / plt.gcf().dpi * len(x) + 2 * m
        margin = m / plt.gcf().get_size_inches()[0]
        plt.gcf().subplots_adjust(left=margin, right=1.0 - margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        #####################################################################################`

    plt.close()

    return fig_name, fig


def plot_max_negativities_graph(
    negativities: Dict[str, Dict[str, str | float]],
    backend_coupling_map: CouplingMap,
    qubit_names: Dict[int, str],
    timestamp: str,
    tomography: Literal["shadow_tomography", "state_tomography"],
    station: Optional[str] = None,
    num_shots: Optional[int] = None,
    num_bootstraps: Optional[int] = None,
    num_RM_samples: Optional[int] = None,
    num_MoMs_samples: Optional[int] = None,
) -> Tuple[str, Figure]:
    """Plots the maximum negativity for each corresponding pair of qubits in a graph layout of the given backend.

    Args:
        negativities (Dict[str, Dict[str, str | float]]): A dictionary (str qubit keys) of dictionaries (keys "value"/"uncertainty") of negativities (float) to plot.
        backend_coupling_map (CouplingMap): The CouplingMap instance.
        qubit_names (Dict[int, str]): A dictionary of qubit names corresponding to qubit indices.
        timestamp (str): The timestamp of the corresponding experiment.
        tomography (Literal["shadow_tomography", "state_tomography"]): The type of tomography that was used.
        station (str): The name of the station to use for the graph layout.
        num_shots (Optional[int]): The number of shots used in the corresponding experiment.
            * Defaults to None: won't be displayed in title.
        num_bootstraps (Optional[int]): The number of bootstraps used if tomography corresponds to state tomography.
            * Defaults to None if the tomography type is "shadow_tomography".
        num_RM_samples (Optional[int]): The number of randomized measurement samples used if tomography corresponds to shadow tomography.
            * Defaults to None if the tomography type is "state_tomography".
        num_MoMs_samples (Optional[int]): The number of Median of Means samples per randomized measurement used if tomography corresponds to shadow tomography.
            * Defaults to None if the tomography type is "shadow_tomography".

    Returns:
        Tuple[str, Figure]: The figure label and the max negativities plot figure.
    """
    num_qubits = len(qubit_names.keys())
    fig_name = (
        f"max_negativities_graph_{station}_{timestamp}"
        if station is not None
        else f"max_negativities_graph_{timestamp}"
    )
    # Sort the negativities by value
    sorted_negativities = dict(sorted(negativities.items(), key=lambda item: item[1]["value"]))

    qubit_pairs = [
        tuple(int(num) for num in x.replace("(", "").replace(")", "").replace("...", "").split(", "))
        for x in sorted_negativities.keys()
    ]
    negativity_values = [a["value"] for a in sorted_negativities.values()]

    negativity_edges = dict(zip(qubit_pairs, negativity_values))

    cmap = plt.cm.get_cmap("winter")

    fig = plt.figure()
    ax = plt.axes()

    if station is not None:
        if station.lower() in GraphPositions.predefined_stations:
            qubit_positions = GraphPositions.predefined_stations[station.lower()]
        else:
            graph_backend = backend_coupling_map.graph.to_undirected(multigraph=False)
            qubit_positions = GraphPositions.create_positions(graph_backend)
    else:
        graph_backend = backend_coupling_map.graph.to_undirected(multigraph=False)
        if num_qubits in (20, 7):
            station = "garnet" if num_qubits == 20 else "deneb"
            qubit_positions = GraphPositions.predefined_stations[station]
        else:
            qubit_positions = GraphPositions.create_positions(graph_backend)

    # Normalize negativity values to the range [0, 1] for color mapping
    norm = plt.Normalize(vmin=cast(float, min(negativity_values)), vmax=cast(float, max(negativity_values)))
    edge_colors = [cmap(norm(negativity_edges[edge])) for edge in qubit_pairs]

    nx.draw_networkx(
        rx_to_nx_graph(backend_coupling_map),
        pos=qubit_positions,
        nodelist=list(range(num_qubits)),
        labels={x: qubit_names[x] for x in range(num_qubits)},
        font_size=6.5,
        edgelist=qubit_pairs,
        width=4.0,
        edge_color=edge_colors,
        node_color="k",
        font_color="w",
        ax=ax,
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5)

    shots_string = "" if num_shots is None else f"Shots per tomography sample: {num_shots}"
    station_string = "IQM Backend" if station is None else station.capitalize()
    if tomography == "shadow_tomography":
        plt.title(
            f"Max entanglement negativities for qubit pairs in {station_string}\n"
            f"{num_RM_samples} local RM samples x {num_MoMs_samples} Median of Means samples\n"
            f"{shots_string}; {timestamp}"
        )
    else:
        plt.title(
            f"Max entanglement negativities for qubit pairs in {station_string}\n"
            f"{shots_string}; Bootstraps: {num_bootstraps}"
            f"\n{timestamp}"
        )
    plt.close()

    return fig_name, fig


def update_pauli_expectations(
    pauli_expectations: Dict[str, Dict[str, float]],
    projected_counts: Dict[str, Dict[str, int]],
    nonId_pauli_label: str,
) -> Dict[str, Dict[str, float]]:
    """Helper function that updates the input Pauli expectations dictionary of dictionaries (projections -> {pauli string: expectation}).

    Args:
        pauli_expectations (Dict[str, Dict[str, float]]): The Pauli expectations dictionary of dictionaries to update.
            * Outermost keys are projected bitstrings; innermost are pauli strings and values are expectation values.
        projected_counts (Dict[str, Dict[str, int]]): The corresponding projected counts dictionary of dictionaries.
        nonId_pauli_label (str): The Pauli label to update expectations of, that should not contain identities.
            * Pauli expectations corresponding to I are inferred and updated from counts corresponding to strings containing Z instead.

    Returns:
        Dict[str, Dict[str, float]]: The updated Pauli expectations dictionary of dictionaries (projections -> {pauli string: expectation}).
    """
    # Get the individual Pauli expectations for each projection
    for projected_bit_string in projected_counts.keys():
        # Ideally the counts should be labeled by Pauli basis measurement!
        # Here by construction they should be ordered as all_pauli_labels,
        # however, this assumed that measurements never got scrambled (which should not happen anyway).
        counts = projected_counts[projected_bit_string]
        get_exp = lambda label, ct=counts: get_Pauli_expectation(ct, label)
        updates = {nonId_pauli_label: get_exp(nonId_pauli_label)}

        if nonId_pauli_label == "ZZ":
            updates.update({x: get_exp(x) for x in ["ZI", "IZ", "II"]})
        if nonId_pauli_label[0] == "Z":
            updates[f"I{nonId_pauli_label[1]}"] = get_exp(f"I{nonId_pauli_label[1]}")
        if nonId_pauli_label[1] == "Z":
            updates[f"{nonId_pauli_label[0]}I"] = get_exp(f"{nonId_pauli_label[0]}I")

        pauli_expectations[projected_bit_string].update(updates)

    return pauli_expectations


def shadow_tomography_analysis(
    dataset: xr.Dataset,
    all_qubit_pairs_per_group: Dict[int, List[Tuple[int, int]]],
    all_qubit_neighbors_per_group: Dict[int, List[List[int]]],
    all_unprojected_qubits: Dict[int, List[int]],
    backend_name: str,
    execution_timestamp: str,
) -> Tuple[Dict[str, Any], List[BenchmarkObservation], Dict[str, Dict[str, str | float]], xr.Dataset]:
    """
    Performs shadow tomography analysis on the given dataset.

    Args:
        dataset (xr.Dataset): The dataset containing the experimental data.
        all_qubit_pairs_per_group (Dict[int, List[Tuple[int, int]]]): Dictionary mapping group indices to lists of qubit pairs.
        all_qubit_neighbors_per_group (Dict[int, List[List[int]]]): Dictionary mapping group indices to lists of neighbor qubit groups.
        all_unprojected_qubits (Dict[int, List[int]]): Dictionary mapping group indices to lists of unprojected qubits.
        backend_name (str): The name of the backend used for the experiment.
        execution_timestamp (str): The timestamp of the experiment execution.

    Returns:
        Tuple[Dict[str, Any], List[BenchmarkObservation], Dict[str, Dict[str, str | float]], xr.Dataset]:
        A tuple containing:
            - A dictionary of plots.
            - A list of benchmark observations.
            - A dictionary of maximum negativities.
            - The updated dataset.
    """
    plots: Dict[str, Any] = {}
    observations: List[BenchmarkObservation] = []
    max_negativities: Dict[str, Dict[str, str | float]] = {}

    execution_results = {}

    num_RMs = dataset.attrs["n_random_unitaries"]
    num_MoMs = dataset.attrs["n_median_of_means"]

    qcvv_logger.info("Fetching Clifford dictionary")
    clifford_1q_dict = import_native_gate_cliffords("1q")
    all_unitaries = dataset.attrs["all_unitaries"]

    shadows_per_projection: Dict[str, Dict[int, Dict[str, List[np.ndarray]]]] = {}
    # shadows_per_projection: qubit_pair -> MoMs -> {Projection, List of shadows}
    MoMs_shadows: Dict[str, Dict[str, np.ndarray]] = {}
    # MoMs_shadows: qubit_pair -> {Projection: MoMs shadow}
    average_shadows_per_projection: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    # average_shadows_per_projection: qubit_pair -> MoMs -> {Projection: shadows}
    all_negativities: Dict[str, Dict[int, Dict[str, float]]] = {}
    # all_negativities: qubit_pair -> MoMs -> {Projection: Negativity}
    MoMs_negativities: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_idx, group in all_qubit_pairs_per_group.items():
        qcvv_logger.info(f"Retrieving shadows for qubit-pair group {group_idx+1}/{len(all_qubit_pairs_per_group)}")
        # Assume only pairs and nearest-neighbors were measured, and each pair in the group uses num_RMs randomized measurements:
        execution_results[group_idx] = xrvariable_to_counts(
            dataset, str(all_unprojected_qubits[group_idx]), num_RMs * num_MoMs * len(group)
        )

        partitioned_counts_MoMs_RMs = split_sequence_in_chunks(execution_results[group_idx], num_RMs * num_MoMs)
        partitioned_counts_RMs = {}

        for pair_idx, qubit_pair in enumerate(group):
            all_negativities[str(qubit_pair)] = {}
            MoMs_negativities[str(qubit_pair)] = {}
            shadows_per_projection[str(qubit_pair)] = {}
            average_shadows_per_projection[str(qubit_pair)] = {}

            partitioned_counts_RMs[pair_idx] = split_sequence_in_chunks(partitioned_counts_MoMs_RMs[pair_idx], num_RMs)

            # Get the neighbor qubits of qubit_pair
            neighbor_qubits = all_qubit_neighbors_per_group[group_idx][pair_idx]
            neighbor_bit_strings_length = len(neighbor_qubits)
            # Generate all possible projection bitstrings for the neighbors, {'0','1'}^{\otimes{N}}
            all_projection_bit_strings = [
                "".join(x) for x in itertools.product(("0", "1"), repeat=neighbor_bit_strings_length)
            ]

            for MoMs in range(num_MoMs):
                qcvv_logger.info(
                    f"Now on qubit pair {qubit_pair} ({pair_idx+1}/{len(group)}) and median of means sample {MoMs+1}/{num_MoMs}"
                )

                # Get all shadows of qubit_pair
                shadows_per_projection[str(qubit_pair)][MoMs] = {
                    projection: [] for projection in all_projection_bit_strings
                }
                for RM_idx, counts in enumerate(partitioned_counts_RMs[pair_idx][MoMs]):
                    # Retrieve both Cliffords (i.e. for each qubit)
                    cliffords_rm = [all_unitaries[group_idx][MoMs][str(q)][RM_idx] for q in qubit_pair]
                    # Organize counts by projection
                    # e.g. counts ~ {'000 00': 31, '000 01': 31, '000 10': 38, '000 11': 41, '001 00': 28, '001 01': 33,
                    #                   '001 10': 31, '001 11': 37, '010 00': 29, '010 01': 32, '010 10': 31, '010 11': 25,
                    #                   '011 00': 36, '011 01': 24, '011 10': 33, '011 11': 32, '100 00': 22, '100 01': 38,
                    #                   '100 10': 34, '100 11': 26, '101 00': 26, '101 01': 26, '101 10': 37, '101 11': 30,
                    #                   '110 00': 36, '110 01': 35, '110 10': 31, '110 11': 35, '111 00': 31, '111 01': 32,
                    #                   '111 10': 37, '111 11': 36}
                    # organize to projected_counts['000'] ~ {'00': 31, '01': 31, '10': 38, '11': 41},
                    #             projected_counts['001'] ~ {'00': 28, '01': 33, '10': 31, '11': 37}
                    #             ...
                    projected_counts = {
                        projection: {
                            b_s[-2:]: b_c
                            for b_s, b_c in counts.items()
                            if b_s[:neighbor_bit_strings_length] == projection
                        }
                        for projection in all_projection_bit_strings
                    }

                    # Get the individual shadow for each projection
                    for projected_bit_string in all_projection_bit_strings:
                        shadows_per_projection[str(qubit_pair)][MoMs][projected_bit_string].append(
                            get_local_shadow(
                                counts=projected_counts[projected_bit_string],
                                unitary_arg=cliffords_rm,
                                subsystem_bit_indices=list(range(2)),
                                clifford_or_haar="clifford",
                                cliffords_1q=clifford_1q_dict,
                            )
                        )

                # Average the shadows for each projection and MoMs sample
                average_shadows_per_projection[str(qubit_pair)][MoMs] = {
                    projected_bit_string: np.mean(
                        shadows_per_projection[str(qubit_pair)][MoMs][projected_bit_string], axis=0
                    )
                    for projected_bit_string in all_projection_bit_strings
                }

                # Compute the negativity of the shadow of each projection
                qcvv_logger.info(
                    f"Computing the negativity of all shadow projections for qubit pair {qubit_pair} ({pair_idx+1}/{len(group)}) and median of means sample {MoMs+1}/{num_MoMs}"
                )
                all_negativities[str(qubit_pair)][MoMs] = {
                    projected_bit_string: get_negativity(
                        average_shadows_per_projection[str(qubit_pair)][MoMs][projected_bit_string], 1, 1
                    )
                    for projected_bit_string in all_projection_bit_strings
                }

            MoMs_negativities[str(qubit_pair)] = {
                projected_bit_string: median_with_uncertainty(
                    [all_negativities[str(qubit_pair)][m][projected_bit_string] for m in range(num_MoMs)]
                )
                for projected_bit_string in all_projection_bit_strings
            }

            MoMs_shadows[str(qubit_pair)] = {
                projected_bit_string: np.median(
                    [average_shadows_per_projection[str(qubit_pair)][m][projected_bit_string] for m in range(num_MoMs)],
                    axis=0,
                )
                for projected_bit_string in all_projection_bit_strings
            }

            all_negativities_list = [
                MoMs_negativities[str(qubit_pair)][projected_bit_string]["value"]
                for projected_bit_string in all_projection_bit_strings
            ]
            all_negativities_uncertainty = [
                MoMs_negativities[str(qubit_pair)][projected_bit_string]["uncertainty"]
                for projected_bit_string in all_projection_bit_strings
            ]

            max_negativity_projection = np.argmax(all_negativities_list)

            max_negativity = {
                "value": all_negativities_list[max_negativity_projection],
                "uncertainty": all_negativities_uncertainty[max_negativity_projection],
            }

            max_negativities[str(qubit_pair)] = {}  # {str(qubit_pair): {"negativity": float, "projection": str}}
            max_negativities[str(qubit_pair)].update(
                {
                    "projection": all_projection_bit_strings[max_negativity_projection],
                }
            )
            max_negativities[str(qubit_pair)].update(max_negativity)

            fig_name, fig = plot_density_matrix(
                matrix=MoMs_shadows[str(qubit_pair)][all_projection_bit_strings[max_negativity_projection]],
                qubit_pair=qubit_pair,
                projection=all_projection_bit_strings[max_negativity_projection],
                negativity=max_negativity,
                backend_name=backend_name,
                timestamp=execution_timestamp,
                tomography="shadow_tomography",
                num_RM_samples=num_RMs,
                num_MoMs_samples=num_MoMs,
            )
            plots[fig_name] = fig

            observations.extend(
                [
                    BenchmarkObservation(
                        name="max_negativity",
                        value=max_negativity["value"],
                        uncertainty=max_negativity["uncertainty"],
                        identifier=BenchmarkObservationIdentifier(qubit_pair),
                    )
                ]
            )

    dataset.attrs.update(
        {
            "median_of_means_shadows": MoMs_shadows,
            "median_of_means_negativities": MoMs_negativities,
            "all_negativities": all_negativities,
            "all_shadows": shadows_per_projection,
        }
    )

    return plots, observations, max_negativities, dataset


def state_tomography_analysis(
    dataset: xr.Dataset,
    all_qubit_pairs_per_group: Dict[int, List[Tuple[int, int]]],
    all_qubit_neighbors_per_group: Dict[int, List[List[int]]],
    all_unprojected_qubits: Dict[int, List[int]],
    backend_name: str,
    execution_timestamp: str,
) -> Tuple[Dict[str, Any], List[BenchmarkObservation], Dict[str, Dict[str, str | float]], xr.Dataset]:
    """
    Performs state tomography analysis on the given dataset.

    Args:
        dataset (xr.Dataset): The dataset containing the experimental data.
        all_qubit_pairs_per_group (Dict[int, List[Tuple[int, int]]]): Dictionary mapping group indices to lists of qubit pairs.
        all_qubit_neighbors_per_group (Dict[int, List[List[int]]]): Dictionary mapping group indices to lists of neighbor qubit groups.
        all_unprojected_qubits (Dict[int, List[int]]): Dictionary mapping group indices to lists of unprojected qubits.
        backend_name (str): The name of the backend used for the experiment.
        execution_timestamp (str): The timestamp of the experiment execution.

    Returns:
        Tuple[Dict[str, Any], List[BenchmarkObservation], Dict[str, Dict[str, float]], xr.Dataset]:
        A tuple containing:
            - A dictionary of plots.
            - A list of benchmark observations.
            - A dictionary of maximum negativities.
            - The updated dataset.
    """
    plots: Dict[str, Any] = {}
    observations: List[BenchmarkObservation] = []
    max_negativities: Dict[str, Dict[str, str | float]] = {}

    execution_results = {}

    num_bootstraps = dataset.attrs["num_bootstraps"]

    tomography_state: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    # tomography_state: group_idx -> qubit_pair -> {projection:numpy array}
    bootstrapped_states: Dict[int, Dict[str, List[np.ndarray]]] = {}
    # bootstrapped_states: group_idx -> qubit_pair -> List of bootstrapped states for max_neg_projection
    tomography_negativities: Dict[int, Dict[str, Dict[str, float]]] = {}
    bootstrapped_negativities: Dict[int, Dict[str, List[float]]] = {}
    bootstrapped_avg_negativities: Dict[int, Dict[str, Dict[str, float]]] = {}
    num_tomo_samples = (
        3**2
    )  # In general 3**n samples suffice (assuming trace-preservation and unitality for the Pauli measurements)
    for group_idx, group in all_qubit_pairs_per_group.items():
        qcvv_logger.info(
            f"Retrieving tomography-reconstructed states with {num_bootstraps} for qubit-pair group {group_idx+1}/{len(all_qubit_pairs_per_group)}"
        )

        # Assume only pairs and nearest-neighbors were measured, and each pair in the group used num_RMs randomized measurements:
        execution_results[group_idx] = xrvariable_to_counts(
            dataset, str(all_unprojected_qubits[group_idx]), num_tomo_samples * len(group)
        )

        tomography_state[group_idx] = {}
        bootstrapped_states[group_idx] = {}
        tomography_negativities[group_idx] = {}
        bootstrapped_negativities[group_idx] = {}
        bootstrapped_avg_negativities[group_idx] = {}

        partitioned_counts = split_sequence_in_chunks(execution_results[group_idx], num_tomo_samples)

        for pair_idx, qubit_pair in enumerate(group):
            # Get the neighbor qubits of qubit_pair
            neighbor_qubits = all_qubit_neighbors_per_group[group_idx][pair_idx]
            neighbor_bit_strings_length = len(neighbor_qubits)
            # Generate all possible projection bitstrings for the neighbors, {'0','1'}^{\otimes{N}}
            all_projection_bit_strings = [
                "".join(x) for x in itertools.product(("0", "1"), repeat=neighbor_bit_strings_length)
            ]

            sqg_pauli_strings = ("Z", "X", "Y")
            all_nonId_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=2)]

            pauli_expectations: Dict[str, Dict[str, float]] = {
                projection: {} for projection in all_projection_bit_strings
            }
            # pauli_expectations: projected_bit_string -> pauli string -> float expectation
            for pauli_idx, counts in enumerate(partitioned_counts[pair_idx]):
                projected_counts = {
                    projection: {
                        b_s[-2:]: b_c for b_s, b_c in counts.items() if b_s[:neighbor_bit_strings_length] == projection
                    }
                    for projection in all_projection_bit_strings
                    if projection in [c[:neighbor_bit_strings_length] for c in counts.keys()]
                }

                pauli_expectations = update_pauli_expectations(
                    pauli_expectations,
                    projected_counts,
                    nonId_pauli_label=all_nonId_pauli_labels[pauli_idx],
                )

            # Remove projections with empty values for pauli_expectations
            # This will happen if certain projection bitstrings were just not measured
            pauli_expectations = {
                projection: expectations for projection, expectations in pauli_expectations.items() if expectations
            }

            tomography_state[group_idx][str(qubit_pair)] = {
                projection: get_tomography_matrix(pauli_expectations=pauli_expectations[projection])
                for projection in pauli_expectations.keys()
            }

            tomography_negativities[group_idx][str(qubit_pair)] = {
                projected_bit_string: get_negativity(
                    tomography_state[group_idx][str(qubit_pair)][projected_bit_string], 1, 1
                )
                for projected_bit_string in pauli_expectations.keys()
            }

            # Extract the max negativity and the corresponding projection - save in dictionary
            all_negativities_list = [
                tomography_negativities[group_idx][str(qubit_pair)][projected_bit_string]
                for projected_bit_string in pauli_expectations.keys()
            ]

            max_negativity_projection_idx = np.argmax(all_negativities_list)
            max_negativity_bitstring = list(pauli_expectations.keys())[max_negativity_projection_idx]

            # Bootstrapping - do only for max projection bitstring
            bootstrapped_pauli_expectations: List[Dict[str, Dict[str, float]]] = [
                {max_negativity_bitstring: {}} for _ in range(num_bootstraps)
            ]
            for pauli_idx, counts in enumerate(partitioned_counts[pair_idx]):
                projected_counts = {
                    b_s[-2:]: b_c
                    for b_s, b_c in counts.items()
                    if b_s[:neighbor_bit_strings_length] == max_negativity_bitstring
                }
                all_bootstrapped_counts = bootstrap_counts(
                    projected_counts, num_bootstraps, include_original_counts=True
                )
                for bootstrap in range(num_bootstraps):
                    bootstrapped_pauli_expectations[bootstrap] = update_pauli_expectations(
                        bootstrapped_pauli_expectations[bootstrap],
                        projected_counts={max_negativity_bitstring: all_bootstrapped_counts[bootstrap]},
                        nonId_pauli_label=all_nonId_pauli_labels[pauli_idx],
                    )

            bootstrapped_states[group_idx][str(qubit_pair)] = [
                get_tomography_matrix(
                    pauli_expectations=bootstrapped_pauli_expectations[bootstrap][max_negativity_bitstring]
                )
                for bootstrap in range(num_bootstraps)
            ]

            bootstrapped_negativities[group_idx][str(qubit_pair)] = [
                get_negativity(bootstrapped_states[group_idx][str(qubit_pair)][bootstrap], 1, 1)
                for bootstrap in range(num_bootstraps)
            ]

            bootstrapped_avg_negativities[group_idx][str(qubit_pair)] = {
                "value": float(np.mean(bootstrapped_negativities[group_idx][str(qubit_pair)])),
                "uncertainty": float(np.std(bootstrapped_negativities[group_idx][str(qubit_pair)])),
            }

            max_negativity = {
                "value": all_negativities_list[max_negativity_projection_idx],
                "bootstrapped_average": bootstrapped_avg_negativities[group_idx][str(qubit_pair)]["value"],
                "uncertainty": bootstrapped_avg_negativities[group_idx][str(qubit_pair)]["uncertainty"],
            }

            max_negativities[str(qubit_pair)] = {}  # {str(qubit_pair): {"negativity": float, "projection": str}}
            max_negativities[str(qubit_pair)].update(
                {
                    "projection": max_negativity_bitstring,
                }
            )
            max_negativities[str(qubit_pair)].update(max_negativity)

            fig_name, fig = plot_density_matrix(
                matrix=tomography_state[group_idx][str(qubit_pair)][max_negativity_bitstring],
                qubit_pair=qubit_pair,
                projection=max_negativity_bitstring,
                negativity=max_negativity,
                backend_name=backend_name,
                timestamp=execution_timestamp,
                tomography="state_tomography",
            )
            plots[fig_name] = fig

            observations.extend(
                [
                    BenchmarkObservation(
                        name="max_negativity",
                        value=max_negativity["value"],
                        uncertainty=max_negativity["uncertainty"],
                        identifier=BenchmarkObservationIdentifier(qubit_pair),
                    )
                ]
            )

        dataset.attrs.update(
            {
                "all_tomography_states": tomography_state,
                "all_negativities": tomography_negativities,
            }
        )

    return plots, observations, max_negativities, dataset


def negativity_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a Graph State benchmark experiment."""
    qcvv_logger.info("Fetching dataset")
    dataset = run.dataset.copy(deep=True)
    qcvv_logger.info("Dataset imported OK")
    backend_name = dataset.attrs["backend_name"]
    coupling_map = dataset.attrs["coupling_map"]
    qubit_names = dataset.attrs["qubit_names"]
    execution_timestamp = dataset.attrs["execution_timestamp"]
    tomography = dataset.attrs["tomography"]
    num_bootstraps = dataset.attrs["num_bootstraps"]
    num_RMs = dataset.attrs["n_random_unitaries"]
    num_MoMs = dataset.attrs["n_median_of_means"]
    num_shots = dataset.attrs["shots"]

    all_qubit_pairs_per_group = dataset.attrs["all_pair_groups"]
    all_qubit_neighbors_per_group = dataset.attrs["all_neighbor_groups"]
    all_unprojected_qubits = dataset.attrs["all_unprojected_qubits"]

    if tomography == "shadow_tomography":
        plots, observations, max_negativities, dataset = shadow_tomography_analysis(
            dataset,
            all_qubit_pairs_per_group,
            all_qubit_neighbors_per_group,
            all_unprojected_qubits,
            backend_name,
            execution_timestamp,
        )
    else:
        plots, observations, max_negativities, dataset = state_tomography_analysis(
            dataset,
            all_qubit_pairs_per_group,
            all_qubit_neighbors_per_group,
            all_unprojected_qubits,
            backend_name,
            execution_timestamp,
        )

    dataset.attrs.update({"max_negativities": max_negativities})

    fig_name, fig = plot_max_negativities(
        negativities=max_negativities,
        backend_name=backend_name,
        qubit_names=qubit_names,
        timestamp=execution_timestamp,
        tomography=tomography,
        num_shots=num_shots,
        num_bootstraps=num_bootstraps,
        num_RM_samples=num_RMs,
        num_MoMs_samples=num_MoMs,
    )
    plots[fig_name] = fig

    fig_name, fig = plot_max_negativities_graph(
        negativities=max_negativities,
        backend_coupling_map=coupling_map,
        qubit_names=qubit_names,
        timestamp=execution_timestamp,
        tomography=tomography,
        num_shots=num_shots,
        num_bootstraps=num_bootstraps,
        num_RM_samples=num_RMs,
        num_MoMs_samples=num_MoMs,
    )
    plots[fig_name] = fig

    qcvv_logger.info("Analysis of Graph State Benchmark experiment concluded!")

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


class GraphStateBenchmark(Benchmark):
    """The Graph States benchmark estimates the bipartite entangelement negativity of native graph states."""

    analysis_function = staticmethod(negativity_analysis)
    name = "graph_states"

    def __init__(self, backend_arg: IQMBackendBase, configuration: "GraphStateConfiguration"):
        """Construct the GraphStateBenchmark class.

        Args:
            backend_arg (IQMBackendBase): the backend to execute the benchmark on
            configuration (GraphStateConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.qubits = configuration.qubits
        self.tomography = configuration.tomography

        self.num_bootstraps = configuration.num_bootstraps
        self.n_random_unitaries = configuration.n_random_unitaries
        self.n_median_of_means = configuration.n_median_of_means

        # Initialize relevant variables for the benchmark
        self.graph_state_circuit = generate_graph_state(self.qubits, self.backend)
        self.coupling_map = set_coupling_map(self.qubits, self.backend, physical_layout="fixed")

        # Initialize the variable to contain the benchmark circuits of each layout
        self.circuits = Circuits()
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

    def add_all_meta_to_dataset(self, dataset: xr.Dataset):
        """Adds all configuration metadata and circuits to the dataset variable

        Args:
            dataset (xr.Dataset): The xarray dataset
        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend_configuration_name"] = self.backend_configuration_name
        dataset.attrs["backend_name"] = self.backend.name
        dataset.attrs["qubit_names"] = {qubit: self.backend.index_to_qubit_name(qubit) for qubit in self.qubits}
        dataset.attrs["coupling_map"] = self.coupling_map

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        # Defined outside configuration - if any

    @timeit
    def add_all_circuits_to_dataset(self, dataset: xr.Dataset):
        """Adds all generated circuits during execution to the dataset variable

        Args:
            dataset (xr.Dataset):  The xarray dataset

        Returns:

        """
        qcvv_logger.info(f"Adding all circuits to the dataset")
        for key, circuit in zip(
            ["transpiled_circuits", "untranspiled_circuits"], [self.transpiled_circuits, self.untranspiled_circuits]
        ):
            dictionary = {}
            for outer_key, outer_value in circuit.items():
                dictionary[str(outer_key)] = {
                    str(inner_key): inner_values for inner_key, inner_values in outer_value.items()
                }
            dataset.attrs[key] = dictionary

    @timeit
    def generate_all_circuit_info_for_graph_state_benchmark(self) -> Dict[str, Any]:
        """
        Generates all circuits and associated information for the Graph State benchmark:
            - Generates native graph states
            - Identifies all pairs of qubits with disjoint neighbors
            - Generates all projected nodes to cover all pairs of qubits with disjoint neighbors

        Returns:
            Dict[str, Any]: A dictionary containing all circuit information for the Graph State benchmark.

        """
        layout_mapping = {
            a._index: b  # pylint: disable=W0212
            for a, b in self.graph_state_circuit.layout.initial_layout.get_virtual_bits().items()
            if b in self.qubits
        }

        # Get unique list of edges - Use layout_mapping to determine the connections between phyical qubits
        graph_edges = [
            (layout_mapping[e[0]], layout_mapping[e[1]])
            for e in list(self.coupling_map.graph.to_undirected(multigraph=False).edge_list())
        ]

        # Find pairs of nodes with disjoint neighbors
        # {idx: [(q1,q2), (q3,q4), ...]}
        pair_groups = find_edges_with_disjoint_neighbors(graph_edges)
        # {idx: [(n11,n12,n13,...), (n21,n22,n23,...), ...]}
        neighbor_groups = {
            idx: [get_neighbors_of_edges([y], graph_edges) for y in x] for idx, x in enumerate(pair_groups)
        }

        # Get all projected nodes to cover all pairs of qubits with disjoint neighbours
        # {idx: [q1,q2,q3,q4, ...]}
        unmeasured_qubit_indices = {idx: [a for b in x for a in b] for idx, x in enumerate(pair_groups)}
        # {idx: [n11,n12,n13,...,n21,n22,n23, ...]}
        projected_nodes = {idx: get_neighbors_of_edges(list(x), graph_edges) for idx, x in enumerate(pair_groups)}

        # Generate copies of circuits to add projections and randomized measurements
        grouped_graph_circuits = {idx: self.graph_state_circuit.copy() for idx in projected_nodes.keys()}

        return {
            "grouped_graph_circuits": grouped_graph_circuits,
            "unmeasured_qubit_indices": unmeasured_qubit_indices,
            "projected_nodes": projected_nodes,
            "pair_groups": dict(enumerate(pair_groups)),
            "neighbor_groups": neighbor_groups,
        }

    def execute(self, backend) -> xr.Dataset:  # pylint: disable=too-many-statements
        """
        Executes the benchmark.
        """
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Routine to generate all
        qcvv_logger.info(f"Identifying qubit pairs and neighbor groups for the Graph State benchmark")
        graph_benchmark_circuit_info, time_circuit_generation = (
            self.generate_all_circuit_info_for_graph_state_benchmark()
        )
        dataset.attrs.update({"time_circuit_generation": time_circuit_generation})

        # pylint: disable=invalid-sequence-index
        grouped_graph_circuits: Dict[int, QuantumCircuit] = graph_benchmark_circuit_info["grouped_graph_circuits"]
        unprojected_qubits = graph_benchmark_circuit_info["unmeasured_qubit_indices"]
        neighbor_qubits = graph_benchmark_circuit_info["projected_nodes"]
        pair_groups = graph_benchmark_circuit_info["pair_groups"]
        neighbor_groups = graph_benchmark_circuit_info["neighbor_groups"]
        # pylint: enable=invalid-sequence-index

        dataset.attrs.update(
            {
                "all_unprojected_qubits": unprojected_qubits,
                "all_projected_qubits": neighbor_qubits,
                "all_pair_groups": pair_groups,
                "all_neighbor_groups": neighbor_groups,
            }
        )

        circuits_untranspiled: Dict[int, List[QuantumCircuit]] = {}
        circuits_transpiled: Dict[int, List[QuantumCircuit]] = {}

        time_circuits = {}
        time_transpilation = {}
        all_graph_submit_results = []

        if self.tomography == "shadow_tomography":
            clifford_1q_dict = import_native_gate_cliffords("1q")

        qcvv_logger.info(f"Performing {self.tomography.replace('_',' ')} of all qubit pairs")

        all_unitaries: Dict[int, Dict[int, Dict[str, List[str]]]] = {}
        # all_unitaries: group_idx -> MoMs -> projection -> List[Clifford labels]
        # Will be empty if state_tomography -> assign Clifford labels in analysis
        for idx, circuit in grouped_graph_circuits.items():
            # It is not clear now that grouping is needed,
            # since it seems like pairs must be measured one at a time
            # (marginalizing any other qubits gives maximally mixed states)
            # however, the same structure is used in case this can still somehow be parallelized
            qcvv_logger.info(f"Now on group {idx + 1}/{len(grouped_graph_circuits)}")
            if self.tomography == "shadow_tomography":
                # Outer loop for each mean to be considered for Median of Means (MoMs) estimators
                all_unitaries[idx] = {m: {} for m in range(self.n_median_of_means)}
                circuits_untranspiled[idx] = []
                circuits_transpiled[idx] = []
                time_circuits[idx] = 0
                time_transpilation[idx] = 0
                for qubit_pair, neighbors in zip(pair_groups[idx], neighbor_groups[idx]):
                    RM_circuits_untranspiled_MoMs = []
                    RM_circuits_transpiled_MoMs = []
                    time_circuits_MoMs = 0
                    for MoMs in range(self.n_median_of_means):
                        # Go though each pair and only project neighbors
                        # all_unitaries[idx][MoMs] = {}
                        qcvv_logger.info(
                            f"Now on qubit pair {qubit_pair} and neighbors {neighbors} for Median of Means sample {MoMs + 1}/{self.n_median_of_means}"
                        )
                        (unitaries_single_pair, rm_circuits_untranspiled_single_pair), time_rm_circuits_single_pair = (
                            local_shadow_tomography(
                                qc=circuit,
                                Nu=self.n_random_unitaries,
                                active_qubits=qubit_pair,
                                measure_other=neighbors,
                                measure_other_name="neighbors",
                                clifford_or_haar="clifford",
                                cliffords_1q=clifford_1q_dict,
                            )
                        )

                        all_unitaries[idx][MoMs].update(unitaries_single_pair)
                        RM_circuits_untranspiled_MoMs.extend(rm_circuits_untranspiled_single_pair)
                        # When using a Clifford dictionary, both the graph state and the RMs are generated natively
                        RM_circuits_transpiled_MoMs.extend(rm_circuits_untranspiled_single_pair)
                        time_circuits_MoMs += time_rm_circuits_single_pair

                        self.transpiled_circuits.circuit_groups.append(
                            CircuitGroup(name=str(qubit_pair), circuits=rm_circuits_untranspiled_single_pair)
                        )

                    time_circuits[idx] += time_circuits_MoMs
                    circuits_untranspiled[idx].extend(RM_circuits_untranspiled_MoMs)
                    circuits_transpiled[idx].extend(RM_circuits_transpiled_MoMs)

                dataset.attrs.update({"all_unitaries": all_unitaries})
            else:  # if self.tomography == "state_tomography" (default)
                circuits_untranspiled[idx] = []
                circuits_transpiled[idx] = []
                time_circuits[idx] = 0
                time_transpilation[idx] = 0
                for qubit_pair, neighbors in zip(pair_groups[idx], neighbor_groups[idx]):
                    qcvv_logger.info(f"Now on qubit pair {qubit_pair} and neighbors {neighbors}")
                    state_tomography_circuits, time_state_tomo_circuits_single_pair = (
                        generate_state_tomography_circuits(
                            qc=circuit,
                            active_qubits=qubit_pair,
                            measure_other=neighbors,
                            measure_other_name="neighbors",
                            native=True,
                        )
                    )

                    self.transpiled_circuits.circuit_groups.append(
                        CircuitGroup(
                            name=str(qubit_pair), circuits=list(cast(dict, state_tomography_circuits).values())
                        )
                    )
                    time_circuits[idx] += time_state_tomo_circuits_single_pair
                    circuits_untranspiled[idx].extend(cast(dict, state_tomography_circuits).values())
                    # When using a native gates in tomo step, both the graph state and the RMs are generated natively
                    circuits_transpiled[idx].extend(cast(dict, state_tomography_circuits).values())

            # Submit for execution in backend - submit all per pair group, irrespective of tomography procedure.
            # A whole group is considered as a single batch.
            # Jobs will only be split in separate submissions if there are batch size limitations (retrieval will occur per batch).
            # It shouldn't be a problem [anymore] that different qubits are being measured in a single batch.
            # Post-processing will take care of separating MoMs samples and identifying all unitary (Clifford) labels.
            sorted_transpiled_qc_list = {tuple(unprojected_qubits[idx]): circuits_transpiled[idx]}
            graph_jobs, time_submit = submit_execute(
                sorted_transpiled_qc_list, backend, self.shots, self.calset_id, self.max_gates_per_batch
            )

            all_graph_submit_results.append(
                {
                    "unprojected_qubits": unprojected_qubits[idx],
                    "neighbor_qubits": neighbor_qubits[idx],
                    "jobs": graph_jobs,
                    "time_submit": time_submit,
                }
            )

        # Retrieve all counts and add to dataset
        for job_idx, job_dict in enumerate(all_graph_submit_results):
            unprojected_qubits = job_dict["unprojected_qubits"]
            # Retrieve counts
            execution_results, time_retrieve = retrieve_all_counts(job_dict["jobs"], identifier=str(unprojected_qubits))

            # Retrieve all job meta data
            all_job_metadata = retrieve_all_job_metadata(job_dict["jobs"])

            # Export all to dataset
            dataset.attrs.update(
                {
                    job_idx: {
                        "time_circuits": time_circuits[job_idx],
                        "time_transpilation": time_transpilation[job_idx],
                        "time_submit": job_dict["time_submit"],
                        "time_retrieve": time_retrieve,
                        "all_job_metadata": all_job_metadata,
                    }
                }
            )

            qcvv_logger.info(f"Adding counts of qubit pairs {unprojected_qubits} to the dataset")
            dataset, _ = add_counts_to_dataset(execution_results, str(unprojected_qubits), dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        # if self.rem:  TODO: add REM functionality

        qcvv_logger.info(f"Graph State benchmark experiment execution concluded !")

        return dataset


class GraphStateConfiguration(BenchmarkConfigurationBase):
    """Graph States Benchmark configuration.

    Attributes:
        benchmark (Type[Benchmark]): GraphStateBenchmark
        qubits (Sequence[int]): The physical qubit layout in which to benchmark graph state generation.
        tomography (Literal["state_tomography", "shadow_tomography"]): Whether to use state or shadow tomography.
            * Default is "state_tomography".
        num_bootstraps (int): The amount of bootstrap samples to use with state tomography.
            * Default is 50.
        n_random_unitaries (int): The number of Haar random single-qubit unitaries to use for (local) shadow tomography.
            * Default is 100.
        n_median_of_means(int): The number of mean samples over n_random_unitaries to generate a median of means estimator for shadow tomography.
            * NB: The total amount of execution calls will be a multiplicative factor of n_random_unitaries x n_median_of_means.
            * Default is 1 (no median of means).

    """

    benchmark: Type[Benchmark] = GraphStateBenchmark
    qubits: Sequence[int]
    tomography: Literal["state_tomography", "shadow_tomography"] = "state_tomography"
    num_bootstraps: int = 50
    n_random_unitaries: int = 100
    n_median_of_means: int = 1
