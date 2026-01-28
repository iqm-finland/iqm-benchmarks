# Copyright 2024 IQM Benchmarks developers
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

"""
Plotting and visualization utility functions
"""
from dataclasses import dataclass
import os
from typing import Dict, List, Literal, Optional, Sequence, Tuple, cast

from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from rustworkx import PyGraph, spring_layout, visualization  # pylint: disable=no-name-in-module

from iqm.benchmarks.entanglement.ghz import get_cx_map, get_edges
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import extract_fidelities_unified, get_iqm_backend, random_hamiltonian_path
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


@dataclass
class GraphPositions:
    """A class to store and generate graph positions for different chip layouts.

    This class contains predefined node positions for various quantum chip topologies and
    provides methods to generate positions for different layout types.

    Attributes:
        garnet_positions (Dict[int, Tuple[float, float]]): Mapping of node indices to (x,y) positions for Garnet chip.
        deneb_positions (Dict[int, Tuple[float, float]]): Mapping of node indices to (x,y) positions for Deneb chip.
        predefined_stations (Dict[str, Dict[int, Tuple[float, float]]]): Mapping of chip names to their position dictionaries.
    """

    garnet_positions = {
        0: (5.0, 7.0),
        1: (6.0, 6.0),
        2: (3.0, 7.0),
        3: (4.0, 6.0),
        4: (5.0, 5.0),
        5: (6.0, 4.0),
        6: (7.0, 3.0),
        7: (2.0, 6.0),
        8: (3.0, 5.0),
        9: (4.0, 4.0),
        10: (5.0, 3.0),
        11: (6.0, 2.0),
        12: (1.0, 5.0),
        13: (2.0, 4.0),
        14: (3.0, 3.0),
        15: (4.0, 2.0),
        16: (5.0, 1.0),
        17: (1.0, 3.0),
        18: (2.0, 2.0),
        19: (3.0, 1.0),
    }

    emerald_positions = {
        0: (10, 10),
        1: (11, 9),
        2: (7, 11),
        3: (8, 10),
        4: (9, 9),
        5: (10, 8),
        6: (11, 7),
        7: (5, 11),
        8: (6, 10),
        9: (7, 9),
        10: (8, 8),
        11: (9, 7),
        12: (10, 6),
        13: (11, 5),
        14: (3, 11),
        15: (4, 10),
        16: (5, 9),
        17: (6, 8),
        18: (7, 7),
        19: (8, 6),
        20: (9, 5),
        21: (10, 4),
        22: (2, 10),
        23: (3, 9),
        24: (4, 8),
        25: (5, 7),
        26: (6, 6),
        27: (7, 5),
        28: (8, 4),
        29: (9, 3),
        30: (10, 2),
        31: (2, 8),
        32: (3, 7),
        33: (4, 6),
        34: (5, 5),
        35: (6, 4),
        36: (7, 3),
        37: (8, 2),
        38: (9, 1),
        39: (1, 7),
        40: (2, 6),
        41: (3, 5),
        42: (4, 4),
        43: (5, 3),
        44: (6, 2),
        45: (7, 1),
        46: (1, 5),
        47: (2, 4),
        48: (3, 3),
        49: (4, 2),
        50: (5, 1),
        51: (1, 3),
        52: (2, 2),
        53: (3, 1),
    }

    sirius_positions = {
        # Even nodes on the bottom
        2: (1, 5),
        4: (3, 5),
        6: (5, 5),
        8: (7, 5),
        10: (9, 5),
        12: (11, 5),
        14: (13, 5),
        16: (15, 5),
        18: (17, 5),
        20: (19, 5),
        22: (21, 5),
        24: (23, 5),
        # Odd nodes on the top
        1: (1, 1),
        3: (3, 1),
        5: (5, 1),
        7: (7, 1),
        9: (9, 1),
        11: (11, 1),
        13: (13, 1),
        15: (15, 1),
        17: (17, 1),
        19: (19, 1),
        21: (21, 1),
        23: (23, 1),
    }
    # Add dummy indices for the resonator
    max_qubit_number = np.max(list(sirius_positions.keys()))
    previous_nodes = list(sirius_positions.keys())
    for node in previous_nodes:
        sirius_positions.update({node + max_qubit_number: (sirius_positions[node][0], 3)})
    # Node 0 in the middle
    sirius_positions.update({0: (16.5, 3)})

    deneb_positions = {
        # Even nodes on the bottom
        2: (1, 5),
        4: (3, 5),
        6: (5, 5),
        # Odd nodes on the top
        1: (1, 1),
        3: (3, 1),
        5: (5, 1),
    }
    # Add dummy indices for the resonator
    max_qubit_number = np.max(list(deneb_positions.keys()))
    previous_nodes = list(deneb_positions.keys())
    for node in previous_nodes:
        deneb_positions.update({node + max_qubit_number: (deneb_positions[node][0], 3)})
    # Node 0 in the middle
    deneb_positions.update({0: (2.5, 3)})

    predefined_stations = {
        "garnet": garnet_positions,
        "fakeapollo": garnet_positions,
        "iqmfakeapollo": garnet_positions,
        "deneb": deneb_positions,
        "fakedeneb": deneb_positions,
        "iqmfakedeneb": deneb_positions,
        "emerald": emerald_positions,
        "sirius": sirius_positions,
        "crystal54": emerald_positions,
        "crystal20": garnet_positions,
        "star": sirius_positions,
        "star24": sirius_positions,
    }

    @staticmethod
    def create_positions(
        graph: PyGraph, topology: Optional[Literal["star", "crystal"]] = None
    ) -> Dict[int, Tuple[float, float]]:
        """Generate node positions for a given graph and topology.

        Args:
            graph (PyGraph): The graph to generate positions for.
            topology (Optional[Literal["star", "crystal"]]): The type of layout to generate. Must be either "star" or "crystal".

        Returns:
            Dict[int, Tuple[float, float]]: A dictionary mapping node indices to (x,y) coordinates.
        """
        n_nodes = len(graph.node_indices())

        if topology == "star":
            # Place center node at (0,0)
            pos = {0: (0.0, 0.0)}

            if n_nodes > 1:
                # Place other nodes in a circle around the center
                angles = np.linspace(0, 2 * np.pi, n_nodes - 1, endpoint=False)
                radius = 1.0

                for i, angle in enumerate(angles, start=1):
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    pos[i] = (x, y)

        # Crystal and other topologies
        else:
            # Fix first node position in bottom right
            fixed_pos = {0: (1.0, 1.0)}  # For more consistent layouts

            # Get spring layout with one fixed position
            pos = {
                int(k): (float(v[0]), float(v[1]))
                for k, v in spring_layout(graph, scale=2, pos=fixed_pos, num_iter=300, fixed={0}).items()
            }
        return pos

    @staticmethod
    def get_positions(
        station: Optional[str] = None, graph: Optional[PyGraph] = None, num_qubits: Optional[int] = None
    ) -> Dict[int, Tuple[float, float]]:
        """Get predefined positions for a specific station or generate positions for a custom graph.

        Args:
            station (Optional[str]): The name of the station to get predefined positions for.
                If None, positions will be generated algorithmically.
            graph (Optional[PyGraph]): The graph to generate positions for if no predefined positions exist.
                Used only when station is None and num_qubits doesn't match any predefined layout.
            num_qubits (Optional[int]): The number of qubits to get a layout for.
                If matches a known system, predefined positions will be used.

        Returns:
            Dict[int, Tuple[float, float]]: A dictionary mapping node indices to (x,y) coordinates.

        Raises:
            ValueError: If none of station, graph, or num_qubits are provided, or if num_qubits doesn't
                match any predefined layout and graph is None.
        """
        if station is not None and station.lower() in GraphPositions.predefined_stations:
            qubit_positions = cast(Dict[int, Tuple[float, float]], GraphPositions.predefined_stations[station.lower()])
        else:
            qubit_station_dict = {6: "deneb", 7: "deneb", 20: "garnet", 24: "sirius", 17: "sirius", 54: "emerald"}
            if num_qubits is not None and num_qubits in qubit_station_dict:
                station = qubit_station_dict[num_qubits]
                qubit_positions = cast(Dict[int, Tuple[float, float]], GraphPositions.predefined_stations[station])
            elif graph is not None:
                qubit_positions = GraphPositions.create_positions(graph)
            else:
                raise ValueError("Either a station name, a graph, or a qubit count must be provided to get positions.")
        return qubit_positions


def draw_graph_edges(
    backend_coupling_map: CouplingMap,
    backend_num_qubits: int,
    edge_list: Sequence[Tuple[int, int]],
    timestamp: str,
    disjoint_layers: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    station: Optional[str] = None,
    qubit_names: Optional[Dict[int, str]] = None,
    is_eplg: Optional[bool] = False,
) -> Tuple[str, Figure]:
    """Draw given edges on a graph within the given backend.

    Args:
        backend_coupling_map (CouplingMap): The coupling map to draw the graph from.
        backend_num_qubits (int): The number of qubits of the respectve backend.
        edge_list (Sequence[Tuple[int, int]]): The edge list of the linear chain.
        timestamp (str): The timestamp to include in the figure name.
        disjoint_layers (Optional[Sequence[Sequence[Tuple[int, int]]]): Sequences of edges defining disjoint layers to draw.
            * Default is None.
        station (Optional[str]): The name of the station.
            * Default is None.
        qubit_names (Optional[Dict[int, str]]): A dictionary mapping qubit indices to their names.
            * Default is None.
        is_eplg (Optional[bool]): A flag indicating if the graph refers to an EPLG experiment.
            * Default is False.

    Returns:
         Tuple[str, Figure]: The figure name and the figure object.
    """
    disjoint = "_disjoint" if disjoint_layers is not None else ""
    fig_name_station = f"_{station.lower()}" if station is not None else ""
    fig_name = f"edges_graph{disjoint}{fig_name_station}_{timestamp}"

    fig = plt.figure()
    ax = plt.axes()

    qubit_positions = GraphPositions.get_positions(station=station, graph=None, num_qubits=backend_num_qubits)

    label_station = station if station is not None else f"{backend_num_qubits}-qubit IQM Backend"
    if disjoint_layers is None:
        nx.draw_networkx(
            rx_to_nx_graph(backend_coupling_map),
            pos=qubit_positions,
            edgelist=edge_list,
            width=4.0,
            edge_color="k",
            node_color="k",
            font_color="w",
            ax=ax,
        )

        plt.title(f"Selected edges in {label_station}\n" f"\n{timestamp}")

    else:
        num_disjoint_layers = len(disjoint_layers)
        colors = plt.colormaps["rainbow"](np.linspace(0, 1, num_disjoint_layers))
        all_edge_colors = [[colors[i]] * len(l) for i, l in enumerate(disjoint_layers)]  # Flatten below
        nx.draw_networkx(
            rx_to_nx_graph(backend_coupling_map),
            pos=qubit_positions,
            labels=(
                {x: qubit_names[x] for x in range(backend_num_qubits)}
                if qubit_names
                else list(range(backend_num_qubits))
            ),
            font_size=6.5 if qubit_names else 10,
            edgelist=[x for y in disjoint_layers for x in y],
            width=4.0,
            edge_color=[x for y in all_edge_colors for x in y],
            node_color="k",
            font_color="w",
            ax=ax,
        )

        is_eplg_string = " for EPLG experiment" if is_eplg else ""
        plt.title(
            f"Selected edges in {label_station.capitalize()}{is_eplg_string}\n"
            f"{len(disjoint_layers)} groups of disjoint layers"
            f"\n{timestamp}"
        )
    ax.set_aspect(0.925)
    plt.gca().invert_yaxis()
    plt.close()

    return fig_name, fig


def evaluate_hamiltonian_paths(
    N: int,
    path_samples: int,
    backend_arg: str | IQMBackendBase,
    url: str,
    max_tries: int = 10,
) -> Dict[int, List[Tuple[int, int]]]:
    """Evaluates Hamiltonian paths according to the product of 2Q gate fidelities on the corresponding edges of the backend graph.

    Args:
        N (int): the number of vertices in the Hamiltonian paths to evaluate.
        path_samples (int): the number of Hamiltonian paths to evaluate.
        backend_arg (str | IQMBackendBase): the backend to evaluate the Hamiltonian paths on with respect to fidelity.
        url (str): the URL address for the backend to retrieve calibration data from.
        max_tries (int): the maximum number of tries to generate a Hamiltonian path.

    Returns:
        Dict[int, List[Tuple[int, int]]]: A dictionary with keys being fidelity products and values being the respective Hamiltonian paths.
    """
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    backend_nx_graph = rx_to_nx_graph(backend.coupling_map)

    all_paths = []
    sample_counter = 0
    tries = 0
    while sample_counter < path_samples and tries <= max_tries:
        h_path = random_hamiltonian_path(backend_nx_graph, N)
        if not h_path:
            qcvv_logger.debug(f"Failed to generate a Hamiltonian path with {N} vertices - retrying...")
            tries += 1
            if tries == max_tries:
                raise RecursionError(
                    f"Max tries to generate a Hamiltonian path with {N} vertices reached - Try with less vertices!\n"
                    f"For EPLG, you may also manually specify qubit pairs."
                )
            continue
        all_paths.append(h_path)
        tries = 0
        sample_counter += 1

    # Get scores for all paths
    # Retrieve fidelity data
    cal_data = extract_fidelities_unified(url, backend)
    two_qubit_fidelity = cal_data[-1]["cz_gate_fidelity"]
    # Rate all the paths
    path_costs = {}  # keys are costs, values are edge paths
    for h_path in all_paths:
        total_cost = 1
        for edge in h_path:
            if len(edge) == 2:
                qb0 = backend.index_to_qubit_name(edge[0]).replace("QB", "")
                qb1 = backend.index_to_qubit_name(edge[1]).replace("QB", "")
                total_cost *= two_qubit_fidelity[[qb0, qb1]]
        path_costs[total_cost] = h_path

    return path_costs


def calculate_node_radii(
    metric_dict: Dict[str, Dict[int, float]], qubit_nodes: List[int], sq_metric: str
) -> np.ndarray:
    """Calculate node radii based on the specified single qubit metric. For the coherence metric, the fidelity is calculated as the idling fidelity of a single qubit gate duration.

    Args:
        metric_dict (Dict[str, Dict[int, float]]): Dictionary containing various qubit metrics.
        qubit_nodes (List[int]): List of qubits to calculate the radius for.
        sq_metric (str): Metric to use for radius calculation.
                         Options: "fidelity", "coherence", or "readout".

    Returns:
        numpy.ndarray: Array of radii values for each qubit node.

    Raises:
        ValueError: If an unsupported metric type is provided.
    """
    if sq_metric == "fidelity":
        radii = -np.log(np.array([metric_dict["fidelity_1qb_gates_averaged"][node] for node in qubit_nodes]))
        if "fidelity_1qb_gates_averaged" not in metric_dict:
            raise ValueError("The metric 'fidelity_1qb_gates_averaged' is not available in the backend metrics.")
    elif sq_metric == "coherence":
        if "t1_time" not in metric_dict or "t2_time" not in metric_dict:
            raise ValueError(
                "At least one of the metrics 't1_time' and 't2_time' is not available in the backend metrics."
            )
        sqg_time = 32e-9
        t1_times = [metric_dict["t1_time"][node] for node in qubit_nodes]
        t2_times = [metric_dict["t2_time"][node] for node in qubit_nodes]
        idle_fidelities = (3 + np.exp(-sqg_time / np.array(t1_times)) + 2 * np.exp(-sqg_time / np.array(t2_times))) / 6
        radii = -np.log(idle_fidelities)
    elif sq_metric == "readout":
        if "single_shot_readout_fidelity" not in metric_dict:
            raise ValueError("The metric 'single_shot_readout_fidelity' is both available in the backend metrics.")
        readout_fidelities = [metric_dict["single_shot_readout_fidelity"][node] for node in qubit_nodes]
        radii = -np.log(readout_fidelities)
    else:
        raise ValueError(
            f"Unsupported single qubit metric: {sq_metric}, supported metrics are: fidelity, coherence, readout"
        )
    return radii


def plot_layout_fidelity_graph(
    iqm_server_url: str,
    backend: IQMBackendBase,
    qubit_layouts: Optional[list[list[int]]] = None,
    quantum_computer: Optional[str] = None,
    sq_metric: Optional[str] = "coherence",
    show_ghz_path: bool = False,
):
    """Plot a graph showing the quantum chip layout with fidelity information.

    Creates a visualization of the quantum chip topology where nodes represent qubits
    and edges represent connections between qubits. Edge thickness indicates gate errors
    (thinner edges mean better fidelity) and selected qubits are highlighted in orange.

    Args:
        iqm_server_url: URL to retrieve calibration data from.
        backend: IQM backend instance.
        qubit_layouts: List of qubit layouts where each layout is a list of qubit indices
        quantum_computer: Name of the quantum computing station to use predefined positions for.
                If None, positions will be generated algorithmically.
        sq_metric: Optional single qubit metric to use for the visualization, can be either "fidelity", "coherence",
                or "readout".
        show_ghz_path: Whether to highlight the edges that are part of the GHZ state creation tree path.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing the graph visualization
    """
    # pylint: disable=unbalanced-tuple-unpacking, disable=too-many-statements

    edges_cal, fidelities_cal, topology, qubit_mapping, metric_dict = extract_fidelities_unified(
        iqm_server_url, backend, quantum_computer
    )

    if topology == "star":
        idx_to_qubit = {idx: qubit for qubit, idx in qubit_mapping.items()}
        qubit_to_idx = qubit_mapping
        qubit_nodes = list(idx_to_qubit.keys())[1:]
        fig, ax = plt.subplots(figsize=(len(qubit_nodes), 3))
    else:
        # For other topologies, qubits are indexed starting from 0 as per the Qiskit convention
        idx_to_qubit = {idx: qubit - 1 for qubit, idx in qubit_mapping.items()}
        qubit_to_idx = {qubit - 1: idx for qubit, idx in qubit_mapping.items()}
        qubit_nodes = list(idx_to_qubit.keys())
        fig, ax = plt.subplots(figsize=(1.5 * np.sqrt(len(qubit_nodes)), 1.5 * np.sqrt(len(qubit_nodes))))

    # Filter out any edges that are not in the backend's coupling map
    fidelities_cal = [
        fidelity
        for edge, fidelity in zip(edges_cal, fidelities_cal, strict=True)
        if (edge[0], edge[1]) in backend.coupling_map or (edge[1], edge[0]) in backend.coupling_map
    ]
    edges_cal = [edge for edge in edges_cal if (edge[0], edge[1])
                 in backend.coupling_map or (edge[1], edge[0]) in backend.coupling_map]

    weights = -np.log(np.array(fidelities_cal))
    calibrated_nodes = list(idx_to_qubit.keys())

    if show_ghz_path and qubit_layouts is not None and any(len(layout) >= 2 for layout in qubit_layouts):
        valid_fidelities = [f for f in fidelities_cal if f < 1.0]
        if valid_fidelities:
            median_fidelity = np.median(valid_fidelities)
            fidelities_cal = [f if f < 1.0 else median_fidelity for f in fidelities_cal]
        from iqm.benchmarks.entanglement.ghz import generate_ghz_spanning_tree, get_cx_map, get_edges

        graph = get_edges(backend.coupling_map, qubit_layouts[0], edges_cal, fidelities_cal)
        cx_map = get_cx_map(qubit_layouts[0], graph)
        print(f"Edge map:", cx_map)

    # Define qubit positions in plot
    qubit_positions = GraphPositions.get_positions(station=quantum_computer, graph=None, num_qubits=len(calibrated_nodes))

    graph = PyGraph()
    nodes = list(set(qubit_positions.keys()))
    graph.add_nodes_from(nodes)
    for edge, weight in zip(edges_cal, weights):
        if topology == "star":
            max_qubit_number = (np.max(list(qubit_positions.keys())) + 1) // 2
            graph.add_edge(idx_to_qubit[edge[0]], idx_to_qubit[edge[0]] + max_qubit_number, weight)
        else:
            graph.add_edge(idx_to_qubit[edge[0]], idx_to_qubit[edge[1]], weight)

    # # Draw the main graph
    visualization.mpl_draw(
        graph,
        ax=ax,
        with_labels=True,
        node_color="none",  # No node color since we're using circles
        pos=qubit_positions,
        labels=lambda node: str(qubit_to_idx[node]) if node in qubit_to_idx else "",
        font_color="white",
        edge_color="white",
    )

    # Draw edges manually with custom styles and colors
    graph_edge_pairs = [list(edge) for edge in graph.edge_list()]
    for _, (edge, weight) in enumerate(zip(graph_edge_pairs, graph.edges(), strict=True)):
        x1, y1 = qubit_positions[edge[0]]
        x2, y2 = qubit_positions[edge[1]]

        # Define edge properties (customize these as needed)
        edge_width = weight / np.max(list(graph.edges())) * 10 + 0.1
        edge_color = "black"
        edge_style = "solid"

        # Highlight edges with zero weight (wrong calibration)
        if weight == 0:
            edge_width = 1
            edge_style = "dashed"
        # Highlight edges that are part of the GHZ path
        if show_ghz_path and qubit_layouts is not None and any(len(layout) >= 2 for layout in qubit_layouts):
            mapped_edge = [qubit_to_idx[edge[0]], qubit_to_idx[edge[1]]]
            if mapped_edge in cx_map or list(reversed(mapped_edge)) in cx_map:
                edge_color = "red"

        ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=edge_width, linestyle=edge_style, zorder=1)

    # Draw nodes as circles with varying radii given by the single qubit metric
    radii = calculate_node_radii(metric_dict, qubit_nodes, sq_metric)
    node_colors = ["darkgray" for _ in range(len(nodes))]
    if qubit_layouts is not None:
        for qb in {qb for layout in qubit_layouts for qb in layout}:
            node_colors[idx_to_qubit[qb]] = "orange"
    max_radius = 0.12 + np.max(radii) / np.max(radii) / 2.5

    for idx, node in enumerate(qubit_nodes):
        position = qubit_positions[idx_to_qubit[node]]
        radius = 0.12 + radii[idx] / np.max(radii) / 2.5
        circle = Circle(position, radius=radius, color=node_colors[idx_to_qubit[node]], fill=True, alpha=1)
        ax.add_patch(circle)

    # Add edge labels using matplotlib's annotate
    # for idx, edge in enumerate(edges_cal):
    for edge, weight in zip(graph_edge_pairs, list(graph.edges()), strict=True):
        x1, y1 = qubit_positions[edge[0]]
        x2, y2 = qubit_positions[edge[1]]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        plt.annotate(
            f"{weight:.1e}",
            xy=(x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )

    # Add horizontal bar representing resonator
    if topology == "star":
        resonator_height = 3
        resonator_thickness = 0.8
        x_min = 0.5
        x_max = qubit_positions[idx_to_qubit[qubit_nodes[-1]]][0] + 0.5
        resonator_width = x_max - x_min

        # Create rectangle with rounded corners
        resonator = FancyBboxPatch(
            (x_min, resonator_height - resonator_thickness / 2),
            resonator_width,
            resonator_thickness,
            boxstyle="round,pad=0.01,rounding_size=0.3",
            color="lightsteelblue",
            zorder=10,
        )
        ax.add_patch(resonator)

        # Add "Resonator" label in the center
        plt.annotate(
            "Resonator",
            xy=((x_min + x_max) / 2, resonator_height),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            zorder=11,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    # Calculate axis limits to ensure all circles are visible
    all_x = [pos[0] for pos in qubit_positions.values()]
    all_y = [pos[1] for pos in qubit_positions.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add padding for circles
    padding = max_radius * 1.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Adjust layout first
    plt.tight_layout(pad=2.0)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.figtext(
        0.5,
        0.99,  # x=0.5 (center), y=0.01 (bottom)
        f"Qubit connectivity with selected qubits in orange\n"
        f"CZ errors -log(F) indicated by edge thickness (thinner is better)\n"
        f"Single qubit errors -log(F) shown as node size with F computed from {sq_metric} metrics",
        fontsize=10,
        ha="center",
        wrap=True,
    )

    plt.show()
    return fig


def rx_to_nx_graph(backend_coupling_map: CouplingMap) -> nx.Graph:
    """Convert the Rustworkx graph returned by a backend to a Networkx graph.

    Args:
        backend_coupling_map (CouplingMap): The coupling map of the backend.

    Returns:
        networkx.Graph: The Networkx Graph corresponding to the backend graph.

    """

    # Generate a Networkx graph
    graph_backend = backend_coupling_map.graph.to_undirected(multigraph=False)
    backend_egdes, backend_nodes = (list(graph_backend.edge_list()), list(graph_backend.node_indices()))
    backend_nx_graph = nx.Graph()
    backend_nx_graph.add_nodes_from(backend_nodes)
    backend_nx_graph.add_edges_from(backend_egdes)

    return backend_nx_graph
