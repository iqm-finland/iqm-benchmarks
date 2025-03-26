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
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests
from qiskit.transpiler import CouplingMap
from rustworkx import PyGraph, spring_layout, visualization  # pylint: disable=no-name-in-module

from iqm.benchmarks.utils import extract_fidelities, get_iqm_backend, random_hamiltonian_path
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


@dataclass
class GraphPositions:
    """A class to store and generate graph positions for different chip layouts.

    This class contains predefined node positions for various quantum chip topologies and
    provides methods to generate positions for different layout types.

    Attributes:
        garnet_positions (Dict[int, Tuple[int, int]]): Mapping of node indices to (x,y) positions for Garnet chip.
        deneb_positions (Dict[int, Tuple[int, int]]): Mapping of node indices to (x,y) positions for Deneb chip.
        predefined_stations (Dict[str, Dict[int, Tuple[int, int]]]): Mapping of chip names to their position dictionaries.
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

    deneb_positions = {
        0: (2.0, 2.0),
        1: (1.0, 1.0),
        3: (2.0, 1.0),
        5: (3.0, 1.0),
        2: (1.0, 3.0),
        4: (2.0, 3.0),
        6: (3.0, 3.0),
    }

    emerald_positions = {
        0: (10.0, 10.0),
        1: (11.0, 9.0),
        2: (7.0, 11.0),
        3: (8.0, 10.0),
        4: (9.0, 9.0),
        5: (10.0, 8.0),
        6: (11.0, 7.0),
        7: (5.0, 11.0),
        8: (6.0, 10.0),
        9: (7.0, 9.0),
        10: (8.0, 8.0),
        11: (9.0, 7.0),
        12: (10.0, 6.0),
        13: (11.0, 5.0),
        14: (3.0, 11.0),
        15: (4.0, 10.0),
        16: (5.0, 9.0),
        17: (6.0, 8.0),
        18: (7.0, 7.0),
        19: (8.0, 6.0),
        20: (9.0, 5.0),
        21: (10.0, 4.0),
        22: (2.0, 10.0),
        23: (3.0, 9.0),
        24: (4.0, 8.0),
        25: (7.0, 5.0),
        26: (8.0, 4.0),
        27: (9.0, 3.0),
        28: (10.0, 2.0),
        29: (2.0, 8.0),
        30: (3.0, 7.0),
        31: (7.0, 3.0),
        32: (8.0, 2.0),
        33: (9.0, 1.0),
        34: (1.0, 7.0),
        35: (2.0, 6.0),
        36: (3.0, 5.0),
        37: (4.0, 4.0),
        38: (5.0, 3.0),
        39: (6.0, 2.0),
        40: (7.0, 1.0),
        41: (1.0, 5.0),
        42: (2.0, 4.0),
        43: (3.0, 3.0),
        44: (4.0, 2.0),
        45: (5.0, 1.0),
        46: (1.0, 3.0),
        47: (2.0, 2.0),
        48: (3.0, 1.0),
    }

    predefined_stations = {"garnet": garnet_positions, "deneb": deneb_positions, "emerald": emerald_positions}

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

    backend_nx_graph = rx_to_nx_graph(backend_arg)

    all_paths = []
    sample_counter = 0
    tries = 0
    while sample_counter < path_samples and tries < max_tries:
        h_path = random_hamiltonian_path(backend_nx_graph, N)
        if not h_path:
            tries += 1
            continue

        all_paths.append(h_path)
        tries = 0
        sample_counter += 1
    if tries == max_tries - 1:
        raise RecursionError(
            f"Max tries to generate a Hamiltonian path with {N} vertices reached - try with less vertices!"
        )

    # Get scores for all paths
    # Retrieve fidelity data
    two_qubit_fidelity = {}

    headers = {"Accept": "application/json", "Authorization": "Bearer " + os.environ["IQM_TOKEN"]}
    r = requests.get(url, headers=headers, timeout=60)
    calibration = r.json()

    edge_dictionary = {}
    for iq in calibration["calibrations"][0]["metrics"][0]["metrics"]:
        temp = list(iq.values())
        two_qubit_fidelity[str(temp[0])] = temp[1]
        two_qubit_fidelity[str([temp[0][1], temp[0][0]])] = temp[1]
        edge_dictionary[str([temp[0][1], temp[0][0]])] = (
            backend.qubit_name_to_index(temp[0][1]),
            backend.qubit_name_to_index(temp[0][0]),
        )

    # Rate all the paths
    path_costs = {}  # keys are costs, values are edge paths
    for h_path in all_paths:
        total_cost = 1
        for edge in h_path:
            if len(edge) == 2:
                total_cost *= two_qubit_fidelity[
                    str([backend.index_to_qubit_name(edge[0]), backend.index_to_qubit_name(edge[1])])
                ]
        path_costs[total_cost] = h_path

    return path_costs


def plot_layout_fidelity_graph(
    cal_url: str, qubit_layouts: Optional[list[list[int]]] = None, station: Optional[str] = None
):
    """Plot a graph showing the quantum chip layout with fidelity information.

    Creates a visualization of the quantum chip topology where nodes represent qubits
    and edges represent connections between qubits. Edge thickness indicates gate errors
    (thinner edges mean better fidelity) and selected qubits are highlighted in orange.

    Args:
        cal_url: URL to retrieve calibration data from
        qubit_layouts: List of qubit layouts where each layout is a list of qubit indices
        station: Name of the quantum computing station to use predefined positions for.
                If None, positions will be generated algorithmically.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing the graph visualization
    """
    edges_cal, fidelities_cal, topology = extract_fidelities(cal_url)
    weights = -np.log(np.array(fidelities_cal))
    edges_graph = [tuple(edge) + (weight,) for edge, weight in zip(edges_cal, weights)]

    graph = PyGraph()

    # Add nodes
    nodes: set[int] = set()
    for edge in edges_graph:
        nodes.update(edge[:2])
    graph.add_nodes_from(list(nodes))

    # Add edges
    graph.add_edges_from(edges_graph)

    # Define qubit positions in plot
    if station is not None and station.lower() in GraphPositions.predefined_stations:
        pos = GraphPositions.predefined_stations[station.lower()]
    else:
        pos = GraphPositions.create_positions(graph, topology)

    # Define node colors
    node_colors = ["lightgrey" for _ in range(len(nodes))]
    if qubit_layouts is not None:
        for qb in {qb for layout in qubit_layouts for qb in layout}:
            node_colors[qb] = "orange"

    # Ensuring weights are in correct order for the plot
    edge_list = graph.edge_list()
    weights_dict = {}
    edge_pos = set()

    # Create a mapping between edge positions as defined in rustworkx and their weights
    for e, w in zip(edge_list, weights):
        pos_tuple = (tuple(pos[e[0]]), tuple(pos[e[1]]))
        weights_dict[pos_tuple] = w
        edge_pos.add(pos_tuple)

    # Get corresponding weights in the same order
    weights_ordered = np.array([weights_dict[edge] for edge in list(edge_pos)])

    plt.subplots(figsize=(6, 6))

    # Draw the graph
    visualization.mpl_draw(
        graph,
        with_labels=True,
        node_color=node_colors,
        pos=pos,
        labels=lambda node: node,
        width=7 * weights_ordered / np.max(weights_ordered),
    )  # type: ignore[call-arg]

    # Add edge labels using matplotlib's annotate
    for edge in edges_graph:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        plt.annotate(
            f"{edge[2]:.1e}",
            xy=(x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )

    plt.gca().invert_yaxis()
    plt.title(
        "Chip layout with selected qubits in orange\n"
        + "and gate errors indicated by edge thickness (thinner is better)"
    )
    plt.show()


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
