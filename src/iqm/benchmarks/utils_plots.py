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
from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
from rustworkx import PyGraph, spring_layout, visualization  # pylint: disable=no-name-in-module

from iqm.benchmarks.utils import extract_fidelities


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

    predefined_stations = {"garnet": garnet_positions, "deneb": deneb_positions}

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

    plt.subplots(figsize=(1.5 * np.sqrt(len(nodes)), 1.5 * np.sqrt(len(nodes))))

    # Draw the graph
    visualization.mpl_draw(
        graph,
        with_labels=True,
        node_color=node_colors,
        pos=pos,
        labels=lambda node: node,
        width=7 * weights / np.max(weights),
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
