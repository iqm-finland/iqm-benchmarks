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
Graph states benchmark
"""

from typing import Dict, List, Sequence, Type

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
import xarray as xr

from iqm.benchmarks import Benchmark
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.utils import set_coupling_map
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def remove_directed_duplicates(cp_map: Type[CouplingMap]) -> List[List[int]]:
    """Remove duplicate edges from a coupling map.

    Args:
        cp_map (CouplingMap):
    Returns:
        List[List[int]]: the edges of the coupling map.
    """
    sorted_cp = [sorted(x) for x in list(cp_map)]
    return [list(x) for x in set(map(tuple, sorted_cp))]


def create_minimal_graph_layers(undirect_cp_map: Sequence[Sequence[int]]) -> Dict[int, List[List[int]]]:
    """Sorts the edges of an undirected coupling map (Sequence[Sequence[int]]),
    arranging them in a dictionary with values being subsets of the coupling map with no overlapping nodes.
    Each item will correspond to a layer of pairs of qubits in which parallel 2Q gates can be applied.

    Args:
         undirect_cp_map (Sequence[Sequnece[int]]): A list of lists of pairs of integers, representing an undirected coupling map (without duplicates).

    Returns:
        Dict[int, List[List[int]]]: A dictionary with values being subsets of the coupling map with no overlapping nodes.
    """
    # Build a conflict graph - Treat the input list as a graph
    # where each sublist is a node, and an edge exists between nodes if they share any integers
    n = len(undirect_cp_map)
    graph = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if set(undirect_cp_map[i]) & set(undirect_cp_map[j]):  # Check for shared integers
                graph[i].add(j)
                graph[j].add(i)

    # Reduce to a graph coloring problem;
    # each color represents a group in the dictionary
    colors = {}
    for node in range(n):
        # Find all used colors among neighbors
        neighbor_colors = {colors[neighbor] for neighbor in graph[node] if neighbor in colors}
        # Assign the smallest unused color
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[node] = color

    # Group by colors - minimize the number of groups
    groups = {}
    for idx, color in colors.items():
        if color not in groups:
            groups[color] = []
        groups[color].append(undirect_cp_map[idx])

    return groups


def generate_graph_state(qubits: Sequence[int], backend: IQMBackendBase | str) -> QuantumCircuit:
    """
    Args:
        qubits (Sequence[int]): A list of integers representing the qubits.
        backend (IQMBackendBase): The backend to target the graph state generating circuit.
    Returns:
        QuantumCircuit: The circuit generating a graph state in the target backend.
    """
    num_qubits = len(qubits)
    qc = QuantumCircuit(num_qubits)
    undirect_coupling_map = remove_directed_duplicates(set_coupling_map(qubits, backend, physical_layout="fixed"))

    # Add all H
    for q in range(num_qubits):
        qc.h(q)

    layers = create_minimal_graph_layers(undirect_coupling_map)

    # Add all CZ
    for layer in layers.values():
        for edge in layer:
            qc.cz(edge[0], edge[1])

    qc_t = transpile(qc, basis_gates=backend.operation_names, optimization_level=3)

    return qc_t


class GraphStatesBenchmark(Benchmark):
    """"""

    # analysis_function = staticmethod(negativity_analysis)
    name = "graph_states"

    def __init__(self, backend: IQMBackendBase, configuration: "GraphStatesConfiguration"):
        """Construct the GHZBenchmark class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend, configuration)

    def execute(self, backend) -> xr.Dataset:
        """
        Executes the benchmark.
        """


class GraphStatesConfiguration(BenchmarkConfigurationBase):
    """Graph States Benchmark configuration

    Attributes:
    """

    benchmark: Type[Benchmark] = GraphStatesBenchmark
