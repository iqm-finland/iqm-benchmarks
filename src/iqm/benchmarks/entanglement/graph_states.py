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
from time import strftime
from typing import Sequence, Type, List

from qiskit import QuantumCircuit, transpile
import xarray as xr

from iqm.benchmarks import Benchmark, BenchmarkCircuit, Circuits
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import (
    find_pairs_with_disjoint_neighbors,
    generate_minimal_edge_layers,
    get_neighbors_of_edges,
    project_neighbouring_qubits,
    set_coupling_map,
    timeit,
    get_active_qubits,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


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
    coupling_map = set_coupling_map(qubits, backend, physical_layout="fixed")
    layers = generate_minimal_edge_layers(coupling_map)
    # Add all H
    for q in range(num_qubits):
        qc.h(q)
    # Add all CZ
    for layer in layers.values():
        for edge in layer:
            qc.cz(edge[0], edge[1])
    # Transpile
    qc_t = transpile(qc, backend=backend, optimization_level=3)
    return qc_t


def generate_all_graph_projected_circuits(
    graph_circuit: QuantumCircuit, backend: IQMBackendBase
) -> List[QuantumCircuit]:
    """Generate all graph circuits with disjoint projected (measured) neighbors for each pair of qubits.

    Args:
        graph_circuit (QuantumCircuit): The graph circuit to generate all projected graph circuits.
        backend (IQMBackendBase): The backend to target the graph state generating circuit.

    Returns:
        List[QuantumCircuit]: A list of projected graph circuits.
    """
    input_qubits = get_active_qubits(graph_circuit)
    # Get projected nodes for each pair of qubits in the graph state
    coupling_map = set_coupling_map(input_qubits, backend, physical_layout="fixed")
    # Get unique list of edges
    graph_edges = list(coupling_map.graph.to_undirected(multigraph=False).edge_list())
    # Find pairs of nodes with disjoint neighbors
    pair_groups = find_pairs_with_disjoint_neighbors(graph_edges)
    # Get all the nodes to be measured for each edge (pair of qubits)
    projected_nodes = [get_neighbors_of_edges(x, graph_edges) for x in pair_groups]
    # Return all the circuits
    return [project_neighbouring_qubits(graph_circuit, x) for x in projected_nodes]


class GraphStatesBenchmark(Benchmark):
    """The Graph States benchmark estimates the bipartite entangelement negativity of native graph states."""

    # analysis_function = staticmethod(negativity_analysis)
    name = "graph_states"

    def __init__(self, backend_arg: IQMBackendBase, configuration: "GraphStatesConfiguration"):
        """Construct the GraphStatesBenchmark class.

        Args:
            backend_arg (IQMBackendBase): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.qubits = configuration.qubits

        # Initialize the variable to contain the QV circuits of each layout
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

    def execute(self, backend) -> xr.Dataset:
        """
        Executes the benchmark.
        """
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Generate native graph state
        graph_state_circuit = generate_graph_state(self.qubits, backend)

        # Generate all projected circuits for all pairs of qubits
        projected_graph_circuits = generate_all_graph_projected_circuits(graph_state_circuit, self.qubits, backend)

        return dataset


class GraphStatesConfiguration(BenchmarkConfigurationBase):
    """Graph States Benchmark configuration.

    Attributes:
        benchmark (Type[Benchmark]): GraphStatesBenchmark
        qubits (Sequence[int]): The physical qubit layout in which to benchmark graph state generation.
    """

    benchmark: Type[Benchmark] = GraphStatesBenchmark
    qubits: Sequence[int]
