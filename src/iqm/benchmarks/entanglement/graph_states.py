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
from typing import Any, Dict, Sequence, Type

from mypyc.analysis.dataflow import AnalysisResult
from qiskit import QuantumCircuit, transpile
import xarray as xr

from iqm.benchmarks import Benchmark, BenchmarkCircuit, BenchmarkRunResult, Circuits
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import BenchmarkAnalysisResult, BenchmarkObservation, add_counts_to_dataset
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.shadow_utils import haar_shadow_tomography
from iqm.benchmarks.utils import (
    find_edges_with_disjoint_neighbors,
    generate_minimal_edge_layers,
    get_neighbors_of_edges,
    perform_backend_transpilation,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    set_coupling_map,
    submit_execute,
    timeit,
    xrvariable_to_counts,
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
    qc_t = transpile(qc, backend=backend, initial_layout=qubits, optimization_level=3)
    return qc_t


def negativity_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a Graph State benchmark experiment.

    Args:
        run (RunResult): A Graph State benchmark experiment run for which analysis result is created.
    Returns:
        AnalysisResult corresponding to Graph State benchmark experiment.
    """
    plots = {}
    observations: list[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)
    backend_name = dataset.attrs["backend_name"]
    execution_timestamp = dataset.attrs["execution_timestamp"]

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


class GraphStateBenchmark(Benchmark):
    """The Graph States benchmark estimates the bipartite entangelement negativity of native graph states."""

    analysis_function = staticmethod(negativity_analysis)
    name = "graph_states"

    def __init__(self, backend_arg: IQMBackendBase, configuration: "GraphStateConfiguration"):
        """Construct the GraphStateBenchmark class.

        Args:
            backend_arg (IQMBackendBase): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.qubits = configuration.qubits
        self.n_random_unitaries = configuration.n_random_unitaries

        # Initialize relevant variables for the benchmark
        self.graph_state_circuit = generate_graph_state(self.qubits, self.backend)
        self.coupling_map = set_coupling_map(self.qubits, self.backend, physical_layout="fixed")

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

        # Get unique list of edges
        graph_edges = list(self.coupling_map.graph.to_undirected(multigraph=False).edge_list())
        # Find pairs of nodes with disjoint neighbors

        pair_groups = find_edges_with_disjoint_neighbors(graph_edges)
        # Get all projected nodes to cover all pairs of qubits with disjoint neighbours
        unmeasured_qubit_indices = {idx: [a for b in x for a in b] for idx, x in enumerate(pair_groups)}
        projected_nodes = {idx: get_neighbors_of_edges(list(x), graph_edges) for idx, x in enumerate(pair_groups)}

        # Generate copies of circuits to add projections and randomized measurements
        grouped_graph_circuits = {idx: self.graph_state_circuit.copy() for idx in projected_nodes.keys()}

        return {
            "grouped_graph_circuits": grouped_graph_circuits,
            "unmeasured_qubit_indices": unmeasured_qubit_indices,
            "projected_nodes": projected_nodes,
            "pair_groups": pair_groups,
        }

    def execute(self, backend) -> xr.Dataset:
        """
        Executes the benchmark.
        """
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Routine to generate all
        graph_benchmark_circuit_info: Dict[str, Any]
        graph_benchmark_circuit_info, time_circuit_generation = (
            self.generate_all_circuit_info_for_graph_state_benchmark()
        )
        dataset.attrs.update({"time_circuit_generation": time_circuit_generation})

        RM_qubits = {}
        neighbor_qubits = {}
        RM_circuits_transpiled = {}
        all_unitaries = {}
        time_RM_circuits = {}
        time_transpilation = {}
        all_graph_submit_results = []

        # Get all local Randomized Measurements

        for idx, circuit in graph_benchmark_circuit_info["grouped_graph_circuits"].items():
            RM_qubits[idx] = graph_benchmark_circuit_info["unmeasured_qubit_indices"][idx]
            neighbor_qubits[idx] = graph_benchmark_circuit_info["projected_nodes"][idx]
            (all_unitaries[idx], RM_circuits), time_RM_circuits[idx] = haar_shadow_tomography(
                circuit, self.n_random_unitaries, RM_qubits, neighbor_qubits, measure_other_name="neighbors"
            )

            # Transpile
            RM_circuits_transpiled[idx], time_transpilation[idx] = perform_backend_transpilation(
                qc_list=RM_circuits,
                backend=backend,
                qubits=self.qubits,
            )

            # Submit for execution in backend
            sorted_transpiled_qc_list = {str(RM_qubits): RM_circuits_transpiled[idx]}
            rm_graph_jobs, time_submit = submit_execute(
                sorted_transpiled_qc_list, backend, self.shots, self.calset_id, self.max_gates_per_batch
            )

            all_graph_submit_results.append(
                {
                    "RM_qubits": RM_qubits,
                    "neighbor_qubits": neighbor_qubits,
                    "jobs": rm_graph_jobs,
                    "time_submit": time_submit,
                }
            )

        # Retrieve all counts and add to dataset
        for job_idx, job_dict in enumerate(all_graph_submit_results):
            RM_qubits = job_dict["RM_qubits"]
            # Retrieve counts
            execution_results, time_retrieve = retrieve_all_counts(job_dict["jobs"], identifier=str(RM_qubits))
            # Retrieve all job meta data
            all_job_metadata = retrieve_all_job_metadata(job_dict["jobs"])

            # Export all to dataset
            dataset.attrs.update(
                {
                    job_idx: {
                        "qubit_pairs": RM_qubits[job_idx],
                        "neighbor_qubits": neighbor_qubits[job_idx],
                        "time_RM_circuits": time_RM_circuits[job_idx],
                        "time_transpilation": time_transpilation[job_idx],
                        "time_submit": job_dict["time_submit"],
                        "time_retrieve": time_retrieve,
                        "all_job_metadata": all_job_metadata,
                    }
                }
            )

            qcvv_logger.info(f"Adding counts of qubit pairs {RM_qubits} to the dataset")
            dataset, _ = add_counts_to_dataset(execution_results, str(RM_qubits), dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        # if self.rem:  TODO: add REM functionality

        qcvv_logger.info(f"Graph State benchmark experiment execution concluded !")

        return dataset


class GraphStateConfiguration(BenchmarkConfigurationBase):
    """Graph States Benchmark configuration.

    Attributes:
        benchmark (Type[Benchmark]): GraphStateBenchmark
        qubits (Sequence[int]): The physical qubit layout in which to benchmark graph state generation.
        n_random_unitaries (int): The number of Haar random single-qubit unitaries to use for (local) shadow tomography.

    """

    benchmark: Type[Benchmark] = GraphStateBenchmark
    qubits: Sequence[int]
    n_random_unitaries: int
