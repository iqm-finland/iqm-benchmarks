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

"""
Graph states benchmark
"""
import itertools
from time import strftime
from typing import Any, Dict, Sequence, Tuple, Type

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
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
from iqm.benchmarks.shadow_utils import get_local_shadow, get_negativity, local_shadow_tomography
from iqm.benchmarks.utils import (  # marginal_distribution,; perform_backend_transpilation,
    find_edges_with_disjoint_neighbors,
    generate_minimal_edge_layers,
    get_neighbors_of_edges,
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


def plot_shadows(
    avg_shadow: np.ndarray,
    qubit_pair: Sequence[int],
    projection: str,
    negativity: float,
    backend_name: str,
    timestamp: str,
    num_RM_samples: int,
) -> Tuple[str, Figure]:
    """Plot shadow density matrices for corresponding qubit pairs, neighbor qubit projections, and negativities.

    Args:

    Returns:
        Tuple[str, Figure]: The figure label and the shadow matrix plot figure.
    """

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 6))
    cmap = "seismic"
    fig_name = str(qubit_pair)

    ax[0].matshow(
        avg_shadow.real, interpolation="nearest", vmin=-np.max(avg_shadow.real), vmax=np.max(avg_shadow.real), cmap=cmap
    )
    ax[0].set_title(r"$\mathrm{Re}(\hat{\rho})$")
    for (i, j), z in np.ndenumerate(avg_shadow.real):
        ax[0].text(
            j,
            i,
            f"{z:0.2f}",
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.3"},
        )

    im1 = ax[1].matshow(
        avg_shadow.imag, interpolation="nearest", vmin=-np.max(avg_shadow.real), vmax=np.max(avg_shadow.real), cmap=cmap
    )
    ax[1].set_title(r"$\mathrm{Im}(\hat{\rho})$")
    for (i, j), z in np.ndenumerate(avg_shadow.imag):
        ax[1].text(
            j,
            i,
            f"{z:0.2f}",
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.3"},
        )

    fig.suptitle(
        f"Average shadow for qubits {qubit_pair} ({num_RM_samples} local RM samples)\n"
        f"Projection: {projection}\nNegativity: {negativity['value']:.4f} +/- {negativity['uncertainty']:.4f}\n"
        f"{backend_name} --- {timestamp}"
    )
    fig.colorbar(im1, shrink=0.5)
    fig.tight_layout(rect=[0, 0.03, 1, 1.25])

    plt.close()

    return fig_name, fig


def plot_max_negativities(
    negativities: Dict[str, Dict[str, float]], backend_name: str, timestamp: str, num_RM_samples: int
) -> Tuple[str, Figure]:
    """Plots the maximum negativity for each corresponding pair of qubits.

    Args:
        negativities (Dict[str, Dict[str,float]]):
        backend_name (str):
        timestamp (str):
        num_RM_samples (int):

    Returns:
        Tuple[str, Figure]: The figure label and the max negativities plot figure.
    """
    fig_name = f"max_negativities_{backend_name}_{timestamp}"
    # Sort the negativities by value
    sorted_negativities = dict(sorted(negativities.items(), key=lambda item: item[1]["value"]))

    x = list(sorted_negativities.keys())
    y = [a["value"] for a in sorted_negativities.values()]
    yerr = [a["uncertainty"] for a in sorted_negativities.values()]

    cmap = plt.cm.get_cmap("winter")

    fig = plt.figure()
    ax = plt.axes()

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        color=cmap(0.15),
        fmt="o",
        alpha=1,
        mec="black",
        markersize=5,
    )
    plt.axhline(0.5, color=cmap(1.0), linestyle="dashed")

    ax.set_xlabel("Qubit pair")
    ax.set_ylabel("Negativity")
    ax.grid()

    plt.xticks(rotation=60)
    plt.title(f"Max negativities for qubit pairs in {backend_name}\n{num_RM_samples} local RM samples\n{timestamp}")
    # plt.legend(fontsize=8)

    ax.set_aspect((2/3)*len(x))

    plt.close()

    return fig_name, fig


def negativity_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:  # pylint: disable=too-many-statements
    """Analysis function for a Graph State benchmark experiment.

    Args:
        run (RunResult): A Graph State benchmark experiment run for which analysis result is created.
    Returns:
        AnalysisResult corresponding to Graph State benchmark experiment.
    """
    plots = {}
    observations: list[BenchmarkObservation] = []
    qcvv_logger.info("Fetching dataset")
    dataset = run.dataset.copy(deep=True)
    qcvv_logger.info("Dataset imported OK")
    backend_name = dataset.attrs["backend_name"]
    execution_timestamp = dataset.attrs["execution_timestamp"]

    # qubits = dataset.attrs["qubits"]
    num_RMs = dataset.attrs["n_random_unitaries"]

    all_qubit_pairs_per_group = dataset.attrs["all_pair_groups"]
    all_qubit_neighbors_per_group = dataset.attrs["all_neighbor_groups"]
    all_RM_qubits = dataset.attrs["all_RM_qubits"]
    # all_projected_qubits = dataset.attrs["all_projected_qubits"]

    all_unitaries = dataset.attrs["all_unitaries"]
    # For graph states benchmark, all_unitaries is a Dict[int, Dict[str, List[str]]] where
    # the keys are the group indices and values are Dict[str, List[str]] with
    # keys being str(qubit) and values being lists of Clifford labels (keys for clifford_1q_dict below) for each RM

    qcvv_logger.info("Fetching Clifford dictionary")
    clifford_1q_dict, _ = import_native_gate_cliffords()

    execution_results = {}

    shadows_per_projection = {}
    average_shadows = {}
    stddev_shadows = {}
    all_negativities = {}  # {str(qubit_pair): {projections: negativities}}
    max_negativities = {}  # {str(qubit_pair): {"negativity": float, "projection": str}}

    for group_idx, group in all_qubit_pairs_per_group.items():
        qcvv_logger.info(f"Retrieving shadows for qubit-pair group {group_idx+1}/{len(all_qubit_pairs_per_group)}")
        # Assume only pairs and nearest-neighbors were measured, and each pair in the group user num_RMs randomized measurements:
        execution_results[group_idx] = xrvariable_to_counts(
            dataset, str(all_RM_qubits[group_idx]), num_RMs * len(group)
        )
        # Organize the counts into Dict[str, Dict[str,int]] with outermost keys being qubit pairs
        partitioned_counts = [
            execution_results[group_idx][i : i + num_RMs] for i in range(0, len(execution_results[group_idx]), num_RMs)
        ]
        marginal_counts = {}
        for pair_idx, qubit_pair in enumerate(group):
            qcvv_logger.info(f"Now on qubit pair {qubit_pair} ({pair_idx+1}/{len(group)})")
            marginal_counts[str(qubit_pair)] = partitioned_counts[pair_idx]
            # Done previously (Marginalize the counts over non-neighbor qubits of the current pair)
            # MARGINALIZING (EVEN NON-NEAREST-NEIGHBORS) SEEMS TO ALWAYS GENERATE A MAXIMALLY-MIXED STATE
            # This has been taken care of, and only pairs and nearest-neighbors were measured
            # qubits_to_marginalize = [
            #     x for x in all_projected_qubits[group_idx] if x not in neighbor_qubits and x not in qubit_pair
            # ]
            # if qubits_to_marginalize:
            #     bits_idx_to_marginalize = [i for i, x in enumerate(all_projected_qubits[group_idx]) if x in qubits_to_marginalize]
            #     marginal_counts = [
            #         marginal_distribution(counts, bits_idx_to_marginalize) for counts in execution_results[group_idx]
            #     ]
            # else:
            #     marginal_counts = execution_results[group_idx]

            all_negativities[str(qubit_pair)] = {}  # {str(qubit_pair): {projections: negativities}}
            max_negativities[str(qubit_pair)] = {}  # {str(qubit_pair): {"negativity": float, "projection": str}}

            # Get the neighbor qubits of qubit_pair
            neighbor_qubits = all_qubit_neighbors_per_group[group_idx][pair_idx]

            # Get all shadows of qubit_pair
            neighbor_bit_strings_length = len(neighbor_qubits)
            all_projection_bit_strings = [
                "".join(x) for x in itertools.product(("0", "1"), repeat=neighbor_bit_strings_length)
            ]
            shadows_per_projection[str(qubit_pair)] = {projection: [] for projection in all_projection_bit_strings}
            for RM_idx, counts in enumerate(marginal_counts[str(qubit_pair)]):
                # NEED TO RETRIEVE BOTH CLIFFORDS
                cliffords_rm = [all_unitaries[group_idx][str(q)][RM_idx] for q in qubit_pair]
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
                        b_s[-2:]: b_c for b_s, b_c in counts.items() if b_s[:neighbor_bit_strings_length] == projection
                    }
                    for projection in all_projection_bit_strings
                }

                # Get the individual shadow for each projection
                for projected_bit_string in all_projection_bit_strings:
                    shadows_per_projection[str(qubit_pair)][projected_bit_string].append(
                        get_local_shadow(
                            counts=projected_counts[projected_bit_string],
                            unitary_arg=cliffords_rm,
                            subsystem_bit_indices=list(range(2)),
                            clifford_or_haar="clifford",
                            cliffords_1q=clifford_1q_dict,
                        )
                    )

            # Average the shadows for each projection
            average_shadows[str(qubit_pair)] = {
                projected_bit_string: np.mean(shadows_per_projection[str(qubit_pair)][projected_bit_string], axis=0)
                for projected_bit_string in all_projection_bit_strings
            }
            stddev_shadows[str(qubit_pair)] = {
                projected_bit_string: np.std(shadows_per_projection[str(qubit_pair)][projected_bit_string], axis=0)
                / np.sqrt(num_RMs)
                for projected_bit_string in all_projection_bit_strings
            }

            # Compute the negativity of the shadow of each projection
            qcvv_logger.info(
                f"Computing the negativity of all shadow projections for qubit pair {qubit_pair} ({pair_idx+1}/{len(group)})"
            )
            all_negativities[str(qubit_pair)] = {
                projected_bit_string: {
                    "value": get_negativity(average_shadows[str(qubit_pair)][projected_bit_string], 1, 1),
                    "uncertainty": get_negativity(stddev_shadows[str(qubit_pair)][projected_bit_string], 1, 1),
                }
                for projected_bit_string in all_projection_bit_strings
            }

            # Extract the max negativity and the corresponding projection - save in dictionary
            all_negativities_list = [
                all_negativities[str(qubit_pair)][projected_bit_string]["value"]
                for projected_bit_string in all_projection_bit_strings
            ]
            all_negativities_uncertainty = [
                all_negativities[str(qubit_pair)][projected_bit_string]["uncertainty"]
                for projected_bit_string in all_projection_bit_strings
            ]

            max_negativity_projection = np.argmax(all_negativities_list)
            max_negativity = {
                "value": all_negativities_list[max_negativity_projection],
                "uncertainty": all_negativities_uncertainty[max_negativity_projection],
            }
            max_negativities[str(qubit_pair)].update(max_negativity)
            max_negativities[str(qubit_pair)].update(
                {
                    "projection": all_projection_bit_strings[max_negativity_projection],
                }
            )

            fig_name, fig = plot_shadows(
                avg_shadow=average_shadows[str(qubit_pair)][all_projection_bit_strings[max_negativity_projection]],
                qubit_pair=qubit_pair,
                projection=all_projection_bit_strings[max_negativity_projection],
                negativity=max_negativity,
                backend_name=backend_name,
                timestamp=execution_timestamp,
                num_RM_samples=num_RMs,
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
            "max_negativities": max_negativities,
            "average_shadows": average_shadows,
            "stddev_shadows": stddev_shadows,
            "all_negativities": all_negativities,
            "all_shadows": shadows_per_projection,
        }
    )

    fig_name, fig = plot_max_negativities(max_negativities, backend_name, execution_timestamp, num_RM_samples=num_RMs)
    plots[fig_name] = fig

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
        # {idx: [(q1,q2), (q3,q4), ...]}
        pair_groups = find_edges_with_disjoint_neighbors(graph_edges)
        # {idx: [{n11,n12,n13,...), (n21,n22,n23,...), ...]}
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

    def execute(self, backend) -> xr.Dataset:
        """
        Executes the benchmark.
        """
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Routine to generate all
        qcvv_logger.info(f"Generating all circuits for the Graph State benchmark")
        graph_benchmark_circuit_info, time_circuit_generation = (
            self.generate_all_circuit_info_for_graph_state_benchmark()
        )
        dataset.attrs.update({"time_circuit_generation": time_circuit_generation})

        # pylint: disable=invalid-sequence-index
        grouped_graph_circuits: Dict[int, QuantumCircuit] = graph_benchmark_circuit_info["grouped_graph_circuits"]
        RM_qubits = graph_benchmark_circuit_info["unmeasured_qubit_indices"]
        neighbor_qubits = graph_benchmark_circuit_info["projected_nodes"]
        pair_groups = graph_benchmark_circuit_info["pair_groups"]
        neighbor_groups = graph_benchmark_circuit_info["neighbor_groups"]
        # pylint: enable=invalid-sequence-index

        dataset.attrs.update(
            {
                "all_RM_qubits": RM_qubits,
                "all_projected_qubits": neighbor_qubits,
                "all_pair_groups": pair_groups,
                "all_neighbor_groups": neighbor_groups,
            }
        )

        RM_circuits_untranspiled = {}
        RM_circuits_transpiled = {}
        all_unitaries = {}
        time_RM_circuits = {}
        time_transpilation = {}
        all_graph_submit_results = []

        clifford_1q_dict, _ = import_native_gate_cliffords()

        # Get all shadow and neighbor-projection measurements
        qcvv_logger.info("Performing Randomized Measurements of all qubit pairs")
        for idx, circuit in grouped_graph_circuits.items():
            # It is not clear now that grouping is needed,
            # since it seems like pairs must be measured one at a time
            # (marginalizing any other qubits gives maximally mixed states)
            # however, the same structure is used in case this can still somehow be parallelized
            qcvv_logger.info(f"Now on group {idx+1}/{len(grouped_graph_circuits)}")
            # Go though each pair and only project neighbors
            # (as opposed to measuring everything and then in postprocessing having to marginalize non-neighbors)
            all_unitaries[idx] = {}
            RM_circuits_untranspiled[idx] = []
            RM_circuits_transpiled[idx] = []
            time_RM_circuits[idx] = 0
            time_transpilation[idx] = 0
            for rms, neighbors in zip(pair_groups[idx], neighbor_groups[idx]):
                qcvv_logger.info(f"Now on RMs {rms} and neighbors {neighbors}")
                (unitaries_single_pair, rm_circuits_untranspiled_single_pair), time_rm_circuits_single_pair = (
                    local_shadow_tomography(
                        qc=circuit,
                        Nu=self.n_random_unitaries,
                        active_qubits=rms,
                        measure_other=neighbors,
                        measure_other_name="neighbors",
                        clifford_or_haar="clifford",
                        cliffords_1q=clifford_1q_dict,
                    )
                )

                all_unitaries[idx].update(unitaries_single_pair)
                RM_circuits_untranspiled[idx].extend(rm_circuits_untranspiled_single_pair)
                time_RM_circuits[idx] += time_rm_circuits_single_pair
                # Transpile
                # rm_circuits_transpiled_single_pair, time_transpilation_single_pair = perform_backend_transpilation(
                #     qc_list=rm_circuits_untranspiled_single_pair,
                #     backend=backend,
                #     qubits=self.qubits,
                #     coupling_map=backend.coupling_map,
                # )
                # When using a Clifford dictionary, both the graph state and the RMs are generated natively
                RM_circuits_transpiled[idx].extend(rm_circuits_untranspiled_single_pair)

                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(name=str(rms), circuits=rm_circuits_untranspiled_single_pair)
                )

            # Submit for execution in backend
            # It shouldn't be a problem [anymore] that different qubits are being measured in a single batch.
            sorted_transpiled_qc_list = {tuple(RM_qubits[idx]): RM_circuits_transpiled[idx]}
            rm_graph_jobs, time_submit = submit_execute(
                sorted_transpiled_qc_list, backend, self.shots, self.calset_id, self.max_gates_per_batch
            )

            all_graph_submit_results.append(
                {
                    "RM_qubits": RM_qubits[idx],
                    "neighbor_qubits": neighbor_qubits[idx],
                    "jobs": rm_graph_jobs,
                    "time_submit": time_submit,
                }
            )

        dataset.attrs.update({"all_unitaries": all_unitaries})

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
