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
CLOPS benchmark
"""

from datetime import datetime
from math import floor, pi
from time import perf_counter, strftime
from typing import Any, Dict, List, Sequence, Tuple, Type

import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector
import xarray as xr

from iqm.benchmarks import Benchmark
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import BenchmarkAnalysisResult, BenchmarkRunResult
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import (
    count_2q_layers,
    count_native_gates,
    perform_backend_transpilation,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    set_coupling_map,
    sort_batches_by_final_layout,
    submit_execute,
    timeit,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates


def plot_times(clops_data: xr.Dataset, observations: Dict[int, Dict[str, Dict[str, float]]]) -> Tuple[str, Figure]:
    """Generate a figure representing the different elapsed times in the CLOPS experiment.

    Args:
        clops_data (xr.Dataset): The dataset including elapsed time data from the CLOPS experiment
        observations (Dict[int, Dict[str, Dict[str, float]]]): The observations from the analysis of the CLOPS experiment.
    Returns:
        str: the name of the figure.
        Figure: the figure.
    """
    # Define the keys for different categories of times
    job_keys = ["submit_total", "compile_total", "execution_total"]
    user_keys = ["user_submit_total", "user_retrieve_total", "assign_parameters_total"]
    total_keys = ["job_total"]

    # Define variables for dataset values
    qubits = clops_data.attrs["qubits"]
    num_qubits = clops_data.attrs["num_qubits"]
    backend_name = clops_data.attrs["backend_name"]
    execution_timestamp = clops_data.attrs["execution_timestamp"]
    clops_time = float(clops_data.attrs["clops_time"])

    all_data = {}
    all_data.update(clops_data.attrs)
    all_data.update(observations[1])

    # Define colors
    cmap = mpl.pyplot.get_cmap("winter")
    colors = [cmap(i) for i in np.linspace(0, 1, len(job_keys) + len(user_keys) + len(total_keys) + 1)]

    # Plotting parameters
    fontsize = 6
    sep = 1
    barsize = 1
    alpha = 0.4

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    fig_name = f"{num_qubits}_qubits_{tuple(qubits)}"

    # Plot total CLOPS time
    x_t = ax1.bar(4 * sep, clops_time, barsize, zorder=0, label="clops time", color=(colors[-1], alpha), edgecolor="k")
    ax1.bar_label(x_t, fmt=f"clops time: {clops_time:.2f}", fontsize=fontsize)

    # Plot user keys
    for i, (key, cumulative_value) in enumerate(zip(user_keys, np.cumsum([all_data[k] for k in user_keys]))):
        x = ax1.bar(
            3 * sep,
            cumulative_value,
            barsize,
            zorder=1 - i / 10,
            label=key,
            color=(colors[len(job_keys) + len(total_keys) + i], alpha),
            edgecolor="k",
        )
        ax1.bar_label(x, fmt=f"{key.replace('_total', ' ').replace('_', ' ')}: {all_data[key]:.2f}", fontsize=fontsize)

    # Plot total keys
    for i, (key, cumulative_value) in enumerate(zip(total_keys, np.cumsum([all_data[k] for k in total_keys]))):
        x = ax1.bar(
            2 * sep,
            cumulative_value,
            barsize,
            zorder=1 - i / 10,
            label=key,
            color=(colors[len(job_keys) + i], alpha),
            edgecolor="k",
        )
        ax1.bar_label(x, fmt=f"{key.replace('_total', ' ')}: {all_data[key]:.2f}", fontsize=fontsize)

    # Plot job keys
    for i, (key, cumulative_value) in enumerate(zip(job_keys, np.cumsum([all_data[k] for k in job_keys]))):
        x = ax1.bar(
            sep, cumulative_value, barsize, zorder=2 - i / 10, label=key, color=(colors[i], alpha), edgecolor="k"
        )
        ax1.bar_label(x, label_type="edge", fmt=f"{key.replace('_total', ' ')}: {all_data[key]:.2f}", fontsize=fontsize)

    # Remove top and bottom spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    # Set axis labels and limits
    ax1.set_ylabel("Total CLOPS time (seconds)")
    ax2.set_ylabel("Total CLOPS time (%)")
    ax1.set_ylim(-0.2, clops_time + 1)
    ax2.set_ylim(-0.2, 100)

    # Set x-ticks and labels
    time_types = ["Remote (components)", "Remote (total)", "Wall-time (components)", "Wall-time (total)"]
    ax1.set_xticks([i * sep + 1 for i in range(4)], time_types, fontsize=fontsize)

    # Set plot title
    if all_data["clops_h"]["value"] == 0:
        plt.title(
            f"Total execution times for CLOPS experiment\n"
            f"{backend_name}, {execution_timestamp}\n"
            f"CLOPS_v {all_data['clops_v']['value']} on qubits {qubits}\n"
        )
    else:
        plt.title(
            f"Total execution times for CLOPS experiment\n"
            f"{backend_name}, {execution_timestamp}\n"
            f"CLOPS_v {all_data['clops_v']['value']} / CLOPS_h {all_data['clops_h']['value']} on qubits {qubits}\n"
        )

    plt.gcf().set_dpi(250)
    plt.close()

    return fig_name, fig


def retrieve_clops_elapsed_times(job_meta: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Retrieve the elapsed times from the CLOPS job metadata

    Args:
        job_meta (Dict[Dict[str, Any]]): A dictionary of CLOPS jobs metadata
    Returns:
        Dict[str, float]: A dictionary of elapsed times of all CLOPS jobs
    """
    all_job_elapsed: Dict = {}
    totals_keys: Dict = {}
    for u_index, update in enumerate(job_meta):
        all_job_elapsed[update] = {}
        for b_index, batch in enumerate(job_meta[update].keys()):
            # ["timestamps"] might be empty if backend is a simulator
            if job_meta[update][batch]["timestamps"] is not None:
                x = job_meta[update][batch]["timestamps"]
                job_time_format = "%Y-%m-%dT%H:%M:%S.%f%z"  # Is it possible to extract this automatically?
                compile_f = datetime.strptime(x["compile_end"], job_time_format)
                compile_i = datetime.strptime(x["compile_start"], job_time_format)
                submit_f = datetime.strptime(x["submit_end"], job_time_format)
                submit_i = datetime.strptime(x["submit_start"], job_time_format)
                execution_f = datetime.strptime(x["execution_end"], job_time_format)
                execution_i = datetime.strptime(x["execution_start"], job_time_format)
                job_f = datetime.strptime(x["job_end"], job_time_format)
                job_i = datetime.strptime(x["job_start"], job_time_format)

                all_job_elapsed[update][batch] = {
                    "job_total": job_f - job_i,
                    "compile_total": compile_f - compile_i,
                    "submit_total": submit_f - submit_i,
                    "execution_total": execution_f - execution_i,
                }

                # Save the keys, will be needed later
                totals_keys = all_job_elapsed[update][batch].keys()

                # Update metadata file - need to turn values ("timedelta" type) into str
                job_meta["update_" + str(u_index + 1)]["batch_job_" + str(b_index + 1)].update(
                    {k: str(all_job_elapsed[update][batch][k]) for k in totals_keys}
                )

    # Add up totals
    overall_elapsed: Dict = {}
    # If backend is a simulator
    if not totals_keys:
        return overall_elapsed

    overall_elapsed = {k: 0 for k in totals_keys}
    overall_elapsed_per_update: Dict = {}
    for update in job_meta:
        overall_elapsed_per_update[update] = {k: 0 for k in totals_keys}
        for batch in job_meta[update].keys():
            for k in totals_keys:
                overall_elapsed_per_update[update][k] += all_job_elapsed[update][batch][k].total_seconds()
        for k in totals_keys:
            overall_elapsed[k] += overall_elapsed_per_update[update][k]

    return overall_elapsed


def clops_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a CLOPS (v or h) experiment

    Args:
        run (RunResult): A CLOPS experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to CLOPS
    """
    plots: Dict[str, Any] = {}
    observations = {}
    dataset = run.dataset

    # Retrieve dataset values
    # backend_name = dataset.attrs["backend_configuration_name"]
    num_circuits = dataset.attrs["num_circuits"]
    num_updates = dataset.attrs["num_updates"]
    num_shots = dataset.attrs["num_shots"]
    depth = dataset.attrs["depth"]

    clops_h_bool = dataset.attrs["clops_h_bool"]

    all_job_meta = dataset.attrs["job_meta_per_update"]

    clops_time = dataset.attrs["clops_time"]
    all_times_parameter_assign = dataset.attrs["all_times_parameter_assign"]
    all_times_submit = dataset.attrs["all_times_submit"]
    all_times_retrieve = dataset.attrs["all_times_retrieve"]

    transpiled_qc_list = []
    for group in run.circuits["transpiled_circuits"].circuit_groups:
        transpiled_qc_list.extend(group.circuits)
    # for _, value in dataset.attrs["transpiled_circuits"].items():
    #     for _, transpiled_circuit in value.items():

    # CLOPS_V
    clops_v: float = num_circuits * num_updates * num_shots * depth / clops_time

    # CLOPS_H
    clops_h: float = 0.0
    counts_depths = {}
    time_count_layers: float = 0.0
    if clops_h_bool:
        # CLOPS_h: count the number of layers made of parallel 2Q gates in the transpiled circuits
        clops_h_depths: List[int]
        qcvv_logger.info("Counting the number of parallel 2Q layer depths in each circuit")
        clops_h_depths, time_count_layers = count_2q_layers(  # pylint: disable=unbalanced-tuple-unpacking
            transpiled_qc_list
        )
        counts_depths = {f"depth_{u}": clops_h_depths.count(u) for u in sorted(set(clops_h_depths))}

        clops_h = num_circuits * num_updates * num_shots * np.mean(clops_h_depths) / clops_time

    processed_results = {
        "clops_v": {"value": int(clops_v), "uncertainty": np.NaN},
        "clops_h": {"value": int(clops_h), "uncertainty": np.NaN},
    }

    dataset.attrs.update(
        {
            "assign_parameters_total": sum(all_times_parameter_assign.values()),
            "user_submit_total": sum(all_times_submit.values()),
            "user_retrieve_total": sum(all_times_retrieve.values()),
            "parallel_2q_layers": counts_depths,
            "time_count_layers": time_count_layers,
        }
    )

    # UPDATE OBSERVATIONS
    observations.update({1: processed_results})

    # PLOT
    # Get all execution elapsed times
    overall_elapsed = retrieve_clops_elapsed_times(all_job_meta)  # will be {} if backend is a simulator
    if overall_elapsed:
        qcvv_logger.info("Total elapsed times from job execution metadata:")
        for k in overall_elapsed.keys():
            dataset.attrs[k] = overall_elapsed[k]
            if overall_elapsed[k] > 60.0:
                qcvv_logger.info(f'\t"{k}": {overall_elapsed[k] / 60.0:.2f} min')
            else:
                qcvv_logger.info(f'\t"{k}": {overall_elapsed[k]:.2f} sec')

        fig_name, fig = plot_times(dataset, observations)
        plots[fig_name] = fig
    else:
        qcvv_logger.info("There is no elapsed-time data associated to jobs (e.g., execution on simulator)")

    # Sort the final dataset
    dataset.attrs = dict(sorted(dataset.attrs.items()))

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


class CLOPSBenchmark(Benchmark):
    """
    CLOPS reflect the speed of execution of parametrized QV circuits.
    """

    analysis_function = staticmethod(clops_analysis)

    name: str = "clops"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "CLOPSConfiguration"):
        """Construct the QuantumVolumeBenchmark class.

        Args:
            backend_arg (IQMBackendBase | str): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        # EXPERIMENT
        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name
        self.qubits = configuration.qubits
        self.num_qubits = len(self.qubits)
        self.depth = self.num_qubits

        self.u_per_layer = floor(self.num_qubits / 2)
        self.num_parameters = self.depth * self.u_per_layer * 15
        self.param_vector = ParameterVector("p", length=self.num_parameters)

        self.num_circuits = configuration.num_circuits
        self.num_updates = configuration.num_updates
        self.num_shots = configuration.num_shots

        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg

        # POST-EXPERIMENT AND VARIABLES TO STORE
        self.clops_h_bool = configuration.clops_h_bool

        self.parameters_per_update: Dict[str, List[float]] = {}
        self.counts_per_update: Dict[str, Dict[str, int]] = {}
        self.job_meta_per_update: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self.time_circuit_generate: float = 0.0
        self.time_transpile: float = 0.0
        self.time_sort_batches: float = 0.0

        self.session_timestamp = strftime("%Y-%m-%d_%H:%M:%S")
        self.execution_timestamp: str = ""

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

        # Defined outside configuration
        dataset.attrs["num_qubits"] = self.num_qubits
        dataset.attrs["depth"] = self.depth
        dataset.attrs["u_per_layer"] = self.u_per_layer
        dataset.attrs["num_parameters"] = self.num_parameters

    def append_parameterized_unitary(
        self,
        qc: QuantumCircuit,
        q0: int,
        q1: int,
        layer: int,
        pair: int,
    ) -> None:
        """Append a decomposed, parametrized SU(4) gate using CX gates to the given quantum circuit.

        Args:
            qc (QuantumCircuit): the quantum circuit to append the SU(4) gate to
            q0 (int): the first qubit involved in the gate
            q1 (int): the second qubit involved in the gate
            layer (int): the QV layer the gate belongs to
            pair (int): the pair index corresponding to the gate
        """
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 0], q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 1], q1)
        qc.sx(q0)
        qc.sx(q1)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 2], q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 3], q1)
        qc.sx(q0)
        qc.sx(q1)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 4], q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 5], q1)
        qc.cx(q1, q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 6], q0)
        qc.sx(q1)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 7], q1)
        qc.cx(q1, q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 8], q0)
        qc.sx(q1)
        qc.cx(q1, q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 9], q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 12], q1)
        qc.sx(q0)
        qc.sx(q1)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 10], q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 13], q1)
        qc.sx(q0)
        qc.sx(q1)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 11], q0)
        qc.rz(self.param_vector[15 * (pair + self.u_per_layer * layer) + 14], q1)

    def generate_single_circuit(self) -> QuantumCircuit:
        """Generate a single parametrized QV quantum circuit, with measurements at the end.

        Returns:
            QuantumCircuit: the QV quantum circuit.
        """
        self.u_per_layer = floor(self.num_qubits / 2)
        qc = QuantumCircuit(self.num_qubits)

        qubits = qc.qubits

        for layer in range(self.depth):
            perm = np.random.permutation(self.num_qubits)
            for pair in range(self.u_per_layer):
                q0 = qubits[perm[pair * 2]]
                q1 = qubits[perm[pair * 2 + 1]]

                self.append_parameterized_unitary(qc, q0, q1, layer, pair)

        qc.measure_all()

        return qc

    @timeit
    def generate_circuit_list(
        self,
    ) -> List[QuantumCircuit]:
        """Generate a list of parametrized QV quantum circuits, with measurements at the end.

        Returns:
            List[QuantumCircuit]: the list of parametrized QV quantum circuits.
        """
        qc_list = [self.generate_single_circuit() for _ in range(self.num_circuits)]
        return qc_list

    def generate_random_parameters(self) -> np.ndarray:
        """Generate an array of as many random parameters as needed by the QV template circuits.

        Returns:
            np.ndarray[np.float64]: the array of random parameters
        """
        return np.random.uniform(low=-pi, high=pi, size=self.num_parameters)

    @timeit
    def assign_random_parameters_to_all(
        self,
        dict_parametrized_circs: Dict[Tuple, List[QuantumCircuit]],
        optimize_sqg: bool,
    ) -> Tuple[List[List[float]], Dict[Tuple, List[QuantumCircuit]]]:
        """

        Args:
            dict_parametrized_circs (Dict[Tuple, List[QuantumCircuit]]): Dictionary with list of int (qubits) as keys and lists of parametrized quantum circuits as values
            optimize_sqg (bool): Whether single qubit gate optimization is applied
        Returns:
            A tuple of dictionaries:
            - lists of list of float parameter values corresponding to param updates
            - dictionary with lists of int (qubits) as keys and lists of quantum circuits as values
        """
        # Store parametrized circuits in a separate dictionary
        sorted_dict_parametrized: Dict[Tuple, List[QuantumCircuit]] = {k: [] for k in dict_parametrized_circs.keys()}
        param_values: List[List[float]] = []
        for k in dict_parametrized_circs.keys():
            for qc in dict_parametrized_circs[k]:
                # Update parameters
                parameters = self.generate_random_parameters()

                # NDonis can't use optimize_sqg as is, yet -> complains about MOVE gate not being IQM native!
                if optimize_sqg and self.backend.name != "IQMNdonisBackend":
                    sorted_dict_parametrized[k].append(
                        optimize_single_qubit_gates(  # Optimize SQG seems worth it AFTER assignment
                            qc.assign_parameters(
                                dict(zip(qc.parameters, parameters)),
                                inplace=False,  # Leave the template intact for next updates
                            )
                        )
                    )
                else:
                    sorted_dict_parametrized[k].append(
                        qc.assign_parameters(
                            dict(zip(qc.parameters, parameters)),
                            inplace=False,  # Leave the template intact for next updates
                        )
                    )

                param_values.append(list(parameters))

        return param_values, sorted_dict_parametrized

    def clops_cycle(
        self,
        backend: IQMBackendBase,
        sorted_transpiled_qc_list: Dict[Tuple, List[QuantumCircuit]],
        update: int,
    ) -> Tuple[float, float, float]:
        """Executes a single CLOPS cycle (parameter assignment and execution) for the given update
        Args:
            backend (IQMBackendBase): the backend to execute the jobs with
            sorted_transpiled_qc_list (Dict[str, List[QuantumCircuit]]): A dictionary of lists of transpiled quantum circuits
            update (int): The current cycle update
        Returns:
            Tuple[float, float, float]: The elapsed times for parameter assignment, submission and retrieval of jobs
        """
        # Assign parameters to all quantum circuits
        qcvv_logger.info(
            f"Update {(update + 1)}/{self.num_updates}\nAssigning random parameters to all {self.num_circuits} circuits"
        )
        (all_param_updates, sorted_transpiled_qc_list_parametrized), time_parameter_assign = (
            self.assign_random_parameters_to_all(sorted_transpiled_qc_list, self.optimize_sqg)
        )

        qcvv_logger.info(f"Executing the corresponding circuit batch")
        # Submit all circuits to execute
        all_jobs, time_submit = submit_execute(
            sorted_transpiled_qc_list_parametrized,
            backend,
            self.num_shots,
            self.calset_id,
            max_gates_per_batch=self.max_gates_per_batch,
        )

        qcvv_logger.info(f"Retrieving counts")
        # Retrieve counts - the precise outputs do not matter
        all_counts, time_retrieve = retrieve_all_counts(all_jobs)
        # Save counts - ensures counts were received and can be inspected
        self.parameters_per_update["parameters_update_" + str(update + 1)] = all_param_updates
        self.counts_per_update["counts_update_" + str(update + 1)] = all_counts
        # Retrieve and save all job metadata
        all_job_metadata = retrieve_all_job_metadata(all_jobs)
        self.job_meta_per_update["update_" + str(update + 1)] = all_job_metadata

        return time_parameter_assign, time_submit, time_retrieve

    def generate_transpiled_clops_templates(self) -> Dict[Tuple, List[QuantumCircuit]]:
        """Generates CLOPS circuit templates transpiled to the backend's physical layout

        Returns:
            Dict[str, QuantumCircuit]: a dictionary of quantum circuits with keys being str(qubit layout)
        """
        # Generate the list of QV circuits

        qc_list, self.time_circuit_generate = self.generate_circuit_list()

        coupling_map = set_coupling_map(self.qubits, self.backend, self.physical_layout)
        # Perform transpilation to backend
        qcvv_logger.info(
            f'Will transpile all {self.num_circuits} circuits according to "{self.physical_layout}" physical layout'
        )

        transpiled_qc_list, self.time_transpile = perform_backend_transpilation(
            qc_list,
            self.backend,
            self.qubits,
            coupling_map=coupling_map,
            qiskit_optim_level=self.qiskit_optim_level,
            optimize_sqg=True,
            routing_method=self.routing_method,
        )

        # Batching
        sorted_transpiled_qc_list = {}
        if self.physical_layout == "fixed":
            sorted_transpiled_qc_list = {tuple(self.qubits): transpiled_qc_list}
        elif self.physical_layout == "batching":
            # Sort circuits according to their final measurement mappings
            (sorted_transpiled_qc_list, _), self.time_sort_batches = sort_batches_by_final_layout(transpiled_qc_list)

        self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=self.qubits, circuits=qc_list))
        for key in sorted_transpiled_qc_list.keys():
            self.transpiled_circuits.circuit_groups.append(CircuitGroup(name=f"{self.qubits}_{key}", circuits=qc_list))

        return sorted_transpiled_qc_list

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y-%m-%d_%H:%M:%S")

        self.circuits = Circuits()
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        qcvv_logger.info(
            "NB: CLOPS should be estimated with same qubit layout and optional inputs used to establish QV!"
        )
        if self.num_circuits != 100 or self.num_updates != 10 or self.num_shots != 100:
            qcvv_logger.info(
                f"NB: CLOPS parameters, by definition, are [num_circuits=100, num_updates=10, num_shots=100]"
                f" You chose"
                f" [num_circuits={self.num_circuits}, num_updates={self.num_updates}, num_shots={self.num_shots}]."
            )

        qcvv_logger.info(
            f"Now generating {self.num_circuits} parametrized circuit templates on qubits {self.qubits}",
        )

        sorted_transpiled_qc_list = self.generate_transpiled_clops_templates()

        # *********************************************
        # Start CLOPS timer
        # *********************************************
        start_clops_timer = perf_counter()
        qcvv_logger.info(f"CLOPS time started")
        all_times_parameter_assign = {}
        all_times_submit = {}
        all_times_retrieve = {}
        for n in range(self.num_updates):
            time_parameter_assign, time_submit, time_retrieve = self.clops_cycle(backend, sorted_transpiled_qc_list, n)
            all_times_parameter_assign["update_" + str(n + 1)] = time_parameter_assign
            all_times_submit["update_" + str(n + 1)] = time_submit
            all_times_retrieve["update_" + str(n + 1)] = time_retrieve
        # *********************************************
        # End CLOPS timer
        # *********************************************
        end_clops_timer = perf_counter()

        # COUNT OPERATIONS
        all_op_counts = count_native_gates(backend, [x for y in list(sorted_transpiled_qc_list.values()) for x in y])

        dataset.attrs.update(
            {
                "clops_time": end_clops_timer - start_clops_timer,
                "all_times_parameter_assign": all_times_parameter_assign,
                "all_times_submit": all_times_submit,
                "all_times_retrieve": all_times_retrieve,
                "time_circuit_generate": self.time_circuit_generate,
                "time_transpile": self.time_transpile,
                "time_sort_batches": self.time_sort_batches,
                "parameters_per_update": self.parameters_per_update,
                "job_meta_per_update": self.job_meta_per_update,
                "counts_per_update": self.counts_per_update,
                "operation_counts": all_op_counts,
            }
        )

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        return dataset


class CLOPSConfiguration(BenchmarkConfigurationBase):
    """CLOPS configuration.

    Attributes:
        benchmark (Type[Benchmark]): CLOPS Benchmark.
        qubits (Sequence[int]): The Sequence (List or Tuple) of physical qubit labels in which to run the benchmark.
                            * The physical qubit layout should correspond to the one used to establish QV.
        num_circuits (int): The number of parametrized circuit layouts.
                            * By definition of arXiv:2110.14108 [quant-ph] set to 100.
        num_updates (int): The number of parameter assignment updates per circuit layout to perform.
                            * By definition of arXiv:2110.14108 [quant-ph] set to 10.
        num_shots (int): The number of measurement shots per circuit to perform.
                            * By definition of arXiv:2110.14108 [quant-ph] set to 100.
        clops_h_bool (bool): Whether a CLOPS value with layer definition of CLOPS_H is estimated.
                            * Default is False
                            * This will not estimate a rigorous CLOPS_H value (as loosely defined in www.ibm.com/quantum/blog/quantum-metric-layer-fidelity)
        qiskit_optim_level (int): The Qiskit transpilation optimization level.
                            * The optimization level should correspond to the one used to establish QV.
                            * Default is 3.
        optimize_sqg (bool): Whether Single Qubit Gate Optimization is performed upon transpilation.
                            * The optimize_sqg value should correspond to the one used to establish QV.
                            * Default is True
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): The Qiskit transpilation routing method to use.
                            * The routing_method value should correspond to the one used to establish QV.
                            * Default is "sabre".
        physical_layout (Literal["fixed", "batching"]): Whether the coupling map is restricted to qubits in the input layout or not.
                            - "fixed": Restricts the coupling map to only the specified qubits.
                            - "batching": Considers the full coupling map of the backend and circuit execution is batched per final layout.
                            * Default is "fixed".
    """

    benchmark: Type[Benchmark] = CLOPSBenchmark
    qubits: Sequence[int]
    num_circuits: int = 100
    num_updates: int = 10
    num_shots: int = 100
    clops_h_bool: bool = False
    qiskit_optim_level: int = 3
    optimize_sqg: bool = True
