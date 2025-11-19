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
Updated CLOPS_h benchmark based on logic of the implementation in the qiskit-device-benchmarking package
"""

from datetime import datetime
from math import floor, pi
from time import perf_counter, strftime
from typing import Any, Dict, List, Sequence, Tuple, Type

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit import ParameterVector
import xarray as xr
from iqm.qiskit_iqm.iqm_job import IQMJob


from iqm.benchmarks import Benchmark
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import (
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
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
from iqm.benchmarks.quantum_volume.clops import plot_times, retrieve_clops_elapsed_times
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def clops_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a CLOPS (v or h) experiment

    Args:
        run (RunResult): A CLOPS experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to CLOPS
    """
    plots: Dict[str, Any] = {}
    obs_dict = {}
    dataset = run.dataset

    # Retrieve dataset values
    qubits = dataset.attrs["qubits"]
    num_circuits = dataset.attrs["num_circuits"]
    num_updates = dataset.attrs["num_updates"]
    num_shots = dataset.attrs["num_shots"]
    depth = dataset.attrs["depth"]

    all_job_meta = dataset.attrs["job_meta_per_update"]

    all_times_parameter_assign = dataset.attrs["all_times_parameter_assign"]
    all_times_submit = dataset.attrs["all_times_submit"]
    all_times_retrieve = dataset.attrs["all_times_retrieve"]

    # Get all execution elapsed times for plot
    overall_elapsed = retrieve_clops_elapsed_times(all_job_meta)  # will be {} if backend is a simulator
    job_time_format = "%Y-%m-%dT%H:%M:%S.%f%z"

    # Determine total number of jobs across all updates
    total_jobs = sum(len(update_meta) for update_meta in all_job_meta.values())

    # Determine start time: skip first job only if there are multiple jobs
    if total_jobs > 1:
        clops_start_meta = all_job_meta["update_1"]["batch_job_2"] if len(all_job_meta["update_1"]) > 1 else all_job_meta["update_2"]["batch_job_1"]
    else:
        clops_start_meta = all_job_meta["update_1"]["batch_job_1"]

    clops_start = datetime.strptime(clops_start_meta["timestamps"]["execution_started"], job_time_format)
    last_update = f"update_{num_updates}"
    clops_end_meta = all_job_meta[last_update][list(all_job_meta[last_update].keys())[-1]]
    clops_end = datetime.strptime(clops_end_meta["timestamps"]["ready"], job_time_format)
    clops_time = (clops_end - clops_start).total_seconds()

    dataset.attrs["clops_time"] = clops_time
    if overall_elapsed:
        qcvv_logger.info("Total elapsed times from job execution metadata:")
        for k in overall_elapsed.keys():
            dataset.attrs[k] = overall_elapsed[k]
            if overall_elapsed[k] > 60.0:
                qcvv_logger.info(f'\t"{k}": {overall_elapsed[k] / 60.0:.2f} min')
            else:
                qcvv_logger.info(f'\t"{k}": {overall_elapsed[k]:.2f} sec')
    else:
        qcvv_logger.info("There is no elapsed-time data associated to jobs (e.g., execution on simulator)")

    # Get circuit counts of updates
    total_circuits = 0
    update_1_circuits = 0  # circuits in the first job, to be disregarded unless it's the only job
    for update, update_meta in all_job_meta.items():
        for job_name, batch_job_meta in update_meta.items():
            num_circuits_in_job = batch_job_meta["circuits_in_batch"]
            total_circuits += num_circuits_in_job
            if update == "update_1" and job_name == "batch_job_1":
                update_1_circuits += num_circuits_in_job


    # CLOPS_h: only disregard first job circuits if there are multiple jobs
    circuits_to_count = total_circuits if total_jobs == 1 else (total_circuits - update_1_circuits)
    clops_h: float = circuits_to_count * num_shots * depth / clops_time

    # Sort the final dataset
    dataset.attrs = dict(sorted(dataset.attrs.items()))


    processed_results = {
        "clops_h": {"value": int(clops_h), "uncertainty": np.NaN},
    }

    dataset.attrs.update(
        {
            "assign_parameters_total": sum(all_times_parameter_assign.values()),
            "user_submit_total": sum(all_times_submit.values()),
            "user_retrieve_total": sum(all_times_retrieve.values()),
            "clops_time": clops_time,
        }
    )

    # UPDATE OBSERVATIONS
    obs_dict.update({1: processed_results})

    if overall_elapsed:
        fig_name, fig = plot_times(dataset, obs_dict)
        plots[fig_name] = fig

    observations = [
        BenchmarkObservation(name="clops_h", value=int(clops_h), identifier=BenchmarkObservationIdentifier(qubits)),
    ]

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


class CLOPSHBenchmark(Benchmark):
    """
    CLOPS_H reflects the speed of execution for native layers of entangling gates and parameterized single-qubit gates.
    """

    analysis_function = staticmethod(clops_analysis)

    name: str = "clops_h"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "CLOPSHConfiguration"):
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
        self.depth = configuration.num_layers
        self.entangling_gate = configuration.entangling_gate

        self.num_circuits = configuration.num_circuits
        self.num_shots = configuration.num_shots

        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg

        # POST-EXPERIMENT AND VARIABLES TO STORE
        self.parameters_per_update: Dict[str, List[float]] = {}
        self.counts: List[IQMJob] = []
        self.job_meta_per_update: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self.time_circuit_generate: float = 0.0
        self.num_updates: int = 0

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



    def generate_single_circuit(self) -> QuantumCircuit:
        """Generate a single parametrized circuit, consisting of alternating layers of native entangling gates and
        parametrized single qubit gates, with measurements at the end.

        Returns:
            QuantumCircuit: the QV quantum circuit.
        """
        qc = QuantumCircuit(self.num_qubits)

        qubits = qc.qubits

        for layer in range(self.depth):
            # 2 qubit gate layer
            reduced_coupling_map = set_coupling_map(self.qubits, self.backend)
            available_edges = set(reduced_coupling_map)
            while len(available_edges) > 0:
                edge = list(available_edges)[np.random.choice(len(available_edges))]
                available_edges.remove(edge)
                edges_to_delete = []
                for ce in list(available_edges):
                    if (edge[0] in ce) or (edge[1] in ce):
                        edges_to_delete.append(ce)
                available_edges.difference_update(set(edges_to_delete))
                if self.entangling_gate in self.backend.architecture.gates:
                    qc.cz(*edge)
                else:
                    raise ValueError(f"{self.entangling_gate} is not natively supported by the backend. Supported gates are {self.backend.architecture.gates}.")
                if "move" in self.backend.architecture.gates:
                    break # In star architectures entangling use move-cz-move sequences which are not parallelizable
            qc.barrier()
            # 1 qubit layer
            for q in qubits:
                angles = [np.random.uniform(0, np.pi * 2) for _ in range(3)]
                # qc.rz(angles[0], q)
                # qc.x(q)
                # qc.rz(angles[1], q)
                # qc.x(q)
                # qc.rz(angles[2], q)

                qc.rz(angles[0],q)
                qc.r(angles[1], 0,q)
                qc.rz(angles[2],q)
            qc.barrier()
        qc.measure_all()

        return qc

    @timeit
    def generate_circuit_list(
            self, n_circuits: int
    ) -> List[QuantumCircuit]:
        """Generate a list of parametrized QV quantum circuits, with measurements at the end.

        Returns:
            List[QuantumCircuit]: the list of parametrized QV quantum circuits.
        """
        qc_list = [self.generate_single_circuit() for _ in range(n_circuits)]
        self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=f"{self.qubits}", circuits=qc_list))
        return qc_list


    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y-%m-%d_%H:%M:%S")

        self.circuits = Circuits()
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        if self.num_circuits != 1000 or self.depth != 100 or self.num_shots != 100:
            qcvv_logger.info(
                f"NB: CLOPS parameters, by definition, are [num_circuits=1000, num_layers=100, num_shots=100]"
                f" You chose"
                f" [num_circuits={self.num_circuits}, num_layers={self.depth}, num_shots={self.num_shots}]."
            )

        all_times_submit = {}
        all_times_retrieve = {}
        all_jobs = []
        all_untranspiled_circuits = []
        all_transpiled_circuits = []
        total_time_circuit_generate = 0.0
        total_time_transpile = 0.0

        n_batches = int(np.ceil(self.num_circuits / self.configuration.max_circuits_per_batch))
        self.num_updates = n_batches

        for update in range(n_batches):
            circuits_in_batch = min(
                self.configuration.max_circuits_per_batch,
                self.num_circuits - update * self.configuration.max_circuits_per_batch
            )

            qcvv_logger.info(
                f"Generating {circuits_in_batch} parametrized circuit templates for batch {update + 1} / {n_batches}",
            )
            qc_list, time_circuit_generate = self.generate_circuit_list(circuits_in_batch)
            all_untranspiled_circuits.extend(qc_list)
            total_time_circuit_generate += time_circuit_generate

            qcvv_logger.info(
                f"Transpiling {circuits_in_batch} circuits on qubits {self.qubits} for batch {update + 1} / {n_batches}",
            )

            transpiled_qc_list, time_transpile = perform_backend_transpilation(
                qc_list,
                self.backend,
                self.qubits,
                coupling_map=self.backend.coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=False, # Necessary to be False, otherwise single qubit gates are optimized away
                routing_method=self.routing_method,
            )
            all_transpiled_circuits.extend(transpiled_qc_list)
            total_time_transpile += time_transpile

            qcvv_logger.info(f"Submitting batch {update + 1} / {n_batches} with {len(transpiled_qc_list)} circuits")

            batch_jobs, time_submit = submit_execute(
                {tuple(self.qubits): transpiled_qc_list},
                backend,
                self.num_shots,
                self.calset_id,
                max_gates_per_batch=self.max_gates_per_batch,
                circuit_compilation_options=self.circuit_compilation_options,
            )
            all_jobs.append(batch_jobs)
            all_times_submit[f"update_{update + 1}"] = time_submit

        self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=f"{self.qubits}", circuits=all_untranspiled_circuits))
        self.transpiled_circuits.circuit_groups.append(CircuitGroup(name=f"{self.qubits}", circuits=all_transpiled_circuits))
        self.time_circuit_generate = total_time_circuit_generate

        qcvv_logger.info(f"Retrieving counts")
        for update in range(n_batches):
            # Retrieve counts - the precise outputs do not matter
            all_counts, time_retrieve = retrieve_all_counts(all_jobs[update])
            # Save counts - ensures counts were received and can be inspected
            self.counts = all_counts
            # Retrieve and save all job metadata
            update_job_metadata = retrieve_all_job_metadata(all_jobs[update])
            self.job_meta_per_update["update_" + str(update + 1)] = update_job_metadata
            all_times_retrieve[f"update_{update + 1}"] = time_retrieve

        # COUNT OPERATIONS
        all_op_counts = count_native_gates(backend, all_transpiled_circuits)

        dataset.attrs.update(
            {
                "time_circuit_generate": total_time_circuit_generate,
                "time_transpile": total_time_transpile,
                "job_meta_per_update": self.job_meta_per_update,
                "operation_counts": all_op_counts,
                "all_times_submit": all_times_submit,
                "all_times_retrieve": all_times_retrieve,
                "num_updates": self.num_updates,
                "all_times_parameter_assign": {},
            }
        )

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        return dataset


class CLOPSHConfiguration(BenchmarkConfigurationBase):
    """CLOPS configuration.
    Attributes:
        benchmark (Type[Benchmark]): CLOPS Benchmark.
        qubits (Sequence[int]): The Sequence (List or Tuple) of physical qubit labels in which to run the benchmark.
                            * The physical qubit layout should correspond to the one used to establish QV.
        num_circuits (int): The number of parametrized circuit layouts.
                            * Default is 1000.
        num_shots (int): The number of measurement shots per circuit to perform.
                            * By definition set to 100.
        num_layers (int): The number of layers in each circuit.
                            * By definition set to 100.
        qiskit_optim_level (int): The Qiskit transpilation optimization level.
                            * The optimization level should correspond to the one used to establish QV.
                            * Default is 3.
        optimize_sqg (bool): Whether Single Qubit Gate Optimization is performed upon transpilation.
                            * The optimize_sqg value should correspond to the one used to establish QV.
                            * Default is True
        entangling_gate (str): The entangling gate to use for the 2-qubit layers. It must be natively supported by the backend.
                            * Default is "cz".
        max_circuits_per_batch (int): Maximum number of circuits to submit in a single batch.
                            * Default is 100, but this depends on the backend restrictions and can be adjusted accordingly.
    """

    benchmark: Type[Benchmark] = CLOPSHBenchmark
    qubits: Sequence[int]
    num_circuits: int = 1000
    num_shots: int = 100
    num_layers: int = 100
    qiskit_optim_level: int = 0
    optimize_sqg: bool = False
    entangling_gate: str = "cz"
    max_circuits_per_batch: int = 100
