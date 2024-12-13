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
'Standard' Clifford Randomized Benchmarking.
"""

from time import strftime
from typing import Any, Dict, List, Sequence, Type

import numpy as np
import xarray as xr

from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
    add_counts_to_dataset,
)
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.randomized_benchmarking_common import (
    exponential_rb,
    fit_decay_lmfit,
    generate_all_rb_circuits,
    generate_fixed_depth_parallel_rb_circuits,
    get_survival_probabilities,
    import_native_gate_cliffords,
    lmfit_minimizer,
    plot_rb_decay,
    submit_parallel_rb_job,
    submit_sequential_rb_jobs,
    survival_probabilities_parallel,
    validate_rb_qubits,
)
from iqm.benchmarks.utils import retrieve_all_counts, retrieve_all_job_metadata, timeit, xrvariable_to_counts
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def clifford_rb_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a Clifford RB experiment

    Args:
        run (RunResult): A Clifford RB experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to Clifford RB
    """
    dataset = run.dataset.copy(deep=True)
    observations: list[BenchmarkObservation] = []
    obs_dict = {}
    plots = {}

    is_parallel_execution = dataset.attrs["parallel_execution"]
    qubits_array = dataset.attrs["qubits_array"]
    num_circuit_samples = dataset.attrs["num_circuit_samples"]
    sequence_lengths = dataset.attrs["sequence_lengths"]

    all_noisy_counts: Dict[str, Dict[int, List[Dict[str, int]]]] = {}
    fidelities: Dict[str, Dict[int, List[float]]] = {str(q): {} for q in qubits_array}

    if is_parallel_execution:
        qcvv_logger.info(f"Post-processing parallel RB for qubits {qubits_array}")
        all_noisy_counts[str(qubits_array)] = {}
        for depth in sequence_lengths:
            identifier = f"qubits_{str(qubits_array)}_depth_{str(depth)}"
            all_noisy_counts[str(qubits_array)][depth] = xrvariable_to_counts(dataset, identifier, num_circuit_samples)

            qcvv_logger.info(f"Depth {depth}")

            # Retrieve the marginalized survival probabilities
            all_survival_probabilities = survival_probabilities_parallel(
                qubits_array, all_noisy_counts[str(qubits_array)][depth]
            )

            # The marginalized survival probabilities will be arranged by qubit layouts
            for qubits_str in all_survival_probabilities.keys():
                fidelities[qubits_str][depth] = all_survival_probabilities[qubits_str]
            # Remaining analysis is the same regardless of whether execution was in parallel or sequential
    else:  # sequential
        qcvv_logger.info(f"Post-processing sequential RB for qubits {qubits_array}")

        for q in qubits_array:
            all_noisy_counts[str(q)] = {}
            num_qubits = len(q)
            fidelities[str(q)] = {}
            for depth in sequence_lengths:
                identifier = f"qubits_{str(q)}_depth_{str(depth)}"
                all_noisy_counts[str(q)][depth] = xrvariable_to_counts(dataset, identifier, num_circuit_samples)

                qcvv_logger.info(f"Qubits {q} and depth {depth}")
                fidelities[str(q)][depth] = get_survival_probabilities(num_qubits, all_noisy_counts[str(q)][depth])
                # Remaining analysis is the same regardless of whether execution was in parallel or sequential

    # All remaining (fitting & plotting) is done per qubit layout
    for qubits_idx, qubits in enumerate(qubits_array):
        # Fit decays
        list_of_fidelities = list(fidelities[str(qubits)].values())
        fit_data, fit_parameters = fit_decay_lmfit(exponential_rb, qubits, list_of_fidelities, "clifford")
        rb_fit_results = lmfit_minimizer(fit_parameters, fit_data, sequence_lengths, exponential_rb)

        average_fidelities = {d: np.mean(fidelities[str(qubits)][d]) for d in sequence_lengths}
        stddevs_from_mean = {
            d: np.std(fidelities[str(qubits)][d]) / np.sqrt(num_circuit_samples) for d in sequence_lengths
        }
        popt = {
            "amplitude": rb_fit_results.params["amplitude_1"],
            "offset": rb_fit_results.params["offset_1"],
            "decay_rate": rb_fit_results.params["p_rb"],
        }
        fidelity = rb_fit_results.params["fidelity_per_clifford"]

        processed_results = {
            "avg_gate_fidelity": {"value": fidelity.value, "uncertainty": fidelity.stderr},
        }

        dataset.attrs[qubits_idx].update(
            {
                "decay_rate": {"value": popt["decay_rate"].value, "uncertainty": popt["decay_rate"].stderr},
                "fit_amplitude": {"value": popt["amplitude"].value, "uncertainty": popt["amplitude"].stderr},
                "fit_offset": {"value": popt["offset"].value, "uncertainty": popt["offset"].stderr},
                "fidelities": fidelities[str(qubits)],
                "avg_fidelities_nominal_values": average_fidelities,
                "avg_fidelities_stderr": stddevs_from_mean,
                "fitting_method": str(rb_fit_results.method),
                "num_function_evals": int(rb_fit_results.nfev),
                "data_points": int(rb_fit_results.ndata),
                "num_variables": int(rb_fit_results.nvarys),
                "chi_square": float(rb_fit_results.chisqr),
                "reduced_chi_square": float(rb_fit_results.redchi),
                "Akaike_info_crit": float(rb_fit_results.aic),
                "Bayesian_info_crit": float(rb_fit_results.bic),
            }
        )

        obs_dict.update({qubits_idx: processed_results})
        observations.extend(
            [
                BenchmarkObservation(
                    name=key,
                    identifier=BenchmarkObservationIdentifier(qubits),
                    value=values["value"],
                    uncertainty=values["uncertainty"],
                )
                for key, values in processed_results.items()
            ]
        )

        # Generate individual decay plots
        fig_name, fig = plot_rb_decay("clifford", [qubits], dataset, obs_dict)
        plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


class CliffordRandomizedBenchmarking(Benchmark):
    """Clifford RB estimates the average gate fidelity of the n-qubit Clifford group"""

    analysis_function = staticmethod(clifford_rb_analysis)

    name: str = "clifford_rb"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "CliffordRBConfiguration"):
        """Construct the CliffordRandomizedBenchmark class

        Args:
            backend_arg (IQMBackendBase | str): the backend to execute Clifford RB on
            configuration (CliffordRBConfiguration): The Clifford RB configuration
        """
        super().__init__(backend_arg, configuration)

        # EXPERIMENT
        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.qubits_array = configuration.qubits_array
        self.sequence_lengths = configuration.sequence_lengths
        self.num_circuit_samples = configuration.num_circuit_samples

        self.parallel_execution = configuration.parallel_execution

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

    def add_all_metadata_to_dataset(self, dataset: xr.Dataset):
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
        dataset.attrs["untranspiled_circuits"] = self.untranspiled_circuits
        dataset.attrs["transpiled_circuits"] = self.transpiled_circuits

    # pylint: disable=too-many-statements
    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")
        validate_rb_qubits(self.qubits_array, backend)

        dataset = xr.Dataset()
        self.add_all_metadata_to_dataset(dataset)

        clifford_1q_dict, clifford_2q_dict = import_native_gate_cliffords()

        # Submit jobs for all qubit layouts
        all_rb_jobs: List[Dict[str, Any]] = []
        time_circuit_generation: Dict[str, float] = {}

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

        # Auxiliary dict from str(qubits) to indices
        qubit_idx: Dict[str, Any] = {}

        if self.parallel_execution:
            # Take the whole qubits_array and do RB in parallel on each qubits_array element
            parallel_untranspiled_rb_circuits = {}
            parallel_transpiled_rb_circuits = {}
            qcvv_logger.info(
                f"Executing parallel RB on qubits {self.qubits_array}."
                f" Will generate and submit all {self.num_circuit_samples} Clifford RB circuits"
                f" for each depth {self.sequence_lengths}"
            )

            time_circuit_generation[str(self.qubits_array)] = 0
            # Generate and submit all circuits
            for seq_length in self.sequence_lengths:
                qcvv_logger.info(f"Sequence length {seq_length}")
                (
                    (parallel_untranspiled_rb_circuits[seq_length], parallel_transpiled_rb_circuits[seq_length]),
                    elapsed_time,
                ) = generate_fixed_depth_parallel_rb_circuits(
                    self.qubits_array,
                    clifford_1q_dict,
                    clifford_2q_dict,
                    seq_length,
                    self.num_circuit_samples,
                    backend,
                )
                time_circuit_generation[str(self.qubits_array)] += elapsed_time

                # Submit all
                flat_qubits_array = [x for y in self.qubits_array for x in y]
                sorted_transpiled_qc_list = {tuple(flat_qubits_array): parallel_transpiled_rb_circuits[seq_length]}
                all_rb_jobs.append(
                    submit_parallel_rb_job(
                        backend,
                        self.qubits_array,
                        seq_length,
                        sorted_transpiled_qc_list,
                        self.shots,
                        self.calset_id,
                        self.max_gates_per_batch,
                    )
                )
                qcvv_logger.info(f"Job for sequence length {seq_length} submitted successfully!")

                self.untranspiled_circuits.circuit_groups.append(
                    CircuitGroup(
                        name=f"{str(self.qubits_array)}_length_{seq_length}",
                        circuits=parallel_untranspiled_rb_circuits[seq_length],
                    )
                )
                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(
                        name=f"{str(self.qubits_array)}_length_{seq_length}",
                        circuits=parallel_transpiled_rb_circuits[seq_length],
                    )
                )
            qubit_idx = {str(self.qubits_array): "parallel_all"}
            dataset.attrs["parallel_all"] = {"qubits": self.qubits_array}
            dataset.attrs.update({q_idx: {"qubits": q} for q_idx, q in enumerate(self.qubits_array)})
        else:
            rb_untranspiled_circuits: Dict[str, Dict[int, List[QuantumCircuit]]] = {}
            rb_transpiled_circuits: Dict[str, Dict[int, List[QuantumCircuit]]] = {}
            for qubits_idx, qubits in enumerate(self.qubits_array):
                qubit_idx[str(qubits)] = qubits_idx
                qcvv_logger.info(
                    f"Executing sequential Clifford RB on qubits {qubits}."
                    f" Will generate and submit all {self.num_circuit_samples} Clifford RB circuits"
                    f" for each depth {self.sequence_lengths}"
                )

                num_qubits = len(qubits)
                rb_untranspiled_circuits[str(qubits)] = {}
                rb_transpiled_circuits[str(qubits)] = {}

                if num_qubits == 1:
                    clifford_dict = clifford_1q_dict
                else:
                    clifford_dict = clifford_2q_dict

                (rb_untranspiled_circuits[str(qubits)], rb_transpiled_circuits[str(qubits)]), time_circuit_generation[
                    str(qubits)
                ] = generate_all_rb_circuits(
                    qubits,
                    self.sequence_lengths,
                    clifford_dict,
                    self.num_circuit_samples,
                    backend,
                    interleaved_gate=None,
                )

                # Submit
                all_rb_jobs.extend(
                    submit_sequential_rb_jobs(
                        qubits,
                        rb_transpiled_circuits[str(qubits)],
                        self.shots,
                        self.backend,
                        self.calset_id,
                        max_gates_per_batch=self.max_gates_per_batch,
                    )
                )
                qcvv_logger.info(
                    f"All jobs for qubits {qubits} and sequence lengths {self.sequence_lengths} submitted successfully!"
                )

                self.untranspiled_circuits.circuit_groups.append(
                    CircuitGroup(name=str(qubits), circuits=rb_untranspiled_circuits[str(qubits)])
                )
                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(name=str(qubits), circuits=rb_transpiled_circuits[str(qubits)])
                )

                dataset.attrs[qubits_idx] = {"qubits": qubits}

        # Retrieve counts of jobs for all qubit layouts
        all_job_metadata = {}
        for job_dict in all_rb_jobs:
            qubits = job_dict["qubits"]
            depth = job_dict["depth"]
            # Retrieve counts
            identifier = f"qubits_{str(qubits)}_depth_{str(depth)}"
            execution_results, time_retrieve = retrieve_all_counts(job_dict["jobs"], identifier)
            # Retrieve all job meta data
            all_job_metadata = retrieve_all_job_metadata(job_dict["jobs"])
            # Export all to dataset
            dataset.attrs[qubit_idx[str(qubits)]].update(
                {
                    f"depth_{str(depth)}": {
                        "time_circuit_generation": time_circuit_generation[str(qubits)],
                        "time_submit": job_dict["time_submit"],
                        "time_retrieve": time_retrieve,
                        "all_job_metadata": all_job_metadata,
                    },
                }
            )

            qcvv_logger.info(f"Adding counts of qubits {qubits} and depth {depth} run to the dataset")
            dataset, _ = add_counts_to_dataset(execution_results, identifier, dataset)

        qcvv_logger.info(f"RB experiment concluded !")
        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        return dataset


class CliffordRBConfiguration(BenchmarkConfigurationBase):
    """Clifford RB configuration.

    Attributes:
        benchmark (Type[Benchmark]): CliffordRandomizedBenchmarking
        qubits_array (Sequence[Sequence[int]]): The array of qubits on which to execute the benchmark.
        sequence_lengths (Sequence[int]): The length of Cliffords sequences with which to execute benchmark.
        num_circuit_samples (int): The number of Cliffords circuits per sequence length.
        shots (int): The number of measurement shots per circuit.
        parallel_execution(bool): Whether the benchmark is executed on all qubits in parallel or not.
                            * Default is False.
    """

    benchmark: Type[Benchmark] = CliffordRandomizedBenchmarking
    qubits_array: Sequence[Sequence[int]]
    sequence_lengths: Sequence[int]
    num_circuit_samples: int
    parallel_execution: bool = False
