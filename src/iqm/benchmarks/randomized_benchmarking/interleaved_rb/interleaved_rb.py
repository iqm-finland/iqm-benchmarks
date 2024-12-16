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
Interleaved Clifford Randomized Benchmarking.
"""

from time import strftime
from typing import Any, Dict, List, Literal, Optional, Sequence, Type

from matplotlib.figure import Figure
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
    validate_irb_gate,
    validate_rb_qubits,
)
from iqm.benchmarks.utils import retrieve_all_counts, retrieve_all_job_metadata, timeit, xrvariable_to_counts
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


# pylint: disable=too-many-statements, too-many-branches
def interleaved_rb_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for an Interleaved RB experiment

    Args:
        run (RunResult): An interleaved RB experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to Interleaved RB
    """
    dataset = run.dataset.copy(deep=True)
    obs_dict: Dict[int, Any] = {}
    observations: list[BenchmarkObservation] = []
    plots: Dict[str, Figure] = {}

    is_parallel_execution = dataset.attrs["parallel_execution"]
    qubits_array = dataset.attrs["qubits_array"]
    num_circuit_samples = dataset.attrs["num_circuit_samples"]
    sequence_lengths = dataset.attrs["sequence_lengths"]

    interleaved_gate = dataset.attrs["interleaved_gate"]
    interleaved_gate_parameters = dataset.attrs["interleaved_gate_params"]

    simultaneous_fit = dataset.attrs["simultaneous_fit"]

    all_noisy_counts: Dict[str, Dict[str, Dict[int, List[Dict[str, int]]]]] = {}
    fidelities: Dict[str, Dict[str, Dict[int, List[float]]]] = {
        str(q): {rb_type: {} for rb_type in ["clifford", "interleaved"]} for q in qubits_array
    }

    if is_parallel_execution:
        qcvv_logger.info(f"Post-processing parallel Interleaved RB for qubits {qubits_array}")
        all_noisy_counts[str(qubits_array)] = {}
        for rb_type in ["clifford", "interleaved"]:
            all_noisy_counts[str(qubits_array)][rb_type] = {}
            for depth in sequence_lengths:
                identifier = f"{rb_type}_qubits_{str(qubits_array)}_depth_{str(depth)}"
                all_noisy_counts[str(qubits_array)][rb_type][depth] = xrvariable_to_counts(
                    dataset, identifier, num_circuit_samples
                )

                # Retrieve the marginalized survival probabilities
                all_survival_probabilities = survival_probabilities_parallel(
                    qubits_array, all_noisy_counts[str(qubits_array)][rb_type][depth]
                )

                # The marginalized survival probabilities will be arranged by qubit layouts
                for qubits_str in all_survival_probabilities.keys():
                    fidelities[qubits_str][rb_type][depth] = all_survival_probabilities[qubits_str]
                # Remaining analysis is the same regardless of whether execution was in parallel or sequential
            qcvv_logger.info(f"Metrics for {rb_type.capitalize()} estimated successfully!")
    else:  # sequential
        qcvv_logger.info(f"Post-processing sequential Interleaved RB for qubits {qubits_array}")

        for q in qubits_array:
            all_noisy_counts[str(q)] = {}
            num_qubits = len(q)
            fidelities[str(q)] = {}
            for rb_type in ["clifford", "interleaved"]:
                all_noisy_counts[str(q)][rb_type] = {}
                fidelities[str(q)][rb_type] = {}
                for depth in sequence_lengths:
                    identifier = f"{rb_type}_qubits_{str(q)}_depth_{str(depth)}"
                    all_noisy_counts[str(q)][rb_type][depth] = xrvariable_to_counts(
                        dataset, identifier, num_circuit_samples
                    )

                    qcvv_logger.info(f"Now on {rb_type.capitalize()} RB with qubits {q} and depth {depth}")
                    fidelities[str(q)][rb_type][depth] = get_survival_probabilities(
                        num_qubits, all_noisy_counts[str(q)][rb_type][depth]
                    )
                    # Remaining analysis is the same regardless of whether execution was in parallel or sequential

    # All remaining (fitting & plotting) is done per qubit layout
    for qubits_idx, qubits in enumerate(qubits_array):
        dataset.attrs[str(qubits)] = {}
        # Fit decays simultaneously
        list_of_fidelities_clifford = list(fidelities[str(qubits)]["clifford"].values())
        list_of_fidelities_interleaved = list(fidelities[str(qubits)]["interleaved"].values())
        fit_data, fit_parameters = fit_decay_lmfit(
            exponential_rb,
            qubits,
            [list_of_fidelities_clifford, list_of_fidelities_interleaved],
            "interleaved",
            simultaneous_fit,
        )
        rb_fit_results = lmfit_minimizer(fit_parameters, fit_data, sequence_lengths, exponential_rb)

        processed_results = {}
        for rb_type in ["interleaved", "clifford"]:
            average_fidelities = {d: np.mean(fidelities[str(qubits)][rb_type][d]) for d in sequence_lengths}
            stddevs_from_mean = {
                d: np.std(fidelities[str(qubits)][rb_type][d]) / np.sqrt(num_circuit_samples) for d in sequence_lengths
            }
            popt = {
                "amplitude": (
                    rb_fit_results.params["amplitude_1"]
                    if rb_type == "clifford"
                    else rb_fit_results.params["amplitude_2"]
                ),
                "offset": (
                    rb_fit_results.params["offset_1"] if rb_type == "clifford" else rb_fit_results.params["offset_2"]
                ),
                "decay_rate": (
                    rb_fit_results.params["p_rb"] if rb_type == "clifford" else rb_fit_results.params["p_irb"]
                ),
            }
            fidelity = (
                rb_fit_results.params["fidelity_per_clifford"]
                if rb_type == "clifford"
                else rb_fit_results.params["interleaved_fidelity"]
            )

            processed_results[rb_type] = {
                "avg_gate_fidelity": {"value": fidelity.value, "uncertainty": fidelity.stderr},
            }

            observations.extend(
                [
                    BenchmarkObservation(
                        name=f"{key}_{rb_type}",
                        identifier=BenchmarkObservationIdentifier(qubits),
                        value=values["value"],
                        uncertainty=values["uncertainty"],
                    )
                    for key, values in processed_results[rb_type].items()
                ]
            )

            dataset.attrs[qubits_idx].update(
                {
                    rb_type: {
                        "decay_rate": {"value": popt["decay_rate"].value, "uncertainty": popt["decay_rate"].stderr},
                        "fit_amplitude": {"value": popt["amplitude"].value, "uncertainty": popt["amplitude"].stderr},
                        "fit_offset": {"value": popt["offset"].value, "uncertainty": popt["offset"].stderr},
                        "fidelities": fidelities[str(qubits)][rb_type],
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
                }
            )

        obs_dict.update({qubits_idx: processed_results})

        # Generate decay plots
        if interleaved_gate_parameters is None:
            interleaved_gate_string = f"{interleaved_gate}"
        else:
            params_string = str(tuple(f"{x:.2f}" for x in interleaved_gate_parameters))
            interleaved_gate_string = f"{interleaved_gate}{params_string}"

        fig_name, fig = plot_rb_decay(
            "irb",
            [qubits],
            dataset,
            obs_dict,
            interleaved_gate=interleaved_gate_string,
        )
        plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


class InterleavedRandomizedBenchmarking(Benchmark):
    """Interleaved RB estimates the average gate fidelity of a specific Clifford gate"""

    analysis_function = staticmethod(interleaved_rb_analysis)

    name: str = "interleaved_clifford_rb"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "InterleavedRBConfiguration"):
        """Construct the InterleavedRandomizedBenchmark class

        Args:
            backend_arg (IQMBackendBase | str): the backend to execute Clifford RB on
            configuration (InterleavedRBConfiguration): The Clifford RB configuration
        """
        super().__init__(backend_arg, configuration)

        # EXPERIMENT
        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.qubits_array = configuration.qubits_array
        self.sequence_lengths = configuration.sequence_lengths
        self.num_circuit_samples = configuration.num_circuit_samples

        self.parallel_execution = configuration.parallel_execution

        self.interleaved_gate = configuration.interleaved_gate
        self.interleaved_gate_params = configuration.interleaved_gate_params
        self.simultaneous_fit = configuration.simultaneous_fit

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
        qcvv_logger.info("Adding all circuits to the dataset")
        dataset.attrs["untranspiled_circuits"] = self.untranspiled_circuits
        dataset.attrs["transpiled_circuits"] = self.transpiled_circuits

    # pylint: disable=too-many-statements
    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")
        validate_rb_qubits(self.qubits_array, backend)

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        clifford_1q_dict, clifford_2q_dict = import_native_gate_cliffords()

        # Submit jobs for all qubit layouts
        all_rb_jobs: Dict[str, List[Dict[str, Any]]] = {}  # Label by Clifford or Interleaved
        time_circuit_generation: Dict[str, float] = {}

        # Initialize the variable to contain the circuits for each layout

        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")
        # Validate and get interleaved gate as a QC
        interleaved_gate_qc = validate_irb_gate(
            self.interleaved_gate, backend, gate_params=self.interleaved_gate_params
        )

        # Auxiliary dict from str(qubits) to indices
        qubit_idx: Dict[str, Any] = {}

        if self.parallel_execution:
            all_rb_jobs["clifford"] = []
            all_rb_jobs["interleaved"] = []
            # Take the whole qubits_array and do RB in parallel on each qubits_array element
            parallel_untranspiled_rb_circuits = {}
            parallel_transpiled_rb_circuits = {}
            parallel_untranspiled_interleaved_rb_circuits = {}
            parallel_transpiled_interleaved_rb_circuits = {}
            qcvv_logger.info(
                f"Now executing parallel Interleaved RB on qubits {self.qubits_array}."
                f" Will generate and submit all {self.num_circuit_samples} Interleaved and Clifford RB circuits"
                f" for each depth {self.sequence_lengths}."
            )

            time_circuit_generation[str(self.qubits_array)] = 0

            # Generate and submit all circuits
            for seq_length in self.sequence_lengths:
                # There are different ways of dealing with submission here:
                # We'll generate Clifford circuits, then Interleaved circuits, for fixed depth
                # Then we'll submit separate jobs but one right after the other, Clifford first
                # Do for all sequence depths

                qcvv_logger.info(f"Generating Clifford RB circuits of sequence length {seq_length}")
                (
                    (parallel_untranspiled_rb_circuits[seq_length], parallel_transpiled_rb_circuits[seq_length]),
                    elapsed_time_untranspiled,
                ) = generate_fixed_depth_parallel_rb_circuits(
                    self.qubits_array,
                    clifford_1q_dict,
                    clifford_2q_dict,
                    seq_length,
                    self.num_circuit_samples,
                    backend,
                    interleaved_gate=None,
                )
                qcvv_logger.info(f"Generating Interleaved RB circuits of sequence length {seq_length}")

                (
                    (
                        parallel_untranspiled_interleaved_rb_circuits[seq_length],
                        parallel_transpiled_interleaved_rb_circuits[seq_length],
                    ),
                    elapsed_time_transpiled,
                ) = generate_fixed_depth_parallel_rb_circuits(
                    self.qubits_array,
                    clifford_1q_dict,
                    clifford_2q_dict,
                    seq_length,
                    self.num_circuit_samples,
                    backend,
                    interleaved_gate=interleaved_gate_qc,
                )

                time_circuit_generation[str(self.qubits_array)] += elapsed_time_untranspiled + elapsed_time_transpiled

                # Submit all
                flat_qubits_array = [x for y in self.qubits_array for x in y]
                sorted_transpiled_rb_qc_list = {tuple(flat_qubits_array): parallel_transpiled_rb_circuits[seq_length]}
                sorted_transpiled_interleaved_rb_qc_list = {
                    tuple(flat_qubits_array): parallel_transpiled_interleaved_rb_circuits[seq_length]
                }
                all_rb_jobs["clifford"].append(
                    submit_parallel_rb_job(
                        backend,
                        self.qubits_array,
                        seq_length,
                        sorted_transpiled_rb_qc_list,
                        self.shots,
                        self.calset_id,
                        self.max_gates_per_batch,
                    )
                )
                all_rb_jobs["interleaved"].append(
                    submit_parallel_rb_job(
                        backend,
                        self.qubits_array,
                        seq_length,
                        sorted_transpiled_interleaved_rb_qc_list,
                        self.shots,
                        self.calset_id,
                        self.max_gates_per_batch,
                    )
                )
                qcvv_logger.info(f"Both jobs for sequence length {seq_length} submitted successfully!")

            self.untranspiled_circuits.circuit_groups.append(
                CircuitGroup(
                    name=str(self.qubits_array),
                    circuits=[parallel_untranspiled_rb_circuits[m] for m in self.sequence_lengths],
                )
            )
            self.transpiled_circuits.circuit_groups.append(
                CircuitGroup(
                    name=str(self.qubits_array),
                    circuits=[parallel_transpiled_rb_circuits[m] for m in self.sequence_lengths],
                )
            )
            self.untranspiled_circuits.circuit_groups.append(
                CircuitGroup(
                    name=f"{str(self.qubits_array)}_interleaved",
                    circuits=[parallel_untranspiled_interleaved_rb_circuits[m] for m in self.sequence_lengths],
                )
            )
            self.transpiled_circuits.circuit_groups.append(
                CircuitGroup(
                    name=f"{str(self.qubits_array)}_interleaved",
                    circuits=[parallel_transpiled_interleaved_rb_circuits[m] for m in self.sequence_lengths],
                )
            )

            qubit_idx = {str(self.qubits_array): "parallel_all"}
            dataset.attrs["parallel_all"] = {"qubits": self.qubits_array}
            dataset.attrs.update({q_idx: {"qubits": q} for q_idx, q in enumerate(self.qubits_array)})
        else:
            rb_untranspiled_circuits: Dict[str, Dict[int, List[QuantumCircuit]]] = {}
            rb_transpiled_circuits: Dict[str, Dict[int, List[QuantumCircuit]]] = {}
            rb_untranspiled_interleaved_circuits: Dict[str, Dict[int, List[QuantumCircuit]]] = {}
            rb_transpiled_interleaved_circuits: Dict[str, Dict[int, List[QuantumCircuit]]] = {}

            all_rb_jobs["clifford"] = []
            all_rb_jobs["interleaved"] = []

            for qubits_idx, qubits in enumerate(self.qubits_array):
                qubit_idx[str(qubits)] = qubits_idx
                qcvv_logger.info(
                    f"Wxecuting sequential Clifford and Interleaved RB circuits on qubits {qubits}."
                    f" Will generate and submit all {self.num_circuit_samples} Clifford RB circuits"
                    f" for each depth {self.sequence_lengths}"
                )
                num_qubits = len(qubits)
                rb_untranspiled_circuits[str(qubits)] = {}
                rb_transpiled_circuits[str(qubits)] = {}
                rb_untranspiled_interleaved_circuits[str(qubits)] = {}
                rb_transpiled_interleaved_circuits[str(qubits)] = {}

                if num_qubits == 1:
                    clifford_dict = clifford_1q_dict
                else:
                    clifford_dict = clifford_2q_dict

                ((rb_untranspiled_circuits[str(qubits)], rb_transpiled_circuits[str(qubits)]), t_clifford) = (
                    generate_all_rb_circuits(
                        qubits,
                        self.sequence_lengths,
                        clifford_dict,
                        self.num_circuit_samples,
                        backend,
                        interleaved_gate=None,
                    )
                )
                (
                    (
                        rb_untranspiled_interleaved_circuits[str(qubits)],
                        rb_transpiled_interleaved_circuits[str(qubits)],
                    ),
                    t_inter,
                ) = generate_all_rb_circuits(
                    qubits,
                    self.sequence_lengths,
                    clifford_dict,
                    self.num_circuit_samples,
                    backend,
                    interleaved_gate=interleaved_gate_qc,
                )

                time_circuit_generation[str(qubits)] = t_clifford + t_inter

                # Submit Clifford then Interleaved
                all_rb_jobs["clifford"].extend(
                    submit_sequential_rb_jobs(
                        qubits,
                        rb_transpiled_circuits[str(qubits)],
                        self.shots,
                        backend,
                        self.calset_id,
                        max_gates_per_batch=self.max_gates_per_batch,
                    )
                )
                all_rb_jobs["interleaved"].extend(
                    submit_sequential_rb_jobs(
                        qubits,
                        rb_transpiled_interleaved_circuits[str(qubits)],
                        self.shots,
                        backend,
                        self.calset_id,
                        max_gates_per_batch=self.max_gates_per_batch,
                    )
                )
                qcvv_logger.info(
                    f"All jobs for qubits {qubits} and sequence lengths {self.sequence_lengths} submitted successfully!"
                )

                self.untranspiled_circuits.circuit_groups.append(
                    CircuitGroup(name=str(qubits), circuits=[rb_untranspiled_circuits[str(qubits)]])
                )
                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(name=str(qubits), circuits=[rb_transpiled_circuits[str(qubits)]])
                )

                self.untranspiled_circuits.circuit_groups.append(
                    CircuitGroup(
                        name=f"{str(qubits)}_interleaved", circuits=[rb_untranspiled_interleaved_circuits[str(qubits)]]
                    )
                )
                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(
                        name=f"{str(qubits)}_interleaved",
                        # name=str(qubits),
                        circuits=[rb_transpiled_interleaved_circuits[str(qubits)]],
                    )
                )

                dataset.attrs[qubits_idx] = {"qubits": qubits}

        # Retrieve counts of jobs for all qubit layouts
        all_job_metadata = {}
        for rb_type in ["clifford", "interleaved"]:
            for job_dict in all_rb_jobs[rb_type]:
                qubits = job_dict["qubits"]
                depth = job_dict["depth"]
                # Retrieve counts
                identifier = f"{rb_type}_qubits_{str(qubits)}_depth_{str(depth)}"
                execution_results, time_retrieve = retrieve_all_counts(job_dict["jobs"], identifier)
                # Retrieve all job meta data
                all_job_metadata = retrieve_all_job_metadata(job_dict["jobs"])
                # Export all to dataset
                dataset.attrs[qubit_idx[str(qubits)]].update(
                    {
                        rb_type: {
                            f"depth_{str(depth)}": {
                                "time_circuit_generation": time_circuit_generation[str(qubits)],
                                "time_submit": job_dict["time_submit"],
                                "time_retrieve": time_retrieve,
                                "all_job_metadata": all_job_metadata,
                            },
                        }
                    }
                )

                qcvv_logger.info(f"Adding counts of qubits {qubits} and depth {depth} run to the dataset")
                dataset, _ = add_counts_to_dataset(execution_results, identifier, dataset)

        qcvv_logger.info(f"Interleaved RB experiment concluded !")
        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        return dataset


class InterleavedRBConfiguration(BenchmarkConfigurationBase):
    """Interleaved RB configuration.
    Attributes:
        benchmark (Type[Benchmark]): InterleavedRandomizedBenchmarking.
        qubits_array (Sequence[Sequence[int]]): The array of physical qubit labels with which to execute IRB.
        sequence_lengths (Sequence[int]): The length of each random Clifford sequence.
        num_circuit_samples (int): The number of circuit samples to generate.
        shots (int): The number of measurement shots with which to execute each circuit sample.
        parallel_execution(bool): Whether the benchmark is executed on all qubits in parallel or not.
                            * Default is False.
        interleaved_gate (str): The name of the gate to interleave.
                            * Should be specified as a qiskit circuit library gate name, e.g., "YGate" or "CZGate".
        interleaved_gate_params (Optional[Sequence[float]]): Any optional parameters entering the gate.
                            * Default is None.
        simultaneous_fit (Sequence[Literal["amplitude", "offset"]]): Optional parameters to fit simultaneously.
                            * Default is ["amplitude", "offset"].
    """

    benchmark: Type[Benchmark] = InterleavedRandomizedBenchmarking
    qubits_array: Sequence[Sequence[int]]
    sequence_lengths: Sequence[int]
    num_circuit_samples: int
    parallel_execution: bool = False
    interleaved_gate: str
    interleaved_gate_params: Optional[Sequence[float]] = None
    simultaneous_fit: Sequence[Literal["amplitude", "offset"]] = ["amplitude", "offset"]
