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
Quantum Volume benchmark
"""

from copy import deepcopy
from time import strftime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mthree.classes import QuasiCollection
from mthree.utils import expval
import numpy as np
from qiskit.circuit.library import QuantumVolume
from qiskit_aer import Aer
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

# import iqm.diqe.executors.dynamical_decoupling.dd_high_level as dd
# from iqm.diqe.executors.dynamical_decoupling.dynamical_decoupling_core import DDStrategy
# from iqm.diqe.mapomatic import evaluate_costs, get_calibration_fidelities, get_circuit, matching_layouts
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.readout_mitigation import apply_readout_error_mitigation
from iqm.benchmarks.utils import (  # execute_with_dd,
    count_native_gates,
    perform_backend_transpilation,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    set_coupling_map,
    sort_batches_by_final_layout,
    submit_execute,
    timeit,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def compute_heavy_output_probabilities(
    execution_results: List[Dict[str, int]],
    ideal_heavy_outputs: List[Dict[str, float]],
) -> List[float]:
    """Compute the HOP of all quantum circuits.

    Args:
        execution_results (List[Dict[str, int]]): counts from execution of all quantum circuits.
        ideal_heavy_outputs (List[Dict[str, float]]): list of ideal heavy output dictionaries.

    Returns:
        List[float]: the HOP of all quantum circuits.
    """
    qv_result = []
    for result, heavy in zip(execution_results, ideal_heavy_outputs):
        qv_result += [expval(result, heavy)]

    return qv_result


def cumulative_hop(hops: List[float]) -> List[float]:
    """
    Computes the cumulative average heavy output probabilities of a QV experiment.
    Args:
        hops (List[float]): The individual heavy output probabilities for each trial.
    Returns:
        List[float]: cumulative average heavy output probabilities for all trials.
    """
    c_h: List[float] = [np.mean(hops[: (i + 1)], dtype=float) for i in range(len(hops))]
    return c_h


def cumulative_std(hops: List[float]) -> List[float]:
    """Computes the cumulative standard deviation heavy output probabilities of a QV experiment.

    Args:
        hops (List[float]): The individual heavy output probabilities for each trial.
    Returns:
        List[float]: cumulative standard deviation heavy output probabilities for all trials.
    """
    c_h = cumulative_hop(hops)
    c_s = [(c_h[i] * ((1 - c_h[i]) / len(hops[: (i + 1)]))) ** 0.5 for i in range(len(hops))]
    return c_s


def get_ideal_heavy_outputs(
    qc_list: List[QuantumCircuit],
    sorted_qc_list_indices: Dict[Tuple[int, ...], List[int]],
) -> List[Dict[str, float]]:
    """Calculate the heavy output bitrstrings of a list of quantum circuits.

    Args:
        qc_list (List[QuantumCircuit]): the list of quantum circuits.
        sorted_qc_list_indices (Dict[Tuple, List[int]]): dictionary of indices (integers) corresponding to those in the original (untranspiled) list of circuits, with keys being final physical qubit measurements
    Returns:
        List[Dict[str, float]]: the list of heavy output dictionaries of each of the quantum circuits.
    """
    simulable_circuits = deepcopy(qc_list)
    ideal_heavy_outputs: List[Dict[str, float]] = []
    ideal_simulator = Aer.get_backend("statevector_simulator")

    # Separate according to sorted indices
    circuit_batches = {
        k: [simulable_circuits[i] for i in sorted_qc_list_indices[k]] for k in sorted_qc_list_indices.keys()
    }

    for k in sorted(
        circuit_batches.keys(),
        key=lambda x: len(circuit_batches[x]),
        reverse=True,
    ):
        for qc in circuit_batches[k]:
            qc.remove_final_measurements()
            ideal_probabilities = ideal_simulator.run(qc).result().get_counts()
            heavy_projectors = heavy_projector(ideal_probabilities)
            if isinstance(heavy_projectors, list):
                ideal_heavy_outputs.extend(heavy_projectors)
            elif isinstance(heavy_projectors, dict):
                ideal_heavy_outputs.append(heavy_projectors)

    return ideal_heavy_outputs


def get_rem_hops(
    all_rem_quasidistro: List[List[QuasiCollection]], ideal_heavy_outputs: List[Dict[str, float]]
) -> List[float]:
    """Computes readout-error-mitigated heavy output probabilities.

    Args:
        all_rem_quasidistro (List[List[QuasiCollection]]): The list of lists of quasiprobability distributions.
        ideal_heavy_outputs (List[Dict[str, float]]): A list of the noiseless heavy output probability dictionaries.

    Returns:
        List[float]: A list of readout-error-mitigated heavy output probabilities.
    """
    qv_result_rem = []
    for rem_quasidistro, heavy in zip(all_rem_quasidistro, ideal_heavy_outputs):
        qv_result_rem += [expval(rem_quasidistro, heavy)]
    return qv_result_rem


def heavy_projector(probabilities: Dict[str, float]) -> Dict[str, float]:
    """
        Project (select) the samples from a given probability distribution onto heavy outputs.

    Args:
        probabilities (Dict[str, float]): A dictionary of bitstrings and associated probabilities.

    Returns:
        Dict[str, float]: The dictionary of heavy output bitstrings, all with weight 1.
    """
    median_prob = np.median(list(probabilities.values()))
    heavy_outputs = {k: 1.0 for k, v in probabilities.items() if v > median_prob}
    return heavy_outputs


def is_successful(
    heavy_output_probabilities: List[float],
    num_sigmas: int = 2,
) -> bool:
    """Check whether a QV benchmark returned heavy output results over the threshold, therefore being successful.

    This condition checks that the average of HOP is above the 2/3 threshold within the number of sigmas given in
    the configuration.

    Args:
        heavy_output_probabilities (List[float]): the HOP of all quantum circuits.
        num_sigmas (int): the number of sigmas to check
    Returns:
        bool: whether the QV benchmark was successful.
    """
    avg = np.mean(heavy_output_probabilities)
    std = (avg * (1 - avg) / len(heavy_output_probabilities)) ** 0.5

    return bool((avg - std * num_sigmas) > 2.0 / 3.0)


def plot_hop_threshold(
    qubits: List[int],
    depth: int,
    qv_result: List[float],
    qv_results_type: str,
    num_sigmas: int,
    backend_name: str,
    timestamp: str,
    in_volumetric: bool = False,
    plot_rem: bool = False,
) -> Tuple[str, Figure]:
    """Generate the figure representing each HOP, the average and the threshold.

    Args:
        qubits (List[int]): the list of qubit labels.
        depth (int): the depth of the QV circuit.
        qv_result (List[float]): the list of HOP.
        qv_results_type (str): whether results come from vanilla or DD execution.
        num_sigmas (int): the number of sigmas to plot.
        backend_name (str): the name of the backend.
        timestamp (str): the execution timestamp.
        in_volumetric (bool): whether the QV benchmark is being executed in the context of a volumetric benchmark.
            Defaults to False.
        plot_rem (bool): whether the plot corresponds to REM corrected data.

    Returns:
        str: the name of the figure.
        Figure: the figure.
    """
    cumul_hop = cumulative_hop(qv_result)
    cumul_std = cumulative_std(qv_result)

    fig = plt.figure()
    ax = plt.axes()

    plt.axhline(2.0 / 3.0, color="red", linestyle="dashed", label="Threshold")

    plt.scatter(
        np.arange(len(qv_result)),
        qv_result,
        marker=".",
        s=6,
        alpha=0.7,
        label="Individual HOP",
    )

    y_up: List[float] = [cumul_hop[i] + num_sigmas * cumul_std[i] for i in range(len(cumul_hop))]

    y_down: List[float] = [cumul_hop[i] - num_sigmas * cumul_std[i] for i in range(len(cumul_hop))]

    plt.fill_between(
        np.arange(len(qv_result)),
        y_up,
        y_down,
        color="b",
        alpha=0.125,
        label=rf"Cumulative {num_sigmas}$\sigma$",
    )

    plt.plot(cumul_hop, color=(0.0, 1.0, 0.5, 1.0), linewidth=2, label="Cumulative HOP")

    plt.ylim(min(qv_result), max(qv_result))
    ax.set_ylabel("Heavy Output Probability (HOP)")
    ax.set_xlabel("QV Circuit Samples (N)")
    plt.legend(loc="lower right")

    plt.margins(x=0, y=0)

    if in_volumetric:
        plt.title(
            f"Quantum Volume ({len(qubits)} qubits, {depth} depth)\nBackend: {backend_name} / {timestamp}",
            fontsize=9,
        )
        fig_name = f"{len(qubits)}_qubits_{depth}_depth"
    else:
        if plot_rem:
            plt.title(
                f"Quantum Volume ({qv_results_type}) with REM on {len(qubits)} qubits ({str(qubits)})\nBackend: {backend_name} / {timestamp}"
            )
            fig_name = f"{qv_results_type}_REM_{len(qubits)}_qubits_{str(qubits)}"
        else:
            plt.title(
                f"Quantum Volume ({qv_results_type}) on {len(qubits)} qubits ({str(qubits)})\nBackend: {backend_name} / {timestamp}"
            )
            fig_name = f"{qv_results_type}_{len(qubits)}_qubits_{str(qubits)}"

    plt.gcf().set_dpi(250)
    plt.close()

    return fig_name, fig


def qv_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a Quantum Volume experiment

    Args:
        run (RunResult): A Quantum Volume experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to Quantum Volume
    """

    plots = {}
    observations: list[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)
    backend_name = dataset.attrs["backend_name"]
    execution_timestamp = dataset.attrs["execution_timestamp"]
    num_circuits = dataset.attrs["num_circuits"]
    num_sigmas = dataset.attrs["num_sigmas"]

    physical_layout = dataset.attrs["physical_layout"]

    # Analyze the results for each qubit layout of the experiment dataset
    qubit_layouts = dataset.attrs["custom_qubits_array"]
    depth = {}
    qv_results_type = {}
    execution_results = {}
    ideal_heavy_outputs = {}
    rem = dataset.attrs["rem"]

    for qubits_idx, qubits in enumerate(qubit_layouts):
        qcvv_logger.info(f"Noiseless simulation and post-processing for layout {qubits}")
        # Retrieve counts
        execution_results[str(qubits)] = xrvariable_to_counts(dataset, str(qubits), num_circuits)

        # Retrieve other dataset values
        sorted_qc_list_indices = dataset.attrs[qubits_idx]["sorted_qc_list_indices"]
        qc_list = run.circuits["untranspiled_circuits"][str(qubits)].circuits

        qv_results_type[str(qubits)] = dataset.attrs[qubits_idx]["qv_results_type"]
        depth[str(qubits)] = len(qubits)

        # Simulate the circuits and get the ideal heavy outputs
        ideal_heavy_outputs[str(qubits)] = get_ideal_heavy_outputs(qc_list, sorted_qc_list_indices)

        # Compute the HO probabilities
        qv_result = compute_heavy_output_probabilities(execution_results[str(qubits)], ideal_heavy_outputs[str(qubits)])

        observations.extend(
            [
                BenchmarkObservation(
                    name="average_heavy_output_probability",
                    value=cumulative_hop(qv_result)[-1],
                    uncertainty=cumulative_std(qv_result)[-1],
                    identifier=BenchmarkObservationIdentifier(qubits),
                ),
                BenchmarkObservation(
                    name="is_succesful",
                    value=is_successful(qv_result, num_sigmas),
                    identifier=BenchmarkObservationIdentifier(qubits),
                ),
                BenchmarkObservation(
                    name="QV_result",
                    value=2 ** len(qubits) if is_successful(qv_result) else 1,
                    identifier=BenchmarkObservationIdentifier(qubits),
                ),
            ]
        )

        dataset.attrs[qubits_idx].update(
            {
                "cumulative_average_heavy_output_probability": cumulative_hop(qv_result),
                "cumulative_stddev_heavy_output_probability": cumulative_std(qv_result),
                "heavy_output_probabilities": qv_result,
            }
        )
        # Remove sorted_qc_list_indices from dataset if using physical_layout = "fixed"
        if physical_layout == "fixed" and rem is None:
            del dataset.attrs[qubits_idx]["sorted_qc_list_indices"]

        fig_name, fig = plot_hop_threshold(
            qubits,
            depth[str(qubits)],
            qv_result,
            qv_results_type[str(qubits)],
            num_sigmas,
            backend_name,
            execution_timestamp,
            plot_rem=False,
        )
        plots[fig_name] = fig

    if not rem:
        return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)

    # When REM is set to True, do the post-processing with the adjusted quasi-probabilities
    mit_shots = dataset.attrs["mit_shots"]
    rem_quasidistros = dataset.attrs["REM_quasidistributions"]
    for qubits_idx, qubits in enumerate(qubit_layouts):
        qcvv_logger.info(f"REM post-processing for layout {qubits} with {mit_shots} shots")

        # Remove sorted_qc_list_indices from dataset if using physical_layout = "fixed"
        if physical_layout == "fixed":
            del dataset.attrs[qubits_idx]["sorted_qc_list_indices"]

        qv_result_rem = get_rem_hops(
            rem_quasidistros[f"REM_quasidist_{str(qubits)}"],
            ideal_heavy_outputs[str(qubits)],
        )

        dataset.attrs[qubits_idx].update(
            {
                "REM_cumulative_average_heavy_output_probability": cumulative_hop(qv_result_rem),
                "REM_cumulative_stddev_heavy_output_probability": cumulative_std(qv_result_rem),
                "REM_heavy_output_probabilities": qv_result_rem,
            }
        )

        # UPDATE OBSERVATIONS
        observations.extend(
            [
                BenchmarkObservation(
                    name="REM_average_heavy_output_probability",
                    value=cumulative_hop(qv_result_rem)[-1],
                    uncertainty=cumulative_std(qv_result_rem)[-1],
                    identifier=BenchmarkObservationIdentifier(qubits),
                ),
                BenchmarkObservation(
                    name="REM_is_succesful",
                    value=is_successful(qv_result_rem, num_sigmas),
                    identifier=BenchmarkObservationIdentifier(qubits),
                ),
                BenchmarkObservation(
                    name="REM_QV_result",
                    value=2 ** len(qubits) if is_successful(qv_result_rem) else 1,
                    identifier=BenchmarkObservationIdentifier(qubits),
                ),
            ]
        )

        fig_name_rem, fig_rem = plot_hop_threshold(
            qubits,
            depth[str(qubits)],
            qv_result_rem,
            qv_results_type[str(qubits)],
            num_sigmas,
            backend_name,
            execution_timestamp,
            plot_rem=True,
        )
        plots[fig_name_rem] = fig_rem

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


class QuantumVolumeBenchmark(Benchmark):
    """
    Quantum Volume reflects the deepest circuit a given number of qubits can execute with meaningful results.
    """

    analysis_function = staticmethod(qv_analysis)

    name: str = "quantum_volume"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "QuantumVolumeConfiguration"):
        """Construct the QuantumVolumeBenchmark class.

        Args:
            backend_arg (IQMBackendBase): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.num_circuits = configuration.num_circuits
        self.num_sigmas = configuration.num_sigmas
        self.choose_qubits_routine = configuration.choose_qubits_routine

        # self.mapomatic_num_qubits = configuration.mapomatic_num_qubits
        # self.mapomatic_num_layouts = configuration.mapomatic_num_layouts
        # self.mapomatic_qv_samples = configuration.mapomatic_qv_samples

        # self.dynamical_decoupling = configuration.dynamical_decoupling
        # self.dd_strategy = configuration.dd_strategy

        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg

        self.rem = configuration.rem
        self.mit_shots = configuration.mit_shots

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

        # The only difference between "custom" and "mapomatic" choose_qubits_routine is
        # that the custom qubit layout array is generated either by user-input or by mapomatic
        if self.choose_qubits_routine == "custom":
            if configuration.custom_qubits_array is None:
                raise ValueError(
                    "If the `choose_qubits_custom` routine is chosen, a `custom_qubits_array` must be specified in `QuantumVolumeConfiguration`."
                )
            self.custom_qubits_array = configuration.custom_qubits_array
        # elif self.choose_qubits_routine == "mapomatic":
        #    self.custom_qubits_array = self.get_mapomatic_average_qv_scores()
        else:
            raise ValueError("The `custom_qubits_array` variable must be either 'custom' or 'mapomatic'")

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

    # def get_mapomatic_average_qv_scores(self) -> List[List[int]]:
    #     """Estimate the average mapomatic scores for N quantum volume circuit samples
    #     Returns:
    #         List[List[object]]: the mapomatic layout scores sorted from the smallest mean*(1+std/sqrt(N)) cost score
    #     """
    #     qv_circ_samples = self.mapomatic_qv_samples
    #
    #     # Get calibration data once
    #     token = input("Trying to access IQM calibration data\nEnter your IQM token (without quote marks): ")
    #     calibration_data = get_calibration_fidelities(self.backend.architecture.name, token)
    #
    #     qcvv_logger.info(
    #         "Evaluating matching layouts for QV",
    #     )
    #
    #     all_scores: Dict[Tuple, List[float]] = {}
    #     for mapomatic_qv_sample in range(qv_circ_samples):
    #         qcvv_logger.info(
    #             f"Estimating layout costs for QV circuit sample {mapomatic_qv_sample+1}"
    #         )
    #         # Generate a representative quantum circuit sample
    #         qv_circ = get_circuit("quantum_volume", self.mapomatic_num_qubits)
    #         # Get all matching layouts
    #         layouts = matching_layouts(
    #             qv_circ,
    #             self.backend,
    #             self.backend.coupling_map,
    #             self.qiskit_optim_level,
    #         )
    #         # Evaluate all layouts
    #         scores = evaluate_costs(
    #             layouts,
    #             qv_circ,
    #             self.backend,
    #             calibration_data,
    #             self.qiskit_optim_level,
    #         )
    #         for score in scores:
    #             all_scores.setdefault(tuple(cast(list[int], score[0])), []).extend([cast(float, score[1])])
    #         qv_circ.clear()
    #
    #     # Consider mean and std of the cost scores as mean*(1+std/sqrt(N))
    #     all_scores_mean_plus_std = {
    #         s_k: np.mean(s_v) * (1 + np.std(s_v) / np.sqrt(qv_circ_samples)) for s_k, s_v in all_scores.items()
    #     }
    #     # Turn into an array ([[tuple_layout0, score0], [tuple_layout1,score1], ... ])
    #     sorted_scores_mean_plus_std = list(sorted(all_scores_mean_plus_std.items(), key=lambda y: y[1]))
    #     # Return indices to list type for layouts ([[list_layout0, score0], [list_layout1,score1], ... ])
    #     layout_meanscore = [[list(s[0]), s[1]] for s in sorted_scores_mean_plus_std]
    #
    #     mapomatic_qv_layouts = [x[0] for x in layout_meanscore[: self.mapomatic_num_layouts]]
    #     mapomatic_qv_costs = [x[1] for x in layout_meanscore[: self.mapomatic_num_layouts]]
    #
    #     qcvv_logger.info(
    #         f"Will execute QV on layouts {mapomatic_qv_layouts}, of mapomatic average costs {mapomatic_qv_costs}"
    #     )
    #
    #     return mapomatic_qv_layouts

    @staticmethod
    def generate_single_circuit(
        num_qubits: int,
        depth: Optional[int] = None,
        classical_permutation: bool = True,
    ) -> QuantumCircuit:
        """Generate a single QV quantum circuit, with measurements at the end.

        Args:
            num_qubits (int): number of qubits of the circuit
            depth (Optional[int]): The depth of the QV circuit. Defaults to None, which makes it equal to the number of
                qubits.
            classical_permutation (bool, optional): Whether permutations are classical, avoiding swapping layers.
                * Defaults to True.

        Returns:
            QuantumCircuit: the QV quantum circuit.
        """
        qc = QuantumVolume(num_qubits, depth=depth, classical_permutation=classical_permutation).decompose()
        qc.measure_all()
        return qc

    @timeit
    def generate_circuit_list(
        self,
        num_qubits: int,
        depth: Optional[int] = None,
        classical_permutations: bool = True,
    ) -> List[QuantumCircuit]:
        """Generate a list of QV quantum circuits, with measurements at the end.

        Args:
            num_qubits (int): the number of qubits of the circuits.
            depth (Optional[int]): The depth of the QV circuit. Defaults to None, which makes it equal to the number of qubits.
            classical_permutations (bool, optional): Whether permutations are classical, avoiding swapping layers.
                * Defaults to True.

        Returns:
            List[QuantumCircuit]: the list of QV quantum circuits.
        """
        qc_list = [
            self.generate_single_circuit(num_qubits, depth=depth, classical_permutation=classical_permutations)
            for _ in range(self.num_circuits)
        ]

        return qc_list

    def get_rem_quasidistro(
        self,
        sorted_transpiled_qc_list: Dict[Tuple, List[QuantumCircuit]],
        sorted_qc_list_indices: Dict[Tuple, List[int]],
        execution_results: List[Dict[str, int]],
        mit_shots: int,
    ) -> List[List[QuasiCollection]]:
        """Computes readout-error-mitigated quasiprobabilities.

        Args:
            sorted_transpiled_qc_list (Dict[Tuple, List[QuantumCircuit]]): A dictionary of lists of quantum circuits, indexed by qubiy layouts.
            sorted_qc_list_indices (Dict[Tuple, List[int]]): dictionary of indices (integers) corresponding to those in the original (untranspiled) list of circuits, with keys being final physical qubit measurements.
            execution_results (List[Dict[str, int]]): counts from execution of all quantum circuits.
            mit_shots (int): The number of measurement shots to estimate the readout calibration errors.
        Returns:
            A list of lists of quasiprobabilities.
        """
        all_rem_quasidistro = []
        for k in sorted(
            sorted_transpiled_qc_list.keys(),
            key=lambda x: len(sorted_transpiled_qc_list[x]),
            reverse=True,
        ):
            counts_corresp_to_circs_k = [execution_results[i] for i in sorted_qc_list_indices[k]]
            all_rem_quasidistro_batch_k, _ = apply_readout_error_mitigation(
                self.backend, sorted_transpiled_qc_list[k], counts_corresp_to_circs_k, mit_shots
            )
            all_rem_quasidistro += all_rem_quasidistro_batch_k

        return all_rem_quasidistro

    def submit_single_qv_job(
        self,
        backend: IQMBackendBase,
        qubits: Sequence[int],
        sorted_transpiled_qc_list: Dict[Tuple[int, ...], List[QuantumCircuit]],
    ) -> Dict[str, Any]:
        """
            Submit jobs for execution in the specified IQMBackend.
        Args:
            backend (IQMBackendBase): the IQM backend to submit the job.
            qubits (List[int]): the qubits to identify the submitted job.
            sorted_transpiled_qc_list (Dict[str, List[QuantumCircuit]]): qubits to submit jobs to.
        Returns:
            Dict with qubit layout, submitted job objects, type (vanilla/DD) and submission time.
        """
        time_submit = 0
        execution_jobs = None
        # qv_results_type = "dynamical decoupling"
        # if not self.dynamical_decoupling:
        qv_results_type = "vanilla"
        # Send to execute on backend
        execution_jobs, time_submit = submit_execute(
            sorted_transpiled_qc_list,
            backend,
            self.shots,
            self.calset_id,
            max_gates_per_batch=self.max_gates_per_batch,
        )
        # else:
        # DD IN DIQE VERSION PREVENTS SUBMITTING JOBS DYNAMICALLY:
        # I.E., IT AUTOMATICALLY RETRIEVES COUNTS
        # TODO: change this when job manager for DD in Pulla is updated! # pylint: disable=fixme
        # raise ValueError("Dynamical decoupling is not yet enabled in the new base")
        # qcvv_logger.info(
        #     f"Now executing {self.num_circuits} circuits with default strategy Dynamical Decoupling"
        # )
        # execution_results, time_retrieve = execute_with_dd(
        #     self.backend, transpiled_qc_list, self.shots, self.dd_strategy
        # )
        qv_results = {
            "qubits": qubits,
            "jobs": execution_jobs,
            "qv_results_type": qv_results_type,
            "time_submit": time_submit,
        }
        return qv_results

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark."""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Submit jobs for all qubit layouts first
        all_qv_jobs: List[Dict[str, Any]] = []
        time_circuit_generation = {}
        time_transpilation = {}
        time_batching = {}
        sorted_qc_list_indices = {}

        # Initialize the variable to contain the QV circuits of each layout
        self.circuits = Circuits()
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")
        all_op_counts = {}

        for qubits in self.custom_qubits_array:  # NB: jobs will be submitted for qubit layouts in the specified order

            num_qubits = len(qubits)
            depth = num_qubits
            qcvv_logger.info(f"Executing QV on qubits {qubits}")

            qc_list, time_circuit_generation[str(qubits)] = self.generate_circuit_list(num_qubits, depth=depth)
            qcvv_logger.info(f"Successfully generated all {self.num_circuits} circuits to be executed")
            # Set the coupling map
            coupling_map = set_coupling_map(qubits, backend, self.physical_layout)
            # Perform transpilation to backend
            qcvv_logger.info(f'Will transpile according to "{self.physical_layout}" physical layout')
            transpiled_qc_list, time_transpilation[str(qubits)] = perform_backend_transpilation(
                qc_list,
                backend=backend,
                qubits=qubits,
                coupling_map=coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
                routing_method=self.routing_method,
            )
            # Batching
            sorted_transpiled_qc_list: Dict[Tuple[int, ...], List[QuantumCircuit]] = {}
            time_batching[str(qubits)] = 0
            if self.physical_layout == "fixed":
                sorted_transpiled_qc_list = {tuple(qubits): transpiled_qc_list}
                sorted_qc_list_indices[str(qubits)] = {tuple(qubits): list(range(len(qc_list)))}
            elif self.physical_layout == "batching":
                # Sort circuits according to their final measurement mappings
                (
                    sorted_transpiled_qc_list,
                    sorted_qc_list_indices[str(qubits)],
                ), time_batching[
                    str(qubits)
                ] = sort_batches_by_final_layout(transpiled_qc_list)
            else:
                raise ValueError("physical_layout must either be \"fixed\" or \"batching\"")

            self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=qc_list))
            self.transpiled_circuits.circuit_groups.append(
                CircuitGroup(name=str(qubits), circuits=sorted_transpiled_qc_list[tuple(qubits)])
            )

            # Count operations
            all_op_counts[str(qubits)] = count_native_gates(backend, transpiled_qc_list)

            # Submit
            all_qv_jobs.append(self.submit_single_qv_job(backend, qubits, sorted_transpiled_qc_list))
            qcvv_logger.info(f"Job for layout {qubits} submitted successfully!")

        # Retrieve counts of jobs for all qubit layouts
        all_job_metadata = {}
        for job_idx, job_dict in enumerate(all_qv_jobs):
            qubits = job_dict["qubits"]
            # Retrieve counts
            execution_results, time_retrieve = retrieve_all_counts(job_dict["jobs"], str(qubits))
            # Retrieve all job meta data
            all_job_metadata = retrieve_all_job_metadata(job_dict["jobs"])

            # Export all to dataset
            dataset.attrs.update(
                {
                    job_idx: {
                        "qubits": qubits,
                        "qv_results_type": job_dict["qv_results_type"],
                        "time_circuit_generation": time_circuit_generation[str(qubits)],
                        "time_transpilation": time_transpilation[str(qubits)],
                        "time_batching": time_batching[str(qubits)],
                        "time_submit": job_dict["time_submit"],
                        "time_retrieve": time_retrieve,
                        "all_job_metadata": all_job_metadata,
                        "sorted_qc_list_indices": {
                            str(key): value for key, value in sorted_qc_list_indices[str(qubits)].items()
                        },
                        "operation_counts": all_op_counts[str(qubits)],
                    }
                }
            )

            qcvv_logger.info(f"Adding counts of {qubits} run to the dataset")
            dataset, _ = add_counts_to_dataset(execution_results, str(qubits), dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        if self.rem:
            rem_quasidistros = {}
            for qubits in self.custom_qubits_array:
                exec_counts = xrvariable_to_counts(dataset, str(qubits), self.num_circuits)
                rem_quasidistros[f"REM_quasidist_{str(qubits)}"] = self.get_rem_quasidistro(
                    {tuple(qubits): self.transpiled_circuits[str(qubits)].circuits},
                    # self.transpiled_circuits[str(qubits)],
                    sorted_qc_list_indices[str(qubits)],
                    exec_counts,
                    self.mit_shots,
                )
            dataset.attrs.update({"REM_quasidistributions": rem_quasidistros})

        qcvv_logger.info(f"QV experiment execution concluded !")
        return dataset


class QuantumVolumeConfiguration(BenchmarkConfigurationBase):
    """Quantum Volume configuration.

    Attributes:
        benchmark (Type[Benchmark]): QuantumVolumeBenchmark
        num_circuits (int): The number of circuits to use.
                            * Should be at least 100 for a meaningful QV experiment.
        num_sigmas (int): The number of sample standard deviations to consider with for the threshold criteria.
                            * Default by consensus is 2
        choose_qubits_routine (Literal["custom"]): The routine to select qubit layouts.
                            * Default is "custom".
        custom_qubits_array (Optional[Sequence[Sequence[int]]]): The physical qubit layouts to perform the benchmark on.
                            * Default is [[0, 2]].
        qiskit_optim_level (int): The Qiskit transpilation optimization level.
                            * Default is 3.
        optimize_sqg (bool): Whether Single Qubit Gate Optimization is performed upon transpilation.
                            * Default is True.
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): The Qiskit transpilation routing method to use.
                            * Default is "sabre".
        physical_layout (Literal["fixed", "batching"]): Whether the coupling map is restricted to qubits in the input layout or not.
                            - "fixed": Restricts the coupling map to only the specified qubits.
                            - "batching": Considers the full coupling map of the backend and circuit execution is batched per final layout.
                            * Default is "fixed"
        rem (bool): Whether Readout Error Mitigation is applied in post-processing.
                    When set to True, both results (readout-unmitigated and -mitigated) are produced.
                            - Default is True.
        mit_shots (int): The measurement shots to use for readout calibration.
                            * Default is 1_000.
    """

    benchmark: Type[Benchmark] = QuantumVolumeBenchmark
    num_circuits: int
    num_sigmas: int = 2
    choose_qubits_routine: Literal["custom"] = "custom"
    custom_qubits_array: Sequence[Sequence[int]]
    qiskit_optim_level: int = 3
    optimize_sqg: bool = True
    rem: bool = True
    mit_shots: int = 1_000
