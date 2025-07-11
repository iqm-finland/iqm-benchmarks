"""
Coherence benchmark
"""

import logging
from time import strftime
from typing import Dict, List, Tuple, Type

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import curve_fit
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
from iqm.benchmarks.utils import (  # execute_with_dd,
    perform_backend_transpilation,
    retrieve_all_counts,
    submit_execute,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def exp_decay(t, A, T, C):
    """
    Calculate the exponential decay at time t.

    Parameters:
    t (float or array-like): The time variable(s) at which to evaluate the decay.
    A (float): The initial amplitude of the decay.
    T (float): The time constant, which dictates the rate of decay.
    C (float): The constant offset added to the decay.

    Returns:
    float or array-like: The value(s) of the exponential decay at time t.
    """
    return A * np.exp(-t / T) + C


def plot_coherence(
    amplitude_list: List[float],
    backend_name: str,
    delays: List[float],
    offset_list: List[float],
    qubit_set: List[int],
    qubit_probs: dict[str, List[float]],
    timestamp: str,
    fitted_t_list: List[float],
    t_err_list: List[float],
    qubit_to_plot: List[int] | None = None,
    coherence_exp: str = "t1",
) -> Tuple[str, Figure]:
    """
    Plot coherence decay (T1 or T2_echo) for each qubit as subplots.

    Args:
        amplitude_list (List[float]): Fitted amplitudes (A) per qubit.
        backend_name (str): Name of the backend used for the experiment.
        delays (List[float]): List of delay times used in the coherence experiments.
        offset_list (List[float]): Fitted offsets (C) for each qubit.
        qubit_set (List[int]): List of qubit indices involved in the experiment.
        qubit_probs (dict[str, List[float]]): Measured probabilities P(1) for each qubit at different delays.
        timestamp (str): Timestamp for labeling the plot.
        fitted_t_list (List[float]): Fitted time constants (T) for each qubit.
        coherence_exp (str): Type of coherence experiment ('t1' or 't2_echo') for labeling and plotting logic.
        qubit_to_plot (list[int] | None): Specific qubits to plot. If None, all qubits in `qubit_set` are plotted.

    Returns:
        Tuple[str, Figure]: Filename of the saved plot and the matplotlib figure object.
    """
    if qubit_to_plot is not None:
        num_qubits = len(qubit_to_plot)
    else:
        num_qubits = len(qubit_set)

    ncols = 3
    nrows = (num_qubits + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, qubit in enumerate(qubit_to_plot or []):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        ydata = np.array(qubit_probs[str(qubit)])
        A = amplitude_list[idx]
        C = offset_list[idx]
        T_fit = fitted_t_list[idx]

        # Plot raw data
        ax.plot(delays, ydata, "o", label="Measured P(1)", color="blue")

        # Plot fit line
        t_fit = np.linspace(min(delays), max(delays), 200)
        fitted_curve = exp_decay(t_fit, A, T_fit, C)
        ax.plot(
            t_fit,
            fitted_curve,
            "--",
            color="orange",
            label=f"Fit (T = {T_fit * 1e6:.1f} ± {t_err_list[idx] * 1e6:.1f} µs)",
        )
        tick_list = np.linspace(min(delays), max(delays), 5)
        ax.set_xticks(tick_list)
        ax.set_xticklabels([f"{d * 1e6:.0f}" for d in tick_list])
        ax.set_title(f"Qubit {qubit}")
        ax.set_xlabel("Delay (µs)")
        ax.set_ylabel("|1> Populatiuon")
        ax.grid(True)
        ax.legend()

    # Remove unused axes
    for j in range(idx + 1, nrows * ncols):  # pylint: disable=undefined-loop-variable
        row, col = divmod(j, ncols)
        fig.delaxes(axs[row][col])

    fig.suptitle(f"{coherence_exp.upper()}_decay_{backend_name}_{timestamp}", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    fig_name = f"{coherence_exp}_{backend_name}_{timestamp}.png"
    plt.close()

    return fig_name, fig


def coherence_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a coherence experiment

    Args:
        run (RunResult): A coherence experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to coherence
    """

    plots = {}
    observations: list[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)

    backend_name = dataset.attrs["backend_name"]
    timestamp = dataset.attrs["execution_timestamp"]
    delays = dataset.attrs["delay_list"]
    coherence_exp = dataset.attrs["experiment"]
    qubit_set = dataset.attrs["qubit_set"]
    tot_circs = len(delays)
    groups = dataset.attrs["group"]
    all_counts_group: List[Dict[str, int]] = []
    qubit_probs: Dict[str, List[float]] = {}

    def calculate_probabilities(counts, nqubits, coherence_exp):
        p0_per_qubit = [0.0 for _ in range(nqubits)]
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            for q in range(nqubits):
                if coherence_exp == "t1":
                    if bitstring[::-1][q] == "1":
                        p0_per_qubit[q] += count
                else:
                    if bitstring[::-1][q] == "0":
                        p0_per_qubit[q] += count
        return p0_per_qubit, total_shots

    qubits_to_plot = dataset.attrs["qubits_to_plot"]
    for group in groups:
        all_counts_group = xrvariable_to_counts(dataset, str(group), tot_circs)
        nqubits = len(group)
        qubit_probs.update({str(q): [] for q in group})

        for counts in all_counts_group:
            p0_per_qubit, total_shots = calculate_probabilities(counts, nqubits, coherence_exp)
            for q_idx, qubit in enumerate(group):
                qubit_probs[str(qubit)].append(p0_per_qubit[q_idx] / total_shots)

    def fit_coherence_model(
        qubit: int, probs: np.ndarray, delays: np.ndarray, coherence_exp: str
    ) -> Tuple[List[BenchmarkObservation], float, float, float, float]:
        """Fit the coherence model and return observations."""
        observations_per_qubit = []
        ydata = probs
        p0 = [0.5, 100e-6, 0.5]
        popt, pcov = curve_fit(exp_decay, delays, ydata, p0=p0)  # pylint: disable=unbalanced-tuple-unpacking
        A, T_fit, C = popt
        perr = np.sqrt(np.diag(pcov))  # Standard deviation errors
        T_fit_err = perr[1]

        observations_per_qubit.extend(
            [
                BenchmarkObservation(
                    name="T1" if coherence_exp == "t1" else "T2_echo",
                    value=T_fit,
                    identifier=BenchmarkObservationIdentifier(qubit),
                    uncertainty=T_fit_err,
                ),
            ]
        )
        return observations_per_qubit, T_fit, T_fit_err, A, C

    qubit_set = [item for sublist in groups for item in sublist]
    amplitude_list = []
    offset_list = []
    fitted_t_list = []
    T_fit_err_list = []
    for qubit in qubit_set:
        probs = np.array(qubit_probs[str(qubit)])
        results = fit_coherence_model(qubit, probs, delays, coherence_exp)
        observations.extend(results[0])
        fitted_t_list.append(results[1])
        T_fit_err_list.append(results[2])
        amplitude_list.append(results[3])
        offset_list.append(results[4])

    fig_name, fig = plot_coherence(
        amplitude_list,
        backend_name,
        delays,
        offset_list,
        qubit_set,
        qubit_probs,
        timestamp,
        fitted_t_list,
        T_fit_err_list,
        qubits_to_plot,
        coherence_exp,
    )
    plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, plots=plots, observations=observations)


class CoherenceBenchmark(Benchmark):
    """
    This benchmark estimates the coherence properties of the qubits and computational resonator.
    """

    analysis_function = staticmethod(coherence_analysis)

    name: str = "coherence"

    def __init__(self, backend_arg: IQMBackendBase, configuration: "CoherenceConfiguration"):
        """Construct the CoherenceBenchmark class.

        Args:
            backend_arg (IQMBackendBase): the backend to execute the benchmark on
            configuration (CoherenceConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name
        self.delays = configuration.delays
        self.shots = configuration.shots
        self.optimize_sqg = configuration.optimize_sqg
        self.coherence_exp = configuration.coherence_exp
        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.qubits_to_plot = configuration.qubits_to_plot

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

        # Initialize the variable to contain all coherence circuits
        self.circuits = Circuits()
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")

    def generate_coherence_circuits(
        self,
        nqubits: int,
    ) -> list[QuantumCircuit]:
        """Generates coherence circuits for the given qubit set and delay times.

        Args:
            nqubits (int): Number of qubits to apply the coherence circuits on.

        Returns:
            list[QuantumCircuit]: List of generated coherence circuits.
        """
        circuits = []
        for delay in self.delays:
            qc = QuantumCircuit(nqubits)
            if self.coherence_exp == "t1":
                self._generate_t1_circuits(qc, nqubits, delay)
            elif self.coherence_exp == "t2_echo":
                self._generate_t2_echo_circuits(qc, nqubits, delay)
            qc.measure_all()
            circuits.append(qc)
        return circuits

    def _generate_t1_circuits(self, qc: QuantumCircuit, nqubits: int, delay: float):
        """Generates T1 coherence circuits.

        Args:
            qc (QuantumCircuit): The quantum circuit to modify.
            nqubits (int): Number of qubits.
            delay (float): Delay time for the circuit.
        """

        for qubit in range(nqubits):
            qc.x(qubit)
            qc.delay(int(delay * 1e9), qubit, unit="ns")

    def _generate_t2_echo_circuits(self, qc: QuantumCircuit, nqubits: int, delay: float):
        """Generates T2 echo coherence circuits.

        Args:
            qc (QuantumCircuit): The quantum circuit to modify.
            nqubits (int): Number of qubits.
            delay (float): Delay time for the circuit.
        """
        half_delay = delay / 2
        for qubit in range(nqubits):
            qc.h(qubit)
            qc.delay(int(half_delay * 1e9), qubit, unit="ns")
            qc.x(qubit)
            qc.delay(int(half_delay * 1e9), qubit, unit="ns")
            qc.h(qubit)

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

    def checkerboard_groups_from_coupling(self, coupling_map: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
        """
        Assign Group A and B to qubits based on a checkerboard pattern
        inferred from the connectivity graph (assumed to be grid-like).

        Args:
            coupling_map (list of tuple): List of 2-qubit connections (edges).

        Returns:
            group_a (list): Qubit indices in Group A.
            group_b (list): Qubit indices in Group B.
        """
        G = nx.Graph()
        G.add_edges_from(coupling_map)
        if not nx.is_bipartite(G):
            raise ValueError("The coupling map is not bipartite (not grid-like).")
        coloring = nx.bipartite.color(G)
        group_a = [q for q, color in coloring.items() if color == 0]
        group_b = [q for q, color in coloring.items() if color == 1]

        return group_a, group_b

    def execute(
        self,
        backend: IQMBackendBase,
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
    ) -> xr.Dataset:
        """Executes the benchmark."""
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        self.circuits = Circuits()
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")

        qubit_set = list(range(backend.num_qubits))
        if self.coherence_exp not in ["t1", "t2_echo"]:
            raise ValueError("coherence_exp must be either 't1' or 't2_echo'.")

        qcvv_logger.debug(f"Executing on {self.coherence_exp}.")
        qcvv_logger.setLevel(logging.WARNING)

        if self.backend.has_resonators():
            qc_coherence = self.generate_coherence_circuits(self.backend.num_qubits)
            effective_coupling_map = self.backend.coupling_map.reduce(qubit_set)
            transpilation_params = {
                "backend": self.backend,
                "qubits": qubit_set,
                "coupling_map": effective_coupling_map,
                "qiskit_optim_level": self.qiskit_optim_level,
                "optimize_sqg": self.optimize_sqg,
                "routing_method": self.routing_method,
            }
            transpiled_qc_list, _ = perform_backend_transpilation(qc_coherence, **transpilation_params)
            sorted_transpiled_qc_list = {tuple(qubit_set): transpiled_qc_list}
            # Execute on the backend
            if self.configuration.use_dd is True:
                raise ValueError("Coherence benchmarks should not be run with dynamical decoupling.")

            jobs, _ = submit_execute(
                sorted_transpiled_qc_list,
                self.backend,
                self.shots,
                self.calset_id,
                max_gates_per_batch=self.max_gates_per_batch,
                max_circuits_per_batch=self.configuration.max_circuits_per_batch,
                circuit_compilation_options=self.circuit_compilation_options,
            )
            qcvv_logger.setLevel(logging.INFO)
            execution_results = retrieve_all_counts(jobs)[0]
            dataset, _ = add_counts_to_dataset(execution_results, str(qubit_set), dataset)
            dataset.attrs.update(
                {
                    "qubit_set": qubit_set,
                    "delay_list": self.delays,
                    "experiment": self.coherence_exp,
                    "group": [qubit_set],
                    "qubits_to_plot": self.qubits_to_plot,
                }
            )

        else:
            transpiled_qc_list = []
            # For crystal topology, we use the checkerboard pattern
            group_a, group_b = self.checkerboard_groups_from_coupling(list(self.backend.coupling_map))
            for group in [group_a, group_b]:
                nqubits_group = len(group)
                qc_coherence = self.generate_coherence_circuits(nqubits_group)
                transpiled_qc_list = transpile(
                    qc_coherence, backend=self.backend, initial_layout=group, optimization_level=self.qiskit_optim_level
                )
                sorted_transpiled_qc_list = {tuple(group): transpiled_qc_list}
                # Execute on the backend
                if self.configuration.use_dd is True:
                    raise ValueError("Coherence benchmarks should not be run with dynamical decoupling.")

                jobs, _ = submit_execute(
                    sorted_transpiled_qc_list,
                    self.backend,
                    self.shots,
                    self.calset_id,
                    max_gates_per_batch=self.max_gates_per_batch,
                    max_circuits_per_batch=self.configuration.max_circuits_per_batch,
                    circuit_compilation_options=self.circuit_compilation_options,
                )
                qcvv_logger.setLevel(logging.INFO)
                execution_results = retrieve_all_counts(jobs)[0]
                dataset, _ = add_counts_to_dataset(execution_results, str(group), dataset)

            dataset.attrs.update(
                {
                    "qubit_set": qubit_set,
                    "delay_list": self.delays,
                    "experiment": self.coherence_exp,
                    "group": [group_a, group_b],
                    "qubits_to_plot": self.qubits_to_plot,
                }
            )

        qcvv_logger.debug(f"Adding counts for {self.coherence_exp} to the dataset")
        self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=self.coherence_exp, circuits=qc_coherence))
        self.transpiled_circuits.circuit_groups.append(
            CircuitGroup(name=self.coherence_exp, circuits=transpiled_qc_list)
        )

        return dataset


class CoherenceConfiguration(BenchmarkConfigurationBase):
    """Coherence configuration.

    Attributes:
        benchmark (Type[Benchmark]): The benchmark class used for coherence analysis, defaulting to CoherenceBenchmark.
        delays (list[float]): List of delay times used in the coherence experiments.
        qiskit_optim_level (int): Qiskit transpilation optimization level, default is 3.
        optimize_sqg (bool): Indicates whether Single Qubit Gate Optimization is applied during transpilation, default is True.
        coherence_exp (str): Specifies the type of coherence experiment, either "t1" or "echo", default is "t1".
    """

    benchmark: Type[Benchmark] = CoherenceBenchmark
    delays: list[float]
    optimize_sqg: bool = True
    qiskit_optim_level: int = 3
    coherence_exp: str = "t1"
    shots: int = 1000
    qubits_to_plot: list[int]
