"""
Qscore benchmark
"""

import itertools
import logging
from time import strftime
from typing import Dict, List, Sequence, Tuple, Type, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from networkx import Graph
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import basinhopping, minimize, curve_fit
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
from qiskit import QuantumRegister, ClassicalRegister
from iqm.qiskit_iqm import IQMCircuit
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.readout_mitigation import apply_readout_error_mitigation
from iqm.benchmarks.utils import (  # execute_with_dd,
    perform_backend_transpilation,
    retrieve_all_counts,
    submit_execute,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase

def exp_decay(t, A, T, C):
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

    Returns:
        Tuple[str, Figure]: Filename of the saved plot and the matplotlib figure object.
    """
    num_qubits = len(qubit_set)
    ncols = 3
    nrows = (num_qubits + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, qubit in enumerate(qubit_set):
        row, col = divmod(i, ncols)
        ax = axs[row][col]
        ydata = np.array(qubit_probs[str(qubit)])
        A = amplitude_list[i]
        C = offset_list[i]
        T_fit = fitted_t_list[i]

        # Plot raw data
        ax.plot(delays, ydata, "o", label="Measured P(1)", color="blue")

        # Plot fit line
        t_fit = np.linspace(min(delays), max(delays), 200)
        fitted_curve = exp_decay(t_fit, A, T_fit, C)
        ax.plot(t_fit, fitted_curve,
                "--", color="orange", label=f"Fit (T = {T_fit * 1e6:.1f} µs)")

        ax.set_title(f"Qubit {qubit}")
        ax.set_xlabel("Delay (s)")
        ax.set_ylabel("|1> Populatiuon")
        ax.grid(True)
        ax.legend()

    # Remove unused axes
    for j in range(i + 1, nrows * ncols):
        row, col = divmod(j, ncols)
        fig.delaxes(axs[row][col])

    fig.suptitle(f"{coherence_exp.upper()}_decay_{backend_name}_{timestamp}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_name = f"{coherence_exp}_{backend_name}_{timestamp}.png"
    plt.close()

    return fig_name, fig


def coherence_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for a QScore experiment

    Args:
        run (RunResult): A QScore experiment run for which analysis result is created
    Returns:
        AnalysisResult corresponding to QScore
    """

    plots = {}
    observations: list[BenchmarkObservation] = []
    dataset = run.dataset.copy(deep=True)

    backend_name = dataset.attrs["backend_name"]
    timestamp = dataset.attrs["execution_timestamp"]
    delays = dataset.attrs["delay_list"]
    coherence_exp = dataset.attrs["experiment"]
    qpu_type = dataset.attrs["qpu_type"]
    qubit_set = dataset.attrs["qubit_set"]
    num_circ = len(delays)*2 if qpu_type == "star" else len(delays)
    all_counts: List[Dict[str, int]] = []
    all_counts.extend(xrvariable_to_counts(dataset, idx, 1)[0] for idx in range(num_circ))
    print(all_counts)

    if qpu_type == "star":
        qubit_set = ["COMPR"] + qubit_set
    nqubits = len(qubit_set)
    qubit_probs = {str(q): [] for q in qubit_set}

    for counts in all_counts:
        total_shots = sum(counts.values())
        p0_per_qubit = [0.0 for _ in range(nqubits)]

        for bitstring, count in counts.items():
            for q in range(nqubits):
                if coherence_exp == "t1":
                    if bitstring[::-1][q] == '1':
                        p0_per_qubit[q] += count
                else:
                    if bitstring[::-1][q] == '0':
                        p0_per_qubit[q] += count
        
        for qubit in qubit_set:
            qubit_probs[str(qubit)].append(p0_per_qubit[qubit] / total_shots)

    def fit_coherence_model(qubit: int, probs: np.ndarray, delays: np.ndarray, coherence_exp: str) -> List[BenchmarkObservation]:
        """Fit the coherence model and return observations."""
        observations_per_qubit = []
        ydata = probs
        p0 = [0.5, 50e-6, 0.5]
        popt, _ = curve_fit(exp_decay, delays, ydata, p0=p0)
        A, T_fit, C = popt
        fit_fn = lambda t, A=A, T=T_fit, C=C: exp_decay(t, A, T, C)

        observations_per_qubit.extend([
            BenchmarkObservation(name="T1" if coherence_exp == "t1" else "T2_echo", value=T_fit, identifier=BenchmarkObservationIdentifier(qubit)),
            # BenchmarkObservation(name="Amplitude", value=A, identifier=BenchmarkObservationIdentifier(qubit)),
            # BenchmarkObservation(name="Offset", value=C, identifier=BenchmarkObservationIdentifier(qubit)),
            # BenchmarkObservation(name="Fit_function", value=fit_fn, identifier=BenchmarkObservationIdentifier(qubit)),
        ])
        return observations_per_qubit, T_fit, A, C
    
    print(qubit_probs)
    
    amplitude_list = []
    offset_list = []
    fitted_t_list = []
    for qubit in qubit_set:
        probs = np.array(qubit_probs[str(qubit)])
        results = fit_coherence_model(qubit, probs, delays, coherence_exp)
        observations.extend(results[0])
        fitted_t_list.append(results[1])
        amplitude_list.append(results[2])
        offset_list.append(results[3])


    fig_name, fig = plot_coherence(
        amplitude_list,
        backend_name,
        delays,
        offset_list,
        qubit_set,
        qubit_probs,
        timestamp,
        fitted_t_list,
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
        """Construct the QScoreBenchmark class.

        Args:
            backend_arg (IQMBackendBase): the backend to execute the benchmark on
            configuration (CoherenceConfiguration): the configuration of the benchmark
        """
        super().__init__(backend_arg, configuration)

        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name
        self.custom_qubits_array = configuration.custom_qubits_array
        self.delays = configuration.delays
        self.shots = configuration.shots
        self.optimize_sqg= configuration.optimize_sqg
        self.qpu_topology = configuration.qpu_topology
        self.coherence_exp = configuration.coherence_exp
        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.REM = configuration.REM
        self.mit_shots = configuration.mit_shots

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""


        # Initialize the variable to contain all QScore circuits
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
        if self.qpu_topology == "star" and self.coherence_exp == "t1":
            circuit_compr = []
            for delay in self.delays:
                comp_r = QuantumRegister(1, 'comp_r') 
                q = QuantumRegister(1, 'q') 
                c = ClassicalRegister(1, 'c') 
                qc = IQMCircuit(comp_r, q, c)
                qc.x(1)
                qc.move(1, 0)  # MOVE into the resonator
                qc.delay(delay, 1, unit="s")
                qc.move(1, 0)  # MOVE out of the resonator
                qc.measure(q, c)
                circuit_compr.append(qc)
            circuits.extend(circuit_compr)

        for delay in self.delays:
            qc = QuantumCircuit(nqubits)
            if self.coherence_exp == "t1":  
                self._generate_t1_circuits(qc, nqubits, delay)
            elif self.coherence_exp == 't2_echo':
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

        qubit_set = self.custom_qubits_array
        if self.custom_qubits_array is None:
            qubit_set = list(range(backend.num_qubits))
        if self.coherence_exp not in ["t1", "t2_echo"]:
            raise ValueError("coherence_exp must be either 't1' or 't2_echo'.")
        
        qcvv_logger.debug(f"Executing on {self.coherence_exp}.")
        nqubits = len(qubit_set)
        qc_coherence = self.generate_coherence_circuits(nqubits)
        qcvv_logger.setLevel(logging.WARNING)

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
        if self.configuration.use_dd == True:
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
        for delay_idx in range(len(self.delays)):
            if self.REM:
                rem_counts = apply_readout_error_mitigation(
                    backend, transpiled_qc_list[delay_idx], [retrieve_all_counts(jobs)[0][delay_idx]], self.mit_shots
                )
                rem_distribution = rem_counts[0].nearest_probability_distribution()
                execution_results = rem_distribution
            else:
                execution_results = retrieve_all_counts(jobs)[0]
                dataset, _ = add_counts_to_dataset(execution_results[delay_idx], str(delay_idx), dataset)


        dataset.attrs.update(
            {
                "qubit_set": qubit_set,
                "delay_list": self.delays,
                "experiment": self.coherence_exp,
                "qpu_type": self.qpu_topology,
            }
        )

        qcvv_logger.debug(f"Adding counts for {self.coherence_exp} to the dataset")
        #dataset, _ = add_counts_to_dataset(execution_results, self.coherence_exp, dataset)
        self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=self.coherence_exp, circuits=qc_coherence))
        self.transpiled_circuits.circuit_groups.append(
            CircuitGroup(name=self.coherence_exp, circuits=transpiled_qc_list)
        )

        return dataset


class CoherenceConfiguration(BenchmarkConfigurationBase):
    """Coherence configuration.

    Attributes:
        benchmark (Type[Benchmark]): The benchmark class used for QScore analysis, defaulting to CoherenceBenchmark.
        custom_qubits_array (list[int]): List of custom qubit indices for benchmarking.
        delays (list[float]): List of delay times used in the coherence experiments.
        qiskit_optim_level (int): Qiskit transpilation optimization level, default is 3.
        optimize_sqg (bool): Indicates whether Single Qubit Gate Optimization is applied during transpilation, default is True.
        qpu_topology (str): Specifies the topology of the QPU, either "crystal" or "star", default is "crystal".
        coherence_exp (str): Specifies the type of coherence experiment, either "t1" or "echo", default is "t1".
    """

    benchmark: Type[Benchmark] = CoherenceBenchmark
    custom_qubits_array: list[int] | None = None
    delays: list[float]
    qiskit_optim_level: int = 3
    optimize_sqg: bool = True
    qiskit_optim_level: int = 3
    qpu_topology: str = "crystal"
    coherence_exp: str = "t1"
    shots: int = 1000
    REM: bool = False
    mit_shots: int = 1000
