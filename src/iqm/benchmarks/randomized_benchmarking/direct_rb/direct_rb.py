"""
Direct Randomized Benchmarking.
"""

from time import strftime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford
import xarray as xr

from iqm.benchmarks import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkCircuit,
    BenchmarkRunResult,
    CircuitGroup,
    Circuits,
)
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import (
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    add_counts_to_dataset,
)
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.randomized_benchmarking_common import (
    edge_grab,
    exponential_rb,
    fit_decay_lmfit,
    get_survival_probabilities,
    lmfit_minimizer,
    plot_rb_decay,
    relabel_qubits_array_from_zero,
    submit_parallel_rb_job,
    survival_probabilities_parallel,
)
from iqm.benchmarks.utils import (
    get_iqm_backend,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    submit_execute,
    timeit,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


@timeit
def generate_drb_circuits(
    qubits: Sequence[int],
    depth: int,
    circ_samples: int,
    backend_arg: IQMBackendBase | str,
    density_2q_gates: float = 0.25,
    two_qubit_gate_ensemble: Optional[Dict[str, float]] = None,
    clifford_sqg_probability: float = 1.0,
    sqg_gate_ensemble: Optional[Dict[str, float]] = None,
    qiskit_optim_level: int = 3,
    routing_method: Literal["basic", "lookahead", "stochastic", "sabre", "none"] = "basic",
) -> Dict[str, List[QuantumCircuit]]:
    """Generates lists of samples of Direct RB circuits, of structure:
       Stabilizer preparation - Layers of canonical randomly sampled gates - Stabilizer measurement

    Args:
        qubits (List[int]): the qubits of the backend.
        depth (int): the depth (number of canonical layers) of the circuit.
        circ_samples (int): the number of circuit samples to generate.
        backend_arg (IQMBackendBase | str): the backend.
        density_2q_gates (float): the expected density of 2Q gates.
        two_qubit_gate_ensemble (Optional[Dict[str, float]]): A dictionary with keys being str specifying 2Q gates, and values being corresponding probabilities.
                * Default is None.
        clifford_sqg_probability (float): Probability with which to uniformly sample Clifford 1Q gates.
                * Default is 1.0.
        sqg_gate_ensemble (Optional[Dict[str, float]]): A dictionary with keys being str specifying 1Q gates, and values being corresponding probabilities.
                * Default is None.
        qiskit_optim_level (int): Qiskit transpiler optimization level.
                * Default is 1.
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): Qiskit transpiler routing method.
                * Default is "basic".
    Returns:
        Dict[str, List[QuantumCircuit]]: a dictionary with keys "transpiled", "untranspiled" and values a list of respective DRB circuits.
    """
    num_qubits = len(qubits)

    # Retrieve backend
    if isinstance(backend_arg, str):
        retrieved_backend = get_iqm_backend(backend_arg)
    else:
        assert isinstance(backend_arg, IQMBackendBase)
        retrieved_backend = backend_arg

    # Check if backend includes MOVE gates and set coupling map
    if "move" in retrieved_backend.operation_names:
        # All-to-all coupling map on the active qubits
        effective_coupling_map = [[x, y] for x in qubits for y in qubits if x != y]
    else:
        effective_coupling_map = retrieved_backend.coupling_map

    # Initialize the list of circuits
    all_circuits = {}
    drb_circuits_untranspiled: List[QuantumCircuit] = []
    drb_circuits_transpiled: List[QuantumCircuit] = []

    # simulator = AerSimulator(method=simulation_method)

    for _ in range(circ_samples):
        # Sample Clifford for stabilizer preparation
        clifford_layer = random_clifford(num_qubits)
        # NB: The DRB paper contains a more elaborated stabilizer compilation algorithm.
        # Not having it WILL be an issue here for larger num qubits !
        # Intended usage, however, is solely for 2-qubit DRB subroutines.

        # Sample the layers using edge grab sampler - different samplers may be conditionally chosen here in the future
        cycle_layers = edge_grab(
            qubits,
            depth,
            backend_arg,
            density_2q_gates,
            two_qubit_gate_ensemble,
            clifford_sqg_probability,
            sqg_gate_ensemble,
        )

        # Initialize the quantum circuit object
        circ = QuantumCircuit(num_qubits)

        # Add the edge Clifford
        circ.compose(clifford_layer.to_instruction(), qubits=list(range(num_qubits)), inplace=True)
        circ.barrier()

        # Add the cycle layers
        for k in range(depth):
            circ.compose(cycle_layers[k], inplace=True)
            circ.barrier()

        # Add the inverse Clifford
        circ.compose(
            Clifford(circ.to_instruction().inverse()).to_instruction(), qubits=list(range(num_qubits)), inplace=True
        )
        # Similarly, here the DRB paper contains a stabilizer measurement, determined in a more elaborated way.
        # The stabilizer measurement should effectively render the circuit to a Pauli gate (here always the identity).
        # Would need to modify this for larger num qubits !
        # Here, for 2-qubit DRB subroutines, it *should* suffice (in principle) to compile the inverse.

        circ_untransp = circ.copy()
        # Add measurements to untranspiled - after!
        # THIS LINE IS ONLY NEEDED IF STABILIZER MEASUREMENT IS NOT TAKEN TO IDENTITY
        # circ_untranspiled = transpile(Clifford(circ_untransp).to_circuit(), simulator)
        circ_untranspiled = circ_untransp
        circ_untranspiled.measure_all()

        # Add measurements to transpiled - before!
        circ.measure_all()

        circ_transpiled = transpile(
            circ,
            backend=retrieved_backend,
            coupling_map=effective_coupling_map,
            optimization_level=qiskit_optim_level,
            initial_layout=qubits,
            routing_method=routing_method,
        )

        drb_circuits_untranspiled.append(circ_untranspiled)
        drb_circuits_transpiled.append(circ_transpiled)

    # Store the circuits
    all_circuits.update(
        {
            "untranspiled": drb_circuits_untranspiled,
            "transpiled": drb_circuits_transpiled,
        }
    )

    return all_circuits


@timeit
def generate_fixed_depth_parallel_drb_circuits(
    qubits_array: Sequence[Sequence[int]],
    depth: int,
    num_circuit_samples: int,
    backend_arg: str | IQMBackendBase,
    assigned_density_2q_gates: Dict[str, float],
    assigned_two_qubit_gate_ensembles: Dict[str, Dict[str, float]],
    assigned_clifford_sqg_probabilities: Dict[str, float],
    assigned_sqg_gate_ensembles: Dict[str, Dict[str, float]],
    qiskit_optim_level: int = 3,
    routing_method: Literal["basic", "lookahead", "stochastic", "sabre", "none"] = "basic",
    eplg: bool = False
) -> Dict[str, List[QuantumCircuit]]:
    """Generates DRB circuits in parallel on multiple qubit layouts.
        The circuits follow a layered pattern with barriers, taylored to measured EPLG (arXiv:2311.05933),
        with layers of random Cliffords interleaved among sampled layers of 2Q gates and sequence inversion.

    Args:
        qubits_array (Sequence[Sequence[int]]): The array of physical qubit layouts on which to generate parallel DRB circuits.
        depth (int): The depth (number of canonical DRB layers) of the circuits.
        num_circuit_samples (int): The number of DRB circuits to generate.
        backend_arg (str | IQMBackendBase): The backend on which to generate the circuits.
        assigned_density_2q_gates (Dict[str, float]): The expected densities of 2-qubit gates in the final circuits per qubit layout.
        assigned_two_qubit_gate_ensembles (Dict[str, Dict[str, float]]): The two-qubit gate ensembles to use in the random DRB circuits per qubit layout.
        assigned_clifford_sqg_probabilities (Dict[str, float]): Probability with which to uniformly sample Clifford 1Q gates per qubit layout.
        assigned_sqg_gate_ensembles (Dict[str, Dict[str, float]]): A dictionary with keys being str specifying 1Q gates, and values being corresponding probabilities per qubit layout.
        qiskit_optim_level (int): Qiskit transpiler optimization level.
                        * Defaults to 1.
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): Qiskit transpiler routing method.
                        * Default is "basic".
        eplg (bool): Whether the circuits belong to an EPLG experiment.
                        * If True a single layer is generated.
                        * Default is False.
    Returns:
        Dict[str, List[QuantumCircuit]]: A dictionary of untranspiled and transpiled lists of parallel (simultaneous) DRB circuits.
    """
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    # Check if backend includes MOVE gates and set coupling map
    flat_qubits_array = [x for y in qubits_array for x in y]
    if "move" in backend.operation_names:
        # All-to-all coupling map on the active qubits
        effective_coupling_map = [[x, y] for x in flat_qubits_array for y in flat_qubits_array if x != y]
    else:
        effective_coupling_map = backend.coupling_map

    # Identify total amount of qubits
    qubit_counts = [len(x) for x in qubits_array]

    # Shuffle qubits_array: we don't want unnecessary qubit registers
    shuffled_qubits_array = relabel_qubits_array_from_zero(qubits_array)
    # The total amount of qubits the circuits will have
    n_qubits = sum(qubit_counts)

    # Generate the circuit samples
    # Initialize the list of circuits
    all_circuits = {}
    drb_circuits_untranspiled: List[QuantumCircuit] = []
    drb_circuits_transpiled: List[QuantumCircuit] = []

    # simulator = AerSimulator(method=simulation_method)

    # Generate the layer if EPLG: this will be repeated in all samples and all depths!
    cycle_layers = {}
    if eplg:
        for q_idx, q in enumerate(shuffled_qubits_array):
            original_qubits = str(qubits_array[q_idx])
            cycle_layers[str(q)] = edge_grab(
                qubits_array[q_idx],
                depth,
                backend_arg,
                assigned_density_2q_gates[original_qubits],
                assigned_two_qubit_gate_ensembles[original_qubits],
                assigned_clifford_sqg_probabilities[original_qubits],
                assigned_sqg_gate_ensembles[original_qubits],
            )

    for _ in range(num_circuit_samples):
        # Initialize the quantum circuit object
        circ = QuantumCircuit(n_qubits)

        # Generate small circuits to track inverses
        local_circs = {str(q): QuantumCircuit(len(q)) for q in shuffled_qubits_array}

        # Sample the layers if EPLG is False.
        if not eplg:
            for q_idx, q in enumerate(shuffled_qubits_array):
                original_qubits = str(qubits_array[q_idx])
                cycle_layers[str(q)] = edge_grab(
                    qubits_array[q_idx],
                    depth,
                    backend_arg,
                    assigned_density_2q_gates[original_qubits],
                    assigned_two_qubit_gate_ensembles[original_qubits],
                    assigned_clifford_sqg_probabilities[original_qubits],
                    assigned_sqg_gate_ensembles[original_qubits],
                )

        # Add the cycle layers
        for k in range(depth):
            # Add the edge Clifford
            # The DRB paper here contains a general stabilizer preparation.
            # We will stick to 1Q Clifford gates for now.
            for q in shuffled_qubits_array:
                for idx, i in enumerate(q):
                    rand_clif = random_clifford(1)
                    circ.compose(rand_clif.to_instruction(), qubits=[i], inplace=True)
                    local_circs[str(q)].compose(rand_clif.to_instruction(), qubits=[idx], inplace=True)
            circ.barrier()

            for q in shuffled_qubits_array:
                circ.compose(cycle_layers[str(q)][k], qubits=q, inplace=True)
                local_circs[str(q)].compose(cycle_layers[str(q)][k], inplace=True)
            circ.barrier()

        # Add the inverse Clifford
        for q in shuffled_qubits_array:
            circ.compose(
                Clifford(local_circs[str(q)].to_instruction().inverse()).to_instruction(), qubits=q, inplace=True
            )
        circ.barrier()
        for q_idx, q in enumerate(shuffled_qubits_array):
            original_qubits = str(qubits_array[q_idx])
            local_register = ClassicalRegister(len(q), original_qubits)
            circ.add_register(local_register)
            circ.measure(q, local_register)
        # Similarly, here the DRB paper contains a stabilizer measurement, determined in a more elaborated way.
        # The stabilizer measurement should effectively render the circuit to a Pauli gate (here always the identity).
        # Would need to modify this for larger num qubits !
        # Here, for 2-qubit DRB subroutines, it *should* suffice (in principle) to compile the inverse.

        circ_untranspiled = circ.copy()

        circ_transpiled = transpile(
            circ,
            backend=backend,
            coupling_map=effective_coupling_map,
            optimization_level=qiskit_optim_level,
            initial_layout=flat_qubits_array,
            routing_method=routing_method,
        )

        drb_circuits_untranspiled.append(circ_untranspiled)
        drb_circuits_transpiled.append(circ_transpiled)

    # Store the circuits
    all_circuits.update(
        {
            "untranspiled": drb_circuits_untranspiled,
            "transpiled": drb_circuits_transpiled,
        }
    )
    return all_circuits


def direct_rb_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Direct RB analysis function

    Args:
        run (BenchmarkRunResult): The result of the benchmark run.

    Returns:
        AnalysisResult corresponding to DRB.
    """

    dataset = run.dataset.copy(deep=True)
    observations: list[BenchmarkObservation] = []
    obs_dict = {}
    plots = {}

    is_parallel_execution = dataset.attrs["parallel_execution"]
    qubits_array = dataset.attrs["qubits_array"]
    depths = dataset.attrs["depths"]

    num_circuit_samples = dataset.attrs["num_circuit_samples"]

    density_2q_gates = dataset.attrs["densities_2q_gates"]
    two_qubit_gate_ensemble = dataset.attrs["two_qubit_gate_ensembles"]

    all_noisy_counts: Dict[str, Dict[int, List[Dict[str, int]]]] = {}

    polarizations: Dict[str, Dict[int, List[float]]] = {str(q): {} for q in qubits_array}

    if is_parallel_execution:
        qcvv_logger.info(f"Post-processing parallel Direct RB on qubits {qubits_array}.")
        all_noisy_counts[str(qubits_array)] = {}
        for depth in depths:
            identifier = f"qubits_{str(qubits_array)}_depth_{str(depth)}"
            all_noisy_counts[str(qubits_array)][depth] = xrvariable_to_counts(dataset, identifier, num_circuit_samples)

            qcvv_logger.info(f"Depth {depth}")

            # Retrieve the marginalized survival probabilities
            all_survival_probabilities = survival_probabilities_parallel(
                qubits_array, all_noisy_counts[str(qubits_array)][depth], separate_registers=True
            )

            # The marginalized survival probabilities will be arranged by qubit layouts
            for qubits_str in all_survival_probabilities.keys():
                polarizations[qubits_str][depth] = all_survival_probabilities[qubits_str]
            # Remaining analysis is the same regardless of whether execution was in parallel or sequential
    else:  # sequential
        qcvv_logger.info(f"Post-processing sequential Direct RB for qubits {qubits_array}")
        for q in qubits_array:
            all_noisy_counts[str(q)] = {}
            num_qubits = len(q)
            polarizations[str(q)] = {}
            for depth in depths:
                identifier = f"qubits_{str(q)}_depth_{str(depth)}"
                all_noisy_counts[str(q)][depth] = xrvariable_to_counts(dataset, identifier, num_circuit_samples)

                qcvv_logger.info(f"Qubits {q} and depth {depth}")
                polarizations[str(q)][depth] = get_survival_probabilities(num_qubits, all_noisy_counts[str(q)][depth])
                # Remaining analysis is the same regardless of whether execution was in parallel or sequential

    # All remaining (fitting & plotting) is done per qubit layout
    for qubits_idx, qubits in enumerate(qubits_array):
        # Fit decays
        list_of_polarizations = list(polarizations[str(qubits)].values())
        fit_data, fit_parameters = fit_decay_lmfit(exponential_rb, qubits, list_of_polarizations, "drb")
        rb_fit_results = lmfit_minimizer(fit_parameters, fit_data, depths, exponential_rb)

        average_polarizations = {d: np.mean(polarizations[str(qubits)][d]) for d in depths}
        stddevs_from_mean = {d: np.std(polarizations[str(qubits)][d]) / np.sqrt(num_circuit_samples) for d in depths}
        popt = {
            "amplitude": rb_fit_results.params["amplitude_1"],
            "offset": rb_fit_results.params["offset_1"],
            "decay_rate": rb_fit_results.params["p_drb"],
        }
        fidelity = rb_fit_results.params["fidelity_drb"]

        processed_results = {
            "avg_gate_fidelity": {"value": fidelity.value, "uncertainty": fidelity.stderr},
        }

        dataset.attrs[qubits_idx].update(
            {
                "decay_rate": {"value": popt["decay_rate"].value, "uncertainty": popt["decay_rate"].stderr},
                "fit_amplitude": {"value": popt["amplitude"].value, "uncertainty": popt["amplitude"].stderr},
                "fit_offset": {"value": popt["offset"].value, "uncertainty": popt["offset"].stderr},
                "polarizations": polarizations[str(qubits)],
                "avg_polarization_nominal_values": average_polarizations,
                "avg_polatization_stderr": stddevs_from_mean,
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
        fig_name, fig = plot_rb_decay(
            identifier="drb",
            qubits_array=[qubits],
            dataset=dataset,
            observations=obs_dict,
            mrb_2q_density=density_2q_gates,  # Misnomer coming from MRB - ignore
            mrb_2q_ensemble=two_qubit_gate_ensemble,
        )
        plots[fig_name] = fig

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


class DirectRandomizedBenchmarking(Benchmark):
    """Direct RB estimates the fidelity of layers of canonical gates"""

    analysis_function = staticmethod(direct_rb_analysis)

    name: str = "direct_rb"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "DirectRBConfiguration"):
        """Construct the DirectRandomizedBenchmarking class

        Args:
            backend_arg (IQMBackendBase | str): _description_
            configuration (MirrorRBConfiguration): _description_
        """
        super().__init__(backend_arg, configuration)

        # EXPERIMENT
        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name

        self.qubits_array = configuration.qubits_array
        self.eplg = configuration.eplg

        # Override if EPLG is True but parallel_execution was set to False
        if self.eplg and not configuration.parallel_execution:
            configuration.parallel_execution = True

        self.parallel_execution = configuration.parallel_execution
        self.depths = configuration.depths
        self.num_circuit_samples = configuration.num_circuit_samples

        self.two_qubit_gate_ensembles = configuration.two_qubit_gate_ensembles
        self.densities_2q_gates = configuration.densities_2q_gates
        self.clifford_sqg_probabilities = configuration.clifford_sqg_probabilities
        self.sqg_gate_ensembles = configuration.sqg_gate_ensembles

        self.qiskit_optim_level = configuration.qiskit_optim_level

        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

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
        dataset.attrs["two_qubit_gate_ensembles"] = self.two_qubit_gate_ensembles
        dataset.attrs["densities_2q_gates"] = self.densities_2q_gates
        dataset.attrs["clifford_sqg_probabilities"] = self.clifford_sqg_probabilities
        dataset.attrs["sqg_gate_ensembles"] = self.sqg_gate_ensembles

    def assign_inputs_to_qubits(self):  # pylint: disable=too-many-branches
        """Assigns all DRB inputs (Optional[Sequence[Any]]) to input qubit layouts.
        Args:
        Returns:

        """
        # Depths - can be modified as in MRB to be qubit layout-dependent
        assigned_drb_depths = self.depths

        # 2Q gate ensemble
        if self.two_qubit_gate_ensembles is None:
            assigned_two_qubit_gate_ensembles = {str(q): {"CZGate": 1.0} for q in self.qubits_array}
        else:
            if len(self.two_qubit_gate_ensembles.values()) != len(self.qubits_array):
                if len(self.two_qubit_gate_ensembles.values()) != 1:
                    qcvv_logger.warning(
                        f"The amount of 2Q gate ensembles ({len(self.two_qubit_gate_ensembles)}) is not the same "
                        f"as the amount of qubit layout configurations ({len(self.qubits_array)}):\n\tWill assign to all the first "
                        f"configuration: {self.two_qubit_gate_ensembles[0]} !"
                    )
                assigned_two_qubit_gate_ensembles = {
                    str(q): self.two_qubit_gate_ensembles[0] for q in self.qubits_array
                }
            else:
                assigned_two_qubit_gate_ensembles = {
                    str(q): self.two_qubit_gate_ensembles[q_idx] for q_idx, q in enumerate(self.qubits_array)
                }

        # Density 2Q gates
        if self.densities_2q_gates is None and self.eplg:
            assigned_density_2q_gates = {str(q): 0.5 for q in self.qubits_array}
        elif self.densities_2q_gates is None:
            assigned_density_2q_gates = {str(q): 0.25 for q in self.qubits_array}
        else:
            if len(self.densities_2q_gates) != len(self.qubits_array):
                if len(self.densities_2q_gates) != 1:
                    qcvv_logger.warning(
                        f"The amount of 2Q gate densities ({len(self.densities_2q_gates)}) is not the same "
                        f"as the amount of qubit layout configurations ({len(self.qubits_array)}):\n\tWill assign to all the first "
                        f"configuration: {self.densities_2q_gates[0]} !"
                    )
                assigned_density_2q_gates = {str(q): self.densities_2q_gates[0] for q in self.qubits_array}
            else:
                assigned_density_2q_gates = {
                    str(q): self.densities_2q_gates[q_idx] for q_idx, q in enumerate(self.qubits_array)
                }

        # clifford_sqg_probabilities
        if self.clifford_sqg_probabilities is None and self.eplg:
            assigned_clifford_sqg_probabilities = {str(q): 0.0 for q in self.qubits_array}
        elif self.clifford_sqg_probabilities is None:
            assigned_clifford_sqg_probabilities = {str(q): 1.0 for q in self.qubits_array}
        else:
            if len(self.clifford_sqg_probabilities) != len(self.qubits_array):
                if len(self.clifford_sqg_probabilities) != 1:
                    qcvv_logger.warning(
                        f"The amount of Clifford 1Q gate sampling probabilities ({len(self.clifford_sqg_probabilities)}) is not the same "
                        f"as the amount of qubit layout configurations ({len(self.qubits_array)}):\n\tWill assign to all the first "
                        f"configuration: {self.clifford_sqg_probabilities[0]} !"
                    )
                assigned_clifford_sqg_probabilities = {
                    str(q): self.clifford_sqg_probabilities[0] for q in self.qubits_array
                }
            else:
                assigned_clifford_sqg_probabilities = {
                    str(q): self.clifford_sqg_probabilities[q_idx] for q_idx, q in enumerate(self.qubits_array)
                }

        # sqg_gate_ensembles
        if self.sqg_gate_ensembles is not None:
            if len(self.sqg_gate_ensembles) != len(self.qubits_array):
                if len(self.sqg_gate_ensembles) != 1:
                    qcvv_logger.warning(
                        f"The amount of 1Q gate ensembles ({len(self.sqg_gate_ensembles)}) is not the same "
                        f"as the amount of qubit layout configurations ({len(self.qubits_array)}):\n\tWill assign to all the first "
                        f"configuration: {self.sqg_gate_ensembles[0]} !"
                    )
                assigned_sqg_gate_ensembles = {str(q): self.sqg_gate_ensembles[0] for q in self.qubits_array}
            else:
                assigned_sqg_gate_ensembles = {
                    str(q): self.sqg_gate_ensembles[q_idx] for q_idx, q in enumerate(self.qubits_array)
                }
        elif self.sqg_gate_ensembles is None and self.eplg:  # No Cliffords and no 1Q gates in layers
            assigned_sqg_gate_ensembles = {str(q): {"IGate": 1.0} for q in self.qubits_array}
        elif self.sqg_gate_ensembles is None:  # all are Clifford
            assigned_sqg_gate_ensembles = {str(q): {"HGate": 0.0} for q in self.qubits_array}
        else:
            assigned_sqg_gate_ensembles = {
                str(q): {"HGate": 1.0 - assigned_clifford_sqg_probabilities[str(q)]} for q in self.qubits_array
            }

        # Reset the configuration values to store in dataset
        self.two_qubit_gate_ensembles = assigned_two_qubit_gate_ensembles
        self.densities_2q_gates = assigned_density_2q_gates
        self.clifford_sqg_probabilities = assigned_clifford_sqg_probabilities
        self.sqg_gate_ensembles = assigned_sqg_gate_ensembles

        return (
            assigned_drb_depths,
            assigned_two_qubit_gate_ensembles,
            assigned_density_2q_gates,
            assigned_clifford_sqg_probabilities,
            assigned_sqg_gate_ensembles,
        )

    def submit_single_drb_job(
        self,
        backend_arg: IQMBackendBase,
        qubits: Sequence[int],
        depth: int,
        sorted_transpiled_circuit_dicts: Dict[Tuple[int, ...], List[QuantumCircuit]],
    ) -> Dict[str, Any]:
        """
        Submit fixed-depth DRB jobs for execution in the specified IQMBackend

        Args:
            backend_arg (IQMBackendBase): the IQM backend to submit the job
            qubits (Sequence[int]): the qubits to identify the submitted job
            depth (int): the depth (number of canonical layers) of the circuits to identify the submitted job
            sorted_transpiled_circuit_dicts (Dict[str, List[QuantumCircuit]]): A dictionary containing all MRB circuits
        Returns:
            Dict with qubit layout, submitted job objects, type (vanilla/DD) and submission time
        """
        # Submit
        # Send to execute on backend
        execution_jobs, time_submit = submit_execute(
            sorted_transpiled_circuit_dicts,
            backend_arg,
            self.shots,
            self.calset_id,
            max_gates_per_batch=self.max_gates_per_batch,
        )
        drb_submit_results = {
            "qubits": qubits,
            "depth": depth,
            "jobs": execution_jobs,
            "time_submit": time_submit,
        }
        return drb_submit_results

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:  # pylint: disable=too-many-statements
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()

        (
            assigned_drb_depths,
            assigned_two_qubit_gate_ensembles,
            assigned_density_2q_gates,
            assigned_clifford_sqg_probabilities,
            assigned_sqg_gate_ensembles,
        ) = self.assign_inputs_to_qubits()

        self.add_all_meta_to_dataset(dataset)

        # Submit jobs for all qubit layouts
        all_drb_jobs: List[Dict[str, Any]] = []
        time_circuit_generation: Dict[str, float] = {}

        # Auxiliary dict from str(qubits) to indices
        qubit_idx: Dict[str, Any] = {}

        # Main execution
        if self.parallel_execution:
            # Take the whole qubits_array and do DRB in parallel on each qubits_array element
            parallel_drb_circuits = {}
            qcvv_logger.info(
                f"Executing parallel Direct RB on qubits {self.qubits_array}."
                f" Will generate and submit all {self.num_circuit_samples} DRB circuits"
                f" for each depth {self.depths}"
            )

            time_circuit_generation[str(self.qubits_array)] = 0
            # Generate and submit all circuits
            for depth in self.depths:
                qcvv_logger.info(f"Depth {depth}")
                parallel_drb_circuits[depth], elapsed_time = generate_fixed_depth_parallel_drb_circuits(
                    qubits_array=self.qubits_array,
                    depth=depth,
                    num_circuit_samples=self.num_circuit_samples,
                    backend_arg=backend,
                    assigned_density_2q_gates=assigned_density_2q_gates,
                    assigned_two_qubit_gate_ensembles=assigned_two_qubit_gate_ensembles,
                    assigned_clifford_sqg_probabilities=assigned_clifford_sqg_probabilities,
                    assigned_sqg_gate_ensembles=assigned_sqg_gate_ensembles,
                    qiskit_optim_level=self.qiskit_optim_level,
                    routing_method=self.routing_method,
                    eplg=self.eplg
                )
                time_circuit_generation[str(self.qubits_array)] += elapsed_time

                # Submit all
                flat_qubits_array = [x for y in self.qubits_array for x in y]
                sorted_transpiled_qc_list = {tuple(flat_qubits_array): parallel_drb_circuits[depth]["transpiled"]}
                all_drb_jobs.append(
                    submit_parallel_rb_job(
                        backend,
                        self.qubits_array,
                        depth,
                        sorted_transpiled_qc_list,
                        self.shots,
                        self.calset_id,
                        self.max_gates_per_batch,
                    )
                )
                qcvv_logger.info(f"Job for depth {depth} submitted successfully!")

                self.untranspiled_circuits.circuit_groups.append(
                    CircuitGroup(
                        name=f"{str(self.qubits_array)}_depth_{depth}",
                        circuits=parallel_drb_circuits[depth]["untranspiled"],
                    )
                )
                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(
                        name=f"{str(self.qubits_array)}_depth_{depth}",
                        circuits=parallel_drb_circuits[depth]["transpiled"],
                    )
                )
            qubit_idx = {str(self.qubits_array): "parallel_all"}
            dataset.attrs["parallel_all"] = {"qubits": self.qubits_array}
            dataset.attrs.update({q_idx: {"qubits": q} for q_idx, q in enumerate(self.qubits_array)})
        else:  # if sequential
            for qubits_idx, qubits in enumerate(self.qubits_array):
                qubit_idx[str(qubits)] = qubits_idx

                qcvv_logger.info(
                    f"Executing DRB on qubits {qubits}."
                    f" Will generate and submit all {self.num_circuit_samples} DRB circuits"
                    f" for depths {assigned_drb_depths}"
                )
                drb_circuits = {}
                drb_transpiled_circuits_lists: Dict[int, List[QuantumCircuit]] = {}
                drb_untranspiled_circuits_lists: Dict[int, List[QuantumCircuit]] = {}
                time_circuit_generation[str(qubits)] = 0
                for depth in assigned_drb_depths:
                    qcvv_logger.info(f"Depth {depth} - Generating all circuits")
                    drb_circuits[depth], elapsed_time = generate_drb_circuits(
                        qubits,
                        depth=depth,
                        circ_samples=self.num_circuit_samples,
                        backend_arg=backend,
                        density_2q_gates=assigned_density_2q_gates[str(qubits)],
                        two_qubit_gate_ensemble=assigned_two_qubit_gate_ensembles[str(qubits)],
                        clifford_sqg_probability=assigned_clifford_sqg_probabilities[str(qubits)],
                        sqg_gate_ensemble=assigned_sqg_gate_ensembles[str(qubits)],
                        qiskit_optim_level=self.qiskit_optim_level,
                        routing_method=self.routing_method,
                    )
                    time_circuit_generation[str(qubits)] += elapsed_time

                    # Generated circuits at fixed depth are (dict) indexed by Pauli sample number, turn into List
                    drb_transpiled_circuits_lists[depth] = drb_circuits[depth]["transpiled"]
                    drb_untranspiled_circuits_lists[depth] = drb_circuits[depth]["untranspiled"]

                    # Submit
                    sorted_transpiled_qc_list = {tuple(qubits): drb_transpiled_circuits_lists[depth]}
                    all_drb_jobs.append(self.submit_single_drb_job(backend, qubits, depth, sorted_transpiled_qc_list))

                    qcvv_logger.info(f"Job for layout {qubits} & depth {depth} submitted successfully!")

                    self.untranspiled_circuits.circuit_groups.append(
                        CircuitGroup(
                            name=f"{str(qubits)}_depth_{depth}", circuits=drb_untranspiled_circuits_lists[depth]
                        )
                    )
                    self.transpiled_circuits.circuit_groups.append(
                        CircuitGroup(name=f"{str(qubits)}_depth_{depth}", circuits=drb_transpiled_circuits_lists[depth])
                    )

                dataset.attrs[qubits_idx] = {"qubits": qubits}

        # Retrieve counts of jobs for all qubit layouts
        for job_dict in all_drb_jobs:
            qubits = job_dict["qubits"]
            depth = job_dict["depth"]
            # Retrieve counts
            execution_results, time_retrieve = retrieve_all_counts(
                job_dict["jobs"], f"qubits_{str(qubits)}_depth_{str(depth)}"
            )
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
            dataset, _ = add_counts_to_dataset(execution_results, f"qubits_{str(qubits)}_depth_{str(depth)}", dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        qcvv_logger.info(f"DRB experiment execution concluded !")

        return dataset


class DirectRBConfiguration(BenchmarkConfigurationBase):
    """Direct RB configuration

    Attributes:
        benchmark (Type[Benchmark]): DirectRandomizedBenchmarking.
        qubits_array (Sequence[Sequence[int]]): The array of physical qubits in which to execute DRB.
        eplg (bool): Whether the DRB experiment is executed as a EPLG subroutine.
                            * If True:
                                - default parallel_execution below is override to True.
                                - default two_qubit_gate_ensembles is {"CZGate": 1.0}.
                                - default densities_2q_gates is 0.5 (probability of sampling 2Q gates is 1).
                                - default clifford_sqg_probabilities is 0.0.
                                - default sqg_gate_ensembles is {"IGate": 1.0}.
                            * Default is False.
        parallel_execution (bool): Whether DRB is executed in parallel for all qubit layouts in qubits_array.
                            * If eplg is False, it executes parallel DRB with MRB gate ensemble and density defaults.
                            * Default is False.
        depths (Sequence[int]): The list of layer depths in which to execute DRB for all qubit layouts in qubits_array.
        num_circuit_samples (int): The number of random-layer DRB circuits to generate.
        shots (int): The number of measurement shots to execute per circuit.
        qiskit_optim_level (int): The Qiskit-level of optimization to use in transpilation.
                            * Default is 1.
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): The routing method to use in transpilation.
                            * Default is "sabre".
        two_qubit_gate_ensembles (Optional[Sequence[Dict[str, float]]]): The two-qubit gate ensembles to use in the random DRB circuits.
                            * Keys correspond to str names of qiskit circuit library gates, e.g., "CZGate" or "CXGate".
                            * Values correspond to the probability for the respective gate to be sampled.
                            * Each Dict[str,float] corresponds to each qubit layout in qubits_array.
                            * If len(two_qubit_gate_ensembles.values()) != len(qubits_array), the first Dict is assinged by default.
                            * Default is None, which assigns {str(q): {"CZGate": 1.0} for q in qubits_array}.
        densities_2q_gates (Optional[Sequence[float]]): The expected densities of 2-qubit gates in the final circuits per qubit layout.
                            * If len(densities_2q_gates) != len(qubits_array), the first density value is assinged by default.
                            * Default is None, which assigns 0.25 to all qubit layouts.
        clifford_sqg_probabilities (Optional[Sequence[float]]): Probability with which to uniformly sample Clifford 1Q gates per qubit layout.
                            * Default is None, which assigns 1.0 to all qubit layouts.
        sqg_gate_ensembles (Optional[Sequence[Dict[str, float]]]): A dictionary with keys being str specifying 1Q gates, and values being corresponding probabilities.
                            * If len(sqg_gate_ensembles) != len(qubits_array), the first ensemble is assinged by default.
                            * Default is None, which leaves only uniform sampling of 1Q Clifford gates.
    """

    benchmark: Type[Benchmark] = DirectRandomizedBenchmarking
    qubits_array: Sequence[Sequence[int]]
    eplg: bool = False
    parallel_execution: bool = False
    depths: Sequence[int]
    num_circuit_samples: int
    qiskit_optim_level: int = 1
    two_qubit_gate_ensembles: Optional[Sequence[Dict[str, float]]] = None
    densities_2q_gates: Optional[Sequence[float]] = None
    clifford_sqg_probabilities: Optional[Sequence[float]] = None
    sqg_gate_ensembles: Optional[Sequence[Dict[str, float]]] = None
