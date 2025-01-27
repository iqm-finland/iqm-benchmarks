import warnings
from time import strftime
from typing import Type, Sequence, Dict, Optional, Literal, List, Any, Tuple

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import xarray as xr
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford, Clifford
from qiskit_aer import AerSimulator

from iqm.benchmarks import Benchmark, BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import add_counts_to_dataset
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.randomized_benchmarking_common import edge_grab
from iqm.benchmarks.utils import retrieve_all_job_metadata, retrieve_all_counts, get_iqm_backend, \
    perform_backend_transpilation, submit_execute


def generate_drb_circuits(
    qubits: List[int],
    depth: int,
    circ_samples: int,
    backend_arg: IQMBackendBase | str,
    density_2q_gates: float = 0.25,
    two_qubit_gate_ensemble: Optional[Dict[str, float]] = None,
    clifford_sqg_probability: float = 1.0,
    sqg_gate_ensemble: Optional[Dict[str, float]] = None,
    qiskit_optim_level: int = 1,
    routing_method: str = "basic",
    simulation_method: Literal["automatic", "statevector", "stabilizer", "extended_stabilizer", "matrix_product_state"] = "automatic"
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
        routing_method (str): Qiskit transpiler routing method.
                * Default is "basic".
        simulation_method (Literal["automatic", "statevector", "stabilizer", "extended_stabilizer", "matrix_product_state"]):
                Qiskit's Aer simulation method
                * Default is "automatic".
    Returns:
        Dict[str, List[QuantumCircuit]]: a dictionary with keys "transpiled", "untranspiled" and values a list of respective DRB circuits.
    """
    num_qubits = len(qubits)

    # Transpile to backend - no optimize SQG should be used!
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

    simulator = AerSimulator(method=simulation_method)

    for _ in range(circ_samples):
        # Sample Clifford for stabilizer preparation
        clifford_layer = random_clifford(num_qubits)
        # NB: The DRB paper contains a more elaborated stabilizer compilation algo.
        # Not having it WILL be an issue here for larger num qubits !
        # Intended usage, however, is solely for 2-qubit subroutines.

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
        print("Using stabilizer circ")
        circ.compose(clifford_layer.to_instruction(), qubits=list(range(num_qubits)), inplace=True)
        circ.barrier()

        # Add the cycle layers
        for k in range(depth):
            circ.compose(cycle_layers[k], inplace=True)
            circ.barrier()

        # Add the inverse Clifford
        circ.compose(transpile(Clifford(circ.to_instruction().inverse()).to_circuit(), AerSimulator(method="stabilizer")), qubits=list(range(num_qubits)), inplace=True)
        # Similarly, here the DRB paper contains a stabilizer measurement, determined in a more elaborated way.
        # Would need to modify this for larger num qubits ! Stabilizer measurement should effectively render the circuit to a Pauli gate.
        # Here, for 2-qubit subroutines, it *should* suffice (in principle) to compile the inverse.

        circ_untransp = circ.copy()
        # Add measurements to untranspiled - after!
        circ_untranspiled = transpile(Clifford(circ_untransp).to_circuit(), simulator)
        circ_untranspiled.measure_all()

        # Add measurements to transpiled - before!
        circ.measure_all()

        circ_transpiled, _ = perform_backend_transpilation(
            [circ],
            backend=retrieved_backend,
            qubits=qubits,
            coupling_map=effective_coupling_map,
            qiskit_optim_level=qiskit_optim_level,
            routing_method=routing_method,
        )

        drb_circuits_untranspiled.append(circ_untranspiled)
        drb_circuits_transpiled.append(circ_transpiled[0])

    # Store the circuits
    all_circuits.update(
        {
            "untranspiled": drb_circuits_untranspiled,
            "transpiled": drb_circuits_transpiled,
        }
    )

    return all_circuits



class DirectRandomizedBenchmarking(Benchmark):
    """Direct RB estimates the fidelity of layers of canonical gates
    """

    # analysis_function = staticmethod(mrb_analysis)

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
        self.depths_array = configuration.depths_array
        self.num_circuit_samples = configuration.num_circuit_samples

        self.two_qubit_gate_ensemble = configuration.two_qubit_gate_ensemble
        self.density_2q_gates = configuration.density_2q_gates
        self.clifford_sqg_probability = configuration.clifford_sqg_probability
        self.sqg_gate_ensemble = configuration.sqg_gate_ensemble

        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.simulation_method = configuration.simulation_method

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


    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

        # Submit jobs for all qubit layouts
        all_drb_jobs: List[Dict[str, Any]] = []
        time_circuit_generation: Dict[str, float] = {}

        # The depths should be assigned to each set of qubits!
        assigned_drb_depths = {}
        if len(self.qubits_array) != len(self.depths_array):
            # If user did not specify a list of depth for each list of qubits, assign the first
            # If the len is not one, the input was incorrect
            if len(self.depths_array) != 1:
                warnings.warn(
                    f"The amount of qubit layouts ({len(self.qubits_array)}) is not the same "
                    f"as the amount of depth configurations ({len(self.depths_array)}):\n\tWill assign to all the first "
                    f"configuration: {self.depths_array[0]} !"
                )
            assigned_drb_depths = {str(q): self.depths_array[0] for q in self.qubits_array}
        else:
            assigned_drb_depths = self.depths_array

        # Auxiliary dict from str(qubits) to indices
        qubit_idx: Dict[str, Any] = {}
        for qubits_idx, qubits in enumerate(self.qubits_array):
            qubit_idx[str(qubits)] = qubits_idx

            qcvv_logger.info(
                f"Executing DRB on qubits {qubits}."
                f" Will generate and submit all {self.num_circuit_samples} DRB circuits"
                f" for each depth {assigned_drb_depths[str(qubits)]}"
            )
            drb_circuits = {}
            drb_transpiled_circuits_lists: Dict[int, List[QuantumCircuit]] = {}
            drb_untranspiled_circuits_lists: Dict[int, List[QuantumCircuit]] = {}
            time_circuit_generation[str(qubits)] = 0
            for depth in assigned_drb_depths[str(qubits)]:
                qcvv_logger.info(f"Depth {depth} - Generating all circuits")
                drb_circuits[depth], elapsed_time = generate_drb_circuits(
                    qubits,
                    depth=depth,
                    circ_samples=self.num_circuit_samples,
                    backend_arg=backend,
                    density_2q_gates=self.density_2q_gates,
                    two_qubit_gate_ensemble=self.two_qubit_gate_ensemble,
                    clifford_sqg_probability=self.clifford_sqg_probability,
                    sqg_gate_ensemble=self.sqg_gate_ensemble,
                    qiskit_optim_level=self.qiskit_optim_level,
                    routing_method=self.routing_method,
                    simulation_method=self.simulation_method
                )
                time_circuit_generation[str(qubits)] += elapsed_time

                # Generated circuits at fixed depth are (dict) indexed by Pauli sample number, turn into List
                drb_transpiled_circuits_lists[depth] = []
                drb_untranspiled_circuits_lists[depth] = []
                for c_s in range(self.num_circuit_samples):
                    drb_transpiled_circuits_lists[depth].extend(drb_circuits[depth][c_s]["transpiled"])
                for c_s in range(self.num_circuit_samples):
                    drb_untranspiled_circuits_lists[depth].extend(drb_circuits[depth][c_s]["untranspiled"])

                # Submit
                sorted_transpiled_qc_list = {tuple(qubits): drb_transpiled_circuits_lists[depth]}
                all_drb_jobs.append(self.submit_single_drb_job(backend, qubits, depth, sorted_transpiled_qc_list))
                qcvv_logger.info(f"Job for layout {qubits} & depth {depth} submitted successfully!")

                self.untranspiled_circuits.circuit_groups.append(
                    CircuitGroup(name=f"{str(qubits)}_depth_{depth}",
                                 circuits=drb_untranspiled_circuits_lists[depth])
                )
                self.transpiled_circuits.circuit_groups.append(
                    CircuitGroup(name=f"{str(qubits)}_depth_{depth}", circuits=drb_transpiled_circuits_lists[depth])
                )

            dataset.attrs[qubits_idx] = {"qubits": qubits}

        # Retrieve counts of jobs for all qubit layouts
        all_job_metadata = {}
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
            dataset, _ = add_counts_to_dataset(execution_results, f"qubits_{str(qubits)}_depth_{str(depth)}",
                                               dataset)

        self.circuits = Circuits([self.transpiled_circuits, self.untranspiled_circuits])

        qcvv_logger.info(f"MRB experiment execution concluded !")

        return dataset

class DirectRBConfiguration(BenchmarkConfigurationBase):
    """Direct RB configuration

    Attributes:
        benchmark (Type[Benchmark]): DirectRandomizedBenchmarking.
        qubits_array (Sequence[Sequence[int]]): The array of physical qubits in which to execute DRB.
        depths_array (Sequence[Sequence[int]]): The array of physical depths in which to execute DRB for a corresponding qubit list.
                            * If len is the same as that of qubits_array, each Sequence[int] corresponds to the depths for the corresponding layout of qubits.
                            * If len is different from that of qubits_array, assigns the first Sequence[int].
        num_circuit_samples (int): The number of random-layer mirror circuits to generate.
        shots (int): The number of measurement shots to execute per circuit.
        qiskit_optim_level (int): The Qiskit-level of optimization to use in transpilation.
                            * Default is 1.
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): The routing method to use in transpilation.
                            * Default is "sabre".
        two_qubit_gate_ensemble (Dict[str, float]): The two-qubit gate ensemble to use in the random mirror circuits.
                            * Keys correspond to str names of qiskit circuit library gates, e.g., "CZGate" or "CXGate".
                            * Values correspond to the probability for the respective gate to be sampled.
                            * Default is {"CZGate": 1.0}.
        density_2q_gates (float): The expected density of 2-qubit gates in the final circuits.
                            * Default is 0.25.
        clifford_sqg_probability (float): Probability with which to uniformly sample Clifford 1Q gates.
                * Default is 1.0.
        sqg_gate_ensemble (Optional[Dict[str, float]]): A dictionary with keys being str specifying 1Q gates, and values being corresponding probabilities.
                * Default is None.
        simulation_method (Literal["automatic", "statevector", "stabilizer", "extended_stabilizer", "matrix_product_state"]):
                            Qiskit's Aer simulation method
                            * Default is "automatic".
    """
    benchmark: Type[Benchmark] = DirectRandomizedBenchmarking
    qubits_array: Sequence[Sequence[int]]
    depths_array: Sequence[Sequence[int]]
    num_circuit_samples: int
    qiskit_optim_level: int = 1
    two_qubit_gate_ensemble: Dict[str, float] = None
    density_2q_gates: float = 0.25
    clifford_sqg_probability: float = 1.0
    sqg_gate_ensemble: Optional[Dict[str, float]] = None
    simulation_method: Literal[
        "automatic", "statevector", "stabilizer", "extended_stabilizer", "matrix_product_state"] = "automatic"