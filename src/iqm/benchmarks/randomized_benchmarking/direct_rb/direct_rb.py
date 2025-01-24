from time import strftime
from typing import Type, Sequence, Dict, Optional, Literal

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
import xarray as xr

from iqm.benchmarks import Benchmark
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase


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

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Executes the benchmark"""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        self.add_all_meta_to_dataset(dataset)

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
    clifford_sqg_probability = 1.0
    sqg_gate_ensemble: Optional[Dict[str, float]] = None
    simulation_method: Literal[
        "automatic", "statevector", "stabilizer", "extended_stabilizer", "matrix_product_state"] = "automatic"