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
Compressive gate set tomography

The full benchmark executes the following steps:
    1) Generation of circuits to measure
    2) Running of said circuits on the backend
    3) Conversion of the ideal gate set into numpy arrays to be processed by mGST
    4) Optional generation of an initialization for mGST
    5) Running mGST to get first gate set estimates
    6) Gauge optimization to the target gates using pyGSTi's gauge optimizer
    7) Optional rerun of mGST on bootstrap data to produce error bars
    8) Generation of error measures and result tables
    9) Generation of matrix plots for the reconstructed gate set
"""

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from qiskit.circuit.library import CZGate, RGate
import xarray as xr

from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import Benchmark, BenchmarkObservationIdentifier, add_counts_to_dataset
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.compressive_gst.gst_analysis import mgst_analysis
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import (
    perform_backend_transpilation,
    retrieve_all_counts,
    set_coupling_map,
    submit_execute,
    timeit,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from mGST import additional_fns, qiskit_interface
from mGST.qiskit_interface import add_idle_gates, remove_idle_wires


class CompressiveGST(Benchmark):
    """
    SPAM-robust characterization of a set of quantum gates
    """

    analysis_function = staticmethod(mgst_analysis)

    def __init__(self, backend: IQMBackendBase, configuration: "GSTConfiguration"):
        """Construct the compressive_gst class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (GSTConfiguration): the configuration of the benchmark
        """
        super().__init__(backend, configuration)

        self.qubit_layouts = parse_layouts(configuration.qubit_layouts)
        self.num_qubits = len(self.qubit_layouts[0])
        self.pdim = 2**self.num_qubits
        self.num_povm = self.pdim

        self.gate_set, self.gate_labels, self.num_gates = parse_gate_set(
            configuration, self.num_qubits, self.qubit_layouts
        )

        if configuration.opt_method not in ["GD", "SFN", "auto"]:
            raise ValueError("Invalid optimization method, valid options are: GD, SFN, auto")
        if configuration.opt_method == "auto":
            if (self.num_qubits == 2 and configuration.rank > 2) or self.num_qubits > 2:
                self.opt_method = "GD"
            else:
                self.opt_method = "SFN"
        else:
            self.opt_method = configuration.opt_method

        if configuration.max_iterations == "auto":
            if self.opt_method == "SFN":
                self.max_iterations = [100, 100]
            if self.opt_method == "GD":
                self.max_iterations = [250, 250]
        elif isinstance(configuration.max_iterations, list):
            self.max_iterations = configuration.max_iterations
        if configuration.batch_size == "auto":
            self.batch_size = 30 * self.pdim
        else:
            self.batch_size = configuration.batch_size

        # Circuit format used by mGST
        self.J = np.empty((self.configuration.num_circuits, self.num_povm))
        self.bootstrap_results = List[Tuple[np.ndarray]]  # List of GST outcomes from bootstrapping

    @staticmethod
    def name() -> str:
        return "compressive_GST"

    @timeit
    def generate_meas_circuits(self) -> None:
        """Generate random circuits from the gate set

        The random circuits are distributed among different depths ranging from L_MIN
        to L_MAX, both are configurable and stored in self.configuration.seq_len_list.
        No transpilation other than mapping to the desired qubits is performed,
        as the gates need to be executed axactly as described for GST to give
        meaningful results

        Returns:
            circuit_gen_transp_time: float
                The time it took to generate and transpile the circuits

        """
        # Calculate number of short and long circuits
        N_short = int(np.ceil(self.configuration.num_circuits / 2))
        N_long = int(np.floor(self.configuration.num_circuits / 2))
        L_MIN, L_CUT, L_MAX = self.configuration.seq_len_list

        # Generate random circuits
        J = additional_fns.random_seq_design(self.num_gates, L_MIN, L_CUT, L_MAX, N_short, N_long)

        # Reduce circuits to exclude negative placeholder values
        gate_circuits = [list(seq[seq >= 0]) for seq in J]

        # Sequences in GST are applied to the state from right to left
        self.J = np.array(J)[:, ::-1]

        unmapped_qubits = list(np.arange(self.num_qubits))
        raw_qc_list = qiskit_interface.get_qiskit_circuits(
            gate_circuits, self.gate_set, self.num_qubits, unmapped_qubits
        )

        for qubits in self.qubit_layouts:
            coupling_map = set_coupling_map(qubits, self.backend, physical_layout="fixed")

            # Perform transpilation to backend
            qcvv_logger.info(
                f"Will transpile all {self.configuration.num_circuits} circuits according to fixed physical layout"
            )
            transpiled_qc_list, _ = perform_backend_transpilation(
                raw_qc_list,
                self.backend,
                qubits,
                coupling_map=coupling_map,
                qiskit_optim_level=0,
                optimize_sqg=False,
                drop_final_rz=False,
            )
            # Saving raw and transpiled circuits in a consistent format with other benchmarks
            self.transpiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=transpiled_qc_list))
            self.untranspiled_circuits.circuit_groups.append(CircuitGroup(name=str(qubits), circuits=raw_qc_list))

    def add_configuration_to_dataset(self, dataset):  # CHECK
        """
        Creates an xarray.Dataset and adds the circuits and configuration metadata to it

        Args:
            self: Source class
        Returns:
            dataset: xarray.Dataset to be used for further data storage
        """
        # Adding configuration entries and class variables, prioritizing the latter in case of conflicts
        for key, value in (self.configuration.__dict__ | self.__dict__).items():
            if key == "benchmark":  # Avoid saving the class objects
                dataset.attrs[key] = value.name()
            elif key == "backend":
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        dataset.attrs["gauge_weights"] = dict({f"G%i" % i: 1 for i in range(self.num_gates)}, **{"spam": 0.1})

    def execute(self, backend) -> xr.Dataset:
        """
        The main GST execution routine
        """

        dataset = xr.Dataset()
        qcvv_logger.info(f"Now generating {self.configuration.num_circuits} random GST circuits...")

        self.circuits = Circuits()
        self.transpiled_circuits = BenchmarkCircuit(name="transpiled_circuits")
        self.untranspiled_circuits = BenchmarkCircuit(name="untranspiled_circuits")
        # Generate circuits
        self.generate_meas_circuits()

        # Submit all
        all_jobs: Dict = {}
        for qubit_layout in self.qubit_layouts:
            transpiled_circuit_dict = {tuple(qubit_layout): self.transpiled_circuits[str(qubit_layout)].circuits}
            all_jobs[str(qubit_layout)], _ = submit_execute(
                transpiled_circuit_dict,
                backend,
                self.configuration.shots,
                self.calset_id,
                max_gates_per_batch=self.configuration.max_gates_per_batch,
            )
        # Retrieve all
        qcvv_logger.info(f"Now executing the corresponding circuit batch")
        for qubit_layout in self.qubit_layouts:
            counts, _ = retrieve_all_counts(all_jobs[str(qubit_layout)])
            dataset, _ = add_counts_to_dataset(counts, str(qubit_layout), dataset)

        self.add_configuration_to_dataset(dataset)
        self.circuits.benchmark_circuits = [self.transpiled_circuits, self.untranspiled_circuits]
        return dataset


class GSTConfiguration(BenchmarkConfigurationBase):
    """Compressive GST configuration base.

    Attributes:
        benchmark (Type[Benchmark]): GHZBenchmark
        qubit_layouts (Sequence[Sequence[int]]): A sequence (e.g., Tuple or List) of sequences of
            physical qubit layouts, as specified by integer labels, where the benchmark is meant to be run.
        gate_set (Union[str, List[Type[QuantumCircuit]]]): The gate set in given either as a list of qiskit quantum
            cirucuits, or as one of the predefined gate sets "1QXYI", "2QXYCZ", "2QXYCZ_extended", "3QXYCZ".
            A gate set should be tomographically complete, meaning from the specified gates and the vacuum state, one
            should be able to prepare states that form a frame for the state space. A practical sufficient condition
            is that the gate set is able the generate all combinations of local Pauli eigenstates.
        num_circuits (int): How many random circuits are generated from the gate set. Guidelines on choosing this value:
            - At least 50 circuits for a single qubit gate set with 3 gates.
            - At least 400 circuits for a two qubit gate set with 6 gates.
            - At least 2000 circuits for a three qubit gate set with 9 gates.
            The number of random circuits needed is expected to grow linearly in the number of gates and exponentially
            in the number of qubits.
        rank (int): The Kraus rank of the reconstruction. Choose rank=1 for coherent error analysis and rank<=dim**2
            generally.
        shots (int): The number of measurement shots per circuit.
            * Default: 1024
        gate_labels (Union[Dict, None]): Names of the gates in the gate set. Used for plots and result tables.
            * Default: None
        seq_len_list (list[int]): Three numbers controling the depth of the random sequences. The first is the minimal
            depth, the last is the maximal depth and the middle number specifies a cutoff depth below which all possible
            sequences are selected.
            * Default: [1, 8, 14]
        from_init (bool): Whether the target gate set is used as an initialization to the mGST algorithm.
            * Default: True
        max_inits (int): If from_init = False, random initial points are tried and this parameter limits the amount of
            retries.
            * Default: 20
        opt_method (str): Which optimization method is used, can be either of "GD" or "SFN", for gradient descent or
            saddle free Newton, respectively.
            * Default: "auto" (Method is automatically selected based on qubit count and rank)
        max_iterations (Union[str, List[int]]): How many iterations to run the optimization algorithm for. Accepted
            values are either "auto" or a list of two integers. The first specifies the number of iterations for the
            optimization on batches or circuit data, while the second specifies the number of iterations on the full
            data for all circuits.
            * Default: "auto" (Numbers are automatically selected based on qubit count and rank)
        convergence_criteria (Union[str, List[float]]): Two parameters which determine when the optimization algorithm
            terminates. The first is a multiplier which specifies how close the cost function should get to a threshold
            for the given shot noise in order for the optimization to be considered "successful". Values in [2,10] are
            usually sensible. The second parameter sets the relative change in cost function between
            two consecutive iterations, below which the algorithm terminates.
            * Default: [4, 1e-4]
        batch_size (Union[str, int]): The number of circuits per batch in the optimization. This hyperparamters is
            automatically set and determines the convergence behaviour. A smaller batch size reduces runtime but can
            lead to erratic jumps in parameter space and lack of convergence.
            * Default: "auto"
        bootstrap_samples (int): The number of times the optimization algorithm is repeated on fake data to estimate
            the uncertainty via bootstrapping.
    """

    benchmark: Type[Benchmark] = CompressiveGST
    qubit_layouts: Union[List[int], List[List[int]]]
    gate_set: Union[str, List[Any]]
    num_circuits: int
    rank: int
    shots: int = 2**10
    gate_labels: Union[List, None] = None
    seq_len_list: list[int] = [1, 8, 14]
    from_init: bool = True
    max_inits: int = 20
    opt_method: str = "auto"
    max_iterations: Union[str, List[int]] = "auto"
    convergence_criteria: Union[str, List[float]] = [4, 1e-4]
    batch_size: Union[str, int] = "auto"
    bootstrap_samples: int = 0
    testing: bool = False


def parse_layouts(qubit_layouts: Union[List[int], List[List[int]]]) -> List[List[int]]:
    """Checks for correct setting of qubit_layouts in the configuration and return a correct type

    Args:
        qubit_layouts: List[List[[int]] or List[int]
            The qubit_layouts on the backend where the gates are defined on

    Returns:
        qubit_layouts: List[List[int]]
            A properly typed qubit_layout if no Error was raised
    """
    if all(isinstance(qubits, int) for qubits in qubit_layouts):
        return [qubit_layouts]  # type: ignore
    if all(isinstance(qubits, List) for qubits in qubit_layouts):
        if all(len(x) == len(qubit_layouts[0]) for x in qubit_layouts):  # type: ignore
            return qubit_layouts  # type: ignore
        raise ValueError(
            "Qubit layouts are of different size, currently only specifying "
            "qubit layouts with the same number of qubits is supported."
        )
    raise ValueError(
        "The qubit_layouts parameters needs to be one of List[int] for a single layout"
        " or List[List[int]] for multiple layouts."
    )


def parse_gate_set(
    configuration: GSTConfiguration, num_qubits: int, qubit_layouts: List[List[int]]
) -> Tuple[List[QuantumCircuit], Dict[str, Dict[int, str]], int]:
    """
    Handles different gate set inputs and produces a valid gate set

    Args:
        configuration: BenchmarkConfigurationBase
            Configuration class containing variables
        num_qubits: int
            The number of qubits on which the gate set is defined
        qubit_layouts: List[List[int]]
            A List of physical qubit layouts, as specified by integer labels, where the benchmark is meant to be run.

    Returns:
        gate_set: List[QuantumCircuit]
            A list of gates defined as quantum circuit objects
        gate_labels_dict: Dict[str, Dict[int, str]]
            The names of gates, i.e. "Rx(pi/2)" for a pi/2 rotation around the x-axis.
        num_gates: int
            The number of gates in the gate set

    """
    if isinstance(configuration.gate_set, str) and configuration.gate_set not in [
        "1QXYI",
        "2QXYCZ",
        "2QXYCZ_extended",
        "3QXYCZ",
    ]:
        raise ValueError(
            "No gate set of the specified name is implemented, please choose among "
            "1QXYI, 2QXYCZ, 2QXYCZ_extended, 3QXYCZ."
        )
    if configuration.gate_set in ["1QXYI", "2QXYCZ", "2QXYCZ_extended", "3QXYCZ"]:
        gate_set, gate_label_dict, num_gates = create_predefined_gate_set(
            configuration.gate_set, num_qubits, qubit_layouts
        )
        return gate_set, gate_label_dict, num_gates

    if isinstance(configuration.gate_set, list):
        gate_label_dict = {}
        gate_set = configuration.gate_set
        num_gates = len(gate_set)
        if configuration.gate_labels is None:
            gate_labels = {i: f"Gate %i" % i for i in range(num_gates)}
        else:
            if len(configuration.gate_labels) != num_gates:
                raise ValueError(
                    f"The number of gate labels (%i) does not match the number of gates (%i)"
                    % (len(configuration.gate_labels), num_gates)
                )
            gate_labels = dict(enumerate(configuration.gate_labels))
        for qubit_layout in qubit_layouts:
            gate_label_dict.update({BenchmarkObservationIdentifier(qubit_layout).string_identifier: gate_labels})
        return gate_set, gate_label_dict, num_gates

    raise ValueError(
        f"Invalid gate set, choose among 1QXYI, 2QXYCZ, 2QXYCZ_extended,"
        f" 3QXYCZ or provide a list of Qiskti circuits to define the gates."
    )


def create_predefined_gate_set(
    gate_set: Union[str, List[Any]], num_qubits: int, qubit_layouts: List[List[int]]
) -> Tuple[List[QuantumCircuit], Dict[str, Dict[int, str]], int]:
    """Create a list of quantum circuits corresponding to a predefined gate set.

    The circuits are assigned to the specified qubit_layouts on the backend only during transipilation, so the qubit labels
    at this stage may not represent the actual qubit labels on the backend.

    Args:
        gate_set: str
            The name of the gate set
    Returns:
        gates: List[QuantumCircuit]
            The gate set as a list of circuits
        gate_labels_dict: Dict[str, Dict[int, str]]
            The names of gates, i.e. "Rx(pi/2)" for a pi/2 rotation around the x-axis.
        num_gates: int
            The number of gates in the gate set

    """
    unmapped_qubits = list(np.arange(num_qubits))

    if gate_set == "1QXYI":
        gate_list = [RGate(1e-10, 0), RGate(0.5 * np.pi, 0), RGate(0.5 * np.pi, np.pi / 2)]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(len(gate_list))]
        gate_qubits = [[0], [0], [0]]
        for i, gate in enumerate(gate_list):
            gates[i].append(gate, gate_qubits[i])
        gate_labels = ["Idle", "Rx(pi/2)", "Ry(pi/2)"]
    elif gate_set == "2QXYCZ":
        gate_qubits = [[0], [1], [0], [1], [0, 1]]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(5)]
        gates[0].append(RGate(0.5 * np.pi, 0), [0])
        gates[1].append(RGate(0.5 * np.pi, 0), [1])
        gates[2].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[3].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[4].append(CZGate(), [0, 1])
        gate_labels = ["Rx(pi/2)", "Rx(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "CZ"]
    elif gate_set == "2QXYCZ_extended":
        gate_qubits = [[0], [1], [0], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(9)]
        gates[0].append(RGate(0.5 * np.pi, 0), [0])
        gates[1].append(RGate(0.5 * np.pi, 0), [1])
        gates[2].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[3].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[4].append(RGate(0.5 * np.pi, 0), [0])
        gates[4].append(RGate(0.5 * np.pi, 0), [1])
        gates[5].append(RGate(0.5 * np.pi, 0), [0])
        gates[5].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[6].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[6].append(RGate(0.5 * np.pi, 0), [1])
        gates[7].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[7].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[8].append(CZGate(), [[0], [1]])
        gate_labels = [
            "Rx(pi/2)",
            "Rx(pi/2)",
            "Ry(pi/2)",
            "Ry(pi/2)",
            "Rx(pi/2)-Rx(pi/2)",
            "Rx(pi/2)-Ry(pi/2)",
            "Ry(pi/2)-Rx(pi/2)",
            "Ry(pi/2)-Ry(pi/2)",
            "CZ",
        ]
    elif gate_set == "3QXYCZ":
        gate_list = [
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, np.pi / 2),
            RGate(0.5 * np.pi, np.pi / 2),
            RGate(0.5 * np.pi, np.pi / 2),
            CZGate(),
            CZGate(),
        ]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(len(gate_list))]
        gate_qubits = [[0], [1], [2], [0], [1], [2], [0, 1], [0, 2]]
        for i, gate in enumerate(gate_list):
            gates[i].append(gate, gate_qubits[i])
        gate_labels = ["Rx(pi/2)", "Rx(pi/2)", "Rx(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "CZ", "CZ"]
    else:
        raise ValueError(
            f"Invalid gate set, choose among 1QXYI, 2QXYCZ, 2QXYCZ_extended,"
            f" 3QXYCZ or provide a list of Qiskti circuits to define the gates."
        )

    gates = add_idle_gates(gates, unmapped_qubits, gate_qubits)
    gates = [remove_idle_wires(qc) for qc in gates]

    gate_label_dict = {}
    for qubit_layout in qubit_layouts:
        layout_label_dict = dict(enumerate(gate_labels))
        iqm_qubits = [f"QB{q + 1}" for q in qubit_layout]
        gate_qubits_iqm = [(iqm_qubits[q] for q in qubits) for qubits in gate_qubits]
        for key, value in layout_label_dict.items():
            layout_label_dict[key] = value + ":" + "-".join(gate_qubits_iqm[key])
        gate_label_dict.update({BenchmarkObservationIdentifier(qubit_layout).string_identifier: layout_label_dict})
    return gates, gate_label_dict, len(gates)
