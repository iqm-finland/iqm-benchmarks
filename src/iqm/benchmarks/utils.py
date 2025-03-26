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
General utility functions
"""
from collections import defaultdict
from functools import wraps
import itertools
from math import floor
import os
from time import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union, cast
import warnings

from more_itertools import chunked
from mthree.utils import final_measurement_mapping
import numpy as np
from numpy.random import Generator
from qiskit import ClassicalRegister, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli
from qiskit.transpiler import CouplingMap
import requests
import xarray as xr

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm import IQMFakeDeneb, transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMProvider


def timeit(f):
    """Calculates the amount of time a function takes to execute

    Args:
        f: The function to add the timing attribute to
    Returns:
        The decorated function execution with logger statement of elapsed time in execution
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        elapsed = te - ts
        if 1.0 <= elapsed <= 60.0:
            qcvv_logger.debug(f'\t"{f.__name__}" took {elapsed:.2f} sec')
        else:
            qcvv_logger.debug(f'\t"{f.__name__}" took {elapsed/60.0:.2f} min')
        return result, elapsed

    return wrap


def bootstrap_counts(
    original_counts: Dict[str, int],
    num_bootstrap_samples: int = 100,
    rgen: Optional[Generator] = None,
    include_original_counts: bool = False,
) -> List[Dict[str, int]]:
    """Returns num_bootstrap_samples resampled copies of the original_counts.

    Args:
        original_counts (Dict[str, int]): The original dictionary of counts to bootstrap from.
        num_bootstrap_samples (int): The number of bootstrapping samples to generate.
            * Default is 100.
        rgen (Optional[Generator]): The random number generator.
            * Default is None: assigns numpy's default_rng().
        include_original_counts (bool): Whether to include the original counts in the returned bootstrapped count samples.
            * Default is False.
    Returns:
        List[Dict[str, int]]: A list of bootstrapped counts.
    """
    if rgen is None:
        rgen = np.random.default_rng()

    keys = list(original_counts.keys())
    values = list(original_counts.values())
    tot_shots = int(sum(values))

    # Pre-calculate cumulative sum and create bins
    cumulative_sum = np.cumsum(values)
    bins = np.insert(cumulative_sum, 0, 0)

    if include_original_counts:
        bs_counts_fast = [original_counts]
    else:
        bs_counts_fast = []

    for _ in range(num_bootstrap_samples):
        # Generate random integers
        random_integers = rgen.integers(low=0, high=tot_shots, size=tot_shots)
        # Bin the random integers
        binned_integers = np.digitize(random_integers, bins) - 1
        # Count occurrences in each bin
        occurrences = np.bincount(binned_integers, minlength=len(keys))
        # Create dictionary mapping keys to occurrence counts
        bs_counts_fast.append(dict(zip(keys, occurrences)))

    return bs_counts_fast


@timeit
def count_2q_layers(circuit_list: List[QuantumCircuit]) -> List[int]:
    """Calculate the number of layers of parallel 2-qubit gates in a list of circuits.

    Args:
        circuit_list (List[QuantumCircuit]): the list of quantum circuits to analyze.

    Returns:
        List[int]: the number of layers of parallel 2-qubit gates in the list of circuits.
    """
    all_number_2q_layers = []
    for circuit in circuit_list:
        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())  # Call the method and convert the result to a list
        parallel_2q_layers = 0

        for layer in layers:
            two_qubit_gates_in_layer = [
                node
                for node in layer["graph"].op_nodes()  # Use op_nodes to get only operation nodes
                if node.op.num_qubits == 2
            ]
            if two_qubit_gates_in_layer:
                parallel_2q_layers += 1
        all_number_2q_layers.append(parallel_2q_layers)

    return all_number_2q_layers


def count_native_gates(
    backend_arg: Union[str, IQMBackendBase], transpiled_qc_list: List[QuantumCircuit]
) -> Dict[str, Dict[str, float]]:
    """Count the number of IQM native gates of each quantum circuit in a list.

    Args:
        backend_arg (str | IQMBackendBase): The backend, either specified as str or as IQMBackendBase.
        transpiled_qc_list: a list of quantum circuits transpiled to ['r','cz','barrier','measure'] gate set.
    Returns:
        Dictionary with
             - outermost keys being native operations.
             - values being Dict[str, float] with mean and standard deviation values of native operation counts.

    """
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    native_operations = backend.operation_names

    if "move" in backend.architecture.gates:
        native_operations.append("move")
    # Some backends may not include "barrier" in the operation_names attribute
    if "barrier" not in native_operations:
        native_operations.append("barrier")

    num_native_operations: Dict[str, List[int]] = {x: [0] for x in native_operations}
    avg_native_operations: Dict[str, Dict[str, float]] = {x: {} for x in native_operations}

    for q in transpiled_qc_list:
        for k in q.count_ops().keys():
            if k not in native_operations:
                raise ValueError(f"Count # of gates: '{k}' is not in the backend's native gate set")
        for op in native_operations:
            if op in q.count_ops().keys():
                num_native_operations[op].append(q.count_ops()[op])

    avg_native_operations.update(
        {
            x: {"Mean": np.mean(num_native_operations[x]), "Std": np.std(num_native_operations[x])}
            for x in native_operations
        }
    )

    return avg_native_operations


@timeit
def generate_state_tomography_circuits(
    qc: QuantumCircuit,
    active_qubits: Sequence[int],
    measure_other: Optional[Sequence[int]] = None,
    measure_other_name: Optional[str] = None,
    native: bool = True,
) -> Dict[str, QuantumCircuit]:
    """Generate all quantum circuits required for a quantum state tomography experiment.

    Args:
        qc (QuantumCircuit): The quantum circuit.
        active_qubits (Sequence[int]): The qubits to perform tomograhy on.
        measure_other (Optional[Sequence[int]]): Whether to measure other qubits in the qc QuantumCircuit.
            * Default is None.
        measure_other_name (Optional[str]): Name of the classical register to assign measure_other.
        native (bool): Whether circuits are prepared using IQM-native gates.
            * Default is True.
    Returns:
        Dict[str, QuantumCircuit]: A dictionary with keys being Pauli (measurement) strings and values the respective circuit.
            * Pauli strings are ordered for qubit labels in increasing order, e.g., "XY" for active_qubits 4, 1 corresponds to "X" measurement on qubit 1 and "Y" measurement on qubit 4.
    """
    num_qubits = len(active_qubits)

    # Organize all Pauli measurements as circuits
    aux_circ = QuantumCircuit(1)
    sqg_pauli_strings = ("Z", "X", "Y")
    pauli_measurements = {p: aux_circ.copy() for p in sqg_pauli_strings}

    # Avoid transpilation, generate either directly in native basis or in H, S
    if native:
        # Z measurement
        pauli_measurements["Z"].r(0, 0, 0)
        # X measurement
        pauli_measurements["X"].r(np.pi / 2, np.pi / 2, 0)
        pauli_measurements["X"].r(np.pi, 0, 0)
        # Y measurement
        pauli_measurements["Y"].r(-np.pi / 2, 0, 0)
        pauli_measurements["Y"].r(np.pi, np.pi / 4, 0)
    else:
        # Z measurement
        pauli_measurements["Z"].id(0)
        # X measurement
        pauli_measurements["X"].h(0)
        # Y measurement
        pauli_measurements["Y"].sdg(0)
        pauli_measurements["Y"].h(0)

    all_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=num_qubits)]
    all_circuits = {P_n: qc.copy() for P_n in all_pauli_labels}
    for P_n in all_pauli_labels:
        all_circuits[P_n].barrier()
        for q_idx, q_active in enumerate(sorted(active_qubits)):
            all_circuits[P_n].compose(pauli_measurements[P_n[q_idx]], qubits=q_active, inplace=True)

        all_circuits[P_n].barrier()

        register_tomo = ClassicalRegister(len(active_qubits), "tomo_qubits")
        all_circuits[P_n].add_register(register_tomo)
        all_circuits[P_n].measure(active_qubits, register_tomo)

        if measure_other is not None:
            if measure_other_name is None:
                measure_other_name = "non_tomo_qubits"
            register_neighbors = ClassicalRegister(len(measure_other), measure_other_name)
            all_circuits[P_n].add_register(register_neighbors)
            all_circuits[P_n].measure(measure_other, register_neighbors)

    return all_circuits


def get_active_qubits(qc: QuantumCircuit) -> List[int]:
    """Extract active qubits from a quantum circuit.

    Args:
        qc (QuantumCircuit): The quantum circuit to extract active qubits from.
    Returns:
        List[int]: A list of active qubits.
    """
    active_qubits = set()
    for instruction in qc.data:
        for qubit in instruction.qubits:
            active_qubits.add(qc.find_bit(qubit).index)
    return list(active_qubits)


# pylint: disable=too-many-branches
def get_iqm_backend(backend_label: str) -> IQMBackendBase:
    """Get the IQM backend object from a backend name (str).

    Args:
        backend_label (str): The name of the IQM backend.
    Returns:
        IQMBackendBase.
    """
    # ****** 5Q star ******
    # Pyrite
    if backend_label.lower() == "pyrite":
        iqm_server_url = "https://cocos.resonance.meetiqm.com/pyrite"
        provider = IQMProvider(iqm_server_url)
        backend_object = provider.get_backend()
    # FakeAdonis
    elif backend_label.lower() in ("iqmfakeadonis", "fakeadonis"):
        backend_object = IQMFakeAdonis()

    # ****** 20Q grid ******
    # Garnet
    elif backend_label.lower() == "garnet":
        iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet"
        provider = IQMProvider(iqm_server_url)
        backend_object = provider.get_backend()
    # FakeApollo
    elif backend_label.lower() in ("iqmfakeapollo", "fakeapollo"):
        backend_object = IQMFakeApollo()

    # ****** 6Q Resonator Star ******
    # Deneb
    elif backend_label.lower() == "deneb":
        iqm_server_url = "https://cocos.resonance.meetiqm.com/deneb"
        provider = IQMProvider(iqm_server_url)
        backend_object = provider.get_backend()
    # FakeDeneb
    elif backend_label.lower() in ("iqmfakedeneb", "fakedeneb"):
        backend_object = IQMFakeDeneb()

    else:
        raise ValueError(f"Backend {backend_label} not supported. Try 'garnet', 'deneb', 'fakeadonis' or 'fakeapollo'.")

    return backend_object


def get_measurement_mapping(circuit: QuantumCircuit) -> Dict[int, int]:
    """
    Extracts the final measurement mapping (qubits to bits) of a quantum circuit.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to extract the measurement mapping from.

    Returns:
        dict: A dictionary where keys are qubits and values are classical bits.
    """
    mapping = {}
    for instruction, qargs, cargs in circuit.data:
        if instruction.name == "measure":
            qubit = circuit.find_bit(qargs[0]).registers[0][1]
            cbit = circuit.find_bit(cargs[0]).registers[0][1]
            mapping[qubit] = cbit
    return mapping


def get_neighbors_of_edges(edges: Sequence[Sequence[int]], graph: Sequence[Sequence[int]]) -> Set[int]:
    """Given a Sequence of edges and a graph, return all neighboring nodes of the edges.

    Args:
        edges (Sequence[Sequence[int]]): A sequence of pairs of integers, representing edges of a graph.
        graph (Sequence[Sequence[int]]): The input graph specified as a sequence of edges (Sequence[int]).
    Returns:
        Sequence[int]: list of all neighboring nodes of the input edges.
    """
    neighboring_nodes = set()
    nodes_in_edges = set()

    for u, v in edges:
        nodes_in_edges.add(u)
        nodes_in_edges.add(v)

    for x, y in graph:
        if x in nodes_in_edges:
            neighboring_nodes.add(y)
        if y in nodes_in_edges:
            neighboring_nodes.add(x)
    neighboring_nodes -= nodes_in_edges

    return neighboring_nodes


def get_Pauli_expectation(counts: Dict[str, int], pauli_label: Literal["I", "X", "Y", "Z"]) -> float:
    """Gets an estimate of a Pauli expectation value for a given set of counts and a Pauli measurement label.

    Args:
        counts (Dict[str, int]): A dictionary of counts.
            * NB: keys are assumed to have a single bitstring, i.e., coming from a single classical register.
        pauli_label (str): A Pauli measurement label, specified as a string of I, X, Y, Z characters.

    Raises:
        ValueError: If Pauli labels are not specified in terms of I, X, Y, Z characters.
    Returns:
        float: The estimate of the Pauli expectation value.
    """
    num_qubits = len(list(counts.keys())[0])
    sqg_pauli_strings = ("I", "Z", "X", "Y")
    all_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=num_qubits)]

    if pauli_label not in all_pauli_labels:
        raise ValueError("pauli_label must be specified as a string made up of characters 'I', 'X', 'Y', or 'Z'.")

    expect = 0
    if "I" not in pauli_label:
        for b, count_b in counts.items():
            if b.count("1") % 2 == 0:
                expect += count_b
            else:
                expect -= count_b
        return expect / sum(counts.values())

    non_I_indices = [idx for idx, P in enumerate(pauli_label) if P != "I"]
    for b, count_b in counts.items():
        b_Z_parity = [1 if b[i] == "1" else 0 for i in non_I_indices]
        if sum(b_Z_parity) % 2 == 0:
            expect += count_b
        else:
            expect -= count_b
    return expect / sum(counts.values())


def get_tomography_matrix(pauli_expectations: Dict[str, float]) -> np.ndarray:
    """Reconstructs a density matrix from given Pauli expectations.

    Args:
        pauli_expectations (Dict[str, float]): A dictionary of Pauli expectations, with keys being Pauli strings.
    Raises:
        ValueError: If not all 4**n Pauli expectations are specified.
    Returns:
        np.ndarray: A tomographically reconstructed density matrix.
    """
    num_qubits = len(list(pauli_expectations.keys())[0])
    sqg_pauli_strings = ("I", "Z", "X", "Y")
    all_pauli_labels = ["".join(x) for x in itertools.product(sqg_pauli_strings, repeat=num_qubits)]
    if set(list(pauli_expectations.keys())) != set(all_pauli_labels):
        raise ValueError(
            f"Pauli expectations are incomplete ({len(list(pauli_expectations.keys()))} out of {len(all_pauli_labels)} expectations)"
        )

    rho = np.zeros([2**num_qubits, 2**num_qubits], dtype=complex)
    for pauli_string, pauli_expectation in pauli_expectations.items():
        rho += 2 ** (-num_qubits) * pauli_expectation * Pauli(pauli_string).to_matrix()
    return rho


def marginal_distribution(prob_dist_or_counts: Dict[str, float | int], indices: Iterable[int]) -> Dict[str, float]:
    """Compute the marginal distribution over specified bits (indices).

    Params:
    - prob_dist (Dict[str, float | int]): A dictionary with keys being bitstrings and values are either probabilities or counts
    - indices (Iterable[int]): List of bit indices to marginalize over

    Returns:
    - dict: A dictionary representing the marginal distribution over the specified bits.
    """
    marginal_dist: Dict[str, float] = defaultdict(float)

    for bitstring, prob in prob_dist_or_counts.items():
        # Extract the bits at the specified indices and form the marginalized bitstring
        marginalized_bitstring = "".join(bitstring[i] for i in sorted(indices))
        # Sum up probabilities for each marginalized bitstring
        marginal_dist[marginalized_bitstring] += prob

    return dict(marginal_dist)


def median_with_uncertainty(observations: Sequence[float]) -> Dict[str, float]:
    """Computes the median of a Sequence of float observations and returns value and propagated uncertainty.
    Reference: https://mathworld.wolfram.com/StatisticalMedian.html

    Args:
        observations (Sequence[float]): a Sequence of floating-point numbers.

    Returns:
        Dict[str, float]: a dictionary with keys "value" and "uncertainty" for the median of the input Sequence.
    """
    median = np.median(observations)
    N = len(observations)
    error_from_mean = np.std(observations) / np.sqrt(N)
    median_uncertainty = error_from_mean * np.sqrt(np.pi * N / (2 * (N - 1)))

    return {"value": float(median), "uncertainty": float(median_uncertainty)}


@timeit
def perform_backend_transpilation(
    qc_list: List[QuantumCircuit],
    backend: IQMBackendBase,
    qubits: Sequence[int],
    coupling_map: List[List[int]],
    basis_gates: Sequence[str] = ("r", "cz"),
    qiskit_optim_level: int = 1,
    optimize_sqg: bool = False,
    drop_final_rz: bool = True,
    routing_method: Optional[str] = "sabre",
) -> List[QuantumCircuit]:
    """
    Transpile a list of circuits to backend specifications.

    Args:
        qc_list (List[QuantumCircuit]): The original (untranspiled) list of quantum circuits.
        backend (IQMBackendBase ): The backend to execute the benchmark on.
        qubits (Sequence[int]): The qubits to target in the transpilation.
        coupling_map (List[List[int]]): The target coupling map to transpile to.
        basis_gates (Tuple[str, ...]): The basis gates.
        qiskit_optim_level (int): Qiskit "optimization_level" value.
        optimize_sqg (bool): Whether SQG optimization is performed taking into account virtual Z.
        drop_final_rz (bool): Whether the SQG optimizer drops a final RZ gate.
        routing_method (Optional[str]): The routing method employed by Qiskit's transpilation pass.

    Returns:
        List[QuantumCircuit]: A list of transpiled quantum circuits.

    Raises:
        ValueError: if Star topology and label 0 is in qubit layout.
    """

    # Helper function considering whether optimize_sqg is done,
    # and whether the coupling map is reduced (whether final physical layout must be fixed onto an auxiliary QC)
    def transpile_and_optimize(qc, aux_qc=None):
        transpiled = transpile(
            qc,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=qiskit_optim_level,
            initial_layout=qubits if aux_qc is None else None,
            routing_method=routing_method,
        )
        if "move" in backend.architecture.gates:
            transpiled = transpile_to_IQM(
                qc, backend=backend, optimize_single_qubits=optimize_sqg, remove_final_rzs=drop_final_rz
            )
        if aux_qc is not None:
            if "move" in backend.architecture.gates:
                if backend.num_qubits in qubits:
                    raise ValueError(
                        f"Label {backend.num_qubits} is reserved for Resonator - "
                        f"Please specify computational qubit labels {np.arange(backend.num_qubits)}"
                    )
                backend_topology = "star"
                transpiled = reduce_to_active_qubits(transpiled, backend_topology, backend.num_qubits)
                transpiled = aux_qc.compose(
                    transpiled, qubits=qubits + [backend.num_qubits], clbits=list(range(qc.num_clbits))
                )
            else:
                transpiled = aux_qc.compose(transpiled, qubits=qubits, clbits=list(range(qc.num_clbits)))

        return transpiled

    qcvv_logger.info(
        f"Transpiling for backend {backend.name} with optimization level {qiskit_optim_level}, "
        f"{routing_method} routing method{' including SQG optimization' if qiskit_optim_level>0 else ''} all circuits"
    )

    if coupling_map == backend.coupling_map:
        transpiled_qc_list = [transpile_and_optimize(qc) for qc in qc_list]
    else:  # The coupling map will be reduced if the physical layout is to be fixed
        if "move" in backend.architecture.gates:
            aux_qc_list = [QuantumCircuit(backend.num_qubits + 1, q.num_clbits) for q in qc_list]
        else:
            aux_qc_list = [QuantumCircuit(backend.num_qubits, q.num_clbits) for q in qc_list]
        transpiled_qc_list = [transpile_and_optimize(qc, aux_qc=aux_qc_list[idx]) for idx, qc in enumerate(qc_list)]

    return transpiled_qc_list


def reduce_to_active_qubits(
    circuit: QuantumCircuit, backend_topology: Optional[str] = None, backend_num_qubits=None
) -> QuantumCircuit:
    """
    Reduces a quantum circuit to only its active qubits.

    Args:
        backend_topology (Optional[str]): The backend topology to execute the benchmark on.
        circuit (QuantumCircuit): The original quantum circuit.
        backend_num_qubits (int): The number of qubits in the backend.

    Returns:
        QuantumCircuit: A new quantum circuit containing only active qubits.
    """
    # Identify active qubits
    active_qubits: list | set = set()
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            cast(set, active_qubits).add(circuit.find_bit(qubit).index)
    if backend_topology == "star" and backend_num_qubits not in active_qubits:
        # For star systems, the resonator must always be there, regardless of whether it MOVE gates on it or not
        cast(set, active_qubits).add(backend_num_qubits)

    # Create a mapping from old qubits to new qubits
    active_qubits = list(set(sorted(active_qubits)))
    qubit_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_qubits)}

    # Create a new quantum circuit with the reduced number of qubits
    reduced_circuit = QuantumCircuit(len(active_qubits))

    # Add classical registers if they exist
    if circuit.num_clbits > 0:
        creg = ClassicalRegister(circuit.num_clbits)
        reduced_circuit.add_register(creg)

    # Copy operations to the new circuit, remapping qubits and classical bits
    for instruction in circuit.data:
        new_qubits = [reduced_circuit.qubits[qubit_map[circuit.find_bit(qubit).index]] for qubit in instruction.qubits]
        new_clbits = [reduced_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        reduced_circuit.append(instruction.operation, new_qubits, new_clbits)

    return reduced_circuit


def remove_directed_duplicates_to_list(cp_map: CouplingMap) -> List[List[int]]:
    """Remove duplicate edges from a coupling map and returns as a list of edges (as a list of pairs of vertices).

    Args:
        cp_map (CouplingMap): A list of pairs of integers, representing a coupling map.
    Returns:
        List[List[int]]: the edges of the coupling map.
    """
    sorted_cp = [sorted(x) for x in list(cp_map)]
    return [list(x) for x in set(map(tuple, sorted_cp))]


@timeit
def retrieve_all_counts(iqm_jobs: List[IQMJob], identifier: Optional[str] = None) -> List[Dict[str, int]]:
    """Retrieve the counts from a list of IQMJob objects.
    Args:
        iqm_jobs (List[IQMJob]): The list of IQMJob objects.
        identifier (Optional[str]): a string identifying the job.
    Returns:
        List[Dict[str, int]]: The counts of all the IQMJob objects.
    """
    if identifier is None:
        qcvv_logger.info(f"Retrieving all counts")
    else:
        qcvv_logger.info(f"Retrieving all counts for {identifier}")
    final_counts = []
    for j in iqm_jobs:
        counts = j.result().get_counts()
        if isinstance(counts, list):
            final_counts.extend(counts)
        elif isinstance(counts, dict):
            final_counts.append(counts)

    return final_counts


def retrieve_all_job_metadata(
    iqm_jobs: List[IQMJob],
) -> Dict[str, Dict[str, Any]]:
    """Retrieve the counts from a list of IQMJob objects.
    Args:
        iqm_jobs List[IQMJob]: The list of IQMJob objects.

    Returns:
        Dict[str, Dict[str, Any]]: Relevant metadata of all the IQMJob objects.
    """
    all_meta = {}

    for index, j in enumerate(iqm_jobs):
        all_attributes_j = dir(j)
        all_meta.update(
            {
                "batch_job_"
                + str(index + 1): {
                    "job_id": j.job_id() if "job_id" in all_attributes_j else None,
                    "backend": j.backend().name if "backend" in all_attributes_j else None,
                    "status": j.status().value if "status" in all_attributes_j else None,
                    "circuits_in_batch": (
                        len(cast(List, j.circuit_metadata)) if "circuit_metadata" in all_attributes_j else None
                    ),
                    "shots": j.metadata["shots"] if "shots" in j.metadata.keys() else None,
                    "timestamps": j.metadata["timestamps"] if "timestamps" in j.metadata.keys() else None,
                }
            }
        )

    return all_meta


def set_coupling_map(
    qubits: Sequence[int], backend: IQMBackendBase, physical_layout: Literal["fixed", "batching"] = "fixed"
) -> CouplingMap:
    """Set a coupling map according to the specified physical layout.

    Args:
        qubits (Sequence[int]): the list of physical qubits to consider.
        backend (IQMBackendBase): the backend from IQM.
        physical_layout (Literal["fixed", "batching"]): the physical layout type to consider.
                - "fixed" sets a coupling map restricted to the input qubits -> results will be constrained to measure those qubits.
                - "batching" sets the coupling map of the backend -> results in a benchmark will be "batched" according to final layouts.
                * Default is "fixed".
    Raises:
        ValueError: if the physical layout is not "fixed" or "batching".
    Returns:
        A coupling map according to the specified physical layout.

    Raises:
        ValueError: if Star topology and label 0 is in qubit layout.
        ValueError: if the physical layout is not "fixed" or "batching".
    """
    if physical_layout == "fixed":
        # if "move" in backend.architecture.gates:
        #     if 0 in qubits:
        #         raise ValueError(
        #             "Label 0 is reserved for Resonator - Please specify computational qubit labels (1,2,...)"
        #         )
        #     return backend.coupling_map.reduce(mapping=[0] + list(qubits))
        return backend.coupling_map.reduce(mapping=qubits)
    if physical_layout == "batching":
        return backend.coupling_map
    raise ValueError('physical_layout must either be "fixed" or "batching"')


def split_sequence_in_chunks(sequence_in: Sequence[Any], split_size: int) -> List[Sequence[Any]]:
    """Split a given Sequence into chunks of a given split size, return as a List of Sequences.

    Args:
        sequence_in (Sequence[Any]): The input list.
        split_size (int): The split size.

    Returns:
        List[Sequence[Any]]: A List of Sequences.
    """
    if split_size > len(sequence_in):
        raise ValueError("The split size should be smaller or equal than the list length")
    if len(sequence_in) % split_size != 0 and (split_size != 1 and split_size != len(sequence_in)):
        qcvv_logger.debug(
            f"Since len(input_list) = {len(sequence_in)} and split_size = {split_size}, the input list will be split into chunks of uneven size!"
        )
        warnings.warn(
            f"Since len(input_list) = {len(sequence_in)} and split_size = {split_size}, the input list will be split into chunks of uneven size!"
        )

    return [sequence_in[i : i + split_size] for i in range(0, len(sequence_in), split_size)]


@timeit
def sort_batches_by_final_layout(
    transpiled_circuit_list: List[QuantumCircuit],
) -> Tuple[Dict[Tuple, List[QuantumCircuit]], Dict[Tuple, List[int]]]:
    """Sort batches of circuits according to the final measurement mapping in their corresponding backend.

    Args:
        transpiled_circuit_list (List[QuantumCircuit]): the list of circuits transpiled to a given backend.
    Returns:
        sorted_circuits (Dict[Tuple, List[QuantumCircuit]]): dictionary, keys: final measured qubits, values: corresponding circuits.
        sorted_indices (Dict[Tuple, List[int]]): dictionary, keys: final measured qubits, values: corresponding circuit indices.
    """
    qcvv_logger.info("Now getting the final measurement maps of all circuits")
    all_measurement_maps = [tuple(final_measurement_mapping(qc).values()) for qc in transpiled_circuit_list]
    unique_measurement_maps = set(tuple(sorted(x)) for x in all_measurement_maps)
    sorted_circuits: Dict[Tuple, List[QuantumCircuit]] = {u: [] for u in unique_measurement_maps}
    sorted_indices: Dict[Tuple, List[int]] = {i: [] for i in unique_measurement_maps}
    for index, qc in enumerate(transpiled_circuit_list):
        final_measurement = all_measurement_maps[index]
        final_measurement = tuple(sorted(final_measurement))
        sorted_circuits[final_measurement].append(qc)
        sorted_indices[final_measurement].append(index)

    if len(sorted_circuits) == 1:
        qcvv_logger.info(f"The routing method generated a single batch of circuits to be measured")
    else:
        qcvv_logger.info(f"The routing method generated {len(sorted_circuits)} batches of circuits to be measured")

    return sorted_circuits, sorted_indices


@timeit
def submit_execute(
    sorted_transpiled_qc_list: Dict[Tuple[int] | str, List[QuantumCircuit]],
    backend: IQMBackendBase,
    shots: int,
    calset_id: Optional[str] = None,
    max_gates_per_batch: Optional[int] = None,
    max_circuits_per_batch: Optional[int] = None,
    circuit_compilation_options: Optional[CircuitCompilationOptions] = None,
) -> List[IQMJob]:
    """Submit function to execute lists of quantum circuits on the specified backend,
        organized as a dictionary with keys being identifiers of a batch (normally qubits) and values corresponding lists of quantum circuits.
        The result is returned as a single list of IQMJob objects.

    Args:
        sorted_transpiled_qc_list (Dict[Tuple[int] | str, List[QuantumCircuit]]): A dictionary of lists of quantum circuits to be executed.
            * The keys (Tuple[int] | str) should correspond to final measured qubits.
            * The values (List[QuantumCircuit]) should be the corresponding list (batch) of quantum circuits.
        backend (IQMBackendBase): the backend to execute the circuits on.
        shots (int): the number of shots per circuit.
        calset_id (Optional[str]): the calibration set ID.
            * Default is None: uses the latest calibration ID.
        max_gates_per_batch (Optional[int]): the maximum number of gates per batch sent to the backend, used to make manageable batches.
            * Default is None.
        max_circuits_per_batch (Optional[int]): the maximum number of circuits per batch sent to the backend, used to make manageable batches.
            * Default is None.
        circuit_compilation_options (CircuitCompilationOptions): Ability to pass a compilation options object,
            enabling execution with dynamical decoupling, among other options - see qiskit-iqm documentation.
            * Default is None.
    Returns:
        List[IQMJob]: a list of IQMJob objects corresponding to the submitted circuits.
    """
    final_jobs = []
    for k in sorted(
        sorted_transpiled_qc_list.keys(),
        key=lambda x: len(sorted_transpiled_qc_list[x]),
        reverse=True,
    ):
        # sorted is so batches are looped from larger to smaller
        qcvv_logger.info(
            f"Submitting batch with {len(sorted_transpiled_qc_list[k])} circuits corresponding to qubits {list(k)}"
        )
        # Divide into batches according to maximum gate count per batch
        if max_gates_per_batch is None and max_circuits_per_batch is None:
            jobs = backend.run(sorted_transpiled_qc_list[k], shots=shots, calibration_set_id=calset_id)
            final_jobs.append(jobs)

        else:
            if max_gates_per_batch is None and max_circuits_per_batch is not None:
                restriction = "max_circuits_per_batch"
                batching_size = max_circuits_per_batch

            elif max_circuits_per_batch is None and max_gates_per_batch is not None:
                restriction = "max_gates_per_batch"
                # Calculate average gate count per quantum circuit
                avg_gates_per_qc = sum(sum(qc.count_ops().values()) for qc in sorted_transpiled_qc_list[k]) / len(
                    sorted_transpiled_qc_list[k]
                )
                batching_size = max(1, floor(max_gates_per_batch / avg_gates_per_qc))

            else:  # Both are not None - select the one rendering the smallest batches.
                # Calculate average gate count per quantum circuit
                avg_gates_per_qc = sum(sum(qc.count_ops().values()) for qc in sorted_transpiled_qc_list[k]) / len(
                    sorted_transpiled_qc_list[k]
                )
                qcvv_logger.warning(
                    "Both max_gates_per_batch and max_circuits_per_batch are not None. Selecting the one giving the smallest batches."
                )
                batching_size = min(max_circuits_per_batch, max(1, floor(max_gates_per_batch / avg_gates_per_qc)))  # type: ignore
                if batching_size == max_circuits_per_batch:
                    restriction = "max_circuits_per_batch"
                else:
                    restriction = "max_gates_per_batch"

            final_batch_jobs = []
            for index, qc_batch in enumerate(chunked(sorted_transpiled_qc_list[k], batching_size)):
                qcvv_logger.info(
                    f"{restriction} restriction: submitting subbatch #{index + 1} with {len(qc_batch)} circuits corresponding to qubits {list(k)}"
                )
                batch_jobs = backend.run(
                    qc_batch,
                    shots=shots,
                    calibration_set_id=calset_id,
                    circuit_compilation_options=circuit_compilation_options,
                )
                final_batch_jobs.append(batch_jobs)
            final_jobs.extend(final_batch_jobs)

    return final_jobs


def xrvariable_to_counts(dataset: xr.Dataset, identifier: str, counts_range: int) -> List[Dict[str, int]]:
    """Retrieve counts from xarray dataset.

    Args:
        dataset (xr.Dataset): the dataset to extract counts from.
        identifier (str): the identifier for the dataset counts.
        counts_range (int): the range of counts to extract (e.g., the amount of circuits that were executed).
    Returns:
        List[Dict[str, int]]: A list of counts dictionaries from the dataset.
    """
    return [
        dict(zip(list(dataset[f"{identifier}_state_{u}"].data), dataset[f"{identifier}_counts_{u}"].data))
        for u in range(counts_range)
    ]


def extract_fidelities(cal_url: str) -> tuple[list[list[int]], list[float], str]:
    """Returns couplings and CZ-fidelities from calibration data URL

    Args:
        cal_url: str
            The url under which the calibration data for the backend can be found
    Returns:
        list_couplings: List[List[int]]
            A list of pairs, each of which is a qubit coupling for which the calibration
            data contains a fidelity.
        list_fids: List[float]
            A list of CZ fidelities from the calibration url, ordered in the same way as list_couplings
        topology: str
            Name of the chip topology layout, currently either "star" or "crystal"
    """
    headers = {"Accept": "application/json", "Authorization": "Bearer " + os.environ["IQM_TOKEN"]}
    r = requests.get(cal_url, headers=headers, timeout=60)
    calibration = r.json()
    cal_keys = {
        el2["key"]: (i, j) for i, el1 in enumerate(calibration["calibrations"]) for j, el2 in enumerate(el1["metrics"])
    }
    list_couplings = []
    list_fids = []
    if "double_move_gate_fidelity" in cal_keys.keys():
        i, j = cal_keys["double_move_gate_fidelity"]
        topology = "star"
    else:
        i, j = cal_keys["cz_gate_fidelity"]
        topology = "crystal"
    for item in calibration["calibrations"][i]["metrics"][j]["metrics"]:
        qb1 = int(item["locus"][0][2:]) if "COMP" not in item["locus"][0] else 0
        qb2 = int(item["locus"][1][2:]) if "COMP" not in item["locus"][1] else 0
        list_couplings.append([qb1 - 1, qb2 - 1])
        list_fids.append(float(item["value"]))
    calibrated_qubits = set(np.array(list_couplings).reshape(-1))
    qubit_mapping = {}
    if topology == "star":
        qubit_mapping.update({-1: len(calibrated_qubits)})  # Place resonator qubit as last qubit
    qubit_mapping.update({qubit: idx for idx, qubit in enumerate(calibrated_qubits)})
    list_couplings = [[qubit_mapping[edge[0]], qubit_mapping[edge[1]]] for edge in list_couplings]

    return list_couplings, list_fids, topology
