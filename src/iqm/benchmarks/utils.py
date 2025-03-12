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
from dataclasses import dataclass
from functools import wraps
from math import floor
import os
import random
from time import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
from more_itertools import chunked
from mthree.utils import final_measurement_mapping
import networkx as nx
import numpy as np
from numpy.random import Generator
from qiskit import ClassicalRegister, transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
import requests
from rustworkx import PyGraph, spring_layout, visualization  # pylint: disable=no-name-in-module
import xarray as xr

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMProvider
from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates


def timeit(f):
    """Calculates the amount of time a function takes to execute.

    Args:
        f: The function to add the timing attribute to.
    Returns:
        The decorated function execution with logger statement of elapsed time in execution.
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

    else:
        raise ValueError(f"Backend {backend_label} not supported. Try 'garnet', 'deneb', 'fakeadonis' or 'fakeapollo'.")

    return backend_object


def evaluate_hamiltonian_paths(
    N: int,
    path_samples: int,
    backend_arg: str | IQMBackendBase,
    url: str,
    max_tries: int = 10,
) -> Dict[int, List[Tuple[int, int]]]:
    """Evaluates Hamiltonian paths according to the product of 2Q gate fidelities on the corresponding edges of the backend graph.

    Args:
        N (int): the number of vertices in the Hamiltonian paths to evaluate.
        path_samples (int): the number of Hamiltonian paths to evaluate.
        backend_arg (str | IQMBackendBase): the backend to evaluate the Hamiltonian paths on with respect to fidelity.
        url (str): the URL address for the backend to retrieve calibration data from.
        max_tries (int): the maximum number of tries to generate a Hamiltonian path.

    Returns:
        Dict[int, List[Tuple[int, int]]]: A dictionary with keys being fidelity products and values being the respective Hamiltonian paths.
    """
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    backend_nx_graph = rx_to_nx_graph(backend_arg)

    all_paths = []
    sample_counter = 0
    tries = 0
    while sample_counter < path_samples and tries < max_tries:
        h_path = random_hamiltonian_path(backend_nx_graph, N)
        if not h_path:
            tries += 1
            continue

        all_paths.append(h_path)
        tries = 0
        sample_counter += 1
    if tries == max_tries - 1:
        raise RecursionError(
            f"Max tries to generate a Hamiltonian path with {N} vertices reached - try with less vertices!"
        )

    # Get scores for all paths
    # Retrieve fidelity data
    two_qubit_fidelity = {}

    headers = {"Accept": "application/json", "Authorization": "Bearer " + os.environ["IQM_TOKEN"]}
    r = requests.get(url, headers=headers, timeout=60)
    calibration = r.json()

    edge_dictionary = {}
    for iq in calibration["calibrations"][0]["metrics"][0]["metrics"]:
        temp = list(iq.values())
        two_qubit_fidelity[str(temp[0])] = temp[1]
        two_qubit_fidelity[str([temp[0][1], temp[0][0]])] = temp[1]
        edge_dictionary[str([temp[0][1], temp[0][0]])] = (
            backend.qubit_name_to_index(temp[0][1]),
            backend.qubit_name_to_index(temp[0][0]),
        )

    # Rate all the paths
    path_costs = {}  # keys are costs, values are edge paths
    for h_path in all_paths:
        total_cost = 1
        for edge in h_path:
            if len(edge) == 2:
                total_cost *= two_qubit_fidelity[
                    str([backend.index_to_qubit_name(edge[0]), backend.index_to_qubit_name(edge[1])])
                ]
        path_costs[total_cost] = h_path

    return path_costs


def marginal_distribution(prob_dist: Dict[str, float], indices: Iterable[int]) -> Dict[str, float]:
    """Compute the marginal distribution over specified bits (indices)

    Params:
    - prob_dist (dict): A dictionary with keys being bitstrings and values are their probabilities
    - indices (list): List of bit indices to marginalize over

    Returns:
    - dict: A dictionary representing the marginal distribution over the specified bits.
    """
    marginal_dist: Dict[str, float] = defaultdict(float)

    for bitstring, prob in prob_dist.items():
        # Extract the bits at the specified indices and form the marginalized bitstring
        marginalized_bitstring = "".join(bitstring[i] for i in indices)
        # Sum up probabilities for each marginalized bitstring
        marginal_dist[marginalized_bitstring] += prob

    return dict(marginal_dist)


@timeit
def perform_backend_transpilation(
    qc_list: List[QuantumCircuit],
    backend: IQMBackendBase,
    qubits: Sequence[int],
    coupling_map: List[List[int]],
    basis_gates: Tuple[str, ...] = ("r", "cz"),
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
        if optimize_sqg:
            transpiled = optimize_single_qubit_gates(transpiled, drop_final_rz=drop_final_rz)
        if "move" in backend.operation_names:
            transpiled = transpile_to_IQM(
                qc, backend=backend, optimize_single_qubits=optimize_sqg, remove_final_rzs=drop_final_rz
            )
        if aux_qc is not None:
            if "move" in backend.operation_names:
                if 0 in qubits:
                    raise ValueError(
                        "Label 0 is reserved for Resonator - Please specify computational qubit labels (1,2,...)"
                    )
                backend_name = "IQMNdonisBackend"
                transpiled = reduce_to_active_qubits(transpiled, backend_name)
                transpiled = aux_qc.compose(transpiled, qubits=[0] + qubits, clbits=list(range(qc.num_clbits)))
            else:
                transpiled = aux_qc.compose(transpiled, qubits=qubits, clbits=list(range(qc.num_clbits)))

        return transpiled

    qcvv_logger.info(
        f"Transpiling for backend {backend.name} with optimization level {qiskit_optim_level}, "
        f"{routing_method} routing method{' and SQG optimization' if optimize_sqg else ''} all circuits"
    )

    if coupling_map == backend.coupling_map:
        transpiled_qc_list = [transpile_and_optimize(qc) for qc in qc_list]
    else:  # The coupling map will be reduced if the physical layout is to be fixed
        aux_qc_list = [QuantumCircuit(backend.num_qubits, q.num_clbits) for q in qc_list]
        transpiled_qc_list = [transpile_and_optimize(qc, aux_qc=aux_qc_list[idx]) for idx, qc in enumerate(qc_list)]

    return transpiled_qc_list


def random_hamiltonian_path(G: nx.Graph, N: int) -> List[Tuple[int, int]]:
    """
    Generates a random Hamiltonian path with N vertices from a given NetworkX graph.

    Args:
        G (networkx.Graph): The input graph.
        N (int): The desired number of vertices in the Hamiltonian path.

    Returns:
        list: A list of edges (tuples of nodes) representing the Hamiltonian path, or an empty list if not possible.
    """
    if N > len(G):
        raise ValueError(
            f"The number of vertices in the Hamiltonian path ({N}) cannot be greater than the number of nodes in the graph ({len(G)})"
        )

    nodes = list(G.nodes)
    random.shuffle(nodes)  # Shuffle nodes to introduce randomness

    for start in nodes:
        path = [start]
        visited = set(path)
        edges = []

        while len(path) < N:
            neighbors = [n for n in G.neighbors(path[-1]) if n not in visited]

            if not neighbors:
                break  # Dead end, stop trying this path

            next_node = random.choice(neighbors)
            edges.append((int(path[-1]), int(next_node)))
            path.append(next_node)
            visited.add(next_node)

        if len(path) == N:
            return edges  # Successfully found a Hamiltonian path of length N

    return []  # No valid path found


def rx_to_nx_graph(backend_arg: str | IQMBackendBase) -> nx.Graph:
    """Convert the Rustworkx graph returned by a backend to a Networkx graph.

    Args:
        backend_arg (str | IQMBackendBase): The backend, either specified as str or as IQMBackendBase.

    Returns:
        networkx.Graph: The Networkx Graph corresponding to the backend graph.

    """
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    # Generate a Networkx graph
    graph_backend = backend.coupling_map.graph.to_undirected(multigraph=False)
    backend_egdes, backend_nodes = (list(graph_backend.edge_list()), list(graph_backend.node_indices()))
    backend_nx_graph = nx.Graph()
    backend_nx_graph.add_nodes_from(backend_nodes)
    backend_nx_graph.add_edges_from(backend_egdes)

    return backend_nx_graph


def reduce_to_active_qubits(circuit: QuantumCircuit, backend_name: Optional[str] = None) -> QuantumCircuit:
    """
    Reduces a quantum circuit to only its active qubits.

    Args:
        backend_name (Optional[str]): The backend name, if any, in which the circuits are defined.
        circuit (QuantumCircuit): The original quantum circuit.

    Returns:
        QuantumCircuit: A new quantum circuit containing only active qubits.
    """
    # Identify active qubits
    active_qubits = set()
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            active_qubits.add(circuit.find_bit(qubit).index)
    if backend_name is not None and backend_name == "IQMNdonisBackend" and 0 not in active_qubits:
        # For star systems, the resonator must always be there, regardless of whether it MOVE gates on it or not
        active_qubits.add(0)

    # Create a mapping from old qubits to new qubits
    active_qubits = set(sorted(active_qubits))
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
    Returns:
        A coupling map according to the specified physical layout.

    Raises:
        ValueError: if Star topology and label 0 is in qubit layout.
        ValueError: if the physical layout is not "fixed" or "batching".
    """
    if physical_layout == "fixed":
        if "move" in backend.operation_names:
            if 0 in qubits:
                raise ValueError(
                    "Label 0 is reserved for Resonator - Please specify computational qubit labels (1,2,...)"
                )
            return backend.coupling_map.reduce(mapping=[0] + list(qubits))
        return backend.coupling_map.reduce(mapping=qubits)
    if physical_layout == "batching":
        return backend.coupling_map
    raise ValueError('physical_layout must either be "fixed" or "batching"')


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
    sorted_transpiled_qc_list: Dict[Tuple, List[QuantumCircuit]],
    backend: IQMBackendBase,
    shots: int,
    calset_id: Optional[str] = None,
    max_gates_per_batch: Optional[int] = None,
    circuit_compilation_options: Optional[CircuitCompilationOptions] = None,
) -> List[IQMJob]:
    """Submit for execute a list of quantum circuits on the specified Backend.

    Args:
        sorted_transpiled_qc_list (Dict[Tuple, List[QuantumCircuit]]): the list of quantum circuits to be executed.
        backend (IQMBackendBase): the backend to execute the circuits on.
        shots (int): the number of shots per circuit.
        calset_id (Optional[str]): the calibration set ID, uses the latest one if None.
        max_gates_per_batch (int): the maximum number of gates per batch sent to the backend, used to make manageable batches.
        circuit_compilation_options (CircuitCompilationOptions): Ability to pass a compilation options object,
            enabling execution with dynamical decoupling, among other options - see qiskit-iqm documentation.
    Returns:
        List[IQMJob]: the IQMJob objects of the executed circuits.

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
        if max_gates_per_batch is None:
            jobs = backend.run(sorted_transpiled_qc_list[k], shots=shots, calibration_set_id=calset_id)
            final_jobs.append(jobs)
        else:
            # Calculate average gate count per quantum circuit
            avg_gates_per_qc = sum(sum(qc.count_ops().values()) for qc in sorted_transpiled_qc_list[k]) / len(
                sorted_transpiled_qc_list[k]
            )
            final_batch_jobs = []
            for index, qc_batch in enumerate(
                chunked(
                    sorted_transpiled_qc_list[k],
                    max(1, floor(max_gates_per_batch / avg_gates_per_qc)),
                )
            ):
                qcvv_logger.info(
                    f"max_gates_per_batch restriction: submitting subbatch #{index+1} with {len(qc_batch)} circuits corresponding to qubits {list(k)}"
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


@dataclass
class GraphPositions:
    """A class to store and generate graph positions for different chip layouts.

    This class contains predefined node positions for various quantum chip topologies and
    provides methods to generate positions for different layout types.

    Attributes:
        garnet_positions (Dict[int, Tuple[int, int]]): Mapping of node indices to (x,y) positions for Garnet chip.
        deneb_positions (Dict[int, Tuple[int, int]]): Mapping of node indices to (x,y) positions for Deneb chip.
        predefined_stations (Dict[str, Dict[int, Tuple[int, int]]]): Mapping of chip names to their position dictionaries.
    """

    garnet_positions = {
        0: (5.0, 7.0),
        1: (6.0, 6.0),
        2: (3.0, 7.0),
        3: (4.0, 6.0),
        4: (5.0, 5.0),
        5: (6.0, 4.0),
        6: (7.0, 3.0),
        7: (2.0, 6.0),
        8: (3.0, 5.0),
        9: (4.0, 4.0),
        10: (5.0, 3.0),
        11: (6.0, 2.0),
        12: (1.0, 5.0),
        13: (2.0, 4.0),
        14: (3.0, 3.0),
        15: (4.0, 2.0),
        16: (5.0, 1.0),
        17: (1.0, 3.0),
        18: (2.0, 2.0),
        19: (3.0, 1.0),
    }

    deneb_positions = {
        0: (2.0, 2.0),
        1: (1.0, 1.0),
        3: (2.0, 1.0),
        5: (3.0, 1.0),
        2: (1.0, 3.0),
        4: (2.0, 3.0),
        6: (3.0, 3.0),
    }

    predefined_stations = {
        "Garnet": garnet_positions,
        "Deneb": deneb_positions,
    }

    @staticmethod
    def create_positions(graph: PyGraph, topology: Optional[str] = None) -> Dict[int, Tuple[float, float]]:
        """Generate node positions for a given graph and topology.

        Args:
            graph: The graph to generate positions for.
            topology: The type of layout to generate. Must be either "star" or "crystal".

        Returns:
            A dictionary mapping node indices to (x,y) coordinates.
        """
        n_nodes = len(graph.node_indices())

        if topology == "star":
            # Place center node at (0,0)
            pos = {0: (0.0, 0.0)}

            if n_nodes > 1:
                # Place other nodes in a circle around the center
                angles = np.linspace(0, 2 * np.pi, n_nodes - 1, endpoint=False)
                radius = 1.0

                for i, angle in enumerate(angles, start=1):
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    pos[i] = (x, y)

        # Crystal and other topologies
        else:
            # Fix first node position in bottom right
            fixed_pos = {0: (1.0, 1.0)}  # For more consistent layouts

            # Get spring layout with one fixed position
            pos = {
                int(k): (float(v[0]), float(v[1]))
                for k, v in spring_layout(graph, scale=2, pos=fixed_pos, num_iter=300, fixed={0}).items()
            }
        return pos


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
        qb1 = int(item["locus"][0][2:]) if item["locus"][0] != "COMP_R" else 0
        qb2 = int(item["locus"][1][2:]) if item["locus"][1] != "COMP_R" else 0
        if topology == "star":
            list_couplings.append([qb1, qb2])
        else:
            list_couplings.append([qb1 - 1, qb2 - 1])
        list_fids.append(float(item["value"]))
    calibrated_qubits = set(np.array(list_couplings).reshape(-1))
    qubit_mapping = {qubit: idx for idx, qubit in enumerate(calibrated_qubits)}
    list_couplings = [[qubit_mapping[edge[0]], qubit_mapping[edge[1]]] for edge in list_couplings]

    return list_couplings, list_fids, topology


def plot_layout_fidelity_graph(
    cal_url: str, qubit_layouts: Optional[list[list[int]]] = None, station: Optional[str] = None
):
    """Plot a graph showing the quantum chip layout with fidelity information.

    Creates a visualization of the quantum chip topology where nodes represent qubits
    and edges represent connections between qubits. Edge thickness indicates gate errors
    (thinner edges mean better fidelity) and selected qubits are highlighted in orange.

    Args:
        cal_url: URL to retrieve calibration data from
        qubit_layouts: List of qubit layouts where each layout is a list of qubit indices
        station: Name of the quantum computing station to use predefined positions for.
                If None, positions will be generated algorithmically.

    Returns:
        matplotlib.figure.Figure: The generated figure object containing the graph visualization
    """
    edges_cal, fidelities_cal, topology = extract_fidelities(cal_url)
    weights = -np.log(np.array(fidelities_cal))
    edges_graph = [tuple(edge) + (weight,) for edge, weight in zip(edges_cal, weights)]

    graph = PyGraph()

    # Add nodes
    nodes: set[int] = set()
    for edge in edges_graph:
        nodes.update(edge[:2])
    graph.add_nodes_from(list(nodes))

    # Add edges
    graph.add_edges_from(edges_graph)

    # Define qubit positions in plot
    if station in GraphPositions.predefined_stations:
        pos = GraphPositions.predefined_stations[station]
    else:
        pos = GraphPositions.create_positions(graph, topology)

    # Define node colors
    node_colors = ["lightgrey" for _ in range(len(nodes))]
    if qubit_layouts is not None:
        for qb in {qb for layout in qubit_layouts for qb in layout}:
            node_colors[qb] = "orange"

    # Ensuring weights are in correct order for the plot
    edge_list = graph.edge_list()
    weights_dict = {}
    edge_pos = set()

    # Create a mapping between edge positions as defined in rustworkx and their weights
    for e, w in zip(edge_list, weights):
        pos_tuple = (tuple(pos[e[0]]), tuple(pos[e[1]]))
        weights_dict[pos_tuple] = w
        edge_pos.add(pos_tuple)

    # Get corresponding weights in the same order
    weights_ordered = np.array([weights_dict[edge] for edge in list(edge_pos)])

    plt.subplots(figsize=(6, 6))

    # Draw the graph
    visualization.mpl_draw(
        graph,
        with_labels=True,
        node_color=node_colors,
        pos=pos,
        labels=lambda node: node,
        width=7 * weights_ordered / np.max(weights_ordered),
    )  # type: ignore[call-arg]

    # Add edge labels using matplotlib's annotate
    for edge in edges_graph:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        plt.annotate(
            f"{edge[2]:.1e}",
            xy=(x, y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )

    plt.gca().invert_yaxis()
    plt.title(
        "Chip layout with selected qubits in orange\n"
        + "and gate errors indicated by edge thickness (thinner is better)"
    )
    plt.show()
