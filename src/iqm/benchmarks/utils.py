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
from math import floor
from time import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, cast

from more_itertools import chunked
from mthree.utils import final_measurement_mapping
import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap
import xarray as xr

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.qiskit_iqm.iqm_provider import IQMProvider
from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates


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
    # FakeAdonis
    if backend_label.lower() in ("iqmfakeadonis", "fakeadonis"):
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
        if backend.name == "IQMNdonisBackend":
            transpiled = transpile_to_IQM(
                transpiled, backend=backend, optimize_single_qubits=optimize_sqg, remove_final_rzs=drop_final_rz
            )
        if aux_qc is not None:
            if backend.name == "IQMNdonisBackend":
                transpiled = reduce_to_active_qubits(transpiled, backend.name)
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
    """
    if physical_layout == "fixed":
        if backend.name == "IQMNdonisBackend":
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
    calset_id: Optional[str],
    max_gates_per_batch: Optional[int],
) -> List[IQMJob]:
    """Submit for execute a list of quantum circuits on the specified Backend.

    Args:
        sorted_transpiled_qc_list (Dict[Tuple, List[QuantumCircuit]]): the list of quantum circuits to be executed.
        backend (IQMBackendBase): the backend to execute the circuits on.
        shots (int): the number of shots per circuit.
        calset_id (Optional[str]): the calibration set ID, uses the latest one if None.
        max_gates_per_batch (int): the maximum number of gates per batch sent to the backend, used to make manageable batches.
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
                batch_jobs = backend.run(qc_batch, shots=shots, calibration_set_id=calset_id)
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
