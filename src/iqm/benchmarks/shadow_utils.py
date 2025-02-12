import random
from typing import List, Optional, Sequence, Tuple, Dict, Literal, cast

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import UnitaryGate

# from qiskit.extensions import UnitaryGate
import scipy.linalg as spl

from iqm.benchmarks.utils import timeit


a = random.SystemRandom().randrange(2**32 - 1)  # Init Random Generator
random_gen = np.random.RandomState(a)


def CUE(random_gen, nh):
    """Prepares single qubit Haar unitary.

    Args:
        random_gen (Int): random generator.
        nh (Int): size of the matrix.
    Returns:
        U (Array): nh x nh CUE matrix
    """
    U = (random_gen.randn(nh, nh) + 1j * random_gen.randn(nh, nh)) / np.sqrt(2)
    q, r = spl.qr(U)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    U = np.multiply(q, ph, q)
    return U


@timeit
def local_shadow_tomography(
    qc: QuantumCircuit,
    Nu: int,
    active_qubits: Sequence[int],
    measure_other: Optional[Sequence[int]] = None,
    measure_other_name: Optional[str] = None,
    clifford_or_haar: Literal["clifford", "haar"] = "clifford",
    cliffords_1q: Optional[Dict[str, QuantumCircuit]] = None,
) -> Tuple[List[np.ndarray], List[QuantumCircuit]]:
    """Prepares the circuits to perform Haar shadow tomography.

    Args:
        qc (QuantumCircuit): The quantum circuit to which random unitaries are appended.
        Nu (Int): Number of local random unitaries used.
        active_qubits (Sequence[int]): The Sequence of active qubits.
        measure_other (Optional[Sequence[int]]): Whether to measure other qubits in the qc QuantumCircuit.
                * Default is None.
        measure_other_name (Optional[str]): Name of the classical register to assign measure_other.
        clifford_or_haar (Literal["clifford", "haar"]): Whether to use Clifford or Haar random 1Q gates.
                * Default is "clifford".
        cliffords_1q (Optional[Dict[str, QuantumCircuit]]): dictionary of 1-qubit Cliffords in terms of IQM-native r and CZ gates
                * Default is None.

    Raises:
        ValueError: If clifford_or_haar is not "clifford" or "haar".
        Exception: If cliffords_1q is None and clifford_or_haar is "clifford".

    Returns:
        Tuple(List[ndarray], List[QuantumCircuit])
        - List[ndarray] | List[str]: Either:
                * List of unitary gates (numpy ndarray) for each random initialisation and qubit, if clifford_or_haar == 'haar'.
                * List of (str) Clifford labels, if clifford_or_haar == 'clifford'.
        - List[QuantumCircuit]: List of tomography circuits.
    """
    if clifford_or_haar not in ["clifford", "haar"]:
        raise ValueError("clifford_or_haar must be either 'clifford' or 'haar'.")
    elif clifford_or_haar == "clifford" and cliffords_1q is None:
        raise Exception("cliffords_1q dictionary must be provided if clifford_or_haar is 'clifford'.")
    elif clifford_or_haar == "clifford":
        # Get the keys of the Clifford dictionaries
        clifford_1q_keys = list(cliffords_1q.keys())

    qclist = []

    if clifford_or_haar == "haar":
        unitaries = np.zeros((Nu, len(active_qubits), 2, 2), dtype=np.complex_)
    else:
        unitaries = []

    for u in range(Nu):
        qc_copy = qc.copy()
        for q_idx, qubit in enumerate(active_qubits):
            if clifford_or_haar == "haar":
                temp_U = CUE(random_gen, 2)
                qc_copy.append(UnitaryGate(temp_U), [qubit])
                unitaries[u, q_idx, :, :] = np.array(temp_U)
            elif clifford_or_haar == "clifford":
                rand_key = random.choice(clifford_1q_keys)
                c_1q = cast(dict, cliffords_1q)[rand_key]
                qc_copy.append(c_1q, [qubit])
                unitaries.append(rand_key)

        qc_copy.barrier()

        register_rm = ClassicalRegister(len(active_qubits), "RMs")
        qc_copy.add_register(register_rm)
        qc_copy.measure(active_qubits, register_rm)

        if measure_other is not None:
            if measure_other_name is None:
                measure_other_name = "non_RMs"
            register_neighbors = ClassicalRegister(len(measure_other), measure_other_name)
            qc_copy.add_register(register_neighbors)
            qc_copy.measure(measure_other, register_neighbors)

        qclist.append(qc_copy)

    return unitaries, qclist


def get_shadow(counts: Dict[str, int], Unitary: np.ndarray, subsystem: Sequence[int]):
    """Constructs shadows for each individual initialisation.

    Args:
        counts (Dict[str, int]): a dictionary of bit-string counts.
        Unitary (Numpy Array): local random unitaries used for a given initialisation.
        subsystem (Sequence[int]): Sequence of qubits to construct the shadow of.
    Returns:
        rhoshadows (Array): shadow of considered subsystem.
    """
    nqubits = len(subsystem)
    rhoshadows = np.zeros([2**nqubits, 2**nqubits], dtype=complex)
    proj = np.zeros((2, 2, 2), dtype=complex)
    proj[0, :, :] = np.array([[1, 0], [0, 0]])
    proj[1, :, :] = np.array([[0, 0], [0, 1]])
    shots = sum(list(counts.values()))
    for bit_strings in counts.keys():
        rho_j = 1
        for j in subsystem:
            s_j = int(bit_strings[::-1][j])
            rho_j = np.kron(
                rho_j,
                3
                * np.einsum('ab,bc,cd', np.transpose(np.conjugate(Unitary[j, :, :])), proj[s_j, :, :], Unitary[j, :, :])
                - np.array([[1, 0], [0, 1]]),
            )

        rhoshadows += rho_j * counts[bit_strings] / shots

    return rhoshadows


def get_negativity(rho, NA, NB):
    """Computes the negativity of a given density matrix.

    Args:
        rho (Array): Density matrix.
        NA (Int): Number of qubits for subsystem A.
        NB (Int): Number of qubits for subsystem B.
    Returns:
        neg (Array): shadow of considered subsystem.
    """
    da = 2**NA
    db = 2**NB
    rho = rho.reshape(da, db, da, db)
    rho_t = np.einsum('ijkl -> kjil', rho)
    rho_t = rho_t.reshape(2 ** (NA + NB), 2 ** (NA + NB))
    eigs = np.linalg.eig(rho_t)
    neg = np.sum(i for i in np.real(eigs[0]) if np.all(i < 0))
    return neg
