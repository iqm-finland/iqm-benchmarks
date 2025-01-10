import random

import numpy as np
from qiskit.circuit.library import UnitaryGate

# from qiskit.extensions import UnitaryGate
import scipy.linalg as spl


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


def haar_shadow_tomography(qc, Nu, active_qubits):
    """Prepares the circuits to perform Haar shadow tomography.

    Args:
        qc (QuantumCircuit): The quantum circuit to which random unitaries are appended.
        Nu (Int): Number of local random unitaries used.
        active_qubits (List[int]): List of active qubits.
    Returns:
        unitary_gates (List[Array]): List of unitary gates for each random initialisation and qubit.
        qclist (List[QuantumCircuit]): List of tomography circuits.
    """
    qclist = []
    unitaries = np.zeros((Nu, len(active_qubits), 2, 2), dtype=np.complex_)

    for iu in range(Nu):
        qc_copy = qc.copy()
        qc_copy.barrier()
        for idx, z in enumerate(active_qubits):
            temp_U = CUE(random_gen, 2)
            qc_copy.append(UnitaryGate(temp_U), [z])
            unitaries[iu, idx, :, :] = np.array(temp_U)

        qc_copy.measure_all()
        qclist.append(qc_copy)

    return unitaries, qclist


def get_shadow(counts, Unitary, subsystem):
    """Constructs shadows for each individual initialisation..

    Args:
        counts (Dict): bit-string counts.
        Unitary (Array): local random unitaries used for a given initialisation.
        subsystem (List[int]): List of qubits to construct the shadow of.
    Returns:
        rhoshadows (Array): shadow of considered subsystem.
    """
    nqubits = len(subsystem)
    rhoshadows = np.zeros([2**nqubits, 2**nqubits], dtype=complex)
    proj = np.zeros((2, 2, 2), dtype=complex)
    proj[0, :, :] = np.array([[1, 0], [0, 0]])
    proj[1, :, :] = np.array([[0, 0], [0, 1]])
    shots = sum(list(counts.values()))
    for ist in counts.keys():
        rhoj = 1
        for j in subsystem:
            sj = int(ist[::-1][j])
            rhoj = np.kron(
                rhoj,
                3
                * np.einsum('ab,bc,cd', np.transpose(np.conjugate(Unitary[j, :, :])), proj[sj, :, :], Unitary[j, :, :])
                - np.array([[1, 0], [0, 1]]),
            )

        rhoshadows += counts[ist] * rhoj / shots

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
