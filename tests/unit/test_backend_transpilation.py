import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.optimize import minimize

from iqm.benchmarks.utils import perform_backend_transpilation, reduce_to_active_qubits, set_coupling_map
from iqm.qiskit_iqm import transpile_to_IQM
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from mGST.additional_fns import multikron


class TestPerformBackendTranspilation(unittest.TestCase):
    """Test cases for the perform_backend_transpilation function."""

    def setUp(self):
        """Set up test environment."""
        self.backend = IQMFakeApollo()
        self.qubit_layout = [1, 4, 5, 6]
        self.coupling_map = set_coupling_map(self.qubit_layout, self.backend, "fixed")

        # Create simple test circuits
        self.test_circuits = []

        # Simple circuit: GHZ state
        qc1 = QuantumCircuit(3)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.cx(1, 2)
        qc1.measure_all()

        # Simple randomized benchmarking circuit
        qc2 = QuantumCircuit(2)
        # Apply a series of Clifford gates
        qc2.h(0)
        qc2.x(1)
        qc2.cx(0, 1)
        qc2.s(0)
        qc2.y(1)
        qc2.cz(0, 1)
        qc2.z(0)
        # Measurement
        qc2.measure_all()

        # Simple CLOPS benchmark circuit - sequence of gates similar to those in CLOPS
        qc3 = QuantumCircuit(3)
        qc3.h([0, 1, 2])  # Parallel operations
        qc3.cx(0, 1)
        qc3.cx(1, 2)
        qc3.rz(0.5, 0)
        qc3.rx(0.3, 1)
        qc3.ry(0.7, 2)
        qc3.barrier()
        qc3.measure_all()

        self.test_circuits = [qc1, qc2, qc3]

    def count_active_qubits(self, circuit):
        """
        Returns the number of qubits on which operations are applied in the circuit.
        """

        active_qubits = set()
        for instruction, qubits, _ in circuit.data:
            for qubit in qubits:
                active_qubits.add(qubit)

        return len(active_qubits)

    def equiv_up_to_local_z(self, op1: "Operator", op2: "Operator", atol: float = 1e-6) -> bool:
        """
        Check if two operators are equivalent up to local Z rotations and global phase at the beginning
        or end of the circuit.

        Args:
            op1: First Qiskit Operator object to compare
            op2: Second Qiskit Operator object to compare
            atol: Absolute tolerance for numerical comparison

        Returns:
            bool: True if operators are equivalent up to local Z rotations
        """

        # First check dimensions match
        if op1.dim != op2.dim:
            return False

        def u_rz(u: np.ndarray, angles: np.ndarray) -> np.ndarray:
            """
            Apply before-and-after local Z rotations and global phase to a unitary matrix.

            Args:
                u: Input unitary matrix
                angles: Rotation angles (first is global phase, others for local Z rotations)

            Returns:
                np.ndarray: Modified unitary matrix
            """
            rz_l = multikron(np.array([np.array([[1, 0], [0, np.exp(1j * angle)]]) for angle in angles[2::2]]))
            rz_r = multikron(np.array([np.array([[1, 0], [0, np.exp(1j * angle)]]) for angle in angles[1::2]]))
            return np.exp(1j * angles[0]) * rz_l @ u @ rz_r

        def dist_up_to_rz(angles: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
            """
            Compute the distance between two unitary matrices up to local Z rotations.

            Args:
                angles: Rotation angles to optimize
                u1: First unitary matrix
                u2: Second unitary matrix

            Returns:
                float: Frobenius norm of the difference between the modified matrices
            """
            u1_rz = u_rz(u1, angles)
            return np.linalg.norm(u1_rz - u2, ord='fro')

        res = minimize(
            dist_up_to_rz,
            x0=np.ones(2 * int(np.log2(op1.dim[0])) + 1),
            args=(op1.to_matrix(), op2.to_matrix()),
            method='L-BFGS-B',
        )

        return res.fun < atol

    def test_basic_transpilation(self):
        """Test basic transpilation functionality."""

        transpiled_circuits, _ = perform_backend_transpilation(
            self.test_circuits,
            self.backend,
            self.qubit_layout,
            self.coupling_map,
            qiskit_optim_level=1,
            optimize_sqg=False,
        )

        # Verify we get the same number of circuits back
        self.assertEqual(len(transpiled_circuits), len(self.test_circuits))

        # Check that the number of active qubits matches
        for circ_transp, circ in zip(transpiled_circuits, self.test_circuits):
            self.assertEqual(self.count_active_qubits(circ), self.count_active_qubits(circ_transp))

        # Check that circuits only use qubits in the layout
        for circ in transpiled_circuits:
            for inst in circ.data:
                for qubit in inst.qubits:
                    self.assertIn(qubit._index, self.qubit_layout)

        # The circuits should be functionally equivalent
        for i in range(len(self.test_circuits)):
            # For small circuits we can check unitary equivalence
            if self.test_circuits[i].num_qubits <= 3:
                reduced_qc = reduce_to_active_qubits(self.test_circuits[i])
                reduced_qc.remove_final_measurements()
                reduced_qc_transp = reduce_to_active_qubits(transpiled_circuits[i])
                reduced_qc_transp.remove_final_measurements()
                op = Operator(reduced_qc)
                op_transp = Operator(reduced_qc_transp)
                self.assertTrue(
                    op.equiv(op_transp), f"Circuit {i} failed unitary equivalence check after transpilation"
                )

    def test_transpilation_with_sqg_optimization(self):
        """Test with and without single-qubit gate optimization."""
        # With SQG optimization
        transpiled_with_opt, _ = perform_backend_transpilation(
            self.test_circuits,
            self.backend,
            self.qubit_layout,
            self.coupling_map,
            qiskit_optim_level=1,
            optimize_sqg=True,
        )

        # Without SQG optimization
        transpiled_without_opt, _ = perform_backend_transpilation(
            self.test_circuits,
            self.backend,
            self.qubit_layout,
            self.coupling_map,
            qiskit_optim_level=1,
            optimize_sqg=False,
        )

        # The circuits should be functionally equivalent
        for i in range(len(self.test_circuits)):
            # For small circuits we can check unitary equivalence, this time up to global phase and local Z rotations
            # since those are ignored in the single qubit gate optimization
            if self.test_circuits[i].num_qubits <= 3:
                reduced_qc_with = reduce_to_active_qubits(transpiled_with_opt[i])
                reduced_qc_with.remove_final_measurements()
                reduced_qc_without = reduce_to_active_qubits(transpiled_without_opt[i])
                reduced_qc_without.remove_final_measurements()
                op_with = Operator(reduced_qc_with)
                op_without = Operator(reduced_qc_without)
                equiv_rz = self.equiv_up_to_local_z(op_with, op_without)
                self.assertTrue(equiv_rz, f"Circuit {i} failed unitary equivalence check after transpilation")

    def test_transpilation_with_parameter_binding(self):
        """Test transpilation with parameter binding."""
        from qiskit.circuit import Parameter

        # Create parameterized circuit
        theta = Parameter('θ')
        param_qc = QuantumCircuit(2)
        param_qc.rx(theta, 0)
        param_qc.cx(0, 1)

        # Bind parameters
        parameter_bindings = {theta: np.pi / 2}

        # Bind parameters before transpilation
        bound_qc = param_qc.assign_parameters(parameter_bindings)

        # Transpile using transpile_to_IQM
        transpiled_circuit = transpile_to_IQM(
            bound_qc,
            backend=self.backend,
            initial_layout=self.qubit_layout[:2],  # Only need 2 qubits
            coupling_map=self.coupling_map,
            optimization_level=1,
            seed_transpiler=42,
        )

        # Check that the output circuit has no free parameters
        self.assertEqual(len(transpiled_circuit.parameters), 0)


if __name__ == '__main__':
    unittest.main()
