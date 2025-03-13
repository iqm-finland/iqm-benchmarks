import unittest
import numpy as np
from qiskit import QuantumCircuit
from iqm.qiskit_iqm import transpile_to_IQM
from qiskit.quantum_info import Operator
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.benchmarks.utils import perform_backend_transpilation, set_coupling_map, reduce_to_active_qubits


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

    def test_basic_transpilation(self):
        """Test basic transpilation functionality."""

        transpiled_circuits, _ = perform_backend_transpilation(
            self.test_circuits,
            self.backend,
            self.qubit_layout,
            self.coupling_map,
            qiskit_optim_level=1,
            optimize_sqg=True,
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
                fidelity = np.abs(op.equiv(op_transp))
                self.assertGreaterEqual(fidelity, 0.9999)

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
            # For small circuits we can check unitary equivalence
            if self.test_circuits[i].num_qubits <= 3:
                reduced_qc_with = reduce_to_active_qubits(transpiled_with_opt[i])
                reduced_qc_with.remove_final_measurements()
                reduced_qc_without = reduce_to_active_qubits(transpiled_without_opt[i])
                reduced_qc_without.remove_final_measurements()
                op_with = Operator(reduced_qc_with)
                op_without = Operator(reduced_qc_without)
                fidelity = np.abs(op_with.equiv(op_without))
                self.assertGreaterEqual(fidelity, 0.9999)

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
            seed_transpiler=42
        )

        # Check that the output circuit has no free parameters
        self.assertEqual(len(transpiled_circuit.parameters), 0)


if __name__ == '__main__':
    unittest.main()
