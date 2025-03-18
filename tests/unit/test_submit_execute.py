import unittest
from unittest.mock import Mock, patch, call

from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_job import IQMJob
from iqm.iqm_client.models import CircuitCompilationOptions
from iqm.benchmarks.utils import submit_execute



class TestSubmitExecute(unittest.TestCase):
    def setUp(self):
        # Create mock backend
        self.mock_backend = Mock(spec=IQMBackendBase)
        self.mock_run_result = Mock(spec=IQMJob)
        self.mock_backend.run.return_value = self.mock_run_result

        # Create test circuits with varying gate counts
        self.create_test_circuits()

        # Create a dict mapping tuple keys to lists of circuits
        self.sorted_circuits = {
            (0, 1): self.qc_list_small,  # 10 circuits with 5 operations each
            (2, 3): self.qc_list_large,  # 5 circuits with 12 operations each
        }

    def create_test_circuits(self):
        # Small circuits (5 operations each)
        self.qc_list_small = []
        for _ in range(10):
            qc = QuantumCircuit(2, 2)
            qc.r(0.1, 0.2, 0)
            qc.r(0.3, 0.4, 1)
            qc.cz(0, 1)
            qc.measure_all()
            # Mock the count_ops method to return a fixed gate count
            qc.count_ops = Mock(return_value={'r': 2, 'cz': 1, 'measure': 2})
            self.qc_list_small.append(qc)

        # Large circuits (12 operations each)
        self.qc_list_large = []
        for _ in range(5):
            qc = QuantumCircuit(2, 2)
            for _ in range(5):
                qc.r(0.1, 0.2, 0)
                qc.r(0.3, 0.4, 1)
            qc.measure_all()
            # Mock the count_ops method to return a fixed gate count
            qc.count_ops = Mock(return_value={'r': 10, 'measure': 2})
            self.qc_list_large.append(qc)

    @patch('iqm.benchmarks.utils.qcvv_logger')
    def test_submit_execute_no_restrictions(self, mock_logger):
        """Test with no batch size restrictions."""
        jobs,_ = submit_execute(self.sorted_circuits, self.mock_backend, shots=1000)

        # Should return after first batch
        self.assertEqual(len(jobs), 2)

        # Check that backend.run was called with each circuit list
        expected_calls = []
        for key in self.sorted_circuits.keys():
            expected_calls.append(call(
                self.sorted_circuits[key], shots=1000, calibration_set_id=None
            ))
        self.mock_backend.run.assert_has_calls(expected_calls, any_order=True)

    @patch('iqm.benchmarks.utils.qcvv_logger')
    def test_submit_execute_max_circuits(self, mock_logger):
        """Test with max_circuits_per_batch restriction."""

        jobs, _ = submit_execute(
            self.sorted_circuits,
            self.mock_backend,
            shots=1000,
            max_circuits_per_batch=2
        )

        # Check that backend.run was called with correct batches
        expected_calls = [
            # First key (2,3) - 5 circuits in batches of 2
            call(self.qc_list_large[0:2], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            call(self.qc_list_large[2:4], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            call([self.qc_list_large[4]], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            # Second key (0,1) - 10 circuits in batches of 2
            call(self.qc_list_small[0:2], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            call(self.qc_list_small[2:4], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            call(self.qc_list_small[4:6], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            call(self.qc_list_small[6:8], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
            call(self.qc_list_small[8:10], shots=1000, calibration_set_id=None, circuit_compilation_options=None),
        ]

        # Verify all calls were made and all circuits were used
        self.mock_backend.run.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(self.mock_backend.run.call_count, 8)

    @patch('iqm.benchmarks.utils.qcvv_logger')
    def test_submit_execute_max_gates(self, mock_logger):
        """Test with max_gates_per_batch restriction."""

        # Each small circuit has 5 operations, each large has 10
        # Set max_gates_per_batch to a value that will test different batch sizes
        jobs,_ = submit_execute(
            self.sorted_circuits,
            self.mock_backend,
            shots=1000,
            max_gates_per_batch=15
        )

        # For large circuits (12 operations each), batches should have 1 circuit per batch
        # For small circuits (5 operations each), batches should have 3 circuits per batch

        expected_batch_sizes_large = [1, 1, 1, 1, 1]  # 5 batches of 1 circuit
        expected_batch_sizes_small = [3, 3, 3, 1]  # 3 batches of 3 circuits and one batch of one circuit

        actual_batch_sizes = [len(args[0]) for args, _ in self.mock_backend.run.call_args_list]

        # Verify we have the right number of batches
        self.assertEqual(len(actual_batch_sizes),
                         len(expected_batch_sizes_large) + len(expected_batch_sizes_small))

        # Verify batch sizes match our expectations
        self.assertEqual(actual_batch_sizes[4:], expected_batch_sizes_large)
        self.assertEqual(actual_batch_sizes[:4], expected_batch_sizes_small)

        # Check that all circuits were included
        total_circuits = sum(actual_batch_sizes)
        self.assertEqual(total_circuits, len(self.qc_list_large) + len(self.qc_list_small))

    @patch('iqm.benchmarks.utils.qcvv_logger')
    def test_submit_execute_both_restrictions(self, mock_logger):
        """Test with both max_gates_per_batch and max_circuits_per_batch restrictions."""

        jobs,_ = submit_execute(
            self.sorted_circuits,
            self.mock_backend,
            shots=1000,
            max_gates_per_batch=15,
            max_circuits_per_batch=2
        )

        # For large circuits:
        # - max_gates_per_batch=15 would allow 1 circuit per batch (12 operations each)
        # - max_circuits_per_batch=2 would allow 2 circuits per batch
        # Should use the more restrictive: 1 circuit per batch

        # For small circuits:
        # - max_gates_per_batch=15 would allow 3 circuits per batch (5 operations each)
        # - max_circuits_per_batch=2 would allow 2 circuits per batch
        # Should use the more restrictive: 2 circuits per batch

        expected_batch_sizes_large = [1, 1, 1, 1, 1]  # 5 batches of 1 circuit
        expected_batch_sizes_small = [2, 2, 2, 2, 2]  # 5 batches of 2 circuits

        actual_batch_sizes = [len(args[0]) for args, _ in self.mock_backend.run.call_args_list]

        # Verify batch sizes match our expectations
        self.assertEqual(actual_batch_sizes[5:], expected_batch_sizes_large)
        self.assertEqual(actual_batch_sizes[:5], expected_batch_sizes_small)

        # Check that all circuits were included
        total_circuits = sum(actual_batch_sizes)
        self.assertEqual(total_circuits, len(self.qc_list_large) + len(self.qc_list_small))

        # Verify batch sizes are within maximum limits
        for batch_size in actual_batch_sizes:
            # Circuits per batch should not exceed max_circuits_per_batch
            self.assertLessEqual(batch_size, 2)

if __name__ == '__main__':
    unittest.main()