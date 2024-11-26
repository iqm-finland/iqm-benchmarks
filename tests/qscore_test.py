"""Tests for volumetric benchmarks"""

from unittest.mock import patch

from iqm.benchmarks.benchmark_experiment import BenchmarkExperiment
from iqm.benchmarks.optimization.qscore import QScoreConfiguration


backend = "IQMFakeAdonis"


class TestQScore:
    @patch('matplotlib.pyplot.figure')
    def test_qscore(self, mock_fig):
        EXAMPLE_QSCORE = QScoreConfiguration(
            num_instances=2,
            num_qaoa_layers=1,
            shots=4,
            calset_id=None,  # calibration set ID, default is None
            min_num_nodes=2,
            max_num_nodes=5,
            use_virtual_node=True,
            use_classically_optimized_angles=True,
            choose_qubits_routine="custom",
            custom_qubits_array=[[2], [2, 0], [2, 0, 1], [2, 0, 1, 3], [2, 0, 1, 3, 4]],
            seed=1,
        )
        EXAMPLE_EXPERIMENT = BenchmarkExperiment(backend, [EXAMPLE_QSCORE])
        EXAMPLE_EXPERIMENT.run_experiment()
        mock_fig.assert_called()
