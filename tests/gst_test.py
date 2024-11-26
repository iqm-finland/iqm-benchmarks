""" Tests for compressive GST benchmark"""

from iqm.benchmark.compressive_gst.compressive_gst import GSTConfiguration
from unittest.mock import patch

# General BenchmarkExperiment
from iqm.benchmark.benchmark_experiment import BenchmarkExperiment

backend = "iqmfakeapollo"

class TestGST:
    @patch('matplotlib.pyplot.figure')
    def test_1q(self, mock_fig):
        Minimal_1Q_GST = GSTConfiguration(
            qubits=[5],
            gate_set="1QXYI",
            num_circuits=10,
            shots=10,
            rank=4,
            bootstrap_samples=0,
            max_iterations=[1, 1]
        )
        EXAMPLE_EXPERIMENT = BenchmarkExperiment(backend, [Minimal_1Q_GST])
        EXAMPLE_EXPERIMENT.run_experiment()
        mock_fig.assert_called()
    @patch('matplotlib.pyplot.figure')
    def test_2q(self, mock_fig):
        Minimal_2Q_GST = GSTConfiguration(
            qubits=[0, 3],
            gate_set="2QXYCZ_extended",
            num_circuits=10,
            shots=10,
            rank=1,
            bootstrap_samples=0,
            max_iterations=[1, 1]
        )
        EXAMPLE_EXPERIMENT = BenchmarkExperiment(backend, [Minimal_2Q_GST])
        EXAMPLE_EXPERIMENT.run_experiment()
        mock_fig.assert_called()
