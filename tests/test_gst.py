"""Tests for compressive GST benchmark"""

from iqm.benchmarks.compressive_gst.compressive_gst import GSTConfiguration, CompressiveGST
from unittest.mock import patch

backend = "iqmfakeapollo"


class TestGST:
    @patch('matplotlib.pyplot.figure')
    def test_1q(self, mock_fig):
        minimal_1Q_config = GSTConfiguration(
            qubit_layouts=[[5], [0]],
            gate_set="1QXYI",
            num_circuits=10,
            shots=10,
            rank=4,
            bootstrap_samples=2,
            max_iterations=[1, 1],
        )
        benchmark = CompressiveGST(backend, minimal_1Q_config)
        benchmark.run()
        result = benchmark.analyze()
        mock_fig.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_2q(self, mock_fig):
        minimal_2Q_GST = GSTConfiguration(
            qubit_layouts=[[1, 0]],
            gate_set="2QXYCZ_extended",
            num_circuits=4,
            shots=10,
            rank=1,
            bootstrap_samples=0,
            max_iterations=[1, 1],
        )
        benchmark = CompressiveGST(backend, minimal_2Q_GST)
        benchmark.run()
        result = benchmark.analyze()
        mock_fig.assert_called()
