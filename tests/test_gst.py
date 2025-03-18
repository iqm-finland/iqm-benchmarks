"""Tests for compressive GST benchmark"""

from unittest.mock import patch

from iqm.benchmarks.compressive_gst.compressive_gst import CompressiveGST, GSTConfiguration
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb


class TestGST:
    backend = IQMFakeApollo()

    @patch('matplotlib.pyplot.figure')
    def test_1q(self, mock_fig):
        minimal_1Q_config = GSTConfiguration(
            qubit_layouts=[[4], [1]],
            gate_set="1QXYI",
            num_circuits=10,
            shots=10,
            rank=4,
            bootstrap_samples=2,
            max_iterations=[1, 1],
            parallel_execution=True,
        )
        benchmark = CompressiveGST(self.backend, minimal_1Q_config)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_2q(self, mock_fig):
        minimal_2Q_GST = GSTConfiguration(
            qubit_layouts=[[2, 3]],
            gate_set="2QXYCZ_extended",
            num_circuits=4,
            shots=10,
            rank=1,
            bootstrap_samples=0,
            max_iterations=[1, 1],
        )
        benchmark = CompressiveGST(self.backend, minimal_2Q_GST)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

class TestGSTDeneb(TestGST):
    backend = IQMFakeDeneb()