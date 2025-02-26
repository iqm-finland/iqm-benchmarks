"""Tests for graph state entanglement benchmark"""

from unittest.mock import patch

from iqm.benchmarks.entanglement.graph_states import *
from iqm.qiskit_iqm.fake_backends.fake_adonis import IQMFakeAdonis
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb

class TestGraphState:
    backend = IQMFakeAdonis()

    @patch('matplotlib.pyplot.figure')
    def test_state_tomo(self, mock_fig):
        MINIMAL_GRAPHSTATE = GraphStateConfiguration(
            qubits=list(range(self.backend.num_qubits)),
            shots=2 ** 12,
            tomography="state_tomography",
            num_bootstraps=10,
        )
        benchmark = GraphStateBenchmark(self.backend, MINIMAL_GRAPHSTATE)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_shadows(self, mock_fig):
        MINIMAL_GRAPHSTATE = GraphStateConfiguration(
            qubits=list(range(self.backend.num_qubits)),
            shots=2 ** 6,
            tomography="shadow_tomography",
            num_bootstraps=2,
            n_random_unitaries=10,
            n_median_of_means=2,
        )
        benchmark = GraphStateBenchmark(self.backend, MINIMAL_GRAPHSTATE)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

class TestGraphStateDeneb(TestGraphState):
    backend = IQMFakeDeneb()