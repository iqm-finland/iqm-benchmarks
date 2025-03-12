"""Tests for GHZ fidelity estimation using the new base class"""

from unittest.mock import patch

from iqm.benchmarks.entanglement.ghz import GHZBenchmark, GHZConfiguration
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb

class TestGHZ:
    backend = IQMFakeApollo()

    @patch('matplotlib.pyplot.figure')
    def test_layouts(self, mock_fig):
        MINIMAL_GHZ = GHZConfiguration(
            state_generation_routine=f"tree",
            custom_qubits_array=[
                [0, 1],
                [1, 3, 4],
                [1, 3, 4, 5],
            ],
            shots=3,
            qiskit_optim_level=3,
            optimize_sqg=True,
            fidelity_routine="coherences",
            num_RMs=10,
            rem=False,
            mit_shots=10,
            use_dd = True
        )
        benchmark = GHZBenchmark(self.backend, MINIMAL_GHZ)
        benchmark.run()
        benchmark.analyze()
        mock_fig.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_state_routine(self, mock_fig):
        for gen_routine in [f"tree", f"naive", "log_depth", "star"]:
            MINIMAL_GHZ = GHZConfiguration(
                state_generation_routine=gen_routine,
                custom_qubits_array=[[2, 3, 4]],
                shots=3,
                qiskit_optim_level=3,
                optimize_sqg=True,
                fidelity_routine="coherences",
                num_RMs=10,
                rem=False,
                mit_shots=10,
            )
            benchmark = GHZBenchmark(self.backend, MINIMAL_GHZ)
            benchmark.run()
            benchmark.analyze()
            mock_fig.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_rem(self, mock_fig):
        for fidelity_routine in [f"coherences", f"randomized_measurements"]:
            MINIMAL_GHZ = GHZConfiguration(
                state_generation_routine=f"tree",
                custom_qubits_array=[[2, 3, 4]],
                shots=3,
                qiskit_optim_level=3,
                optimize_sqg=True,
                fidelity_routine=fidelity_routine,
                num_RMs=10,
                rem=True,
                mit_shots=10,
            )
            benchmark = GHZBenchmark(self.backend, MINIMAL_GHZ)
            benchmark.run()
            benchmark.analyze()
            mock_fig.assert_called()

class TestGHZDeneb(TestGHZ):
    backend = IQMFakeDeneb()