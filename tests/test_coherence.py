"""Tests for coherence estimation"""

from iqm.benchmarks.coherence.coherence import *
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb


class TestCoherence:
    backend = IQMFakeApollo()
    
    def test_coherence_t1(self):
        EXAMPLE_COHERENCE = CoherenceConfiguration(
            delays = list(np.linspace(0, 100e-6, 2)),
            qiskit_optim_level = 3,
            optimize_sqg = True,
            coherence_exp = "t1",
            qubits_to_plot=list(range(self.backend.num_qubits)),
        )
        benchmark = CoherenceBenchmark(self.backend, EXAMPLE_COHERENCE)
        benchmark.run()
        benchmark.analyze()

    def test_coherence_t2_echo(self):
        EXAMPLE_COHERENCE = CoherenceConfiguration(
            delays = list(np.linspace(0, 100e-6, 2)),
            qiskit_optim_level = 3,
            optimize_sqg = True,
            coherence_exp = "t2_echo",
            qubits_to_plot=list(range(self.backend.num_qubits)),
        )
        benchmark = CoherenceBenchmark(self.backend, EXAMPLE_COHERENCE)
        benchmark.run()
        benchmark.analyze()


class TestCoherenceDeneb(TestCoherence):
    backend = IQMFakeDeneb()

