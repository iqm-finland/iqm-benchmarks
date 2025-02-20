"""Tests for volumetric benchmarks"""

from iqm.benchmarks.quantum_volume.clops import CLOPSBenchmark, CLOPSConfiguration
from iqm.benchmarks.quantum_volume.quantum_volume import QuantumVolumeBenchmark, QuantumVolumeConfiguration
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb


class TestQV:
    backend = IQMFakeApollo()

    def test_qv(self):
        EXAMPLE_QV = QuantumVolumeConfiguration(
            num_circuits=5,
            shots=2**5,
            calset_id=None,
            num_sigmas=2,
            choose_qubits_routine="custom",
            custom_qubits_array=[[2, 3]],
            qiskit_optim_level=3,
            optimize_sqg=True,
            routing_method="sabre",
            physical_layout="fixed",
            max_gates_per_batch=60_000,
            rem=True,
            mit_shots=10,
        )
        benchmark = QuantumVolumeBenchmark(self.backend, EXAMPLE_QV)
        benchmark.run()
        benchmark.analyze()

    def test_clops(self):
        EXAMPLE_CLOPS = CLOPSConfiguration(
            qubits=[2, 3],
            num_circuits=4,  # By definition set to 100
            num_updates=2,  # By definition set to 10
            num_shots=2**5,  # By definition set to 100
            calset_id=None,
            clops_h_bool=True,
            qiskit_optim_level=3,
            optimize_sqg=True,
            routing_method="sabre",
            physical_layout="fixed",
        )
        benchmark = CLOPSBenchmark(self.backend, EXAMPLE_CLOPS)
        benchmark.run()
        benchmark.analyze()

class TestQVDeneb(TestQV):
    backend = IQMFakeDeneb()