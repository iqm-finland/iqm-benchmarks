"""Tests for volumetric benchmarks"""

from iqm.benchmarks.quantum_volume.clops import CLOPSBenchmark, CLOPSConfiguration
from iqm.benchmarks.quantum_volume.clops_h import CLOPSHBenchmark, CLOPSHConfiguration

from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb

import os
from iqm.qiskit_iqm import IQMProvider
os.environ[
    "IQM_TOKEN"
] = ""
iqm_server_url = "https://cocos.resonance.meetiqm.com/sirius"

provider = IQMProvider(iqm_server_url)
backend = provider.get_backend()

class TestQV:
    # backend = IQMFakeApollo()

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

    def test_clops_h(self):
        EXAMPLE_CLOPSH = CLOPSHConfiguration(
            qubits=[2, 3],
            num_circuits=12,  # By definition set to 5000
            num_layers=5,  # By definition set to 100
            num_shots=100,  # By definition set to 100
            max_circuits_per_batch = 4,
            calset_id=None,
            routing_method="sabre",
            physical_layout="fixed",
        )
        benchmark = CLOPSHBenchmark(self.backend, EXAMPLE_CLOPSH)
        benchmark.run()
        benchmark.analyze()


class TestQVDeneb(TestQV):
    backend = IQMFakeDeneb()
