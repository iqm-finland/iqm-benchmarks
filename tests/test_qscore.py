"""Tests for Qscore estimation"""

from iqm.benchmarks.optimization.qscore import *
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb


class TestQScore:
    backend = IQMFakeApollo()
    qpu_topology = "crystal"
    custom_qubits_array = [[0, 1, 3], [0, 1, 2, 3] , [0, 1, 2, 3, 4]]

    def test_qscore(self):
        EXAMPLE_QSCORE = QScoreConfiguration(
            num_instances=2,
            num_qaoa_layers=1,
            shots=4,
            calset_id=None,  # calibration set ID, default is None
            min_num_nodes=2,
            max_num_nodes=4,
            use_virtual_node=True,
            use_classically_optimized_angles=True,
            choose_qubits_routine="custom",
            custom_qubits_array=self.custom_qubits_array,
            seed=1,
            REM=True,
            mit_shots=10,
            qpu_topology=self.qpu_topology
        )
        benchmark = QScoreBenchmark(self.backend, EXAMPLE_QSCORE)
        benchmark.run()
        benchmark.analyze()

class TestQScoreDeneb(TestQScore):
    backend = IQMFakeDeneb()
    qpu_topology = "star"
    custom_qubits_array = [[1], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5]]