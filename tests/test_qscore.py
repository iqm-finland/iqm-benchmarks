"""Tests for Qscore estimation"""

from iqm.benchmarks.optimization.qscore import *
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb

class TestQScore:
    backend = IQMFakeApollo()
    custom_qubits_array = [[0, 1, 2, 3], [0, 2, 3, 4, 8], [0, 1, 2, 3, 4, 8]]


    def test_qscore_combinations(self):
        combinations = [
            (True, "custom"),
            (True, "naive"),
            (False, "custom"),
            (False, "naive"),
        ]
        
        for use_virtual_node, choose_qubits_routine in combinations:
            EXAMPLE_QSCORE = QScoreConfiguration(
                num_instances=10,
                num_qaoa_layers=1,
                shots=4,
                calset_id=None,
                min_num_nodes=4,
                max_num_nodes=6,
                use_virtual_node=use_virtual_node,
                use_classically_optimized_angles=True,
                choose_qubits_routine=choose_qubits_routine,
                custom_qubits_array=self.custom_qubits_array,
                seed=399477,
                num_trials=2,
                REM=True,
                mit_shots=10,
            )
            benchmark = QScoreBenchmark(self.backend, EXAMPLE_QSCORE)
            benchmark.run()
            benchmark.analyze()


class TestQScoreDeneb(TestQScore):
    backend = IQMFakeDeneb()
    custom_qubits_array = [[1, 3, 4, 5], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]
