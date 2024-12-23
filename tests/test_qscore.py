"""Tests for volumetric benchmarks"""

from iqm.benchmarks.optimization.qscore import *

backend = "IQMFakeAdonis"

class TestQScore:
    def test_qscore(self):
        EXAMPLE_QSCORE = QScoreConfiguration(
            num_instances=2,
            num_qaoa_layers=1,
            shots=4,
            calset_id=None,  # calibration set ID, default is None
            min_num_nodes=2,
            max_num_nodes=5,
            use_virtual_node=True,
            use_classically_optimized_angles=True,
            choose_qubits_routine="custom",
            custom_qubits_array=[[2, 0], [2, 0, 1], [2, 0, 1, 3], [2, 0, 1, 3, 4]],
            seed=1,
        )
        benchmark = QScoreBenchmark(backend, EXAMPLE_QSCORE)
        benchmark.run()
        benchmark.analyze()
