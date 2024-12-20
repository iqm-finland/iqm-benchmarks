"""Tests for GHZ fidelity estimation using the new base class"""

import numpy as np

from iqm.benchmarks.entanglement.ghz import GHZBenchmark, GHZConfiguration
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo


backend = IQMFakeApollo()


class TestGHZ:
    def test_layouts(self):
        MINIMAL_GHZ = GHZConfiguration(
            state_generation_routine=f"tree",
            custom_qubits_array=[
                [10, 15],
                [0, 1, 3],
                [0, 1, 3, 4],
                # [0,1,2,3,4],
                # [0,1,2,3,4,5],
                # [0,1,2,3,4,5,6],
            ],
            shots=3,
            qiskit_optim_level=3,
            optimize_sqg=True,
            fidelity_routine="coherences",
            num_RMs=10,
            rem=False,
            mit_shots=10,
        )
        benchmark = GHZBenchmark(backend, MINIMAL_GHZ)
        benchmark.run()
        benchmark.analyze()

    def test_state_routine(self):
        for gen_routine in [f"tree", f"naive", "log_depth"]:
            MINIMAL_GHZ = GHZConfiguration(
                state_generation_routine=gen_routine,
                custom_qubits_array=[[0, 1, 2, 3]],
                shots=3,
                qiskit_optim_level=3,
                optimize_sqg=True,
                fidelity_routine="coherences",
                num_RMs=10,
                rem=False,
                mit_shots=10,
            )
            benchmark = GHZBenchmark(backend, MINIMAL_GHZ)
            benchmark.run()
            benchmark.analyze()

    def test_rem(self):
        for fidelity_routine in [f"coherences", f"randomized_measurements"]:
            MINIMAL_GHZ = GHZConfiguration(
                state_generation_routine=f"tree",
                custom_qubits_array=[[0, 1, 2, 3]],
                shots=3,
                qiskit_optim_level=3,
                optimize_sqg=True,
                fidelity_routine=fidelity_routine,
                num_RMs=10,
                rem=True,
                mit_shots=10,
            )
            benchmark = GHZBenchmark(backend, MINIMAL_GHZ)
            benchmark.run()
            benchmark.analyze()
