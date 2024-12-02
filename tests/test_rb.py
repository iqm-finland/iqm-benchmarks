"""Tests for mirror RB"""

import numpy as np

from iqm.benchmarks.randomized_benchmarking.clifford_rb.clifford_rb import (
    CliffordRandomizedBenchmarking,
    CliffordRBConfiguration,
)
from iqm.benchmarks.randomized_benchmarking.interleaved_rb.interleaved_rb import (
    InterleavedRandomizedBenchmarking,
    InterleavedRBConfiguration,
)
from iqm.benchmarks.randomized_benchmarking.mirror_rb.mirror_rb import (
    MirrorRandomizedBenchmarking,
    MirrorRBConfiguration,
)


backend = "fakeapollo"


class TestRB:
    def test_mrb(self):
        EXAMPLE_MRB = MirrorRBConfiguration(
            qubits_array=[[0, 1], [0, 3]],
            depths_array=[[2**m for m in range(4)]],
            num_circuit_samples=2,
            num_pauli_samples=2,
            shots=2**4,
            qiskit_optim_level=1,
            routing_method="sabre",
            two_qubit_gate_ensemble={"CZGate": 0.8, "iSwapGate": 0.2},
            density_2q_gates=0.25,
        )
        benchmark = MirrorRandomizedBenchmarking(backend, EXAMPLE_MRB)
        benchmark.run()
        benchmark.analyze()

    def test_irb(self):
        EXAMPLE_IRB_1Q = InterleavedRBConfiguration(
            qubits_array=[[0]],
            sequence_lengths=[2 ** (m + 1) - 1 for m in range(4)],
            num_circuit_samples=2,
            shots=2**4,
            parallel_execution=True,
            interleaved_gate="RGate",
            interleaved_gate_params=[np.pi, 0],
            simultaneous_fit=["amplitude", "offset"],
        )
        benchmark = InterleavedRandomizedBenchmarking(backend, EXAMPLE_IRB_1Q)
        benchmark.run()
        benchmark.analyze()

    def test_crb(self):
        EXAMPLE_CRB_1Q = CliffordRBConfiguration(
            qubits_array=[[3]],
            sequence_lengths=[2 ** (m + 1) - 1 for m in range(4)],
            num_circuit_samples=2,
            shots=2**4,
            calset_id=None,
            parallel_execution=False,
        )
        benchmark = CliffordRandomizedBenchmarking(backend, EXAMPLE_CRB_1Q)
        benchmark.run()
        benchmark.analyze()
