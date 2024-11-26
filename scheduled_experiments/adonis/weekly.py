# Copyright 2024 IQM Benchmarks developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AUTOMATED WEEKLY EXPERIMENT CONFIGURATION EXAMPLE
"""

from iqm.benchmarks.randomized_benchmarking.clifford_rb.clifford_rb import CliffordRBConfiguration
from iqm.benchmarks.randomized_benchmarking.interleaved_rb.interleaved_rb import InterleavedRBConfiguration
from iqm.benchmarks.randomized_benchmarking.mirror_rb.mirror_rb import MirrorRBConfiguration
from iqm.benchmarks.quantum_volume.clops import CLOPSConfiguration
from iqm.benchmarks.quantum_volume.quantum_volume import QuantumVolumeConfiguration
from iqm.benchmarks.entanglement.ghz import GHZConfiguration

WEEKLY_CRB = CliffordRBConfiguration(
    qubits_array=[[0,2]],
    sequence_lengths=[2**(m+1)-1 for m in range(7)],
    num_circuit_samples=25,
    shots=2**8,
    calset_id=None,
    parallel_execution=False,
)

WEEKLY_IRB = InterleavedRBConfiguration(
    qubits_array=[[0,2]],
    sequence_lengths=[2**(m+1)-1 for m in range(7)],
    num_circuit_samples=25,
    shots=2**8,
    calset_id=None,
    parallel_execution=False,
    interleaved_gate = "CZGate",
    interleaved_gate_params = None,
    simultaneous_fit = ["amplitude", "offset"],
)

WEEKLY_MRB = MirrorRBConfiguration(
    qubits_array=[[0,2],
                  [0,1,2],
                  [0,1,2,3],
                  [0,1,2,3,4]],
    depths_array=[[2**m for m in range(7)]],
    num_circuit_samples=10,
    num_pauli_samples=5,
    shots=2**8,
    qiskit_optim_level=1,
    routing_method="sabre",
    two_qubit_gate_ensemble={"CZGate": 1.0},
    density_2q_gates=0.35,
    calset_id=None,
)

WEEKLY_QV = QuantumVolumeConfiguration(
    num_circuits=500,
    shots=2**8,
    calset_id=None,
    num_sigmas=2,
    choose_qubits_routine="custom",
    custom_qubits_array=[[0,1,2,3,4]],
    #dynamical_decoupling=True,  # Default is False
    #dd_strategy=dd_custom,  # Default is dd.DDStrategy()
    qiskit_optim_level=3,
    optimize_sqg=True,
    routing_method="sabre",
    physical_layout="fixed",
    max_gates_per_batch=60_000,
    rem=True,
    mit_shots=1_000,
)

WEEKLY_CLOPS = CLOPSConfiguration(
    qubits=[0,1,2,3,4],
    num_circuits=100, # By definition set to 100
    num_updates=10, # By definition set to 10
    num_shots=100, # By definition set to 100
    calset_id=None,
    clops_h_bool=True,
    qiskit_optim_level=3,
    optimize_sqg=True,
    routing_method="sabre",
    physical_layout="fixed",
)

WEEKLY_GHZ = GHZConfiguration(
    state_generation_routine="tree",  # "naive", "log_depth" (default)
    custom_qubits_array=[[0,1,2,3,4]],
    shots=2**10,
    qiskit_optim_level=3,
    optimize_sqg=True,
    fidelity_routine="coherences",
    rem=True,
    mit_shots=1000,
)
