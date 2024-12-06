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

"""
Graph states benchmark
"""

from typing import Sequence, Type

from qiskit import QuantumCircuit, transpile
import xarray as xr

from iqm.benchmarks import Benchmark
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.utils import set_coupling_map
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


def generate_graph_state(qubits: Sequence[int], backend: IQMBackendBase | str) -> QuantumCircuit:
    """ """
    num_qubits = len(qubits)
    qc = QuantumCircuit(num_qubits)
    coupling_map = set_coupling_map(qubits, backend, physical_layout="fixed")
    sorted_cp = [sorted(x) for x in list(coupling_map)]

    # Add all H
    for q in range(num_qubits):
        qc.h(q)
    # Add all CZ
    for c in [list(i) for i in set(map(tuple, sorted_cp))]:
        qc.cz(c[0], c[1])
    qc_t = transpile(qc, basis_gates=backend.operation_names)

    return qc_t


class GraphStatesBenchmark(Benchmark):
    """"""

    # analysis_function = staticmethod(negativity_analysis)
    name = "graph_states"

    def __init__(self, backend: IQMBackendBase, configuration: "GraphStatesConfiguration"):
        """Construct the GHZBenchmark class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend, configuration)

    def execute(self, backend) -> xr.Dataset:
        """
        Executes the benchmark.
        """


class GraphStatesConfiguration(BenchmarkConfigurationBase):
    """Graph States Benchmark configuration

    Attributes:
    """

    benchmark: Type[Benchmark] = GraphStatesBenchmark
