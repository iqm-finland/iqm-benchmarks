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
This module contains classes to easily interact with quantum circuits
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional, TypeAlias
import warnings

from qiskit.circuit import Qubit
from qiskit.circuit.quantumregister import QuantumRegister

from iqm.qiskit_iqm.iqm_circuit import IQMCircuit


QubitLayout: TypeAlias = tuple[tuple[Qubit]]
QubitLayoutIndices: TypeAlias = tuple[tuple[int]]


@dataclass
class CircuitGroup:
    circuits: List[IQMCircuit] = field(default_factory=list)
    name: Optional[str] = field(default="")

    @property
    def qubit_layouts_by_index(self) -> QubitLayoutIndices:
        return tuple(map(lambda x: tuple(q._index for q in x.qubits), self.circuits))

    @property
    def qubit_layouts(self, by_index: bool = True) -> QubitLayout:
        qubit_layouts = tuple(map(lambda x: tuple(x.qubits), self.circuits))
        return qubit_layouts

    @property
    def qubits(self) -> set[int]:
        qubit_set = set()
        for circuit in self.circuits:
            qubit_set.add(*circuit.qubits)
        return qubit_set

    def add_circuit(self, circuit: IQMCircuit):
        self.circuits.append(circuit)

    @property
    def circuit_names(self) -> list[str]:
        benchmark_circuit_names = list(map(lambda x: x.name, self.circuits))
        return benchmark_circuit_names

    def get_circuits_by_name(self, name: str) -> List[IQMCircuit]:
        benchmark_circuit_names = filter(lambda x: x.name == name, self.circuits)
        return next(benchmark_circuit_names, None)

    def __getitem__(self, key: str) -> IQMCircuit:
        return self.get_circuits_by_name(key)


@dataclass
class BenchmarkCircuit:
    name: str
    circuit_groups: List[CircuitGroup] = field(default_factory=list)

    def get_circuit_group_by_name(self, name: str) -> CircuitGroup:
        benchmark_circuit_names = filter(lambda x: x.name == name, self.circuit_groups)
        next_value = next(benchmark_circuit_names, None)
        return next_value

    @property
    def qubit_indices(self) -> set[int]:
        qubit_set = set()
        for circuit in self.circuit_groups:
            for qubit_index in map(lambda x: x._index, circuit.qubits):
                qubit_set.add(qubit_index)
        return qubit_set

    @property
    def qubits(self) -> set[Qubit]:
        qubit_set = set()
        for circuit in self.circuit_groups:
            qubit_set.add(*circuit.qubits)
        return qubit_set

    @property
    def qubit_layouts_by_index(self) -> set[QubitLayoutIndices]:
        layout_set = set()
        for circuit in self.circuit_groups:
            layout_set.add(*circuit.qubit_layouts_by_index)
        return layout_set

    @property
    def qubit_layouts(self) -> set[QubitLayout]:
        layout_set = set()
        for circuit in self.circuit_groups:
            layout_set.add(*circuit.qubit_layouts)
        return layout_set

    def __getitem__(self, key: str) -> List[CircuitGroup]:
        return self.get_circuit_group_by_name(key)


@dataclass
class Circuits:
    benchmark_circuits: List[BenchmarkCircuit] = field(default_factory=list)

    def __getitem__(self, key: str) -> List[BenchmarkCircuit]:
        return self.get_benchmark_circuits_by_name(key)

    def get_benchmark_circuits_by_name(self, name: str) -> List[BenchmarkCircuit]:
        benchmark_circuit_names = filter(lambda x: x.name == name, self.benchmark_circuits)
        return next(benchmark_circuit_names, None)
