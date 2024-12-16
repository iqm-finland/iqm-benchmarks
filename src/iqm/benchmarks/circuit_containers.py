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

from dataclasses import dataclass, field
from typing import List, Optional, TypeAlias

from qiskit.circuit import Qubit

from iqm.qiskit_iqm.iqm_circuit import IQMCircuit


QubitLayout: TypeAlias = tuple[Qubit, ...]
QubitLayouts: TypeAlias = tuple[QubitLayout, ...]
QubitLayoutIndices: TypeAlias = tuple[tuple[int, ...], ...]


# pylint: disable=protected-access
@dataclass
class CircuitGroup:
    """Group of `IQMCircuits`. It represents a list of circuits with a common purpose, typically executed in a batch.

    Attributes:
        circuits: List of `IQMCircuit`.
        name: Name of the group.
    """

    circuits: List[IQMCircuit] = field(default_factory=list)
    name: Optional[str] = field(default="")

    @property
    def qubit_layouts_by_index(self) -> QubitLayoutIndices:
        """The qubit layouts contained in all the circuits, where the qubits are represented by an integer.

        Returns:
            All qubit layouts.
        """
        return tuple(tuple(int(qubit._index) for qubit in layout) for layout in self.qubit_layouts)

    @property
    def qubit_layouts(self) -> QubitLayouts:
        """The qubit layouts contained in all the circuits, where the qubits are represented by an instance of `Qubit`.

        Returns:
            All qubit layouts.
        """
        qubit_layouts = tuple(map(lambda x: tuple(x._data.active_bits()[0]), self.circuits))
        return qubit_layouts

    @property
    def qubits(self) -> set[Qubit]:
        """The set of active qubits in all of the circuits

        Returns:
            A set of `Qubit`
        """
        qubit_set = set()
        for circuit in self.circuits:
            for qubit in circuit._data.active_bits()[0]:
                qubit_set.add(qubit)
        return qubit_set

    def __setitem__(self, key: str, value: IQMCircuit) -> None:
        value.name = key
        return self.add_circuit(value)

    def add_circuit(self, circuit: IQMCircuit):
        """Adds a circuit to the internal list

        Args:
            circuit: Circuit to add.
        """
        self.circuits.append(circuit)

    @property
    def circuit_names(self) -> list[str]:
        """Name of all the circuits contained in the group as a string

        Returns:
            List of strings
        """
        benchmark_circuit_names = list(map(lambda x: x.name, self.circuits))
        return benchmark_circuit_names

    def get_circuits_by_name(self, name: str) -> Optional[IQMCircuit]:
        """Returns a list of the internal circuits that share the same name

        Args:
            name: Name of the circuit to return as a string

        Returns:
            The circuit with the desired name as a string, or None if they are not found.
        """
        benchmark_circuit_names = filter(lambda x: x.name == name, self.circuits)
        return next(benchmark_circuit_names, None)

    def __getitem__(self, key: str) -> Optional[IQMCircuit]:
        return self.get_circuits_by_name(key)


@dataclass
class BenchmarkCircuit:
    """A class grouping a list of `CircuitGroup` into a single purpose. This can typically represent, for example,
    all of the transpiled circuits that are executed.

    Attributes:
        name: Name of the `BenchmarkCircuit`.
        circuit_groups: List of `CircuitGroup` contained inside

    """

    name: str
    circuit_groups: List[CircuitGroup] = field(default_factory=list)

    def get_circuit_group_by_name(self, name: str) -> Optional[CircuitGroup]:
        """Gets a `CircuitGroup` by name.

        Args:
            name: name of the group.

        Returns:
            The desired `CircuitGroup`, or None if it does not exist.

        """
        benchmark_circuit_names = filter(lambda x: x.name == name, self.circuit_groups)
        next_value = next(benchmark_circuit_names, None)
        return next_value

    @property
    def groups(self) -> List[CircuitGroup]:
        """The circuit groups inside."""
        return self.circuit_groups

    @property
    def group_names(self) -> List[str | None]:
        """Names of all the contained `CircuitGroup`."""
        return list(map(lambda x: x.name, self.circuit_groups)) if len(self.circuit_groups) > 0 else []

    @property
    def qubit_indices(self) -> set[int]:
        """Set of all the qubits used in all the `CircuitGroup`, represented as an integer."""
        qubit_set = set()
        for circuit in self.circuit_groups:
            for qubit_index in map(lambda x: x._index, circuit.qubits):
                qubit_set.add(qubit_index)
        return qubit_set

    @property
    def qubits(self) -> set[Qubit]:
        """Set of all the qubits used in all the `CircuitGroup`, represented as an instance of `Qubit`."""
        qubit_set: set[Qubit] = set()
        for qubit in map(lambda x: x.qubits, self.circuit_groups):
            qubit_set = qubit_set.union(qubit)
        return qubit_set

    @property
    def qubit_layouts_by_index(self) -> set[QubitLayoutIndices]:
        """Set of all the qubit layouts used, where the qubits are represented as an integer."""
        layout_set = set()
        for layout in map(lambda x: x.qubit_layouts_by_index, self.circuit_groups):
            layout_set.add(layout)
        return layout_set

    @property
    def qubit_layouts(self) -> set[QubitLayouts]:
        """Set of all the qubit layouts used, where the qubits are represented as an instance of `Qubit`."""
        layout_set = set()
        for layout in map(lambda x: x.qubit_layouts, self.circuit_groups):
            layout_set.add(layout)
        return layout_set

    def __setitem__(self, key: str, value: CircuitGroup) -> None:
        value.name = key
        return self.circuit_groups.append(value)

    def __getitem__(self, key: str) -> CircuitGroup | None:
        return self.get_circuit_group_by_name(key)


@dataclass
class Circuits:
    """Container for all the `BenchmarkCircuit` that are generated in a single benchmark execution.

    Attributes:
        benchmark_circuits: List of `BenchmarkCircuit` contained.
    """

    benchmark_circuits: List[BenchmarkCircuit] = field(default_factory=list)

    def __setitem__(self, key: str, value: BenchmarkCircuit) -> None:
        value.name = key
        return self.benchmark_circuits.append(value)

    def __getitem__(self, key: str) -> BenchmarkCircuit | None:
        return self.get_benchmark_circuits_by_name(key)

    def get_benchmark_circuits_by_name(self, name: str) -> Optional[BenchmarkCircuit]:
        """Returned the `BenchmarkCircuit` by name.

        Args:
            name: Name of the requested `BenchmarkCircuit` as a string

        Returns:
            The requested `BenchmarkCircuit` if it exists, None otherwise
        """
        benchmark_circuit_names = filter(lambda x: x.name == name, self.benchmark_circuits)
        return next(benchmark_circuit_names, None)
