import pytest
from qiskit import QuantumRegister
from qiskit.circuit.quantumcircuit import Qubit

from iqm.benchmarks import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.qiskit_iqm.iqm_circuit import IQMCircuit


def test_circuit_group():
    qreg = QuantumRegister(size=3, name='test_register')
    qc = IQMCircuit(
        qreg,
        name='test_circuit',
    )
    qc.x(0)
    circuit_group = CircuitGroup(circuits=[qc])
    circuit_group.add_circuit(qc)
    assert circuit_group.circuits == [qc, qc]
    assert circuit_group.qubits == set([Qubit(QuantumRegister(3, 'test_register'), 0)])
    assert circuit_group.qubit_layouts == tuple(
        [(Qubit(QuantumRegister(3, 'test_register'), 0),), (Qubit(QuantumRegister(3, 'test_register'), 0),)]
    )
    assert circuit_group.qubit_layouts_by_index == ((0,), (0,))
    assert circuit_group.circuit_names == ['test_circuit', 'test_circuit']


def test_benchmark_circuit():
    qreg = QuantumRegister(size=2, name='test_register')
    qc = IQMCircuit(
        qreg,
        name='test_circuit',
    )
    qc.x(0)
    circuit_group = CircuitGroup(name='circuit_group', circuits=[qc])

    benchmark_circuit = BenchmarkCircuit(name='test', circuit_groups=[circuit_group])

    # print(benchmark_circuit['circuit_group'].name)
    assert benchmark_circuit.circuit_groups == [circuit_group]
    assert benchmark_circuit['circuit_group'] == circuit_group
    assert benchmark_circuit.qubit_indices == set([0])
    assert benchmark_circuit.qubit_layouts_by_index == {((0,),)}
    assert benchmark_circuit.qubit_layouts == {((Qubit(QuantumRegister(2, name='test_register'), 0),),)}


def test_circuits():
    qreg = QuantumRegister(size=1, name='test_register')
    qc = IQMCircuit(
        qreg,
        name='test_circuit',
    )
    circuit_group = CircuitGroup(name='circuit_group', circuits=[qc])

    benchmark_circuit = BenchmarkCircuit(name='test', circuit_groups=[circuit_group])
    circuits = Circuits([benchmark_circuit])
    assert circuits['test'] == benchmark_circuit
