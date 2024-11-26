#
# Qiskit Pass Manger - Examples
#

# Dynamic Decoupling
#
# Some circuits can be sparse or have long idle periods.
# Dynamical decoupling can echo away static ZZ errors during those idling periods.
# Set up a pass manager to add the decoupling pulses to the circuit before executing

from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionDurations, PassManager
from qiskit.transpiler.passes import ALAPSchedule, DynamicalDecoupling


def do_transform(circuit, backend):
    print("  ... performing dynamic decoupling.")
    durations = InstructionDurations.from_backend(backend)
    dd_sequence = [XGate(), XGate()]
    pm = PassManager([ALAPSchedule(durations), DynamicalDecoupling(durations, dd_sequence)])
    return pm.run(circuit)
