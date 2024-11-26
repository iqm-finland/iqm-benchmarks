# IQM Benchmark

The IQM Benchmark is a suite of quantum characterization, verification, and validation (QCVV) tools for quantum computing. It is designed to be a comprehensive tool for benchmarking quantum hardware. The suite is designed to be modular, allowing users to easily add new benchmarks and customize existing ones. The suite is designed to be easy to use, with a simple API that allows users to run benchmarks with a single command.


Below is a list of the benchmarks currently available in the suite:
* Randomized Benchmarking: A suite of randomized benchmarking protocols for characterizing the performance of quantum gates (Clifford Randomized Benchmarking, Mirror Randomized Benchmarking, Interleaved Randomized Benchmarking).
* Quantum Volume: A benchmark for characterizing the performance of quantum computers.
* Q-Score: A benchmark that estimates the size of combinatorial optimization problems a given number of qubits can execute with meaningful results.
* GHZ State Benchmarking: A benchmark for characterizing the performance of multi-qubit entangled states.
* QED-C App-Oriented Benchmarks: A suite of benchmarks by the Quantum Economic Development Consortium for characterizing the performance of quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, Hidden Shift; Quantum Fourier Transform, Grover's Search; Phase Estimation, Amplitude Estimation, HHL Linear Solver; Monte Carlo, Hamiltonian (and HamLib) Simulation, Variational Quantum Eigensolver, Shor's Order Finding Algorithm; MaxCut, Hydrogen-Lattice).

The project is split into different benchmarks, all sharing the `Benchmark` class or the legacy `BenchmarkBase` class. Each individual benchmark takes as an argument their own `BenchmarkConfigurationBase` class. All the (legacy) benchmarks executed at once are wrapped by the `BenchmarkExperiment` class, which handles dependencies among the benchmarks, storing the results, producing the plots...



## Installation _(latest release)_

Usually it makes sense to use a new Conda environment (e.g. ``iqm-benchmark``) to isolate your setup from the global python installation. That way, you can play around without messing the rest of your system.

Start a terminal in your machine, and type

````
conda create -n iqm-benchmark python=3.11.2
conda activate iqm-benchmark
````

Then, you can install the latest release of the IQM Benchmark by running:
```bash
$ pip install iqm-benchmark
```

If you have already installed `iqm-benchmark` and want to get the latest release you can add the --upgrade flag:

```bash
pip install iqm-benchmark --upgrade
```

## Development mode _(latest changes: recommended)_

To install in development mode with all required dependencies, you can instead clone the [repository](https://www.github.com/iqm-finland/iqm-benchmark) and from the project directory run

```bash
python -m pip install -e ".[develop,testing,visualization]" --upgrade --upgrade-strategy=eager
```

To run the tests, you can use the following command:

```bash
tox -e test
```

## Characterize Physical Hardware

The IQM Benchmark suite is designed to be used with real quantum hardware. To use the suite, you will need to have access to a quantum computer. The suite is designed to work with both IQM Resonance (IQM's quantum cloud service) and on-prem devices, but can be easily adapted to work with other quantum computing platforms.

To use the suite with IQM Resonance, you will need to set up an account and obtain an API token. You can then set the `IQM_TOKEN` environment variable to your API token. The suite will automatically use this token to authenticate with IQM Resonance.

```python
import os
os.environ["IQM_TOKEN"] = "your_token"
```

### Using a Jupyter notebook or Python script

You can easily set up one or more benchmarks by defining a configuration for them. For example, for Randomized, Interleaved and Mirror Benchmarking, or Quantum Volume:

```python
from iqm.benchmark.randomized_benchmarking.interleaved_rb.interleaved_rb import InterleavedRBConfiguration
from iqm.benchmark.randomized_benchmarking.mirror_rb.mirror_rb import MirrorRBConfiguration
from iqm.benchmark.quantum_volume.quantum_volume import QuantumVolumeConfiguration

EXAMPLE_IRB = InterleavedRBConfiguration(
    qubits_array=[[3,4],[8,9]],
    sequence_lengths=[2**(m+1)-1 for m in range(7)],
    num_circuit_samples=30,
    shots=2**10,
    calset_id=None,
    parallel_execution=True,
    interleaved_gate = "iSwapGate",
    interleaved_gate_params = None,
    simultaneous_fit = ["amplitude", "offset"],
)

EXAMPLE_MRB = MirrorRBConfiguration(
    qubits_array=[[0,1],
                  [0,1,3,4],
                  [0,1,3,4,8,9],
                  [0,1,3,4,8,9,13,14],
                  [0,1,3,4,8,9,13,14,17,18]],
    depths_array=[[2**m for m in range(9)],
                  [2**m for m in range(8)],
                  [2**m for m in range(7)],
                  [2**m for m in range(6)],
                  [2**m for m in range(5)]],
    num_circuit_samples=10,
    num_pauli_samples=5,
    shots=2**8,
    two_qubit_gate_ensemble={"CZGate": 0.7, "iSwapGate": 0.3}, # {GATE: PROBABILITY}
    density_2q_gates=0.25,
    calset_id=None,
)

EXAMPLE_QV = QuantumVolumeConfiguration(
    num_circuits=500,
    shots=2**8,
    calset_id=None,
    num_sigmas=2,
    choose_qubits_routine="custom",
    custom_qubits_array=[[0,1,2,3], [0,1,3,4]],
    qiskit_optim_level=3,
    optimize_sqg=True,
    routing_method="sabre",
    physical_layout="fixed",
    max_gates_per_batch=60_000,
    rem=True,
    mit_shots=1_000,
)
```

In order to execute them, you must specify a backend.
* for IQM Resonance this can be given as a simple string, such as "garnet" (together with your IQM Token environment variable)
* and for an on-prem device and IQM Resonance this can be defined using the URL of the quantum computer.

Also, you need to reference the benchmark configuration you want to run:

```python
from iqm.benchmark.randomized_benchmarking.mirror_rb.mirror_rb import *
# import os
# os.environ["IQM_TOKEN"] = "your_token"

backend = IQMProvider("https://example-station.qc.iqm.fi/cocos/").get_backend()

EXAMPLE_EXPERIMENT = MirrorRandomizedBenchmarking(backend, EXAMPLE_MRB)
EXAMPLE_EXPERIMENT.run()
```

Full examples on how to run benchmarks and analyze the results can be found in the `examples` folder.

### Application-Oriented Benchmarks (QED-C)

The [app-oriented benchmarks from the QED-C](https://github.com/SRI-International/QC-App-Oriented-Benchmarks) are also hosted within the IQM benchmark suite under the `src/QEDC_app_oriented` folder, and can be easily executed in IQM's hardware with the `benchmarks_qiskit_IQM` Jupyter notebook.

### Scheduled benchmarks using a CI/CD Pipeline

This repository can be setup to perform a scheduled (weekly, daily...) benchmark from a Gitlab pipeline, executed on a real device. An example configuration is given in the `scheduled_experiments` folder.
