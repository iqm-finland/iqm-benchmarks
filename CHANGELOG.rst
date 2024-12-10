=========
Changelog
=========

Version 1.9
===========
* Fixed bug (overwriting observations) in Quantum Volume.
* Fixed small bug in CLOPS when calling plots in simulator execution.

Version 1.8
===========
* Changed compressive GST to operate under the new base class and added multiple qubit layouts.
* Added plot to GHZ benchmark and applied small fixes.
* Added tutorial notebook for the GHZ benchmark.

Version 1.7
===========
* Remove explicit dependency on qiskit, instead taking it from qiskit-on-iqm.

Version 1.6
===========
* Minor change in dependencies for compatibility.

Version 1.5
===========
* fit results are no longer `BenchmarkObservation`, and instead are moved into the datasets.

Version 1.4
===========

* Renames:

  * AnalysisResult -> BenchmarkAnalysisResult
  * RunResult -> BenchmarkRunResult

* Adds BenchmarkObservation class, and modifies BenchmarkAnalysisResult so observations now accepts a list[BenchmarkObservation].
* Adds BenchmarkObservationIdentifier class.
* Rebases RandomizedBenchmarking benchmarks, QuantumVolume, GHZ and CLOPS to use the new Observation class.
* Fixes serialization of some circuits.
* Adds AVAILABLE_BENCHMARKS to map a benchmark name to its class in __init__.
* Adds benchmarks and configurations to __init__ for public import.
* Other fixes.

Version 1.3
===========

* Further improvements to type hints, docstrings, and error messages.

Version 1.2
===========

* Minor improvements to type hints, docstrings, and error messages.

Version 1.1
===========

* Fixed bug preventing execution on a generic IQM Backend.
* Randomized Benchmarking (Clifford, Interleaved and Mirror), Quantum Volume, CLOPS and GHZ state fidelity all functioning exclusively under new Benchmark base class.
* Updated separate example Jupyter notebooks.

Version 1.0
===========

* Published Randomized Benchmarking (Clifford, Interleaved and Mirror), Quantum Volume, CLOPS and GHZ state fidelity all functioning exclusively under new Benchmark base class.
* Updated separate example Jupyter notebooks.
