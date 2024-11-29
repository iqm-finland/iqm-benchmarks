=========
Changelog
=========

Version 1.4
===========

* Adds Observation class, and modifies AnalysisResult so observations now accepts a list[Observation].
* Adds ObservationIdentifier class.
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
