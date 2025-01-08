=========
Changelog
=========

Version 2.6
===========
* Fixed bugs including wrong GHZ plot x-Axis labels and incorrect transpiled and untranspiled circuit storage for mGST.
* Added note about optional dependency "mgst".
* Improved display and calculation method for Hamiltonian parameter output of rank 1 compressive GST.

Version 2.5
===========
* Changed simulation method for MRB to 'stabilizer' and simulation circuits are compiled in circuit generation stage.

Version 2.4
===========
* Changed Qscore to operate under the new base class.

Version 2.3
===========
* Reverted QV simulation circuits to untranspiled ones (fixes bug giving all HOPs equal to zero).

Version 2.2
===========
* Added Clifford RB example notebook to docs. `#20 <https://github.com/iqm-finland/iqm-benchmarks/pull/20>`_

Version 2.1
===========
* Fixed bug in RB plots for individual decays.

Version 2.0
===========
* Adds `Circuits`, `BenchmarkCircuit` and `CircuitGroup` as a way to easily store and interact with multiple quantum circuits.
* `BenchmarkRunResult` now takes a `circuits` argument, expecting an instance of `Circuits`. `QuantumCircuit` instances can now exist there instead of inside xarray Datasets. All analysis methods should also expect to use an instance of `BenchmarkRunResult`.
* Ported all of the benchmarks subclassing from `Benchmark` to use the new containers.
* Updates the usage of `qiskit.QuantumCircuit` to `iqm.qiskit_iqm.IQMCircuit` in many places.

Version 1.12
===========
* Miscellaneous small bugs fixed.

Version 1.11
===========
* Relaxes dependencies to allow for ranges.

Version 1.10
===========
* Added API docs building and publishing.

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
