=========
Changelog
=========

Version 2.32
============
* Fixed incorrect edge width assignment in qubit selection plot.

Version 2.31
============
* Fixed a bug in the bootstrapping functionality of the GST benchmark and updated the respective Jupyter tutorial.

Version 2.30
============
* Updated CLOPS value and plot reporting, making explicit offline values, such as time spent in transpilation and in parameter assigning.

Version 2.29
============
* Fixed header levels in `example_graphstate` notebook for correct pages rendering.

Version 2.28
============
* Added graph state (bipartite entanglement negativity) benchmark.

Version 2.27
============
* Qiskit on IQM dependency updated to > 17.0.

Version 2.26
============
* Changed benchmark observation names and identifiers to be more consistent with guidelines.

Version 2.25
============
* Added optional configuration parameter (`max_circuits_per_batch`) to specify the maximum amount of circuits per batch.

Version 2.24
============
* Added rustworkx dependency range to fix wrong edge thickness assignment in qubit selection plot.

Version 2.23
============
* Added dynamical decoupling parameter option to configurations of all benchmarks.
* Added visual aid plot for qubit selection (see, e.g., GHZ example notebook).
* Included option to run GST in parallel if the specified qubits don't overlap
* Small runtime improvements in the GST benchmark.
* Changed tensor order in GST outputs from Qiskit (bottom to top) to standard (top to bottom) order.

Version 2.22
============
* Fix for QScore errors when custom_qubits_array is specified.

Version 2.21
============
* Function to bootstrap counts added to utils file.

Version 2.20
============
* Standardizes observations for CLOPS and Mirror RB.

Version 2.19
============
* All functional tests extended to the fake Deneb backend.
* Added backend transpilation to REM calibration circuits to fix errors with REM on fake Deneb.

Version 2.18
============
* Added notebook to benchmark IQM Star QPUs and bug fixes done for Qscore.

Version 2.17
============
* Update installation command for development mode. `#41 <https://github.com/iqm-finland/iqm-benchmarks/pull/41>`_

Version 2.16
============
* Added readout error mitigation for Qscore benchmark.

Version 2.15
============
* Added optimal GHZ circuit generation and corresponding example notebook for all-to-all connected QPU topology.

Version 2.14
============
* Added devices folder in docs with notebook to benchmark IQM Spark.

Version 2.13
============
* Move all example notebooks to docs. `#30 <https://github.com/iqm-finland/iqm-benchmarks/pull/30>`_

Version 2.12
============
* Added compatibility with IQM-Deneb by adapting the transpilation behavior in several benchmarks.

Version 2.11
============
* Report average native single-qubit gate fidelity estimates in observations of 1Q Clifford RB and 1Q IRB, and display in plots of 1Q Clifford RB.

Version 2.10
============
* Fix docs publishing by CI.

Version 2.9
===========
* Add optional security-scanned lockfile.

Version 2.8
===========
* Fixed a bug where optional dependencies related to gst were imported with other benchmarks, leading to a ModuleNotFoundError.

Version 2.7
===========
* Fixed bugs in Qscore and enabled benchmark execution for pyrite.

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
============
* Miscellaneous small bugs fixed.

Version 1.11
============
* Relaxes dependencies to allow for ranges.

Version 1.10
============
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
