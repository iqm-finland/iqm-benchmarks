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
IQM's Python Library Benchmarking Suite QCVV.
"""

from importlib.metadata import PackageNotFoundError, version

from .benchmark_definition import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
from .circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from .compressive_gst.compressive_gst import CompressiveGST, GSTConfiguration
from .entanglement.ghz import GHZBenchmark, GHZConfiguration
from .quantum_volume.clops import CLOPSBenchmark, CLOPSConfiguration
from .quantum_volume.quantum_volume import QuantumVolumeBenchmark, QuantumVolumeConfiguration
from .randomized_benchmarking.clifford_rb.clifford_rb import CliffordRandomizedBenchmarking, CliffordRBConfiguration
from .randomized_benchmarking.interleaved_rb.interleaved_rb import (
    InterleavedRandomizedBenchmarking,
    InterleavedRBConfiguration,
)
from .randomized_benchmarking.mirror_rb.mirror_rb import MirrorRandomizedBenchmarking, MirrorRBConfiguration


AVAILABLE_BENCHMARKS = {
    GHZBenchmark.name: GHZBenchmark,
    CLOPSBenchmark.name: CLOPSBenchmark,
    QuantumVolumeBenchmark.name: QuantumVolumeBenchmark,
    CliffordRandomizedBenchmarking.name: CliffordRandomizedBenchmarking,
    InterleavedRandomizedBenchmarking.name: InterleavedRandomizedBenchmarking,
    MirrorRandomizedBenchmarking.name: MirrorRandomizedBenchmarking,
}

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "iqm-benchmarks"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
