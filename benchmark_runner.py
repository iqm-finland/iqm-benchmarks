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

import argparse

from iqm.benchmarks.randomized_benchmarking.interleaved_rb.interleaved_rb import *
from iqm.benchmarks.quantum_volume.quantum_volume import *
from iqm.benchmarks.entanglement.ghz import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser("iqm.benchmark")
    parser.add_argument(
        "--backend",
        help="The type of backend (device) to be characterized",
        required=False,
        type=str,
        default="IQMFakeBackend",
    )
    args = parser.parse_args()

    if args.backend is not None and type(args.backend) is str and args.backend != "":
        backend = args.backend
    else:
        backend = "IQMFakeBackend"

    print(f"USING BACKEND: {backend}")

    from scheduled_experiments.adonis.weekly import *

    InterleavedRandomizedBenchmarking(backend, WEEKLY_IRB).run(backend)
    QuantumVolumeBenchmark(backend, WEEKLY_QV).run(backend)
    GHZBenchmark(backend, WEEKLY_GHZ).run(backend)
