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
Generic Benchmark class
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Literal, Optional, OrderedDict, Type

from matplotlib.figure import Figure
from pydantic import BaseModel

from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


class BenchmarkBase(ABC):
    """
    The base implementation of all benchmarks, from which they inherit.
    """

    def __init__(
        self,
        backend: IQMBackendBase,
        configuration: "BenchmarkConfigurationBase",
    ):
        """Construct the BenchmarkBase class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (BenchmarkConfigurationBase): the configuration of the benchmark
        """
        self.backend = backend
        self.configuration = configuration
        self.serializable_configuration = deepcopy(self.configuration)
        self.serializable_configuration.benchmark = self.name()

        self.shots = self.configuration.shots
        self.calset_id = self.configuration.calset_id
        self.max_gates_per_batch = self.configuration.max_gates_per_batch

        self.routing_method = self.configuration.routing_method
        self.physical_layout = self.configuration.physical_layout

        self.raw_data: Dict = {}
        self.job_meta: Dict = {}
        self.results: Dict = {}
        self.raw_results: Dict = {}

        self.untranspiled_circuits: Dict[str, Dict[int, list]] = {}
        self.transpiled_circuits: Dict[str, Dict[int, list]] = {}

        self.figures: Dict[str, Figure] = {}

    @classmethod
    @abstractmethod
    def name(cls):
        """Return the name of the benchmark."""

    @abstractmethod
    def execute_full_benchmark(self):
        """Execute the full benchmark on the given backend."""

    @staticmethod
    def check_requirements(all_benchmarks: OrderedDict[str, "BenchmarkBase"]) -> OrderedDict[str, "BenchmarkBase"]:
        """Check whether the requirements for the benchmark are met, returning a valid benchmark dictionary."""
        return all_benchmarks

    def generate_requirements(self, all_benchmarks: OrderedDict[str, "BenchmarkBase"]) -> None:
        """Generate the required attributes for execution."""


class BenchmarkConfigurationBase(BaseModel):
    """Benchmark configuration base.

    Attributes:
        benchmark (Type[BenchmarkBase]): the benchmark configuration.
        shots (int): the number of shots to use in circuit execution.
                * Default for all benchmarks is 2**8.
        max_gates_per_batch (Optional[int]): the maximum number of gates per circuit batch.
                * Default for all benchmarks is None.
        calset_id (Optional[str]): the calibration ID to use in circuit execution.
                * Default for all benchmarks is None (uses last available calibration ID).
        routing_method (Literal["basic", "lookahead", "stochastic", "sabre", "none"]): the Qiskit routing method to use in transpilation.
                * Default for all benchmarks is "sabre".
        physical_layout (Literal["fixed", "batching"]): whether physical layout is constrained during transpilation to selected physical qubits.
                - "fixed": physical layout is constrained during transpilation to the selected initial physical qubits.
                - "batching": physical layout is allowed to use any other physical qubits, and circuits are batched according to final measured qubits.
                * Default for all benchmarks is "fixed".
    """

    benchmark: Type[BenchmarkBase]
    shots: int = 2**8
    max_gates_per_batch: Optional[int] = None
    calset_id: Optional[str] = None
    routing_method: Literal["basic", "lookahead", "stochastic", "sabre", "none"] = "sabre"
    physical_layout: Literal["fixed", "batching"] = "fixed"
