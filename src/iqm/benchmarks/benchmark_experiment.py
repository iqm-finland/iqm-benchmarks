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
Generic Benchmark Experiment class
"""

from copy import deepcopy
from json import dump
from pathlib import Path
import pickle
from time import strftime
from typing import List, Optional, OrderedDict, Union

from iqm.benchmarks.benchmark import BenchmarkBase, BenchmarkConfigurationBase
from iqm.benchmarks.utils import get_iqm_backend
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


class BenchmarkExperiment:
    """
    A Benchmark Experiment wraps the execution of one or more benchmarks, checks their requirements are met, stores the
    execution results and plots relevant figures.
    """

    def __init__(
        self,
        backend: Union[str, IQMBackendBase],
        benchmark_configurations: List[BenchmarkConfigurationBase],
        device_id: Optional[str] = None,
    ):
        """Construct the BenchmarkExperiment class.

        Args:
            backend (str | IQMBackendBase): the backend to execute the benchmarks on
            benchmark_configurations (List[BenchmarkConfigurationBase]): the configuration(s) of the benchmark(s)
            device_id (Optional[str], optional): the identifier of the device. Defaults to None.

        Raises:
            ValueError: backend not supported. Try 'garnet' or 'iqmfakeadonis'
        """
        self.timestamp = strftime("%Y%m%d-%H%M%S")

        self.device_id = device_id if device_id is not None else backend

        if isinstance(backend, str):
            self.backend = get_iqm_backend(backend)

        else:
            assert isinstance(backend, IQMBackendBase)
            self.backend = backend

        benchmarks: OrderedDict[str, BenchmarkBase] = OrderedDict(
            (config.benchmark.name(), config.benchmark(self.backend, config)) for config in benchmark_configurations
        )

        for benchmark in benchmarks.values():
            benchmarks_copy = deepcopy(benchmarks)
            benchmarks_copy = benchmark.check_requirements(benchmarks_copy)
        self.benchmarks = benchmarks_copy

    def run_experiment(self) -> None:
        """Run the Benchmark experiment, and store the configuration, raw data, results and figures."""

        for name, benchmark in self.benchmarks.items():
            print("\nNow executing " + name)
            # Create the directory for results
            results_dir = f"Outputs/{self.device_id}/{self.timestamp}/{name}/"
            Path(results_dir).mkdir(parents=True, exist_ok=True)

            # Execute the current benchmark
            benchmark.generate_requirements(self.benchmarks)
            benchmark.execute_full_benchmark()

            # Create configuration JSON file
            with open(
                f"{results_dir}{self.device_id}_{self.timestamp}_{name}_configuration.json",
                "w",
                encoding="utf-8",
            ) as f_configuration:
                dump(benchmark.serializable_configuration.model_dump(), f_configuration)

            # Create untranspiled circuit files
            if benchmark.untranspiled_circuits:
                for key_qubits in benchmark.untranspiled_circuits.keys():
                    for key_depth in benchmark.untranspiled_circuits[key_qubits].keys():
                        with open(
                            f"{results_dir}{self.device_id}_{self.timestamp}_{name}_qubits_{key_qubits}_depth_{key_depth}_untranspiled.pkl",
                            "wb",
                        ) as f_circuits:
                            pickle.dump(
                                benchmark.untranspiled_circuits[key_qubits][key_depth],
                                f_circuits,
                            )

            # Create transpiled circuit files
            for key_qubits in benchmark.transpiled_circuits.keys():
                for key_depth in benchmark.transpiled_circuits[key_qubits].keys():
                    with open(
                        f"{results_dir}{self.device_id}_{self.timestamp}_{name}_qubits_{key_qubits}_depth_{key_depth}_transpiled.pkl",
                        "wb",
                    ) as f_circuits:
                        pickle.dump(
                            benchmark.transpiled_circuits[key_qubits][key_depth],
                            f_circuits,
                        )

            # Create raw result pickle files
            with open(
                f"{results_dir}{self.device_id}_{self.timestamp}_{name}_raw_results.pkl",
                "wb",
            ) as f_raw_results:
                pickle.dump(
                    benchmark.raw_results,
                    f_raw_results,
                )

            # Create raw data JSON file
            with open(
                f"{results_dir}{self.device_id}_{self.timestamp}_{name}_raw_data.json",
                "w",
                encoding="utf-8",
            ) as f_raw_data:
                dump(benchmark.raw_data, f_raw_data)

            # Create job metadata JSON file
            with open(
                f"{results_dir}{self.device_id}_{self.timestamp}_{name}_job_metadata.json",
                "w",
                encoding="utf-8",
            ) as f_job_metadata:
                dump(benchmark.job_meta, f_job_metadata)

            # Create results JSON file
            with open(
                f"{results_dir}{self.device_id}_{self.timestamp}_{name}_results.json",
                "w",
                encoding="utf-8",
            ) as f_results:
                dump(benchmark.results, f_results)

            # Create figures
            Path(f"{results_dir}figures/").mkdir(parents=True, exist_ok=True)
            for fig_name, fig in benchmark.figures.items():
                fig.savefig(
                    f"{results_dir}figures/{self.device_id}_{self.timestamp}_{name}_{fig_name}",
                    dpi=250,
                    bbox_inches="tight",
                )

            # Save benchmark
            self.benchmarks[benchmark.name()] = benchmark
