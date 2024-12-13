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
This module implements the base class for defining a benchmark.
"""

from abc import ABC, abstractmethod
import copy
from copy import deepcopy
from dataclasses import dataclass, field
import functools
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import xarray as xr

from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, Circuits
from iqm.benchmarks.utils import get_iqm_backend, timeit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from iqm.qiskit_iqm.iqm_provider import IQMBackend, IQMFacadeBackend


@dataclass
class BenchmarkObservationIdentifier:
    """Identifier for observations for ease of use

    Attributes:
        qubit_indices: list containing the indices of the qubits the observation was executed on.
    """

    qubit_indices: list[int]

    @property
    def string_identifier(self) -> str:
        """String version of the qubit indices for ease of use

        Returns:
            A string of the qubit indices
        """
        return str(self.qubit_indices)


@dataclass
class BenchmarkObservation:
    """Dataclass to store the main results of a single run of a Benchmark

    Attributes:
        name: name of the observation
        value: value of the observation
        identifier: identifier, which should be a string of the qubit layout
        uncertainty: uncertainty of the observation
    """

    name: str
    value: Any
    identifier: BenchmarkObservationIdentifier
    uncertainty: Optional[Any] = None


@dataclass
class BenchmarkRunResult:
    """
    A dataclass that stores the results of a single run of a Benchmark.

    RunResult should contain enough information that the Benchmark can be analyzed based on those
    results.
    """

    dataset: xr.Dataset
    circuits: Circuits


@dataclass
class BenchmarkAnalysisResult:
    """
    A dataclass storing the results of the analysis.

    The result consists of a dataset, plots, and observations. Plots are defined as a dictionary
    that maps a plot name to a figure. Observations are key-value pairs of data that contain the
    main results of the benchmark.
    """

    dataset: xr.Dataset
    circuits: Optional[Circuits] = field(default=None)
    plots: dict[str, Figure] = field(default_factory=lambda: ({}))
    observations: list[BenchmarkObservation] = field(default_factory=lambda: [])

    def plot(self, plot_name: str):
        """
        Plots the given figure.

        Args:
            plot_name: Name of the figure to be plotted.
        """
        return self.plots[plot_name]

    def plot_all(self):
        """
        Plots all the figures defined for the analysis.
        """
        for fig in self.plots.values():
            show_figure(fig)
            plt.show()

    @classmethod
    def from_run_result(cls, run: BenchmarkRunResult):
        """
        Creates a new ``AnalysisResult`` from a ``RunResult``.

        Args:
            run: A run for which analysis result is created.
        """
        return cls(dataset=run.dataset, circuits=Circuits)


def default_analysis_function(result: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """
    The default analysis that only pass the result through.
    """
    return BenchmarkAnalysisResult.from_run_result(result)


def merge_datasets_dac(datasets: List[xr.Dataset]) -> xr.Dataset:
    """
    Merges a list of datasets recursively to minimize dataset sizes during merge

    Args:
        datasets: List[xr.Dateset]
            A list of xarray datasets
    Returns:
        A list containing a single merged dataset
    """
    if len(datasets) == 1:
        return datasets[0]
    datasets_new = []
    for i in range(0, len(datasets), 2):
        if i == (len(datasets) - 1):
            datasets_new.append(datasets[i])
        else:
            datasets_new.append(xr.merge(datasets[i : i + 2]))
    return merge_datasets_dac(datasets_new)


@timeit
def add_counts_to_dataset(counts: List[Dict[str, int]], identifier: str, dataset: xr.Dataset):
    """Adds the counts from a cortex job result to the given dataset.
    If counts with the same identifier are already present in the old dataset, then both counts are added together.

    Args:
        counts (List[Dict[str, int]]): A list of dictionaries with counts of bitstrings.
        identifier (str): A string to identify the current data, for instance the qubit layout.
        dataset (xr.Dataset): Dataset to add results to.
    Returns:
        dataset_merged: xarray.Dataset
            A merged dataset where the new counts are added the input dataset
    """
    if not isinstance(counts, list):
        counts = [counts]
    datasets = []
    for ii, _ in enumerate(counts):
        ds_temp = xr.Dataset()
        counts_dict = dict(counts[ii])
        counts_array = xr.DataArray(list(counts_dict.values()), {f"{identifier}_state_{ii}": list(counts_dict.keys())})
        if f"{identifier}_counts_{ii}" in dataset.variables:
            counts_array = counts_array + dataset[f"{identifier}_counts_{ii}"]
        counts_array = counts_array.sortby(counts_array[f"{identifier}_state_{ii}"])
        ds_temp.update({f"{identifier}_counts_{ii}": counts_array})
        datasets.append(ds_temp)
    dataset_new = merge_datasets_dac(datasets)
    dataset_merged = dataset.merge(dataset_new, compat="override")
    return dataset_merged


def show_figure(fig):
    """
    Shows a closed figure.

    Args:
        fig: Figure to show.
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


class Benchmark(ABC):
    """
    A base class for running cortex-based Benchmark experiments.

    In order to write a new benchmark, it is recommended to derive a class
    from this baseclass. A new benchmark is defined by deriving from this base
    class and implementing the execute method. Additionally, a custom analysis
    for the benchmark can be implemented by giving the pointer to the analysis
    method in ``analysis_function`` field. The given analysis_function should
    accept ``AnalysisResult`` as its input and return the final result.
    """

    analysis_function: Callable[[BenchmarkRunResult], BenchmarkAnalysisResult] = staticmethod(default_analysis_function)
    default_options: dict[str, Any] | None = None
    options: dict[str, Any] | None = None

    def __init__(self, backend: Union[str, IQMBackendBase], configuration: BenchmarkConfigurationBase, **kwargs):

        # Ported from BenchmarkBase   # CHECK
        self.configuration = configuration
        self.serializable_configuration = deepcopy(self.configuration)
        self.serializable_configuration.benchmark = self.name

        self.circuits = Circuits()

        if isinstance(backend, str):
            self.backend = get_iqm_backend(backend)
        else:
            assert isinstance(backend, IQMBackendBase)
            self.backend = backend

        self.shots = self.configuration.shots
        self.calset_id = self.configuration.calset_id
        self.max_gates_per_batch = self.configuration.max_gates_per_batch

        self.routing_method = self.configuration.routing_method
        self.physical_layout = self.configuration.physical_layout

        self.transpiled_circuits: BenchmarkCircuit
        self.untranspiled_circuits: BenchmarkCircuit

        # From exa_support MR
        self.options = copy.copy(self.default_options) if self.default_options else {}
        self.options.update(kwargs)
        self.runs: list[BenchmarkRunResult] = []

    @classmethod
    @abstractmethod
    def name(cls):
        """Return the name of the benchmark."""

    @abstractmethod
    def execute(self, backend: IQMBackend | IQMFacadeBackend | str) -> xr.Dataset:
        """
        Executes the benchmark.

        This method should be overridden by deriving classes.

        Args:
            backend: Qiskit backend used to execute benchmarks.

        Returns:
            An xarray dataset which contains to results of the benchmark execution.
            The dataset should contain all the information necessary for analysing
            the benchmark results.
        """

    def run(self, calibration_set_id: str | uuid.UUID | None = None) -> BenchmarkRunResult:
        """
        Runs the benchmark using the given backend and calibration_set_id.

        Args:
            calibration_set_id:
                CalibrationSetId used to initialize the backend or None for the latest calibration set.

        Returns:
            RunResult: The result of the benchmark run.
        """
        backend_for_execute = copy.copy(self.backend)
        backend_for_execute.run = functools.partial(
            self.backend.run, calibration_set_id=calibration_set_id
        )  # type: ignore
        dataset = self.execute(backend_for_execute)
        run = BenchmarkRunResult(dataset, self.circuits)
        self.runs.append(run)
        return run

    def analyze(self, run_index=-1) -> BenchmarkAnalysisResult:
        """
        The default analysis for the benchmark.

        Internally uses the function defined by the attribute ``analysis_function``
        to perform the analysis. This function makes a shallow copy of the dataset produced by
        run. Therefore, it is recommended to not make changes to the data of the dataset but
        just the structure of the array.

        Args:
            run_index: Index for the run to analyze.

        Returns:
            An analysis result constructed from the run and updated by the analysis method defined by
            the ``analysis_function`` field.
        """
        run = self.runs[run_index]
        updated_result = self.analysis_function(run)
        return updated_result
