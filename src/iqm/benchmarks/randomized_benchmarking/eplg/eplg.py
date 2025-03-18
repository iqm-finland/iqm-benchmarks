"""
Error Per Layered Gate (EPLG).
"""

from time import strftime
from typing import Optional, Sequence, Tuple, Type

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
from uncertainties import ufloat
import xarray as xr

from iqm.benchmarks import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkCircuit,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.randomized_benchmarking.direct_rb.direct_rb import (
    DirectRandomizedBenchmarking,
    DirectRBConfiguration,
    direct_rb_analysis,
)
from iqm.benchmarks.utils import GraphPositions, evaluate_hamiltonian_paths, rx_to_nx_graph
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


# Move to plot utils file
def draw_linear_chain_graph(
    backend: IQMBackendBase,
    edge_list: Sequence[Tuple[int, int]],
    disjoint_layers: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    timestamp: Optional[str] = None,
    station: Optional[str] = None,
) -> Tuple[str, Figure]:
    """Draw a linear chain graph on the given backend.

    Args:
        backend (IQMBackendBase): The backend to draw the graph on.
        edge_list (Sequence[Tuple[int, int]]): The edge list of the linear chain.
        disjoint_layers (Optional[Sequence[Sequence[Tuple[int, int]]]): The edges defining disjoint layers to draw.
        timestamp (Optional[str]): The timestamp to include in the figure name.
        station (Optional[str]): The name of the station.

    Returns:
         Tuple[str, Figure]: The figure name and the figure object.
    """
    disjoint = "_disjoint" if disjoint_layers is not None else ""
    fig_name = (
        f"linear_chain_graph{disjoint}_{station}_{timestamp}"
        if timestamp is not None
        else f"linear_chain_graph{disjoint}_{station}"
    )

    fig = plt.figure()
    ax = plt.axes()

    if station is not None:
        if station.lower() in GraphPositions.predefined_stations:
            qubit_positions = GraphPositions.predefined_stations[station.lower()]
        else:
            graph_backend = backend.coupling_map.graph.to_undirected(multigraph=False)
            qubit_positions = GraphPositions.create_positions(graph_backend)
    else:
        graph_backend = backend.coupling_map.graph.to_undirected(multigraph=False)
        if backend.num_qubits in (20, 49):
            station = "garnet" if backend.num_qubits == 20 else "emerald"
            qubit_positions = GraphPositions.predefined_stations[station]
        else:
            qubit_positions = GraphPositions.create_positions(graph_backend, station)

    if disjoint_layers is None:
        nx.draw_networkx(
            rx_to_nx_graph(backend),
            pos=qubit_positions,
            edgelist=edge_list,
            width=4.0,
            edge_color="k",
            node_color="k",
            font_color="w",
            ax=ax,
        )

    else:
        num_disjoint_layers = len(disjoint_layers)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_disjoint_layers))
        all_edge_colors = [[colors[i]] * len(l) for i, l in enumerate(disjoint_layers)]  # Flatten below
        nx.draw_networkx(
            rx_to_nx_graph(backend),
            pos=qubit_positions,
            edgelist=[x for y in disjoint_layers for x in y],
            width=4.0,
            edge_color=[x for y in all_edge_colors for x in y],
            node_color="k",
            font_color="w",
        )

    return fig_name, fig


def eplg_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """EPLG analysis function

    Args:
        run (BenchmarkRunResult): The result of the benchmark run.

    Returns:
        AnalysisResult corresponding to DRB.
    """

    result_direct_rb = direct_rb_analysis(run)

    dataset = result_direct_rb.dataset.copy(deep=True)
    observations = result_direct_rb.observations
    plots = {}

    num_edges = len(observations)
    num_qubits = dataset.attrs["chain_length"]

    total_mean = []
    fid_product = [1, 0]
    for obs in observations:
        fid_product[0] *= obs.value
        fid_product[1] += obs.uncertainty
        total_mean.append(ufloat(obs.value, obs.uncertainty))

    LF = ufloat(fid_product[0], fid_product[1])
    observations.append(
        BenchmarkObservation(
            name="Layer Fidelity",
            identifier=BenchmarkObservationIdentifier(f"(n_qubits={num_qubits})"),
            value=LF.nominal_value,
            uncertainty=LF.std_dev,
        )
    )

    observations.append(
        BenchmarkObservation(
            name="EPLG",
            identifier=BenchmarkObservationIdentifier(f"(n_qubits={num_qubits})"),
            value=(1 - LF ** (1 / num_edges)).nominal_value,
            uncertainty=LF.std_dev,
        )
    )

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations, plots=plots)


class EPLGBenchmark(Benchmark):
    """EPLG estimates the layer fidelity of native 2Q gate layers"""

    analysis_function = staticmethod(eplg_analysis)

    name: str = "EPLG"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "EPLGConfiguration"):
        """Construct the EPLG class

        Args:
            backend_arg (IQMBackendBase | str): _description_
            configuration (MirrorRBConfiguration): _description_
        """
        super().__init__(backend_arg, configuration)
        # EXPERIMENT
        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name
        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

        self.drb_depths = configuration.drb_depths
        self.drb_circuit_samples = configuration.drb_circuit_samples

        self.custom_qubits_array = configuration.custom_qubits_array

        self.chain_length = configuration.chain_length
        self.chain_path_samples = configuration.chain_path_samples
        self.num_disjoint_layers = configuration.num_disjoint_layers
        self.calibration_url = configuration.calibration_url

    def add_all_meta_to_dataset(self, dataset: xr.Dataset):
        """Adds all configuration metadata and circuits to the dataset variable

        Args:
            dataset (xr.Dataset): The xarray dataset
        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend"] = self.backend
        dataset.attrs["backend_configuration_name"] = self.backend_configuration_name
        dataset.attrs["backend_name"] = self.backend.name

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        # Defined outside configuration - if any

    def validate_custom_qubits_array(self):
        """Validates that the custom qubits array input forms a linear chain (Hamiltonian path).

        Raises:
        """
        if self.custom_qubits_array is not None:
            # TODO: Implement custom qubits array - have to validate that the qubits constitute a linear chain (Hamiltonian path).
            raise NotImplementedError

    def validate_random_chain_inputs(self):
        """Validates inputs for chain sampling.

        Raises:
        """
        # Check chain length
        if self.chain_length is None:
            self.chain_length = self.backend.num_qubits
        elif self.chain_length > self.backend.num_qubits:
            raise ValueError("The chain length cannot exceed the number of qubits in the backend.")

        # Check path samples
        if self.chain_path_samples is None:
            self.chain_path_samples = 20
        elif self.chain_path_samples < 1:
            raise ValueError("The number of chain path samples must be a positive integer.")

        # Check calibration URL - this is a temporary solution, normally the backend should be enough to specify this
        if self.calibration_url is None:
            raise ValueError("The calibration URL must be specified if custom qubits array is not specified.")

        if self.num_disjoint_layers is None:
            self.num_disjoint_layers = 2
        elif self.num_disjoint_layers < 1:
            raise ValueError("The number of disjoint layers must be a positive integer.")

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Execute the EPLG Benchmark"""

        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")

        dataset = xr.Dataset()
        dataset_eplg = xr.Dataset()

        self.add_all_meta_to_dataset(dataset_eplg)

        if self.custom_qubits_array:
            self.validate_custom_qubits_array()
        else:
            self.validate_random_chain_inputs()

            qcvv_logger.info("Generating linear chain path")
            h_path_costs = evaluate_hamiltonian_paths(
                self.chain_length, self.chain_path_samples, self.backend, self.calibration_url
            )
            qcvv_logger.info("Extracting the path that maximizes total 2Q calibration fidelity")
            max_cost_path = h_path_costs[max(h_path_costs.keys())]

            all_disjoint = [max_cost_path[i :: self.num_disjoint_layers] for i in range(self.num_disjoint_layers)]

            # Execute parallel DRB in all disjoint layers
            drb_config = DirectRBConfiguration(
                qubits_array=all_disjoint,
                is_eplg=True,
                depths=self.drb_depths,
                num_circuit_samples=self.drb_circuit_samples,
                shots=self.shots,
                max_gates_per_batch=self.max_gates_per_batch,
            )

            benchmarks_direct_rb = DirectRandomizedBenchmarking(backend, drb_config)

            run_direct_rb = benchmarks_direct_rb.run()

            dataset = run_direct_rb.dataset

            dataset.attrs.update(dataset_eplg.attrs)

        return dataset


class EPLGConfiguration(BenchmarkConfigurationBase):
    """EPLG Configuration

    Attributes:
        drb_depths (Sequence[int]): The layer depths to consider for the parallel DRB.
        drb_circuit_samples (int): The number of circuit samples to consider for the parallel DRB.
        custom_qubits_array (Optional[Sequence[int]]): The custom qubits array to consider.
                * If not specified, will proceed to generate linear chains at random, selecting the one with the highest total 2Q gate fidelity.
                * Default is None.
        chain_length (Optional[int]): The length of the linear chain of 2Q gates to consider, corresponding to the number of qubits, if custom_qubits_array not specified.
                * Default is None: assigns the number of qubits in the backend, lowering by 1 each time a chain is not successfuly found.
        chain_path_samples (int): The number of chain path samples to consider, if custom_qubits_array not specified.
                * Default is None: assigns 20 path samples (arbitrary).
        calibration_url (Optional[str]): The URL of the IQM station to retrieve calibration data from.
                * It must be specified if custom_qubits_array is not specified.
                * Default is None - raises an error if custom_qubits_array is not specified.
        num_disjoint_layers (Optional[int]): The number of disjoint layers to consider.
                * Default is None: assigns 2 disjoint layers (arbitrary).

    """

    benchmark: Type[Benchmark] = EPLGBenchmark
    drb_depths: Sequence[int]
    drb_circuit_samples: int
    custom_qubits_array: Optional[Sequence[int]] = None
    chain_length: Optional[int] = None
    chain_path_samples: Optional[int] = None
    num_disjoint_layers: Optional[int] = None
    calibration_url: Optional[str] = None
