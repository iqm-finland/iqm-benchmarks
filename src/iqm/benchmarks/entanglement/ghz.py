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
GHZ state benchmark
"""

from io import BytesIO
from itertools import chain
import json
from time import strftime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import networkx
from networkx import Graph, all_pairs_shortest_path, is_connected, minimum_spanning_tree
import numpy as np
import pycurl
from qiskit import QuantumRegister, transpile
from qiskit.quantum_info import random_clifford
from qiskit.transpiler import CouplingMap
from qiskit_aer import Aer
from scipy.spatial.distance import hamming
import xarray as xr

from iqm.benchmarks.benchmark import BenchmarkConfigurationBase
from iqm.benchmarks.benchmark_definition import (
    Benchmark,
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
    add_counts_to_dataset,
)
from iqm.benchmarks.circuit_containers import BenchmarkCircuit, CircuitGroup, Circuits
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.readout_mitigation import apply_readout_error_mitigation
from iqm.benchmarks.utils import (
    perform_backend_transpilation,
    reduce_to_active_qubits,
    retrieve_all_counts,
    set_coupling_map,
    submit_execute,
    timeit,
    xrvariable_to_counts,
)
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


@timeit
def append_rms(
    circuit: QuantumCircuit,
    num_rms: int,
    backend: IQMBackendBase,
    # optimize_sqg: bool = False,
) -> List[QuantumCircuit]:
    """
    Appends 1Q Clifford gates sampled uniformly at random to all qubits in the given circuit.
    Args:
        circuit (QuantumCircuit):
        num_rms (int):
        backend (Optional[IQMBackendBase]): whether Cliffords are decomposed for the given backend
    Returns:
        List[QuantumCircuit] of the original circuit with 1Q Clifford gates appended to it
    """
    rm_circuits: list[QuantumCircuit] = []
    for _ in range(num_rms):
        rm_circ = circuit.copy()
        # It shouldn't matter if measurement bits get scrambled
        rm_circ.remove_final_measurements()

        active_qubits = set()
        data = rm_circ.data
        for instruction in data:
            for qubit in instruction[1]:
                active_qubits.add(rm_circ.find_bit(qubit)[0])

        for q in active_qubits:
            if backend is not None:
                rand_clifford = random_clifford(1).to_circuit()
            else:
                rand_clifford = random_clifford(1).to_instruction()
            rm_circ.compose(rand_clifford, qubits=[q], inplace=True)

        rm_circ.measure_active()
        rm_circuits.append(transpile(rm_circ, basis_gates=backend.operation_names))

    return rm_circuits


def fidelity_ghz_randomized_measurements(
    dataset: xr.Dataset, qubit_layout, ideal_probabilities: List[Dict[str, int]], num_qubits: int, circuits: Circuits
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Estimates GHZ state fidelity through cross-correlations of RMs.
    Implementation of Eq. (34) in https://arxiv.org/abs/1812.02624

    Arguments:
        dataset (xr.Dataset):
        qubit_layout: List[int]: The subset of system-qubits used in the protocol
        ideal_probabilities (List[Dict[str, int]]):
        num_qubits (int): Number of qubits
        circuits (Circuits): Instance of `Circuits` containing transpiled circuits
    Returns:
        values: dict[str, Any]
            The fidelities
        uncertainties: dict[str, Any]
            The uncertainties for the fidelities
    """
    idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    # List for each RM contribution to the fidelity
    fid_rm = []

    # Loop through RMs and add each contribution
    num_rms = len(circuits["transpiled_circuits"][f"{idx}_native_ghz"].circuits)
    for u in range(num_rms):
        # Probability estimates for noisy measurements
        probabilities_sample = {}
        c_keys = dataset[f"{idx}_state_{u}"].data  # measurements[u].keys()
        num_shots_noisy = sum(dataset[f"{idx}_counts_{u}"].data)
        for k, key in enumerate(c_keys):
            probabilities_sample[key] = dataset[f"{idx}_counts_{u}"].data[k] / num_shots_noisy
        # Keys for corresponding ideal probabilities
        c_id_keys = ideal_probabilities[u].keys()

        p_sum = []
        for sb in c_id_keys:
            for sa in c_keys:
                exponent = hamming(list(sa), list(sb)) * num_qubits
                p_sum.append(np.power(-2, -exponent) * probabilities_sample[sa] * ideal_probabilities[u][sb])
        fid_rm.append((2**num_qubits) * sum(p_sum))
    values = {"fidelity": np.mean(fid_rm)}
    uncertainties = {"fidelity": np.std(fid_rm) / np.sqrt(num_rms)}

    if dataset.attrs["rem"]:
        fid_rm_rem = []
        for u in range(num_rms):
            # Probability estimates for noisy measurements
            probabilities_sample = {}
            c_keys = dataset[f"{idx}_rem_state_{u}"].data  # measurements[u].keys()
            num_shots_noisy = sum(dataset[f"{idx}_rem_counts_{u}"].data)
            for k, key in enumerate(c_keys):
                probabilities_sample[key] = dataset[f"{idx}_rem_counts_{u}"].data[k] / num_shots_noisy
            # Keys for corresponding ideal probabilities
            c_id_keys = ideal_probabilities[u].keys()

            p_sum = []
            for sb in c_id_keys:
                for sa in c_keys:
                    exponent = hamming(list(sa), list(sb)) * num_qubits
                    p_sum.append(np.power(-2, -exponent) * probabilities_sample[sa] * ideal_probabilities[u][sb])
            fid_rm_rem.append((2**num_qubits) * sum(p_sum))
        values = values | {"fidelity_rem": np.mean(fid_rm_rem)}
        uncertainties = uncertainties | {"fidelity_rem": np.std(fid_rm_rem) / np.sqrt(num_rms)}
    return values, uncertainties


def fidelity_ghz_coherences(dataset: xr.Dataset, qubit_layout: List[int], circuits: Circuits) -> list[Any]:
    """
    Estimates the GHZ state fidelity based on the multiple quantum coherences method based on [Mooney, 2021]

    Args:
        dataset: xr.Dataset
            An xarray dataset containing the measurement data
        qubit_layout: List[int]
            The subset of system-qubits used in the protocol
        circuits: Circuits
            Instance of `Circuits` containing transpiled circuits

    Returns:
        dict[str, dict[str, Any]]: The ghz fidelity or, if rem=True, fidelity and readout error mitigated fidelity
    """

    num_qubits = len(qubit_layout)
    phases = [np.pi * i / (num_qubits + 1) for i in range(2 * num_qubits + 2)]
    idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    transpiled_circuits = circuits["transpiled_circuits"]
    num_shots = dataset.attrs["shots"]
    num_circuits = len(transpiled_circuits[f"{qubit_layout}_native_ghz"].circuits)

    # Computing the phase acquired by the |11...1> component for each interval
    complex_coefficients = np.exp(1j * num_qubits * np.array(phases))

    # Loading the counts from the dataset
    counts = xrvariable_to_counts(dataset, f"{idx}", num_circuits)
    all_zero_probability_list = []  # An ordered list for storing the probabilities of returning to the |00..0> state
    for count in counts[1:]:
        if "0" * num_qubits in count.keys():
            probability = count["0" * num_qubits] / num_shots
        else:
            probability = 0
        all_zero_probability_list.append(probability)

    # Extracting coherence parameter i_n using the fourier transform
    i_n = np.abs(np.dot(complex_coefficients, np.array(all_zero_probability_list))) / (len(phases))

    # Extracting the probabilities of the 00...0 and 11...1 bit strings
    probs_direct = {label: count / num_shots for label, count in counts[0].items()}

    # Computing GHZ state fidelity from i_n and the probabilities according to the method in [Mooney, 2021]
    p0 = probs_direct["0" * num_qubits] if "0" * num_qubits in probs_direct.keys() else 0
    p1 = probs_direct["1" * num_qubits] if "1" * num_qubits in probs_direct.keys() else 0
    fidelity = (p0 + p1) / 2 + np.sqrt(i_n)

    # Same procedure for error mitigated data
    if dataset.attrs["rem"]:
        probs_mit = xrvariable_to_counts(dataset, f"{idx}_rem", num_circuits)
        all_zero_probability_list_mit = []
        for prob in probs_mit[1:]:
            if "0" * num_qubits in prob.keys():
                probability = prob["0" * num_qubits]
            else:
                probability = 0
            all_zero_probability_list_mit.append(probability)
        i_n_mit = np.abs(np.dot(complex_coefficients, np.array(all_zero_probability_list_mit))) / (len(phases))
        probs_direct_mit = dict(probs_mit[0].items())
        p0_mit = probs_direct_mit["0" * num_qubits] if "0" * num_qubits in probs_direct_mit.keys() else 0
        p1_mit = probs_direct_mit["1" * num_qubits] if "1" * num_qubits in probs_direct_mit.keys() else 0
        fidelity_mit = (p0_mit + p1_mit) / 2 + np.sqrt(i_n_mit)
        return [fidelity, fidelity_mit]
    return [fidelity]


def fidelity_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analyze counts and compute the state fidelity

    Args:
        run: RunResult
            The RunResult object containing a dataset with counts and benchmark parameters

    Returns:
        AnalysisResult
            An object containing the dataset, plots, and observations
    """
    dataset = run.dataset
    routine = dataset.attrs["fidelity_routine"]
    qubit_layouts = dataset.attrs["custom_qubits_array"]
    backend_name = dataset.attrs["backend_name"]

    observation_list: list[BenchmarkObservation] = []
    for qubit_layout in qubit_layouts:
        match routine:
            case "randomized_measurements":
                ideal_simulator = Aer.get_backend("statevector_simulator")
                ideal_probabilities = []
                idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
                all_circuits = run.circuits["transpiled_circuits"][f"{idx}_native_ghz"].circuits
                for qc in all_circuits:
                    qc_copy = qc.copy()
                    qc_copy.remove_final_measurements()
                    deflated_qc = reduce_to_active_qubits(qc_copy, backend_name)
                    ideal_probabilities.append(
                        dict(sorted(ideal_simulator.run(deflated_qc).result().get_counts().items()))
                    )
                values, uncertainties = fidelity_ghz_randomized_measurements(
                    dataset, qubit_layout, ideal_probabilities, len(qubit_layout), run.circuits
                )
                observation_list.extend(
                    [
                        BenchmarkObservation(
                            name=key,
                            identifier=BenchmarkObservationIdentifier(qubit_layout),
                            value=value,
                            uncertainty=uncertainties[key],
                        )
                        for key, value in values.items()
                    ]
                )
            case "coherences":
                fidelity = fidelity_ghz_coherences(dataset, qubit_layout, run.circuits)
                observation_list.extend(
                    [
                        BenchmarkObservation(
                            name="fidelity", identifier=BenchmarkObservationIdentifier(qubit_layout), value=fidelity[0]
                        )
                    ]
                )
                if len(fidelity) > 1:
                    observation_list.append(
                        BenchmarkObservation(
                            name="fidelity_rem",
                            identifier=BenchmarkObservationIdentifier(qubit_layout),
                            value=fidelity[1],
                        )
                    )
    plots = {"All layout fidelities": plot_fidelities(observation_list, dataset, qubit_layouts)}
    return BenchmarkAnalysisResult(dataset=dataset, observations=observation_list, plots=plots)


def generate_ghz_linear(num_qubits: int) -> QuantumCircuit:
    """
    Generates a GHZ state by applying a Hadamard and a series of CX gates in a linear fashion.
    The construction is symmetrized to halve the circuit depth.
    Args:
        num_qubits: the number of qubits of the GHZ state

    Returns:
        A quantum circuit generating a GHZ state of n qubits
    """
    s = int(num_qubits / 2)
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_linear")
    qc.h(s)

    for m in range(s, 0, -1):
        qc.cx(m, m - 1)
        if num_qubits % 2 == 0 and m == s:
            continue
        qc.cx(num_qubits - m - 1, num_qubits - m)
    qc.measure_all()
    return qc


def generate_ghz_log_cruz(num_qubits: int) -> QuantumCircuit:
    """
    Generates a GHZ state in log-depth according to https://arxiv.org/abs/1807.05572
    Args:
        num_qubits: the number of qubits of the GHZ state

    Returns:
        A quantum circuit generating a GHZ state of n qubits
    """
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_log_Cruz")
    qc.h(0)

    for m in range(num_qubits):
        for k in range(2**m):
            if ((2**m) + k) >= num_qubits:
                break
            qc.cx(k, 2**m + k)
    qc.measure_all()
    return qc


def generate_ghz_log_mooney(num_qubits: int) -> QuantumCircuit:
    """
    Generates a GHZ state in log-depth according to https://arxiv.org/abs/2101.08946
    Args:
        num_qubits: the number of qubits of the GHZ state

    Returns:
        A quantum circuit generating a GHZ state of n qubits
    """
    quantum_register = QuantumRegister(num_qubits)
    qc = QuantumCircuit(quantum_register, name="GHZ_log_Mooney")
    qc.h(0)

    aux_n = int(np.ceil(np.log2(num_qubits)))
    for m in range(aux_n, 0, -1):
        for k in range(0, num_qubits, 2**m):
            if k + 2 ** (m - 1) >= num_qubits:
                continue
            qc.cx(k, k + 2 ** (m - 1))
    qc.measure_all()
    return qc


def generate_ghz_spanning_tree(
    graph: Graph,
    qubit_layout: List[int],
    n_state: int | None = None,
) -> Tuple[QuantumCircuit, List[int]]:
    """
    Generates a GHZ state in log-depth by computing a minimal spanning tree for a given coupling map.

    Args:
        graph: networkx.Graph
            A graph of the backend coupling map
        qubit_layout: List[int]
            The subset of system-qubits used in the protocol, indexed from 0
        n_state: int
            The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
            the number of qubits in qubit_layout

    Returns:
        qc: QuantumCircuit
            A quantum circuit generating a GHZ state of n qubits
        participating_qubits: List[int]
            The list of qubits on which the GHZ state is defined. This is a subset of qubit_layout with size n_state
    """

    cx_map = get_cx_map(qubit_layout, graph)
    if n_state is None:
        n_state = len(cx_map) + 1
    participating_qubits = set(qubit for pair in cx_map[: n_state - 1] for qubit in pair)

    relabeling = {idx_old: idx_new for idx_new, idx_old in enumerate(participating_qubits)}
    n_state_register = QuantumRegister(n_state)
    qc = QuantumCircuit(n_state_register, name="ghz")
    qc.h([relabeling[cx_map[0][0]]])
    for _, pair in zip(np.arange(n_state - 1), cx_map):
        relabeled_pair = [relabeling[pair[0]], relabeling[pair[1]]]
        qc.barrier(relabeled_pair)
        qc.cx(*relabeled_pair)
    qc.measure_active()
    return qc, list(participating_qubits)


def extract_fidelities(cal_url: str, qubit_layout: List[int]) -> Tuple[List[List[int]], List[float]]:
    """Returns couplings and CZ-fidelities from calibration data URL

    Args:
        cal_url: str
            The url under which the calibration data for the backend can be found
        qubit_layout: List[int]
            The subset of system-qubits used in the protocol, indexed from 0
    Returns:
        list_couplings: List[List[int]]
            A list of pairs, each of which is a qubit coupling for which the calibration
            data contains a fidelity.
        list_fids: List[float]
            A list of CZ fidelities from the calibration url, ordered in the same way as list_couplings
    """

    byteobj = BytesIO()  # buffer creation
    curlobj = pycurl.Curl()  # pylint: disable=c-extension-no-member
    curlobj.setopt(curlobj.URL, f"{cal_url}")  # type: ignore
    curlobj.setopt(curlobj.WRITEDATA, byteobj)  # type: ignore
    curlobj.perform()  # perform file transfer
    curlobj.close()  # end of session
    body = byteobj.getvalue()
    res = json.loads(body.decode())

    qubit_mapping = {qubit: idx for idx, qubit in enumerate(qubit_layout)}
    list_couplings = []
    list_fids = []
    for key in res["metrics"]:
        if "irb.cz" in key:
            idx_1 = key.index(".QB")
            idx_2 = key.index("__QB")
            idx_3 = key.index(".fidelity")
            qb1 = int(key[idx_1 + 3 : idx_2]) - 1
            qb2 = int(key[idx_2 + 4 : idx_3]) - 1
            if all([qb1 in qubit_layout, qb2 in qubit_layout]):
                list_couplings.append([qubit_mapping[qb1], qubit_mapping[qb2]])
                list_fids.append(float(res["metrics"][key]["value"]))
    return list_couplings, list_fids


def get_edges(
    coupling_map: CouplingMap,
    qubit_layout: List[int],
    edges_cal: Optional[List[List[int]]] = None,
    fidelities_cal: Optional[List[float]] = None,
):
    """Produces a networkx.Graph from coupling map fidelity information, with edges given by couplings
        and edge weights given by fidelities

    Args:
        coupling_map (CouplingMap):
            The list pairs on which 2-qubit gates are natively supported
        qubit_layout (List[int]):
            The subset of system-qubits used in the protocol, indexed from 0
        edges_cal (Optional[List[List[int]]]):
            A coupling map of qubit pairs that have CZ fidelities in the calibration data
        fidelities_cal (Optional[List[float]]):
            A list of CZ fidelities ordered in the same way as edges_cal

    Returns:
        graph: networkx.Graph
            The final weighted graph for the given calibration or coupling map
    """
    edges_coupling = list(coupling_map.get_edges())[::2]
    edges_patch = []
    for idx, edge in enumerate(edges_coupling):
        if edge[0] in qubit_layout and edge[1] in qubit_layout:
            edges_patch.append([edge[0], edge[1]])

    if fidelities_cal is not None:
        fidelities_cal = list(
            np.minimum(np.array(fidelities_cal), np.ones(len(fidelities_cal)))
        )  # get rid of > 1 fidelities
        fidelities_patch = []
        for edge in edges_patch:
            for idx, edge_2 in enumerate(cast(List[int], edges_cal)):
                if edge == edge_2:
                    fidelities_patch.append(fidelities_cal[idx])
        weights = -np.log(np.array(fidelities_patch))
    else:
        weights = np.ones(len(edges_patch))
    graph = Graph()
    for idx, edge in enumerate(edges_patch):
        graph.add_edge(*edge, weight=weights[idx])
    if not is_connected(graph):
        print("Warning: The subgraph of selected qubit_layout is not connected.")
    return graph


def get_cx_map(qubit_layout: List[int], graph: networkx.Graph) -> list[list[int]]:
    """Calculate the cx_map based on participating qubits and the 2QB gate fidelities between them.

    Uses networkx graph algorithms to calculate the minimal spanning tree of the subgraph defined by qubit_layout.
    The weights are -log(CZ fidelity) for each edge. Then, finds the qubit in the most central position
    by calculating the distances between all qubits. Next, adds CX applications to the list, starting
    from the central qubit, such that the smallest number of layers is executed (most parallel).

    Args:
        qubit_layout: List[int]
            The subset of system-qubits used in the protocol, indexed from 0
        graph: networkx.Graph
            The connectivity graph with edge weight given by CZ fidelities
    Returns:
        cx_map: List[List[int]]
            A list of CX gates for the GHZ generation circuit, starting from the first gate to be applied
    """

    rev_mapping = {component: component for i, component in enumerate(qubit_layout)}
    span = minimum_spanning_tree(graph)
    path = dict(all_pairs_shortest_path(span))
    all_distances = [max(len(path[target_qubit][qubit]) for qubit in qubit_layout) for target_qubit in qubit_layout]
    shortest_distance = min(all_distances)
    central_qubit = qubit_layout[np.argmin(all_distances)]
    all_paths = list(path[central_qubit].values())
    all_paths.sort(key=lambda x: -len(x))
    already_entangled = [central_qubit]
    cx_map = []
    for i in range(1, shortest_distance):
        for qubit_path in all_paths:
            if i < len(qubit_path):
                new_qubit = qubit_path[i]
            else:
                continue
            if new_qubit in already_entangled:
                continue
            cx_map.append([rev_mapping[qubit_path[i - 1]], rev_mapping[new_qubit]])
            already_entangled.append(new_qubit)
    return cx_map


def plot_fidelities(
    observations: List[BenchmarkObservation], dataset: xr.Dataset, qubit_layouts: List[List[int]]
) -> Figure:
    """Plots all the fidelities stored in the observations into a single plot of fidelity vs. number of qubits

    Parameters
    ----------
    observations: List[BenchmarkObservation]
        A list of Observations, each assumed to be a fidelity
    dataset: xr.Dataset
        The experiment dataset containing results and metadata
    qubit_layouts
        The list of qubit layouts as given by the user. This is used to name the layouts in order for identification
        in the plot.
    Returns
    -------
    fig :Figure
        The figure object with the fidelity plot.
    """
    timestamp = dataset.attrs["execution_timestamp"]
    backend_name = dataset.attrs["backend_name"]

    fig, ax = plt.subplots()
    layout_short = {str(qubit_layout): f" L{i}" for i, qubit_layout in enumerate(qubit_layouts)}
    recorded_labels = []
    x_positions = []
    cmap = plt.cm.get_cmap("winter")
    # colors = [cmap(0), cmap(1)]
    for i, obs in enumerate(observations):
        label = "With REM" if "rem" in obs.name else "Unmitigated"
        if label in recorded_labels:
            label = "_nolegend_"
        else:
            recorded_labels.append(label)
        identifier = obs.identifier.string_identifier
        x = len(
            identifier.strip("[]").replace('"', "").replace(" ", "").split(",")
        )  # pylint: disable=inconsistent-quotes
        y = obs.value
        ax.errorbar(
            x,
            y,
            yerr=obs.uncertainty,
            capsize=4,
            color=cmap(0.85) if "rem" in obs.name else cmap(0.15),
            label=label,
            fmt="o",
            alpha=1,
            mec="black",
            markersize=5,
        )
        x_positions.append(x)
        ax.annotate(layout_short[identifier], (x, y))

    ax.set_xticks(x_positions, labels=[str(x) for x in x_positions])
    ax.grid()

    ax.axhline(0.5, linestyle="--", color="red", label="GME threshold")
    ax.set_ylim((0, 1))
    ax.set_title(f"GHZ fidelities of all qubit layouts\nbackend: {backend_name} --- {timestamp}")
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Fidelity")
    ax.legend(framealpha=0.5, fontsize=8)
    plt.gcf().set_dpi(250)
    plt.close()
    return fig


class GHZBenchmark(Benchmark):
    """The GHZ Benchmark estimates the quality of generated Greenberger–Horne–Zeilinger states"""

    analysis_function = staticmethod(fidelity_analysis)
    name = "ghz"

    def __init__(self, backend: IQMBackendBase, configuration: "GHZConfiguration"):
        """Construct the GHZBenchmark class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (QuantumVolumeConfiguration): the configuration of the benchmark
        """
        super().__init__(backend, configuration)

        self.state_generation_routine = configuration.state_generation_routine
        if configuration.custom_qubits_array:
            self.custom_qubits_array = configuration.custom_qubits_array
        else:
            self.custom_qubits_array = [list(set(chain(*backend.coupling_map)))]
        self.qubit_counts: Sequence[int] | List[int]
        if not configuration.qubit_counts:
            self.qubit_counts = [len(layout) for layout in self.custom_qubits_array]
        else:
            if any(np.max(configuration.qubit_counts) > [len(layout) for layout in self.custom_qubits_array]):
                raise ValueError("The maximum given qubit count is larger than the size of the smallest qubit layout.")
            self.qubit_counts = configuration.qubit_counts

        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg
        self.fidelity_routine = configuration.fidelity_routine
        self.num_RMs = configuration.num_RMs
        self.rem = configuration.rem
        self.mit_shots = configuration.mit_shots
        self.cal_url = configuration.cal_url
        self.timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

    def generate_native_ghz(self, qubit_layout: List[int], qubit_count: int, routine: str) -> CircuitGroup:
        """
        Generate a circuit preparing a GHZ state,
        according to a given routine and transpiled to the native gate set and topology.

        Args:
            qubit_layout: List[int]
                The subset of system-qubits used in the protocol, indexed from 0
            qubit_count: int
                The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
                the number of qubits in qubit_layout
            routine: str
                The routine to generate the GHZ circuit

        Returns:
            QuantumCircuit implementing GHZ native state
        """
        circuit_group = CircuitGroup(name=f"{qubit_layout}_native_ghz")
        fixed_coupling_map = set_coupling_map(qubit_layout, self.backend, "fixed")
        ghz_native_transpiled: List[QuantumCircuit]

        if routine == "naive":
            ghz: QuantumCircuit = generate_ghz_linear(qubit_count)
            circuit_group.add_circuit(ghz)
            ghz_native_transpiled, _ = perform_backend_transpilation(
                [ghz],
                self.backend,
                qubit_layout,
                fixed_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
            final_ghz = ghz_native_transpiled
        elif routine == "tree":
            if self.cal_url:
                edges_cal, fidelities_cal = extract_fidelities(self.cal_url, qubit_layout)
                graph = get_edges(self.backend.coupling_map, qubit_layout, edges_cal, fidelities_cal)
            else:
                graph = get_edges(self.backend.coupling_map, qubit_layout)
            ghz, _ = generate_ghz_spanning_tree(graph, qubit_layout, qubit_count)
            circuit_group.add_circuit(ghz)
            ghz_native_transpiled, _ = perform_backend_transpilation(
                [ghz],
                self.backend,
                qubit_layout,
                fixed_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
            final_ghz = ghz_native_transpiled
        else:
            ghz_log = [generate_ghz_log_cruz(qubit_count), generate_ghz_log_mooney(qubit_count)]
            ghz_native_transpiled, _ = perform_backend_transpilation(
                ghz_log,
                self.backend,
                qubit_layout,
                fixed_coupling_map,
                qiskit_optim_level=self.qiskit_optim_level,
                optimize_sqg=self.optimize_sqg,
            )
            # Use either the circuit with min depth after transpilation or min #2q gates
            if ghz_native_transpiled[0].depth() == ghz_native_transpiled[1].depth():
                index_min_2q = np.argmin([c.count_ops()["cz"] for c in ghz_native_transpiled])
                final_ghz = ghz_native_transpiled[index_min_2q]
                circuit_group.add_circuit(ghz_log[index_min_2q])
            else:
                index_min_depth = np.argmin([c.depth() for c in ghz_native_transpiled])
                final_ghz = ghz_native_transpiled[index_min_depth]
                circuit_group.add_circuit([ghz_log[index_min_depth]])
        self.circuits["untranspiled_circuits"].circuit_groups.append(circuit_group)
        return CircuitGroup(name=f"{qubit_layout}_native_ghz", circuits=[final_ghz[0]])

    def generate_coherence_meas_circuits(self, qubit_layout: List[int], qubit_count: int) -> List[QuantumCircuit]:
        """
        Takes a given GHZ circuit and outputs circuits needed to measure fidelity via mult. q. coherences method

        Args:
            qubit_layout: List[int]
                The subset of system-qubits used in the protocol, indexed from 0
            qubit_count: int
                The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
                the number of qubits in qubit_layout
        Returns:
             qc_list_transpiled: List[QuantumCircuit]
                A list of transpiled quantum circuits to be measured
        """

        idx = BenchmarkObservationIdentifier(qubit_layout).string_identifier
        qc_list = self.circuits["untranspiled_circuits"][f"{qubit_layout}_native_ghz"].circuits

        qc = qc_list[0].copy()
        qc.remove_final_measurements()
        qc_inv = qc.inverse()
        phases = [np.pi * i / (qubit_count + 1) for i in range(2 * qubit_count + 2)]
        for phase in phases:
            qc_phase = qc.copy()
            qc_phase.barrier()
            for qubit, _ in enumerate(qubit_layout):
                qc_phase.p(phase, qubit)
            qc_phase.barrier()
            qc_phase.compose(qc_inv, inplace=True)
            qc_phase.measure_active()
            qc_list.append(qc_phase)

        fixed_coupling_map = set_coupling_map(qubit_layout, self.backend, "fixed")
        qc_list_transpiled, _ = perform_backend_transpilation(
            qc_list,
            self.backend,
            qubit_layout,
            fixed_coupling_map,
            qiskit_optim_level=self.qiskit_optim_level,
            optimize_sqg=self.optimize_sqg,
        )
        circuit_group = CircuitGroup(name=idx, circuits=qc_list)
        self.circuits["untranspiled_circuits"].circuit_groups.append(circuit_group)
        return qc_list_transpiled

    def generate_readout_circuit(self, qubit_layout: List[int], qubit_count: int) -> CircuitGroup:
        """
        A wrapper for the creation of different circuits to estimate the fidelity

        Args:
            qubit_layout: List[int]
                The subset of system-qubits used in the protocol, indexed from 0
            qubit_count: int
                The number of qubits for which a GHZ state should be created. This values should be smaller or equal to
                the number of qubits in qubit_layout

        Returns:
             all_circuits_list: List[QuantumCircuit]
                A list of transpiled quantum circuits to be measured
        """
        # Generate the list of circuits

        qcvv_logger.info(f"Now generating a {len(qubit_layout)}-qubit GHZ state on qubits {qubit_layout}")
        transpiled_ghz_group: CircuitGroup = self.generate_native_ghz(
            qubit_layout, qubit_count, self.state_generation_routine
        )

        match self.fidelity_routine:
            case "randomized_measurements":
                all_circuits_list, _ = append_rms(
                    transpiled_ghz_group.circuits[0], cast(int, self.num_RMs), self.backend
                )
                transpiled_ghz_group.circuits = all_circuits_list
            case "coherences":
                all_circuits_list = self.generate_coherence_meas_circuits(qubit_layout, qubit_count)
                transpiled_ghz_group.circuits = all_circuits_list
        self.circuits["transpiled_circuits"].circuit_groups.append(transpiled_ghz_group)
        return transpiled_ghz_group

    def add_configuration_to_dataset(self, dataset: xr.Dataset):  # CHECK
        """
        Creates a xarray.Dataset and adds the circuits and configuration metadata to it.

        Args:
            dataset (xr.Dataset):
        Returns:
            xr.Dataset: dataset to be used for further data storage
        """

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        dataset.attrs[f"backend_name"] = self.backend.name
        dataset.attrs[f"execution_timestamp"] = self.execution_timestamp
        dataset.attrs["fidelity_routine"] = self.fidelity_routine

    def execute(self, backend) -> xr.Dataset:
        """
        Executes the benchmark.
        """
        self.execution_timestamp = strftime("%Y%m%d-%H%M%S")
        aux_custom_qubits_array = cast(List[List[int]], self.custom_qubits_array).copy()
        dataset = xr.Dataset()

        # Submit all
        all_jobs: Dict = {}

        self.circuits = Circuits()
        self.circuits.benchmark_circuits.append(BenchmarkCircuit(name="transpiled_circuits"))
        self.circuits.benchmark_circuits.append(BenchmarkCircuit(name="untranspiled_circuits"))
        for qubit_layout in aux_custom_qubits_array:
            Id = BenchmarkObservationIdentifier(qubit_layout)
            idx = Id.string_identifier
            # for qubit_count in self.qubit_counts[idx]:
            qubit_count = len(qubit_layout)
            circuit_group: CircuitGroup = self.generate_readout_circuit(qubit_layout, qubit_count)
            transpiled_circuit_dict = {tuple(qubit_layout): circuit_group.circuits}
            all_jobs[idx], _ = submit_execute(
                transpiled_circuit_dict,
                backend,
                self.shots,
                self.calset_id,
                max_gates_per_batch=self.max_gates_per_batch,
            )

        # Retrieve all
        qcvv_logger.info(f"Retrieving counts and adding counts to dataset...")
        for qubit_layout in aux_custom_qubits_array:
            # for qubit_count in self.qubit_counts[idx]:
            Id = BenchmarkObservationIdentifier(qubit_layout)
            idx = Id.string_identifier
            qubit_count = len(qubit_layout)
            counts, _ = retrieve_all_counts(all_jobs[idx])
            dataset, _ = add_counts_to_dataset(counts, idx, dataset)
            if self.rem:
                qcvv_logger.info(f"Applying readout error mitigation")
                circuit_group = self.circuits["transpiled_circuits"][f"{idx}_native_ghz"]
                rem_results, _ = apply_readout_error_mitigation(backend, circuit_group.circuits, counts, self.mit_shots)
                rem_results_dist = [counts_mit.nearest_probability_distribution() for counts_mit in rem_results]
                dataset, _ = add_counts_to_dataset(rem_results_dist, f"{idx}_rem", dataset)

        self.add_configuration_to_dataset(dataset)
        return dataset


class GHZConfiguration(BenchmarkConfigurationBase):
    """GHZ state configuration

    Attributes:
        benchmark (Type[Benchmark]): GHZBenchmark
        state_generation_routine (str): The routine to construct circuits generating a GHZ state. Possible values:
                                        - "tree" (default): Optimized GHZ state generation circuit in log depth that
                                        takes the qubit coupling and CZ fidelities into account. The algorithm creates
                                        a minimal spanning tree for the qubit layout and chooses an initial qubit
                                        that minimizes largest weighted distance to all other qubits.
                                        - "log": Optimized circuit with parallel application of CX gates such that the
                                        number of CX gates scales logarithmically in the system size. This
                                        implementation currently does not take connectivity on the backend into account.
                                        - "naive": Applies the naive textbook circuit with #CX gates scaling linearly in
                                        the system size.
                                        * If other is specified, assumes "log".
        custom_qubits_array (Optional[Sequence[Sequence[int]]]): A sequence (e.g., Tuple or List) of sequences of
        physical qubit layouts, as specified by integer labels, where the benchmark is meant to be run.
                                        * If None, takes all qubits specified in the backend coupling map.
        qubit_counts (Optional[Sequence[int]]): CURRENTLY NOT SUPPORTED, A sequence (e.g., Tuple or List) of integers
        denoting number of qubits
        for which the benchmark is meant to be run. The largest qubit count provided here has to be smaller than the
        smalles given qubit layout.
        qiskit_optim_level (int): The optimization level used for transpilation to backend architecture.
            * Default: 3
        optimize_sqg (bool): Whether consecutive single qubit gates are optimized for reduced gate count via
        iqm.qiskit_iqm.iqm_transpilation.optimize_single_qubit_gates
            * Default: True
        fidelity_routine (str): The method with which the fidelity is estimated. Possible values:
            - "coherences": The multiple quantum coherences method as in [Mooney, 2021]
            - "randomized_measurements": Fidelity estimation via randomized measurements outlined in
            https://arxiv.org/abs/1812.02624
            * Default is "coherences"
        num_RMs (Optional[int]): The number of randomized measurements used if the respective fidelity routine is chosen
            * Default: 100
        rem (bool): Boolean flag determining if readout error mitigation is used
            * Default: True
        mit_shots (int): Total number of shots for readout error mitigation
            * Default: 1000
        cal_url (Optional[str]): Optional URL where the calibration data for the selected backend can be retrieved from
            The calibration data is used for the "tree" state generation routine to prioritize couplings with high
            CZ fidelity.
            * Default: None
    """

    benchmark: Type[Benchmark] = GHZBenchmark
    state_generation_routine: str = "tree"
    custom_qubits_array: Optional[Sequence[Sequence[int]]] = None
    qubit_counts: Optional[Sequence[int]] = None
    shots: int = 2**10
    qiskit_optim_level: int = 3
    optimize_sqg: bool = True
    fidelity_routine: str = "coherences"
    num_RMs: Optional[int] = 100
    rem: bool = True
    mit_shots: int = 1_000
    cal_url: Optional[str] = None
