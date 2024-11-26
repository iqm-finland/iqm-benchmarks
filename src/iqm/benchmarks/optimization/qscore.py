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
Q-score benchmark
"""

import itertools
from time import strftime
from typing import Callable, Dict, List, Optional, Tuple, Type

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from networkx import Graph
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import basinhopping, minimize

from iqm.benchmarks.benchmark import BenchmarkBase, BenchmarkConfigurationBase
from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import perform_backend_transpilation, retrieve_all_counts, submit_execute, timeit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


class QScoreBenchmark(BenchmarkBase):
    """
    Q-score estimates the size of combinatorial optimization problems a given number of qubits can execute with meaningful results.
    """

    def __init__(self, backend: IQMBackendBase, configuration: "QScoreConfiguration"):
        """Construct the QScoreBenchmark class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (QScoreConfiguration): the configuration of the benchmark
        """
        super().__init__(backend, configuration)

        self.num_instances = configuration.num_instances
        self.num_qaoa_layers = configuration.num_qaoa_layers
        self.min_num_nodes = configuration.min_num_nodes
        self.max_num_nodes = configuration.max_num_nodes
        self.use_virtual_node = configuration.use_virtual_node
        self.use_classically_optimized_angles = configuration.use_classically_optimized_angles
        self.choose_qubits_routine = configuration.choose_qubits_routine
        self.qiskit_optim_level = configuration.qiskit_optim_level
        self.optimize_sqg = configuration.optimize_sqg
        self.timestamp = strftime("%Y%m%d-%H%M%S")
        self.seed = configuration.seed

        self.graph_physical: Graph
        self.virtual_nodes: List[Tuple[int, int]]
        self.node_to_qubit: Dict[int, int]
        self.qubit_to_node: Dict[int, int]

        if self.use_classically_optimized_angles and self.num_qaoa_layers > 1:
            raise ValueError("If the `use_classically_optimized_angles` is chosen, the `num_qaoa_layers` must be 1.")

        if self.use_virtual_node and self.num_qaoa_layers > 1:
            raise ValueError("If the `use_virtual_node` is chosen, the `num_qaoa_layers` must be 1.")

        if self.choose_qubits_routine == "custom":
            self.custom_qubits_array = configuration.custom_qubits_array

    @staticmethod
    def name() -> str:
        return "qscore"

    def generate_maxcut_ansatz(  # pylint: disable=too-many-branches
        self,
        graph: Graph,
        theta: list[float],
    ) -> QuantumCircuit:
        """Generate an ansatz circuit for QAOA MaxCut, with measurements at the end.

        Args:
            graph (networkx graph): the MaxCut problem graph
            theta (list[float]): the variational parameters for QAOA, first gammas then betas

        Returns:
            QuantumCircuit: the QAOA ansatz quantum circuit.
        """
        gamma = theta[: self.num_qaoa_layers]
        beta = theta[self.num_qaoa_layers :]

        num_qubits = self.graph_physical.number_of_nodes()

        if self.graph_physical.number_of_nodes() != graph.number_of_nodes():
            num_qubits = self.graph_physical.number_of_nodes()
            # re-label the nodes to be between 0 and _num_qubits
            self.node_to_qubit = {node: qubit for qubit, node in enumerate(list(self.graph_physical.nodes))}
            self.qubit_to_node = dict(enumerate(list(self.graph_physical.nodes)))
        else:
            num_qubits = graph.number_of_nodes()
            self.node_to_qubit = {node: node for node in list(self.graph_physical.nodes)}  # no relabeling
            self.qubit_to_node = self.node_to_qubit

        # in case the graph is trivial: return empty circuit
        if num_qubits == 0:
            return QuantumCircuit(1)

        qaoa_qc = QuantumCircuit(num_qubits)
        for i in range(0, num_qubits):
            qaoa_qc.h(i)
        for layer in range(self.num_qaoa_layers):
            for edge in self.graph_physical.edges():
                i = self.node_to_qubit[edge[0]]
                j = self.node_to_qubit[edge[1]]
                qaoa_qc.rzz(2 * gamma[layer], i, j)

            # include edges of the virtual node as rz terms
            for vn in self.virtual_nodes:
                for edge in graph.edges(vn[0]):
                    # exclude edges between virtual nodes
                    edges_between_virtual_nodes = list(itertools.combinations([i[0] for i in self.virtual_nodes], 2))
                    if set(edge) not in list(map(set, edges_between_virtual_nodes)):
                        # The value of the fixed node defines the sign of the rz gate
                        sign = 1.0
                        if vn[1] == 1:
                            sign = -1.0
                        qaoa_qc.rz(sign * 2.0 * gamma[layer], self.node_to_qubit[edge[1]])

            for i in range(0, num_qubits):
                qaoa_qc.rx(2 * beta[layer], i)
        qaoa_qc.measure_all()

        return qaoa_qc

    @staticmethod
    def cost_function(x: str, graph: Graph) -> int:
        """Returns the number of cut edges in a graph (with minus sign).

        Args:
            x (str): solution bitstring.
            graph (networkx graph): the MaxCut problem graph.

        Returns:
            obj (float): number of cut edges multiplied by -1.
        """

        obj = 0
        for i, j in graph.edges():
            if x[i] != x[j]:
                obj += 1

        return -1 * obj

    def compute_expectation_value(self, counts: Dict[str, int], graph: Graph) -> float:
        """Computes expectation value based on measurement results.

        Args:
            counts (Dict[str, int]): key as bitstring, val as count
            graph (networkx) graph: the MaxCut problem graph

        Returns:
            avg (float): expectation value of the cut edges for number of counts
        """

        avg = 0
        sum_count = 0
        for bitstring_aux, count in counts.items():
            bitstring_aux_list = list(bitstring_aux)[::-1]  # go from qiskit endianness to networkx endianness

            # map the qubits back to nodes
            bitstring = [""] * (len(bitstring_aux_list) + len(self.virtual_nodes))
            for qubit, node in self.qubit_to_node.items():
                bitstring[node] = bitstring_aux_list[qubit]

            # insert virtual node(s) to bitstring
            for virtual_node in self.virtual_nodes:
                if virtual_node[0] is not None:
                    bitstring[virtual_node[0]] = str(virtual_node[1])

            obj = self.cost_function("".join(bitstring), graph)
            avg += obj * count
            sum_count += count

        return avg / sum_count

    def create_objective_function(self, graph: Graph, qubit_set: List[int]) -> Callable:
        """
        Creates a function that maps the parameters to the parametrized circuit,
        runs it and computes the expectation value.

        Args:
            graph (networkx graph): the MaxCut problem graph.
            qubit_set (List[int]): indices of the used qubits.
        Returns:
            callable: function that gives expectation value of the cut edges from counts sampled from the ansatz
        """

        def objective_function(theta):
            qc = self.generate_maxcut_ansatz(graph, theta)

            if len(qc.count_ops()) == 0:
                counts = {"": 1.0}  # to handle the case of physical graph with no edges

            else:
                coupling_map = self.backend.coupling_map.reduce(qubit_set)
                transpiled_qc_list, _ = perform_backend_transpilation(
                    [qc],
                    backend=self.backend,
                    qubits=qubit_set,
                    coupling_map=coupling_map,
                    qiskit_optim_level=self.qiskit_optim_level,
                    optimize_sqg=self.optimize_sqg,
                    routing_method=self.routing_method,
                )

                sorted_transpiled_qc_list = {tuple(qubit_set): transpiled_qc_list}
                # Execute on the backend
                jobs, _ = submit_execute(
                    sorted_transpiled_qc_list,
                    self.backend,
                    self.shots,
                    self.calset_id,
                    max_gates_per_batch=self.max_gates_per_batch,
                )

                counts = retrieve_all_counts(jobs)[0][0]

            return self.compute_expectation_value(counts, graph)

        return objective_function

    @staticmethod
    def calculate_optimal_angles_for_QAOA_p1(graph: Graph) -> List[float]:
        """
        Calculates the optimal angles for single layer QAOA MaxCut ansatz.

        Args:
            graph (networkx graph): the MaxCut problem graph.

        Returns:
            List[float]: optimal angles gamma and beta.

        """

        def get_Zij_maxcut_p1(edge_ij, gamma, beta):
            """
            Calculates <p1_QAOA | Z_i Z_j | p1_QAOA>, assuming ij is edge of G.
            """
            i, j = edge_ij
            di = graph.degree[i]
            dj = graph.degree[j]

            first = np.cos(2 * gamma) ** (di - 1) + np.cos(2 * gamma) ** (dj - 1)
            first *= 0.5 * np.sin(4 * beta) * np.sin(2 * gamma)

            node_list = list(graph.nodes).copy()
            node_list.remove(i)
            node_list.remove(j)
            f1 = 1
            f2 = 1
            for k in node_list:
                if graph.has_edge(i, k) and graph.has_edge(j, k):  # ijk is triangle
                    f1 *= np.cos(4 * gamma)
                elif graph.has_edge(i, k) or graph.has_edge(j, k):  # ijk is no triangle
                    f1 *= np.cos(2 * gamma)
                    f2 *= np.cos(2 * gamma)
            second = 0.5 * np.sin(2 * beta) ** 2 * (f1 - f2)
            return first - second

        def get_expected_zz_edgedensity(x):
            gamma = x[0]
            beta = x[1]
            # pylint: disable=consider-using-generator
            return sum([get_Zij_maxcut_p1(edge, gamma, beta) for edge in graph.edges]) / graph.number_of_edges()

        bounds = [(0.0, np.pi / 2), (-np.pi / 4, 0.0)]
        x_init = [0.15, -0.28]

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        res = basinhopping(get_expected_zz_edgedensity, x_init, minimizer_kwargs=minimizer_kwargs, niter=10, T=2)

        return res.x

    def run_QAOA(self, graph: Graph, qubit_set: List[int]) -> float:
        """
        Solves the cut size of MaxCut for a graph using QAOA.
        The result is average value sampled from the optimized ansatz.

        Args:
            graph (networkx graph): the MaxCut problem graph.
            qubit_set (List[int]): indices of the used qubits.

        Returns:
            float: the expectation value of the maximum cut size.

        """

        objective_function = self.create_objective_function(graph, qubit_set)

        if self.use_classically_optimized_angles:
            if self.graph_physical.number_of_edges() != 0:
                opt_angles = self.calculate_optimal_angles_for_QAOA_p1(self.graph_physical)
            else:
                opt_angles = [1.0, 1.0]
            res = minimize(objective_function, opt_angles, method="COBYLA", tol=1e-5, options={"maxiter": 0})
        else:
            # Good initial angles from from Wurtz et.al. "The fixed angle conjecture for QAOA on regular MaxCut graphs." arXiv preprint arXiv:2107.00677 (2021).
            OPTIMAL_INITIAL_ANGLES = {
                "1": [-0.616, 0.393 / 2],
                "2": [-0.488, 0.898 / 2, 0.555 / 2, 0.293 / 2],
                "3": [-0.422, 0.798 / 2, 0.937 / 2, 0.609 / 2, 0.459 / 2, 0.235 / 2],
                "4": [-0.409, 0.781 / 2, 0.988 / 2, 1.156 / 2, 0.600 / 2, 0.434 / 2, 0.297 / 2, 0.159 / 2],
                "5": [-0.36, -0.707, -0.823, -1.005, -1.154, 0.632 / 2, 0.523 / 2, 0.390 / 2, 0.275 / 2, 0.149 / 2],
                "6": [
                    -0.331,
                    -0.645,
                    -0.731,
                    -0.837,
                    -1.009,
                    -1.126,
                    0.636 / 2,
                    0.535 / 2,
                    0.463 / 2,
                    0.360 / 2,
                    0.259 / 2,
                    0.139 / 2,
                ],
                "7": [
                    -0.310,
                    -0.618,
                    -0.690,
                    -0.751,
                    -0.859,
                    -1.020,
                    -1.122,
                    0.648 / 2,
                    0.554 / 2,
                    0.490 / 2,
                    0.445 / 2,
                    0.341 / 2,
                    0.244 / 2,
                    0.131 / 2,
                ],
                "8": [
                    -0.295,
                    -0.587,
                    -0.654,
                    -0.708,
                    -0.765,
                    -0.864,
                    -1.026,
                    -1.116,
                    0.649 / 2,
                    0.555 / 2,
                    0.500 / 2,
                    0.469 / 2,
                    0.420 / 2,
                    0.319 / 2,
                    0.231 / 2,
                    0.123 / 2,
                ],
                "9": [
                    -0.279,
                    -0.566,
                    -0.631,
                    -0.679,
                    -0.726,
                    -0.768,
                    -0.875,
                    -1.037,
                    -1.118,
                    0.654 / 2,
                    0.562 / 2,
                    0.509 / 2,
                    0.487 / 2,
                    0.451 / 2,
                    0.403 / 2,
                    0.305 / 2,
                    0.220 / 2,
                    0.117 / 2,
                ],
                "10": [
                    -0.267,
                    -0.545,
                    -0.610,
                    -0.656,
                    -0.696,
                    -0.729,
                    -0.774,
                    -0.882,
                    -1.044,
                    -1.115,
                    0.656 / 2,
                    0.563 / 2,
                    0.514 / 2,
                    0.496 / 2,
                    0.496 / 2,
                    0.436 / 2,
                    0.388 / 2,
                    0.291 / 2,
                    0.211 / 2,
                    0.112 / 2,
                ],
                "11": [
                    -0.257,
                    -0.528,
                    -0.592,
                    -0.640,
                    -0.677,
                    -0.702,
                    -0.737,
                    -0.775,
                    -0.884,
                    -1.047,
                    -1.115,
                    0.656 / 2,
                    0.563 / 2,
                    0.516 / 2,
                    0.504 / 2,
                    0.482 / 2,
                    0.456 / 2,
                    0.421 / 2,
                    0.371 / 2,
                    0.276 / 2,
                    0.201 / 2,
                    0.107 / 2,
                ],
            }

            theta = OPTIMAL_INITIAL_ANGLES[str(self.num_qaoa_layers)]
            bounds = [(-np.pi, np.pi)] * self.num_qaoa_layers + [(0.0, np.pi)] * self.num_qaoa_layers

            res = minimize(
                objective_function,
                theta,
                bounds=bounds,
                method="COBYLA",
                tol=1e-5,
                options={"maxiter": 300},
            )

        return -res.fun

    @staticmethod
    def is_successful(
        approximation_ratio: float,
    ) -> bool:
        """Check whether a Q-score benchmark returned approximation ratio above beta*, therefore being successful.

        This condition checks that the mean approximation ratio is above the beta* = 0.2 threshold.

        Args:
            approximation_ratio (float): the mean approximation ratio of all problem graphs

        Returns:
            bool: whether the Q-score benchmark was successful
        """
        return bool(approximation_ratio > 0.2)

    @staticmethod
    def choose_qubits_naive(num_qubits: int) -> list[int]:
        """Choose the qubits to execute the circuits on, sequentially starting at qubit 0.

        Args:
            num_qubits (int): the number of qubits to choose.

        Returns:
            list[int]: the list of qubits to execute the circuits on.
        """
        if num_qubits == 2:
            return [0, 2]

        return list(range(num_qubits))

    def choose_qubits_custom(self, num_qubits: int) -> list[int]:
        """Choose the qubits to execute the circuits on, according to elements in custom_qubits_array matching num_qubits number of qubits

        Args:
            num_qubits (int): the number of qubits to choose

        Returns:
            list[int]: the list of qubits to execute the circuits on
        """
        if self.custom_qubits_array is None:
            raise ValueError(
                "If the `choose_qubits_custom` routine is chosen, a `custom_qubits_array` must be specified in `QScoreConfiguration`."
            )
        selected_qubits = [qubit_layout for qubit_layout in self.custom_qubits_array if len(qubit_layout) == num_qubits]
        # User may input more than one num_qubits layouts
        if len(selected_qubits) > 1:
            chosen_qubits = selected_qubits[0]
            self.custom_qubits_array.remove(chosen_qubits)
            # The execute_single_benchmark call must be looped through a COPY of custom_qubits_array
        else:
            chosen_qubits = selected_qubits[0]
        return chosen_qubits

    def plot_approximation_ratios(
        self, list_of_num_nodes: list[int], list_of_cut_sizes: list[list[float]]
    ) -> tuple[str, Figure]:
        """Generate the figure of approximation ratios vs number of nodes,
            including standard deviation and the acceptance threshold.

        Args:
            list_of_num_nodes (list[int]): list of problem graph sizes.
            list_of_cut_sizes (list[list[float]]): the list of lists of maximum average cut sizes of problem graph instances for each problem size.

        Returns:
            str: the name of the figure.
            Figure: the figure.
        """

        LAMBDA = 0.178

        approximation_ratio_lists = {
            num_nodes: (np.array(cut_sizes) - num_nodes * (num_nodes - 1) / 8) / (LAMBDA * num_nodes ** (3 / 2))
            for num_nodes, cut_sizes in zip(list_of_num_nodes, list_of_cut_sizes)
        }

        avg_approximation_ratios = {
            num_nodes: np.mean(approximation_ratios)
            for num_nodes, approximation_ratios in approximation_ratio_lists.items()
        }

        std_of_approximation_ratios = {
            num_nodes: np.std(approximation_ratios) / np.sqrt(len(approximation_ratios) - 1)
            for num_nodes, approximation_ratios in approximation_ratio_lists.items()
        }

        nodes = list(avg_approximation_ratios.keys())
        ratios = list(avg_approximation_ratios.values())
        std = list(std_of_approximation_ratios.values())

        fig = plt.figure()
        ax = plt.axes()

        plt.axhline(0.2, color="red", linestyle="dashed", label="Threshold")
        plt.errorbar(
            nodes, ratios, yerr=std, fmt="-o", capsize=10, markersize=8, color="#759DEB", label="Approximation ratio"
        )

        ax.set_ylabel(r"Q-score ratio $\beta(n)$")
        ax.set_xlabel("Number of nodes $(n)$")
        plt.xticks(range(min(nodes), max(nodes) + 1))
        plt.legend(loc="lower right")
        plt.grid(True)

        if self.use_virtual_node and self.use_classically_optimized_angles:
            title = f"Q-score, {self.num_instances} instances, with virtual node and classically optimized angles\nBackend: {self.backend.name} / {self.timestamp}"
        elif self.use_virtual_node and not self.use_classically_optimized_angles:
            title = f"Q-score, {self.num_instances} instances, with virtual node \nBackend: {self.backend.name} / {self.timestamp}"
        elif not self.use_virtual_node and self.use_classically_optimized_angles:
            title = f"Q-score, {self.num_instances} instances, with classically optimized angles\nBackend: {self.backend.name} / {self.timestamp}"
        else:
            title = f"Q-score, {self.num_instances} instances \nBackend: {self.backend.name} / {self.timestamp}"

        plt.title(
            title,
            fontsize=9,
        )
        fig_name = f"{max(nodes)}_nodes_{self.num_instances}_instances.png"

        # Show plot if verbose is True
        plt.gcf().set_dpi(250)
        plt.show()

        plt.close()

        return fig_name, fig

    def execute_single_benchmark(
        self,
        num_nodes: int,
    ) -> tuple[bool, float, list[float], list[int]]:
        """Execute a single benchmark, for a given number of qubits.

        Args:
            num_nodes (int): number of nodes in the MaxCut problem graphs.

        Returns:
            bool: whether the benchmark was successful.
            float: approximation_ratio.
            list[float]: the list of maximum average cut sizes of problem graph instances.
            list[int]: the set of qubits the Q-score benchmark was executed on.
        """

        cut_sizes: list[float] = []
        seed = self.seed

        for i in range(self.num_instances):
            graph = nx.generators.erdos_renyi_graph(num_nodes, 0.5, seed=seed)
            print(f"graph: {graph}")
            self.graph_physical = graph.copy()
            self.virtual_nodes = []
            if self.use_virtual_node:
                virtual_node, _ = max(
                    graph.degree(), key=lambda x: x[1]
                )  # choose the virtual node as the most connected node
                self.virtual_nodes.append(
                    (virtual_node, 1)
                )  # the second element of the tuple is the value assigned to the virtual node
                self.graph_physical.remove_node(self.virtual_nodes[0][0])
            # See if there are any non-connected nodes if so, remove them also
            # and set them to be opposite value to the possible original virtual node
            for node in self.graph_physical.nodes():
                if self.graph_physical.degree(node) == 0:
                    self.virtual_nodes.append((node, 0))
            for vn in self.virtual_nodes:
                if self.graph_physical.has_node(vn[0]):
                    self.graph_physical.remove_node(vn[0])

            # Graph with no edges has cut size = 0
            if graph.number_of_edges() == 0:
                cut_sizes.append(0)
                seed += 1
                qcvv_logger.info(f"Graph {i+1}/{self.num_instances} had no edges: cut size = 0.")
                continue

            # Choose the qubit layout
            qubit_set = []
            if self.choose_qubits_routine.lower() == "naive":
                qubit_set = self.choose_qubits_naive(num_nodes)
            elif self.choose_qubits_routine.lower() == "custom" or self.choose_qubits_routine.lower() == "mapomatic":
                qubit_set = self.choose_qubits_custom(num_nodes)
            else:
                raise ValueError('choose_qubits_routine must either be "naive" or "custom".')

            # Solve the maximum cut size with QAOA
            cut_sizes.append(self.run_QAOA(graph, qubit_set))
            seed += 1
            qcvv_logger.info(f"Solved the MaxCut on graph {i+1}/{self.num_instances}.")

        average_cut_size = np.mean(cut_sizes) - num_nodes * (num_nodes - 1) / 8
        average_best_cut_size = 0.178 * pow(num_nodes, 3 / 2)
        approximation_ratio = float(average_cut_size / average_best_cut_size)

        self.raw_data[num_nodes] = {
            "qubit_set": qubit_set,
            "cut_sizes": cut_sizes,
        }
        self.results[num_nodes] = {
            "qubit_set": qubit_set,
            "is_successful": str(self.is_successful(approximation_ratio)),
            "approximation_ratio": approximation_ratio,
        }

        # Return whether the single Q-score Benchmark was successful and its mean approximation ratio
        # and cut sizes for all instances.
        return self.is_successful(approximation_ratio), approximation_ratio, cut_sizes, qubit_set

    @timeit
    def execute_full_benchmark(self) -> tuple[int, list[float], list[list[float]]]:
        """Execute the full benchmark, starting with self.min_num_nodes nodes up to failure.

        Returns:
            int: the Q-score of the device.
            list[float]: the list of approximation rations over problem graph instances for each problem size.
            list[list[float]]: the list of lists of maximum average cut sizes of problem graph instances for each problem size.
        """
        qscore = 0

        if self.max_num_nodes is None:
            if self.use_virtual_node:
                max_num_nodes = self.backend.num_qubits + 1
            else:
                max_num_nodes = self.backend.num_qubits
        else:
            max_num_nodes = self.max_num_nodes

        approximation_ratios = []
        list_of_cut_sizes = []

        for num_nodes in range(self.min_num_nodes, max_num_nodes + 1):
            qcvv_logger.info(f"Executing on {self.num_instances} random graphs with {num_nodes} nodes.")
            is_successful, approximation_ratio, cut_sizes = self.execute_single_benchmark(num_nodes)[0:3]
            approximation_ratios.append(approximation_ratio)

            list_of_cut_sizes.append(cut_sizes)
            if is_successful:
                qcvv_logger.info(
                    f"Q-Score = {num_nodes} passed with:\nApproximation ratio (Beta): {approximation_ratio:.4f}; Avg MaxCut size: {np.mean(cut_sizes):.4f}"
                )
                qscore = num_nodes
                continue

            qcvv_logger.info(
                f"Q-Score = {num_nodes} failed with \napproximation ratio (Beta): {approximation_ratio:.4f} < 0.2; Avg MaxCut size: {np.mean(cut_sizes):.4f}"
            )

        self.results["qscore"] = qscore
        fig_name, fig = self.plot_approximation_ratios(
            list(range(self.min_num_nodes, max_num_nodes + 1)), list_of_cut_sizes
        )
        self.figures[fig_name] = fig
        return num_nodes, approximation_ratios, list_of_cut_sizes


class QScoreConfiguration(BenchmarkConfigurationBase):
    """Q-score configuration."""

    benchmark: Type[BenchmarkBase] = QScoreBenchmark
    num_instances: int
    num_qaoa_layers: int = 1
    min_num_nodes: int = 2
    max_num_nodes: Optional[int] = None
    use_virtual_node: bool = True
    use_classically_optimized_angles: bool = True
    choose_qubits_routine: str = "naive"
    min_num_qubits: int = 2  # If choose_qubits_routine is "naive"
    custom_qubits_array: Optional[list[list[int]]] = None
    qiskit_optim_level: int = 3
    optimize_sqg: bool = True
    seed: int = 1
