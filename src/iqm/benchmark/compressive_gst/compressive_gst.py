# Copyright 2024 IQM Benchmark developers
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
Compressive gate set tomography
"""

from itertools import product
from time import perf_counter, strftime
from typing import Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pygsti.tools import change_basis
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, RGate

from iqm.benchmark.benchmark import BenchmarkBase, BenchmarkConfigurationBase
from iqm.benchmark.logging_config import qcvv_logger
from iqm.benchmark.utils import (
    perform_backend_transpilation,
    retrieve_all_counts,
    retrieve_all_job_metadata,
    set_coupling_map,
    submit_execute,
    timeit,
)
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase
from mGST import additional_fns, algorithm, compatibility, qiskit_interface
from mGST.low_level_jit import contract
from mGST.qiskit_interface import add_idle_gates, qiskit_gate_to_operator, remove_idle_wires
from mGST.reporting import figure_gen, reporting


# pylint: disable=too-many-lines


class compressive_GST(BenchmarkBase):
    """
    SPAM-robust characterization of a set of quantum gates
    """

    def __init__(self, backend: IQMBackendBase, configuration: "GSTConfiguration"):
        """Construct the compressive_gst class.

        Args:
            backend (IQMBackendBase): the backend to execute the benchmark on
            configuration (GSTConfiguration): the configuration of the benchmark
        """
        super().__init__(backend, configuration)

        self.qubits = configuration.qubits
        self.num_qubits = len(self.qubits)
        self.pdim = 2**self.num_qubits
        self.num_povm = self.pdim
        self.gate_set, self.gate_labels, self.num_gates = parse_gate_set(configuration)
        self.seq_len_list = configuration.seq_len_list
        self.num_circuits = configuration.num_circuits
        self.shots = configuration.shots
        self.rank = configuration.rank
        self.from_init = configuration.from_init
        self.max_inits = configuration.max_inits
        self.convergence_criteria = configuration.convergence_criteria
        self.bootstrap_samples = configuration.bootstrap_samples
        self.weights = dict({f"G%i" % i: 1 for i in range(self.num_gates)}, **{"spam": 1})
        self.max_gates_per_batch = configuration.max_gates_per_batch
        self.timestamp = strftime("%Y%m%d-%H%M%S")

        if configuration.opt_method not in ["GD", "SFN", "auto"]:
            raise ValueError("Invalid optimization method, valid options are: GD, SFN, auto")
        if configuration.opt_method == "auto":
            if (self.num_qubits == 2 and self.rank > 2) or self.num_qubits > 2:
                self.opt_method = "GD"
            else:
                self.opt_method = "SFN"
        else:
            self.opt_method = configuration.opt_method

        if configuration.max_iterations == "auto":
            if self.opt_method == "SFN":
                self.max_iterations = [100, 100]
            if self.opt_method == "GD":
                self.max_iterations = [250, 250]
        elif isinstance(configuration.max_iterations, list):
            self.max_iterations = configuration.max_iterations
        if configuration.batch_size == "auto":
            self.batch_size = 30 * self.pdim
        else:
            self.batch_size = configuration.batch_size

        Pauli_labels_loc = ["I", "X", "Y", "Z"]
        Pauli_labels_rep = [Pauli_labels_loc for _ in range(int(np.log2(self.pdim)))]
        separator = ""
        self.Pauli_labels = [separator.join(map(str, x)) for x in product(*Pauli_labels_rep)]

        std_labels_loc = ["0", "1"]
        std_labels_rep = [std_labels_loc for _ in range(int(np.log2(self.pdim)))]
        self.std_labels = [separator.join(map(str, x)) for x in product(*std_labels_rep)]

        self.y, self.J = (
            np.empty((self.num_povm, self.num_circuits)),
            np.empty((self.num_circuits, self.num_povm)),
        )  # format used by mGST
        self.bootstrap_results = List[Tuple[np.ndarray]]  # List of GST outcomes from bootstrapping
        self.raw_qc_list = List[QuantumCircuit]  # List of circuits before transpilation
        self.transpiled_qc_list = List[QuantumCircuit]  # List of transpiled circuits to be executed

    @staticmethod
    def name() -> str:
        return "compressive_GST"

    @timeit
    def generate_meas_circuits(self) -> float:
        """Generate random circuits from the gate set

        The random circuits are distributed among different depths ranging from L_MIN
        to L_MAX, both are configurable and stored in self.seq_len_list.
        No transpilation other than mapping to the desired qubits is performed,
        as the gates need to be executed axactly as described for GST to give
        meaningful results

        Returns:
            circuit_gen_transp_time: float
                The time it took to generate and transpile the circuits

        """
        start_timer_data = perf_counter()
        # Calculate number of short and long circuits
        N_short = int(np.ceil(self.num_circuits / 2))
        N_long = int(np.floor(self.num_circuits / 2))
        L_MIN, L_CUT, L_MAX = self.seq_len_list

        # Generate random circuits
        J = additional_fns.random_seq_design(self.num_gates, L_MIN, L_CUT, L_MAX, N_short, N_long)

        # Reduce circuits to exclude negative placeholder values
        gate_circuits = [list(seq[seq >= 0]) for seq in J]

        # Sequences in GST are applied to the state from right to left
        self.J = np.array(J)[:, ::-1]

        unmapped_qubits = list(np.arange(self.num_qubits))
        self.raw_qc_list = qiskit_interface.get_qiskit_circuits(
            gate_circuits, self.gate_set, self.num_qubits, unmapped_qubits
        )

        coupling_map = set_coupling_map(self.qubits, self.backend, physical_layout="fixed")

        # Perform transpilation to backend
        qcvv_logger.info(f"Will transpile all {self.num_circuits} circuits according to fixed physical layout")
        self.transpiled_qc_list, _ = perform_backend_transpilation(
            self.raw_qc_list,
            self.backend,
            self.qubits,
            coupling_map=coupling_map,
            qiskit_optim_level=0,
            optimize_sqg=False,
            drop_final_rz=False,
        )
        circuit_gen_transp_time = perf_counter() - start_timer_data

        # Saving raw and transpiled circuits in a consistent format with other benchmarks
        self.untranspiled_circuits.update({str(self.qubits): {str(self.seq_len_list[-1]): self.raw_qc_list}})
        self.transpiled_circuits.update({str(self.qubits): {str(self.seq_len_list[-1]): self.transpiled_qc_list}})
        return circuit_gen_transp_time

    def job_counts_to_mGST_format(self, result_dict):
        """Turns the dictionary of outcomes obtained from qiskit backend
            into the format which is used in mGST

        Parameters
        ----------
        result_dict: (dict of str: int)

        Returns
        -------
        y : numpy array
            2D array of measurement outcomes for sequences in J;
            Each column contains the outcome probabilities for a fixed sequence

        """
        basis_dict_list = []
        for result in result_dict:
            # Translate dictionary entries of bitstring on the full system to the decimal representation of bitstrings on the active qubits
            basis_dict = {
                entry: int("".join([entry[::-1][i] for i in range(self.num_qubits)][::-1]), 2) for entry in result
            }
            # Sort by index:
            basis_dict = dict(sorted(basis_dict.items(), key=lambda item: item[1]))
            basis_dict_list.append(basis_dict)
        y = []
        for i in range(len(result_dict)):
            row = [result_dict[i][key] for key in basis_dict_list[i]]
            if len(row) < self.num_povm:
                missing_entries = list(np.arange(self.num_povm))
                for given_entry in basis_dict_list[i].values():
                    missing_entries.remove(given_entry)
                for missing_entry in missing_entries:
                    row.insert(missing_entry, 0)  # 0 measurement outcomes in not recorded entry
            y.append(row / np.sum(row))
        y = np.array(y).T
        return y

    def run_circuits(self):
        """
        Runs circuits on simulator/hardware and collects counts using utils functions.

        Returns:
            time_submit: float
                The time it took to submit to jobs.
            time_retrieve: float
                The time it took to retrieve counts.
        """

        qcvv_logger.info(f"Now executing the corresponding circuit batch")
        # Submit all circuits to execute
        all_jobs, time_submit = submit_execute(
            {tuple(self.qubits): self.transpiled_qc_list},
            self.backend,
            self.shots,
            self.calset_id,
            max_gates_per_batch=self.max_gates_per_batch,
        )

        # Retrieve counts - the precise outputs do not matter
        all_counts, time_retrieve = retrieve_all_counts(all_jobs)
        # Save counts - ensures counts were received and can be inspected
        self.raw_data = all_counts
        self.y = self.job_counts_to_mGST_format(self.raw_data)
        # Retrieve and save all job metadata
        all_job_metadata = retrieve_all_job_metadata(all_jobs)
        self.job_meta = all_job_metadata

        return time_submit, time_retrieve

    def dataframe_to_figure(self, df, row_labels=None, col_width=2, fontsize=12):
        """Turns a pandas DataFrame into a figure
        This is needed to conform with the standard file saving routine of QCVV.

        Args:
            df: Pandas DataFrame
                A dataframe table containing GST results
            row_labels: List[str]
                The row labels for the dataframe
            precision: int
                The number of digits for entries in the dataframe
            col_width: int
                Used to control cell width in the table
            fontsize: int
                Font size of text/numbers in table cells

        Returns:
            figure: Matplotlib figure object
                A figure representing the dataframe.
        """

        if row_labels is None:
            row_labels = np.arange(df.shape[0])

        row_height = fontsize / 70 * 2
        n_cols = df.shape[1]
        n_rows = df.shape[0]
        figsize = np.array([n_cols + 1, n_rows + 1]) * np.array([col_width, row_height])

        fig, ax = plt.subplots(figsize=figsize)

        fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        data_array = (df.to_numpy(dtype="str")).copy()
        column_names = [str(column) for column in df.columns].copy()
        table = ax.table(
            cellText=data_array,
            colLabels=column_names,
            rowLabels=row_labels,
            cellLoc="center",
            colColours=["#7FA1C3" for _ in range(n_cols)],
            bbox=[0, 0, 1, 1],
        )
        table.set_fontsize(fontsize)
        table.set_figure(fig)
        return fig

    def bootstrap_errors(self, K, X, E, rho, target_mdl, parametric=False):
        """Resamples circuit outcomes a number of times and computes GST estimates for each repetition
        All results are then returned in order to compute bootstrap-error bars for GST estimates.
        Parametric bootstrapping uses the estimated gate set to create a newly sampled data set.
        Non-parametric bootstrapping uses the initial dataset and resamples according to the
        corresp. outcome probabilities.
        Each bootstrap run is initialized with the estimated gate set in order to save processing time.

        Parameters
        ----------
        K : numpy array
            Each subarray along the first axis contains a set of Kraus operators.
            The second axis enumerates Kraus operators for a gate specified by the first axis.
        X : 3D numpy array
            Array where reconstructed CPT superoperators in standard basis are stacked along the first axis.
        E : numpy array
            Current POVM estimate
        rho : numpy array
            Current initial state estimate
        target_mdl : pygsti model object
            The target gate set
        parametric : bool
            If set to True, parametric bootstrapping is used, else non-parametric bootstrapping. Default: False

        Returns
        -------
        X_array : numpy array
            Array containing all estimated gate tensors of different bootstrapping repretitions along first axis
        E_array : numpy array
            Array containing all estimated POVM tensors of different bootstrapping repretitions along first axis
        rho_array : numpy array
            Array containing all estimated initial states of different bootstrapping repretitions along first axis
        df_g_array : numpy array
            Contains gate quality measures of bootstrapping repetitions
        df_o_array : numpy array
            Contains SPAM and other quality measures of bootstrapping repetitions

        """
        if parametric:
            y = np.real(np.array([[E[i].conj() @ contract(X, j) @ rho for j in self.J] for i in range(self.num_povm)]))
        else:
            y = self.y
        X_array = np.zeros((self.bootstrap_samples, *X.shape)).astype(complex)
        E_array = np.zeros((self.bootstrap_samples, *E.shape)).astype(complex)
        rho_array = np.zeros((self.bootstrap_samples, *rho.shape)).astype(complex)
        df_g_list = []
        df_o_list = []

        for i in range(self.bootstrap_samples):
            y_sampled = additional_fns.sampled_measurements(y, self.shots).copy()
            _, X_, E_, rho_, _ = algorithm.run_mGST(
                y_sampled,
                self.J,
                self.seq_len_list[-1],
                self.num_gates,
                self.pdim**2,
                self.rank,
                self.num_povm,
                self.batch_size,
                self.shots,
                method=self.opt_method,
                max_inits=self.max_inits,
                max_iter=0,
                final_iter=self.max_iterations[1],
                threshold_multiplier=self.convergence_criteria[0],
                target_rel_prec=self.convergence_criteria[1],
                init=[K, E, rho],
                testing=False,
            )

            X_opt, E_opt, rho_opt = reporting.gauge_opt(X_, E_, rho_, target_mdl, self.weights)
            df_g, df_o = reporting.report(X_opt, E_opt, rho_opt, self.J, y_sampled, target_mdl, self.gate_labels)
            df_g_list.append(df_g.values)
            df_o_list.append(df_o.values)

            X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)

            X_array[i] = X_opt_pp
            E_array[i] = E_opt_pp
            rho_array[i] = rho_opt_pp

        return (X_array, E_array, rho_array, np.array(df_g_list), np.array(df_o_list))

    def generate_non_gate_results(self, df_o):
        """
        Creates error bars (if bootstrapping was used) and formats results for non-gate errors.
        The resulting tables are also turned into figures, so that they can be saved automatically.

        Args:
            df_o: Pandas DataFrame
                A dataframe containing the non-gate quality metrics (SPAM errors and fit quality)

        Returns:
            df_o_final: Oandas DataFrame
                The final formated results
        """
        # filename = f"{self.results_dir}{self.device_id}_{self.timestamp}_spam_errors.html"
        if self.bootstrap_samples > 0:
            _, _, _, _, df_o_array = self.bootstrap_results
            df_o_array[df_o_array == -1] = np.nan
            percentiles_o_low, percentiles_o_high = np.nanpercentile(df_o_array, [2.5, 97.5], axis=0)
            df_o_final = DataFrame(
                {
                    f"Mean TVD: estimate - data": reporting.number_to_str(
                        df_o.values[0, 1].copy(), [percentiles_o_high[0, 1], percentiles_o_low[0, 1]], precision=5
                    ),
                    f"Mean TVD: target - data": reporting.number_to_str(
                        df_o.values[0, 2].copy(), [percentiles_o_high[0, 2], percentiles_o_low[0, 2]], precision=5
                    ),
                    f"POVM - diamond dist.": reporting.number_to_str(
                        df_o.values[0, 3].copy(), [percentiles_o_high[0, 3], percentiles_o_low[0, 3]], precision=5
                    ),
                    f"State - trace dist.": reporting.number_to_str(
                        df_o.values[0, 4].copy(), [percentiles_o_high[0, 4], percentiles_o_low[0, 4]], precision=5
                    ),
                },
                index=[""],
            )
        else:
            df_o_final = DataFrame(
                {
                    f"Mean TVD: estimate - data": reporting.number_to_str(df_o.values[0, 1].copy(), precision=5),
                    f"Mean TVD: target - data": reporting.number_to_str(df_o.values[0, 2].copy(), precision=5),
                    f"POVM - diamond dist.": reporting.number_to_str(df_o.values[0, 3].copy(), precision=5),
                    f"State - trace dist.": reporting.number_to_str(df_o.values[0, 4].copy(), precision=5),
                },
                index=[""],
            )
        # fig = self.dataframe_to_figure(df_o_final, [""])#self.dataframe_to_figure(df_o_final, [""])
        # self.figures["spam_and_outcome_errors"] = fig
        return df_o_final

    def generate_unit_rank_gate_results(self, df_g, X_opt, K_target):
        """
        Produces all result tables for Kraus rank 1 estimates and turns them into figures.

        This includes parameters of the Hamiltonian generators in the Pauli basis for all gates,
        as well as the usual performance metrics (Fidelities and Diamond distances). If bootstrapping
        data is available, error bars will also be generated.

        Args:
            df_g: Pandas DataFrame
                The dataframe with properly formated results
            X_opt: 3D numpy array
                The gate set after gauge optimization
            K_target: 4D numpy array
                The Kraus operators of all target gates, used to compute distance measures.

        Returns:
            df_g_final: Pandas DataFrame
                The dataframe with properly formated results of standard gate errors
            df_g_rotation Pandas DataFrame
                A dataframe containing Hamiltonian (rotation) parameters

        """
        if self.bootstrap_samples > 0:
            X_array, E_array, rho_array, df_g_array, _ = self.bootstrap_results
            df_g_array[df_g_array == -1] = np.nan
            percentiles_g_low, percentiles_g_high = np.nanpercentile(df_g_array, [2.5, 97.5], axis=0)

            df_g_final = DataFrame(
                {
                    r"Avg. gate fidelity": [
                        reporting.number_to_str(
                            df_g.values[i, 0], [percentiles_g_high[i, 0], percentiles_g_low[i, 0]], precision=5
                        )
                        for i in range(len(self.gate_labels))
                    ],
                    r"Diamond distance": [
                        reporting.number_to_str(
                            df_g.values[i, 1], [percentiles_g_high[i, 1], percentiles_g_low[i, 1]], precision=5
                        )
                        for i in range(self.num_gates)
                    ],
                }
            )

            U_opt = reporting.phase_opt(X_opt, K_target)
            pauli_coeffs = reporting.compute_sparsest_Pauli_Hamiltonian(U_opt)

            bootstrap_pauli_coeffs = np.zeros((len(X_array), self.num_gates, self.pdim**2))
            for i, X_ in enumerate(X_array):
                X_std, _, _ = compatibility.pp2std(X_, E_array[i], rho_array[i])
                U_opt_ = reporting.phase_opt(
                    np.array([change_basis(X_std[j], "pp", "std") for j in range(self.num_gates)]), K_target
                )
                pauli_coeffs_ = reporting.compute_sparsest_Pauli_Hamiltonian(U_opt_)
                bootstrap_pauli_coeffs[i, :, :] = pauli_coeffs_
            pauli_coeffs_low, pauli_coeffs_high = np.nanpercentile(bootstrap_pauli_coeffs, [2.5, 97.5], axis=0)

            df_g_rotation = DataFrame(
                np.array(
                    [
                        [
                            reporting.number_to_str(
                                pauli_coeffs[i, j], [pauli_coeffs_high[i, j], pauli_coeffs_low[i, j]], precision=5
                            )
                            for i in range(self.num_gates)
                        ]
                        for j in range(self.pdim**2)
                    ]
                ).T
            )

            df_g_rotation.columns = [f"h_%s" % label for label in self.Pauli_labels]
            df_g_rotation.rename(index=self.gate_labels, inplace=True)

        else:
            df_g_final = DataFrame(
                {
                    "Avg. gate fidelity": [
                        reporting.number_to_str(df_g.values[i, 0], precision=5) for i in range(self.num_gates)
                    ],
                    "Diamond distance": [
                        reporting.number_to_str(df_g.values[i, 1], precision=5) for i in range(self.num_gates)
                    ],
                }
            )
            U_opt = reporting.phase_opt(X_opt, K_target)
            pauli_coeffs = reporting.compute_sparsest_Pauli_Hamiltonian(U_opt)

            df_g_rotation = DataFrame(
                np.array(
                    [
                        [reporting.number_to_str(pauli_coeffs[i, j], precision=5) for i in range(self.num_gates)]
                        for j in range(self.pdim**2)
                    ]
                ).T
            )
            df_g_rotation.columns = [f"h_%s" % label for label in self.Pauli_labels]
            df_g_final.rename(index=self.gate_labels, inplace=True)

        # fig_g = self.dataframe_to_figure(df_g_final, self.gate_labels)
        # fig_rotation = self.dataframe_to_figure(df_g_rotation, self.gate_labels)
        # self.figures["gate_errors"] = fig_g
        # self.figures["hamiltonian_parameters"] = fig_rotation
        return df_g_final, df_g_rotation

    def generate_gate_results(self, df_g, X_opt, E_opt, rho_opt, max_evals=6):
        """
        Produces all result tables for arbitrary Kraus rank estimates and turns them into figures.

        Args:
            df_g: Pandas DataFrame
                The dataframe with properly formated results
            X_opt: 3D numpy array
                The gate set after gauge optimization
            E_opt: 3D numpy array
                An array containg all the POVM elements as matrices after gauge optimization
            rho_opt: 2D numpy array
                The density matrix after gauge optmization
            max_evals: int
                The maximum number of eigenvalues of the Choi matrices which are returned.

        Returns:
            df_g_final: Pandas DataFrame
                The dataframe with properly formated results of standard gate errors
            df_g_evals Pandas DataFrame
                A dataframe containing eigenvalues of the Choi matrices for each gate

        """
        n_evals = np.min([max_evals, self.pdim**2])
        X_opt_pp, _, _ = compatibility.std2pp(X_opt, E_opt, rho_opt)
        df_g_evals = reporting.generate_Choi_EV_table(X_opt, n_evals, self.gate_labels).copy()

        if self.bootstrap_samples > 0:
            X_array, E_array, rho_array, df_g_array, _ = self.bootstrap_results
            df_g_array[df_g_array == -1] = np.nan
            percentiles_g_low, percentiles_g_high = np.nanpercentile(df_g_array, [2.5, 97.5], axis=0)
            bootstrap_unitarities = np.array([reporting.unitarities(X_array[i]) for i in range(self.bootstrap_samples)])
            percentiles_u_low, percentiles_u_high = np.nanpercentile(bootstrap_unitarities, [2.5, 97.5], axis=0)
            X_array_std = [
                compatibility.pp2std(X_array[i], E_array[i], rho_array[i])[0] for i in range(self.bootstrap_samples)
            ]
            bootstrap_evals = np.array(
                [
                    reporting.generate_Choi_EV_table(X_array_std[i], n_evals, self.gate_labels)
                    for i in range(self.bootstrap_samples)
                ]
            )
            percentiles_evals_low, percentiles_evals_high = np.nanpercentile(bootstrap_evals, [2.5, 97.5], axis=0)
            eval_strs = [
                [
                    reporting.number_to_str(
                        df_g_evals.values[i, j],
                        [percentiles_evals_high[i, j], percentiles_evals_low[i, j]],
                        precision=5,
                    )
                    for i in range(self.num_gates)
                ]
                for j in range(n_evals)
            ]

            df_g_final = DataFrame(
                {
                    r"Avg. gate fidelity": [
                        reporting.number_to_str(
                            df_g.values[i, 0], [percentiles_g_high[i, 0], percentiles_g_low[i, 0]], precision=5
                        )
                        for i in range(self.num_gates)
                    ],
                    r"Diamond distance": [
                        reporting.number_to_str(
                            df_g.values[i, 1], [percentiles_g_high[i, 1], percentiles_g_low[i, 1]], precision=5
                        )
                        for i in range(self.num_gates)
                    ],
                    r"Unitarity": [
                        reporting.number_to_str(
                            reporting.unitarities(X_opt_pp)[i],
                            [percentiles_u_high[i], percentiles_u_low[i]],
                            precision=5,
                        )
                        for i in range(self.num_gates)
                    ],
                }
            )

        else:
            df_g_final = DataFrame(
                {
                    "Avg. gate fidelity": [
                        reporting.number_to_str(df_g.values[i, 0].copy(), precision=5)
                        for i in range(len(self.gate_labels))
                    ],
                    "Diamond distance": [
                        reporting.number_to_str(df_g.values[i, 1].copy(), precision=5)
                        for i in range(len(self.gate_labels))
                    ],
                    "Unitarity": [
                        reporting.number_to_str(reporting.unitarities(X_opt_pp)[i], precision=5)
                        for i in range(len(self.gate_labels))
                    ],
                    # "Entanglemen fidelity to depol. channel": [reporting.number_to_str(reporting.eff_depol_params(X_opt_pp)[i], precision=5)
                    #                                            for i in range(len(gate_labels))],
                    # "Min. spectral distances": [number_to_str(df_g.values[i, 2], precision=5) for i in range(len(gate_labels))]
                }
            )
            eval_strs = [
                [reporting.number_to_str(df_g_evals.values[i, j].copy(), precision=5) for i in range(self.num_gates)]
                for j in range(n_evals)
            ]

        df_g_evals_final = DataFrame(eval_strs).T
        df_g_evals_final.rename(index=self.gate_labels, inplace=True)

        # fig_g = self.dataframe_to_figure(df_g_final, self.gate_labels)
        # fig_choi = self.dataframe_to_figure(df_g_evals_final, self.gate_labels)
        # self.figures["gate_errors"] = fig_g
        # self.figures["choi_evals"] = fig_choi
        return df_g_final, df_g_evals_final

    @timeit
    def execute_full_benchmark(self):
        """
        The main GST execution routine

        The following steps are executed:
            1) Generation of circuits to measure
            2) Running of said circuits on the backend
            3) Conversion of the ideal gate set into numpy arrays to be processed by mGST
            4) Optional generation of an initialization for mGST
            5) Running mGST to get first gate set estimates
            6) Gauge optimization to the target gates using pyGSTi"s gauge optimizer
            7) Generation of error measures and result tables
            8) Generation of matrix plots for the reconstructed gate set
        """

        qcvv_logger.info(f"Now generating {self.num_circuits} random circuits on qubits {self.qubits}")

        # Generate and run circuits on backend
        circuit_gen_transp_time = self.generate_meas_circuits()

        submit_time, retrieve_time = self.run_circuits()
        # data_collection_time = submit_time + retrieve_time

        start_timer_GST = perf_counter()
        K_target = qiskit_gate_to_operator(self.gate_set)
        X_target = np.einsum("ijkl,ijnm -> iknlm", K_target, K_target.conj()).reshape(
            (self.num_gates, self.pdim**2, self.pdim**2)
        )  # tensor of superoperators

        rho_target = (
            np.kron(additional_fns.basis(self.pdim, 0).T.conj(), additional_fns.basis(self.pdim, 0))
            .reshape(-1)
            .astype(np.complex128)
        )

        # Computational basis measurement:
        E_target = np.array(
            [
                np.kron(additional_fns.basis(self.pdim, i).T.conj(), additional_fns.basis(self.pdim, i)).reshape(-1)
                for i in range(self.pdim)
            ]
        ).astype(np.complex128)
        target_mdl = compatibility.arrays_to_pygsti_model(X_target, E_target, rho_target, basis="std")

        # Run mGST
        if self.from_init:
            K_init = additional_fns.perturbed_target_init(X_target, self.rank)
            init_params = [K_init, E_target, rho_target]
        else:
            init_params = None

        K, X, E, rho, _ = algorithm.run_mGST(
            self.y,
            self.J,
            self.seq_len_list[-1],
            self.num_gates,
            self.pdim**2,
            self.rank,
            self.num_povm,
            self.batch_size,
            self.shots,
            method=self.opt_method,
            max_inits=self.max_inits,
            max_iter=self.max_iterations[0],
            final_iter=self.max_iterations[1],
            threshold_multiplier=self.convergence_criteria[0],
            target_rel_prec=self.convergence_criteria[1],
            init=init_params,
            testing=False,
        )

        main_mGST_time = perf_counter() - start_timer_GST
        start_timer_gauge = perf_counter()

        # Gauge optimization
        X_opt, E_opt, rho_opt = reporting.gauge_opt(X, E, rho, target_mdl, self.weights)
        gauge_optimization_time = perf_counter() - start_timer_gauge

        # Quick report
        start_timer_report = perf_counter()
        df_g, df_o = reporting.quick_report(X_opt, E_opt, rho_opt, self.J, self.y, target_mdl, self.gate_labels)
        report_gen_time = perf_counter() - start_timer_report

        ### Bootstrap
        if self.bootstrap_samples > 0:
            self.bootstrap_results = self.bootstrap_errors(K, X, E, rho, target_mdl)

        _, df_o_full = reporting.report(X_opt, E_opt, rho_opt, self.J, self.y, target_mdl, self.gate_labels)
        df_o_final = self.generate_non_gate_results(df_o_full)

        ### Result table generation
        if self.rank == 1:
            df_g_final, df_g_rotation = self.generate_unit_rank_gate_results(df_g, X_opt, K_target)
            self.results.update({"hamiltonian_parameters": df_g_rotation.to_dict()})
        else:
            df_g_final, df_g_evals = self.generate_gate_results(df_g, X_opt, E_opt, rho_opt)
            self.results.update({"choi_evals": df_g_evals.to_dict()})

        # Saving results
        self.results.update(
            {
                "qubit_set": self.qubits,
                "quick_metrics": {"Gates": df_g.to_dict(), "Outcomes and SPAM": df_o.to_dict()},
                "full_metrics": {"Gates": df_g_final.to_dict(), "Outcomes and SPAM": df_o_final.to_dict()},
                "main_mGST_time": main_mGST_time,
                "submit_time": submit_time,
                "retrieve_time": retrieve_time,
                "gauge_optimization_time": gauge_optimization_time,
                "circuit_generation_time": circuit_gen_transp_time,
            }
        )

        self.raw_results.update(
            {
                "raw_Kraus_operators": K,
                "raw_gates": X,
                "raw_POVM": E.reshape((self.num_povm, self.pdim, self.pdim)),
                "raw_state": rho.reshape((self.pdim, self.pdim)),
                "gauge_opt_gates": X_opt,
                "gauge_opt_POVM": E_opt.reshape((self.num_povm, self.pdim, self.pdim)),
                "gauge_opt_state": rho_opt.reshape((self.pdim, self.pdim)),
                "mGST_circuits": self.J,
                "mGST_outcome_probs": self.y,
            }
        )

        ### Process matrix plots
        X_opt_pp, _, _ = compatibility.std2pp(X_opt, E_opt, rho_opt)
        X_target_pp, _, _ = compatibility.std2pp(X_target, E_target, rho_target)

        figures = figure_gen.generate_gate_err_pdf(
            "", X_opt_pp, X_target_pp, basis_labels=self.Pauli_labels, gate_labels=self.gate_labels, return_fig=True
        )
        for i, figure in enumerate(figures):
            self.figures[f"process_mat_plot_%i" % i] = figure

        self.figures["SPAM_matrices_real"] = figure_gen.generate_spam_err_std_pdf(
            "",
            E_opt.real,
            rho_opt.real,
            E_target.real,
            rho_target.real,
            basis_labels=self.std_labels,
            title=f"Real part of state and measurement effects in the standard basis",
            return_fig=True,
        )
        self.figures["SPAM_matrices_imag"] = figure_gen.generate_spam_err_std_pdf(
            "",
            E_opt.imag,
            rho_opt.imag,
            E_target.imag,
            rho_target.imag,
            basis_labels=self.std_labels,
            title=f"Imaginary part of state and measurement effects in the standard basis",
            return_fig=True,
        )
        # plt.show()
        plt.close("all")

        ## Result display
        total_pp_time = main_mGST_time + gauge_optimization_time + report_gen_time
        qcvv_logger.info(
            f"Results for {self.backend.name} with qubits "
            f"{self.qubits}\nTotal GST post-processing time: {total_pp_time / 60:.2f} min\nOverview results: \n"
            + df_g.to_string()
            + f"\n"
            + df_o.to_string()
        )


class GSTConfiguration(BenchmarkConfigurationBase):
    """Compressive GST configuration base.

    Attributes:
        benchmark (Type[Benchmark]): GHZBenchmark
        qubit_layouts (Sequence[Sequence[int]]): A sequence (e.g., Tuple or List) of sequences of
            physical qubit layouts, as specified by integer labels, where the benchmark is meant to be run.
        gate_set (Union[str, List[Type[QuantumCircuit]]]): The gate set in given either as a list of qiskit quantum
            cirucuits, or as one of the predefined gate sets "1QXYI", "2QXYCZ", "2QXYCZ_extended", "3QXYCZ".
            A gate set should be tomographically complete, meaning from the specified gates and the vacuum state, one
            should be able to prepare states that form a frame for the state space. A practical sufficient condition
            is that the gate set is able the generate all combinations of local Pauli eigenstates.
        num_circuits (int): How many random circuits are generated from the gate set. Guidelines on choosing this value:
            - At least 50 circuits for a single qubit gate set with 3 gates.
            - At least 400 circuits for a two qubit gate set with 6 gates.
            - At least 2000 circuits for a three qubit gate set with 9 gates.
            The number of random circuits needed is expected to grow linearly in the number of gates and exponentially
            in the number of qubits.
        rank (int): The Kraus rank of the reconstruction. Choose rank=1 for coherent error analysis and rank<=dim**2
            generally.
        shots (int): The number of measurement shots per circuit.
            * Default: 1024
        gate_labels (Union[Dict, None]): Names of the gates in the gate set. Used for plots and result tables.
            * Default: None
        seq_len_list (list[int]): Three numbers controling the depth of the random sequences. The first is the minimal
            depth, the last is the maximal depth and the middle number specifies a cutoff depth below which all possible
            sequences are selected.
            * Default: [1, 8, 14]
        from_init (bool): Whether the target gate set is used as an initialization to the mGST algorithm.
            * Default: True
        max_inits (int): If from_init = False, random initial points are tried and this parameter limits the amount of
            retries.
            * Default: 20
        opt_method (str): Which optimization method is used, can be either of "GD" or "SFN", for gradient descent or
            saddle free Newton, respectively.
            * Default: "auto" (Method is automatically selected based on qubit count and rank)
        max_iterations (Union[str, List[int]]): How many iterations to run the optimization algorithm for. Accepted
            values are either "auto" or a list of two integers. The first specifies the number of iterations for the
            optimization on batches or circuit data, while the second specifies the number of iterations on the full
            data for all circuits.
            * Default: "auto" (Numbers are automatically selected based on qubit count and rank)
        convergence_criteria (Union[str, List[float]]): Two parameters which determine when the optimization algorithm
            terminates. The first is a multiplier which specifies how close the cost function should get to a threshold
            for the given shot noise in order for the optimization to be considered "successful". Values in [2,10] are
            usually sensible. The second parameter sets the relative change in cost function between
            two consecutive iterations, below which the algorithm terminates.
            * Default: [4, 1e-4]
        batch_size (Union[str, int]): The number of circuits per batch in the optimization. This hyperparamters is
            automatically set and determines the convergence behaviour. A smaller batch size reduces runtime but can
            lead to erratic jumps in parameter space and lack of convergence.
            * Default: "auto"
        bootstrap_samples (int): The number of times the optimization algorithm is repeated on fake data to estimate
            the uncertainty via bootstrapping.
    """

    benchmark: Type[BenchmarkBase] = compressive_GST
    qubits: List[Union[int, List[int]]]
    gate_set: Union[str, List[Type[QuantumCircuit]]]
    num_circuits: int
    shots: int
    rank: int
    gate_labels: Union[Dict, None] = None
    seq_len_list: list = [1, 8, 14]
    from_init: bool = True
    max_inits: int = 10
    opt_method: str = "auto"
    max_iterations: Union[str, List[int]] = "auto"
    convergence_criteria: List[float] = [4, 1e-4]
    batch_size: Union[str, int] = "auto"
    bootstrap_samples: int = 0


def parse_gate_set(configuration):
    """
    Handles different gate set inputs and produces a valid gate set

    Args:
        configuration: BenchmarkConfigurationBase
            Configuration class containing variables

    Returns:
        gate_set: List[QuantumCircuit]
            A list of gates defined as quantum circuit objects
        gate_labels: Dict[int: str]
            A dictionary with gate names
        num_gates: int
            The number of gates in the gate set

    """
    if isinstance(configuration.gate_set, str) and configuration.gate_set not in [
        "1QXYI",
        "2QXYCZ",
        "2QXYCZ_extended",
        "3QXYCZ",
    ]:
        raise ValueError(
            "No gate set of the specified name is implemented, please choose among "
            "1QXYI, 2QXYCZ, 2QXYCZ_extended, 3QXYCZ."
        )
    if configuration.gate_set in ["1QXYI", "2QXYCZ", "2QXYCZ_extended", "3QXYCZ"]:
        gate_set, gate_labels = create_predefined_gate_set(configuration.gate_set, configuration.qubits)
        num_gates = len(gate_set)
        return gate_set, gate_labels, num_gates

    if isinstance(configuration.gate_set, list):
        gate_set = configuration.gate_set
        num_gates = len(gate_set)
        if configuration.gate_labels is None:
            gate_labels = {i: f"Gate %i" % i for i in range(num_gates)}
        else:
            if configuration.gate_labels:
                if len(configuration.gate_labels) != num_gates:
                    raise ValueError(
                        f"The number of gate labels (%i) does not match the number of gates (%i)"
                        % (len(configuration.gate_labels), num_gates)
                    )
                gate_labels = configuration.gate_labels
        return gate_set, gate_labels, num_gates

    raise ValueError(
        f"Invalid gate set, choose among 1QXYI, 2QXYCZ, 2QXYCZ_extended,"
        f" 3QXYCZ or provide a list of Qiskti circuits to define the gates."
    )


def create_predefined_gate_set(gate_set, qubits) -> Tuple[List[QuantumCircuit], Dict]:
    """Create a list of quantum circuits corresponding to a predefined gate set.

    The circuits are assigned to the specified qubits on the backend only during transipilation, so the qubit labels
    at this stage may not represent the actual qubit labels on the backend.
    Args:
        gate_set: str
            The name of the gate set
        qubits: List[int]
            The qubits on the backend where the gates are defined on
    Returns:
        gates: List[QuantumCircuit]
            The gate set as a list of circuits
        gate_labels_dict: Dict[str]
            The names of gates, i.e. "Rx(pi/2)" for a pi/2 rotation around the x-axis.

    """
    num_qubits = len(qubits)
    qubit_mapping = dict(enumerate(qubits))
    unmapped_qubits = list(np.arange(num_qubits))

    if gate_set == "1QXYI":
        gate_list = [RGate(1e-10, 0), RGate(0.5 * np.pi, 0), RGate(0.5 * np.pi, np.pi / 2)]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(len(gate_list))]
        gate_qubits = [[0], [0], [0]]
        for i, gate in enumerate(gate_list):
            gates[i].append(gate, gate_qubits[i])
        gate_labels = ["Idle", "Rx(pi/2)", "Ry(pi/2)"]
    elif gate_set == "2QXYCZ":
        gate_qubits = [[0], [1], [0], [1], [0, 1]]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(5)]
        gates[0].append(RGate(0.5 * np.pi, 0), [0])
        gates[1].append(RGate(0.5 * np.pi, 0), [1])
        gates[2].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[3].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[4].append(CZGate(), [0, 1])
        gate_labels = ["Rx(pi/2)", "Rx(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "CZ"]
    elif gate_set == "2QXYCZ_extended":
        gate_qubits = [[0], [1], [0], [1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(9)]
        gates[0].append(RGate(0.5 * np.pi, 0), [0])
        gates[1].append(RGate(0.5 * np.pi, 0), [1])
        gates[2].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[3].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[4].append(RGate(0.5 * np.pi, 0), [0])
        gates[4].append(RGate(0.5 * np.pi, 0), [1])
        gates[5].append(RGate(0.5 * np.pi, 0), [0])
        gates[5].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[6].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[6].append(RGate(0.5 * np.pi, 0), [1])
        gates[7].append(RGate(0.5 * np.pi, np.pi / 2), [0])
        gates[7].append(RGate(0.5 * np.pi, np.pi / 2), [1])
        gates[8].append(CZGate(), [[0], [1]])
        gate_labels = [
            "Rx(pi/2)",
            "Rx(pi/2)",
            "Ry(pi/2)",
            "Ry(pi/2)",
            "Rx(pi/2)--Rx(pi/2)",
            "Rx(pi/2)--Ry(pi/2)",
            "Ry(pi/2)--Rx(pi/2)",
            "Ry(pi/2)--Ry(pi/2)",
            "CZ",
        ]
    elif gate_set == "3QXYCZ":
        gate_list = [
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, 0),
            RGate(0.5 * np.pi, np.pi / 2),
            RGate(0.5 * np.pi, np.pi / 2),
            RGate(0.5 * np.pi, np.pi / 2),
            CZGate(),
            CZGate(),
        ]
        gates = [QuantumCircuit(num_qubits, 0) for _ in range(len(gate_list))]
        gate_qubits = [[0], [1], [2], [0], [1], [2], [0, 1], [0, 2]]
        for i, gate in enumerate(gate_list):
            gates[i].append(gate, gate_qubits[i])
        gate_labels = ["Rx(pi/2)", "Rx(pi/2)", "Rx(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "Ry(pi/2)", "CZ", "CZ"]
    gates = add_idle_gates(gates, unmapped_qubits, gate_qubits)
    gates = [remove_idle_wires(qc) for qc in gates]
    gate_qubits_mapped = [[qubit_mapping[i] for i in qlist] for qlist in gate_qubits]
    gate_labels = [gate_labels[i] + ":" + str(gate_qubits_mapped[i]) for i in range(len(gates))]
    gate_label_dict = {i: gate_labels[i] for i in range(len(gate_labels))}
    return gates, gate_label_dict
