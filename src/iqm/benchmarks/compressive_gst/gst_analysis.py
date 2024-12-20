"""
Data analysis code for compressive gate set tomography
"""

from itertools import product
from time import perf_counter
from typing import Any, List, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from numpy import ndarray
import numpy as np
from pandas import DataFrame
from pygsti.models.model import Model
from pygsti.tools import change_basis
import xarray as xr

from iqm.benchmarks.benchmark_definition import (
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
from mGST import additional_fns, algorithm, compatibility
from mGST.low_level_jit import contract
from mGST.qiskit_interface import qiskit_gate_to_operator
from mGST.reporting import figure_gen, reporting


def dataframe_to_figure(
    df: DataFrame, row_labels: Union[List[str], None] = None, col_width: float = 2, fontsize: int = 12
) -> Figure:
    """Turns a pandas DataFrame into a figure
    This is needed to conform with the standard file saving routine of QCVV.

    Args:
        df: Pandas DataFrame
            A dataframe table containing GST results
        row_labels: List[str]
            The row labels for the dataframe
        col_width: int
            Used to control cell width in the table
        fontsize: int
            Font size of text/numbers in table cells

    Returns:
        figure: Matplotlib figure object
            A figure representing the dataframe.
    """

    if row_labels is None:
        row_labels = list(np.arange(df.shape[0]))

    row_height = fontsize / 70 * 2
    n_cols = df.shape[1]
    n_rows = df.shape[0]
    figsize = np.array([n_cols + 1, n_rows + 1]) * np.array([col_width, row_height])

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")
    data_array = (df.to_numpy(dtype="str")).copy()
    column_names = df.columns.tolist()
    table = ax.table(
        cellText=data_array,
        colLabels=column_names,
        rowLabels=row_labels,
        cellLoc="center",
        colColours=["#7FA1C3" for _ in range(n_cols)],
        bbox=Bbox([[0, 0], [1, 1]]),
    )
    table.set_fontsize(fontsize)
    table.set_figure(fig)
    return fig


def bootstrap_errors(
    dataset: xr.Dataset,
    y: ndarray,
    K: ndarray,
    X: ndarray,
    E: ndarray,
    rho: ndarray,
    target_mdl: Model,
    identifier: str,
    parametric: bool = False,
) -> tuple[Any, Any, Any, Any, Any]:
    """Resamples circuit outcomes a number of times and computes GST estimates for each repetition
    All results are then returned in order to compute bootstrap-error bars for GST estimates.
    Parametric bootstrapping uses the estimated gate set to create a newly sampled data set.
    Non-parametric bootstrapping uses the initial dataset and resamples according to the
    corresp. outcome probabilities.
    Each bootstrap run is initialized with the estimated gate set in order to save processing time.

    Parameters
    ----------
    dataset: xarray.Dataset
        A dataset containing counts from the experiment and configurations
    qubit_layout: List[int]
        The list of qubits for the current GST experiment
    y: ndarray
        The circuit outcome probabilities as a num_povm x num_circuits array
    K : ndarray
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    X : 3D ndarray
        Array where reconstructed CPT superoperators in standard basis are stacked along the first axis.
    E : ndarray
        Current POVM estimate
    rho : ndarray
        Current initial state estimate
    target_mdl : pygsti model object
        The target gate set
    identifier : str
        The string identifier of the current benchmark
    parametric : bool
        If set to True, parametric bootstrapping is used, else non-parametric bootstrapping. Default: False

    Returns
    -------
    X_array : ndarray
        Array containing all estimated gate tensors of different bootstrapping repetitions along first axis
    E_array : ndarray
        Array containing all estimated POVM tensors of different bootstrapping repetitions along first axis
    rho_array : ndarray
        Array containing all estimated initial states of different bootstrapping repetitions along first axis
    df_g_array : ndarray
        Contains gate quality measures of bootstrapping repetitions
    df_o_array : ndarray
        Contains SPAM and other quality measures of bootstrapping repetitions

    """
    if parametric:
        y = np.real(
            np.array(
                [
                    [E[i].conj() @ contract(X, j) @ rho for j in dataset.attrs["J"]]
                    for i in range(dataset.attrs["num_povm"])
                ]
            )
        )
    X_array = np.zeros((dataset.attrs["bootstrap_samples"], *X.shape)).astype(complex)
    E_array = np.zeros((dataset.attrs["bootstrap_samples"], *E.shape)).astype(complex)
    rho_array = np.zeros((dataset.attrs["bootstrap_samples"], *rho.shape)).astype(complex)
    df_g_list = []
    df_o_list = []

    for i in range(dataset.attrs["bootstrap_samples"]):
        y_sampled = additional_fns.sampled_measurements(y, dataset.attrs["shots"]).copy()
        _, X_, E_, rho_, _ = algorithm.run_mGST(
            y_sampled,
            dataset.attrs["J"],
            dataset.attrs["seq_len_list"][-1],
            dataset.attrs["num_gates"],
            dataset.attrs["pdim"] ** 2,
            dataset.attrs["rank"],
            dataset.attrs["num_povm"],
            dataset.attrs["batch_size"],
            dataset.attrs["shots"],
            method=dataset.attrs["opt_method"],
            max_inits=dataset.attrs["max_inits"],
            max_iter=0,
            final_iter=dataset.attrs["max_iterations"][1],
            threshold_multiplier=dataset.attrs["convergence_criteria"][0],
            target_rel_prec=dataset.attrs["convergence_criteria"][1],
            init=[K, E, rho],
            testing=False,
        )

        X_opt, E_opt, rho_opt = reporting.gauge_opt(X_, E_, rho_, target_mdl, dataset.attrs[f"gauge_weights"])
        df_g, df_o = reporting.report(
            X_opt,
            E_opt,
            rho_opt,
            dataset.attrs["J"],
            y_sampled,
            target_mdl,
            dataset.attrs["gate_labels"][identifier],
        )
        df_g_list.append(df_g.values)
        df_o_list.append(df_o.values)

        X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)

        X_array[i] = X_opt_pp
        E_array[i] = E_opt_pp
        rho_array[i] = rho_opt_pp

    return X_array, E_array, rho_array, np.array(df_g_list), np.array(df_o_list)


def generate_non_gate_results(
    dataset: xr.Dataset, qubit_layout: List[int], df_o: DataFrame
) -> tuple[DataFrame, Figure]:
    """
    Creates error bars (if bootstrapping was used) and formats results for non-gate errors.
    The resulting tables are also turned into figures, so that they can be saved automatically.

    Args:
        dataset: xr.Dataset
            A dataset containing counts from the experiment and configurations
        qubit_layout: List[int]
                The list of qubits for the current GST experiment
        df_o: Pandas DataFrame
            A dataframe containing the non-gate quality metrics (SPAM errors and fit quality)

    Returns:
        df_o_final: Pandas DataFrame
            The final formated results
    """
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    if dataset.attrs["bootstrap_samples"] > 0:
        _, _, _, _, df_o_array = dataset.attrs["results_layout_" + identifier]["bootstrap_data"]
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
    fig = dataframe_to_figure(df_o_final, [""])  # dataframe_to_figure(df_o_final, [""])
    return df_o_final, fig


def generate_unit_rank_gate_results(
    dataset: xr.Dataset, qubit_layout: List[int], df_g: DataFrame, X_opt: ndarray, K_target: ndarray
) -> Tuple[DataFrame, DataFrame, Figure, Figure]:
    """
    Produces all result tables for Kraus rank 1 estimates and turns them into figures.

    This includes parameters of the Hamiltonian generators in the Pauli basis for all gates,
    as well as the usual performance metrics (Fidelities and Diamond distances). If bootstrapping
    data is available, error bars will also be generated.

    Args:
        dataset: xarray.Dataset
            A dataset containing counts from the experiment and configurations
        qubit_layout: List[int]
            The list of qubits for the current GST experiment
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
        fig_g: Figure
            A table in Figure format of gate results (fidelities etc.)
        fig_rotation: Figure
            A table in Figure format of gate Hamiltonian parameters


    """
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    pauli_labels = generate_basis_labels(dataset.attrs["pdim"], basis="Pauli")
    if dataset.attrs["bootstrap_samples"] > 0:
        X_array, E_array, rho_array, df_g_array, _ = dataset.attrs["results_layout_" + identifier]["bootstrap_data"]
        df_g_array[df_g_array == -1] = np.nan
        percentiles_g_low, percentiles_g_high = np.nanpercentile(df_g_array, [2.5, 97.5], axis=0)

        df_g_final = DataFrame(
            {
                r"Avg. gate fidelity": [
                    reporting.number_to_str(
                        df_g.values[i, 0], [percentiles_g_high[i, 0], percentiles_g_low[i, 0]], precision=5
                    )
                    for i in range(len(dataset.attrs["gate_labels"][identifier]))
                ],
                r"Diamond distance": [
                    reporting.number_to_str(
                        df_g.values[i, 1], [percentiles_g_high[i, 1], percentiles_g_low[i, 1]], precision=5
                    )
                    for i in range(dataset.attrs["num_gates"])
                ],
            }
        )

        U_opt = reporting.phase_opt(X_opt, K_target)
        pauli_coeffs = reporting.compute_sparsest_Pauli_Hamiltonian(U_opt)

        bootstrap_pauli_coeffs = np.zeros((len(X_array), dataset.attrs["num_gates"], dataset.attrs["pdim"] ** 2))
        for i, X_ in enumerate(X_array):
            X_std, _, _ = compatibility.pp2std(X_, E_array[i], rho_array[i])
            U_opt_ = reporting.phase_opt(
                np.array([change_basis(X_std[j], "pp", "std") for j in range(dataset.attrs["num_gates"])]), K_target
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
                        for i in range(dataset.attrs["num_gates"])
                    ]
                    for j in range(dataset.attrs["pdim"] ** 2)
                ]
            ).T
        )

        df_g_rotation.columns = [f"h_%s" % label for label in pauli_labels]
        df_g_rotation.rename(index=dataset.attrs["gate_labels"][identifier], inplace=True)

    else:
        df_g_final = DataFrame(
            {
                "Avg. gate fidelity": [
                    reporting.number_to_str(df_g.values[i, 0], precision=5) for i in range(dataset.attrs["num_gates"])
                ],
                "Diamond distance": [
                    reporting.number_to_str(df_g.values[i, 1], precision=5) for i in range(dataset.attrs["num_gates"])
                ],
            }
        )
        U_opt = reporting.phase_opt(X_opt, K_target)
        pauli_coeffs = reporting.compute_sparsest_Pauli_Hamiltonian(U_opt)

        df_g_rotation = DataFrame(
            np.array(
                [
                    [
                        reporting.number_to_str(pauli_coeffs[i, j], precision=5)
                        for i in range(dataset.attrs["num_gates"])
                    ]
                    for j in range(dataset.attrs["pdim"] ** 2)
                ]
            ).T
        )
        df_g_rotation.columns = [f"h_%s" % label for label in pauli_labels]
        df_g_rotation.rename(index=dataset.attrs["gate_labels"][identifier], inplace=True)
        df_g_final.rename(index=dataset.attrs["gate_labels"][identifier], inplace=True)

    fig_g = dataframe_to_figure(df_g_final, dataset.attrs["gate_labels"][identifier])
    fig_rotation = dataframe_to_figure(df_g_rotation, dataset.attrs["gate_labels"][identifier])
    return df_g_final, df_g_rotation, fig_g, fig_rotation


def generate_gate_results(
    dataset: xr.Dataset,
    qubit_layout: List[int],
    df_g: DataFrame,
    X_opt: ndarray,
    E_opt: ndarray,
    rho_opt: ndarray,
    max_evals: int = 6,
) -> Tuple[DataFrame, DataFrame, Figure, Figure]:
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
        df_g_evals_final Pandas DataFrame
            A dataframe containing eigenvalues of the Choi matrices for all gates
        fig_g: Figure
            A table in Figure format of gate results (fidelities etc.)
        fig_choi: Figure
            A table in Figure format of eigenvalues of the Choi matrices of all gates

    """
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    n_evals = np.min([max_evals, dataset.attrs["pdim"] ** 2])
    X_opt_pp, _, _ = compatibility.std2pp(X_opt, E_opt, rho_opt)
    df_g_evals = reporting.generate_Choi_EV_table(X_opt, n_evals, dataset.attrs["gate_labels"][identifier])

    if dataset.attrs["bootstrap_samples"] > 0:
        X_array, E_array, rho_array, df_g_array, _ = dataset.attrs["results_layout_" + identifier]["bootstrap_data"]
        df_g_array[df_g_array == -1] = np.nan
        percentiles_g_low, percentiles_g_high = np.nanpercentile(df_g_array, [2.5, 97.5], axis=0)
        bootstrap_unitarities = np.array(
            [reporting.unitarities(X_array[i]) for i in range(dataset.attrs["bootstrap_samples"])]
        )
        percentiles_u_low, percentiles_u_high = np.nanpercentile(bootstrap_unitarities, [2.5, 97.5], axis=0)
        X_array_std = [
            compatibility.pp2std(X_array[i], E_array[i], rho_array[i])[0]
            for i in range(dataset.attrs["bootstrap_samples"])
        ]
        bootstrap_evals = np.array(
            [
                reporting.generate_Choi_EV_table(X_array_std[i], n_evals, dataset.attrs["gate_labels"][identifier])
                for i in range(dataset.attrs["bootstrap_samples"])
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
                for i in range(dataset.attrs["num_gates"])
            ]
            for j in range(n_evals)
        ]

        df_g_final = DataFrame(
            {
                r"Avg. gate fidelity": [
                    reporting.number_to_str(
                        df_g.values[i, 0], [percentiles_g_high[i, 0], percentiles_g_low[i, 0]], precision=5
                    )
                    for i in range(dataset.attrs["num_gates"])
                ],
                r"Diamond distance": [
                    reporting.number_to_str(
                        df_g.values[i, 1], [percentiles_g_high[i, 1], percentiles_g_low[i, 1]], precision=5
                    )
                    for i in range(dataset.attrs["num_gates"])
                ],
                r"Unitarity": [
                    reporting.number_to_str(
                        reporting.unitarities(X_opt_pp)[i],
                        [percentiles_u_high[i], percentiles_u_low[i]],
                        precision=5,
                    )
                    for i in range(dataset.attrs["num_gates"])
                ],
            }
        )

    else:
        df_g_final = DataFrame(
            {
                "Avg. gate fidelity": [
                    reporting.number_to_str(df_g.values[i, 0].copy(), precision=5)
                    for i in range(len(dataset.attrs["gate_labels"][identifier]))
                ],
                "Diamond distance": [
                    reporting.number_to_str(df_g.values[i, 1].copy(), precision=5)
                    for i in range(len(dataset.attrs["gate_labels"][identifier]))
                ],
                "Unitarity": [
                    reporting.number_to_str(reporting.unitarities(X_opt_pp)[i], precision=5)
                    for i in range(len(dataset.attrs["gate_labels"][identifier]))
                ],
                # "Entanglemen fidelity to depol. channel": [reporting.number_to_str(reporting.eff_depol_params(X_opt_pp)[i], precision=5)
                #                                            for i in range(len(gate_labels))],
                # "Min. spectral distances": [number_to_str(df_g.values[i, 2], precision=5) for i in range(len(gate_labels))]
            }
        )
        eval_strs = [
            [
                reporting.number_to_str(df_g_evals.values[i, j].copy(), precision=5)
                for i in range(dataset.attrs["num_gates"])
            ]
            for j in range(n_evals)
        ]

    df_g_evals_final = DataFrame(eval_strs).T
    df_g_evals_final.rename(index=dataset.attrs["gate_labels"][identifier], inplace=True)

    fig_g = dataframe_to_figure(df_g_final, dataset.attrs["gate_labels"][identifier])
    fig_choi = dataframe_to_figure(df_g_evals_final, dataset.attrs["gate_labels"][identifier])
    return df_g_final, df_g_evals_final, fig_g, fig_choi


def generate_basis_labels(pdim: int, basis: Union[str, None] = None) -> List[str]:
    """Generate a list of labels for the Pauli basis or the standard basis

    Args:
        pdim: int
            Physical dimension
        basis: str
            Which basis the labels correspond to, currently default is standard basis and "Pauli" can be choose
            for Pauli basis labels like "II", "IX", "XX", ...

    Returns:
        labels: List[str]
            A list of all string combinations for the given dimension and basis
    """
    separator = ""
    if basis == "Pauli":
        pauli_labels_loc = ["I", "X", "Y", "Z"]
        pauli_labels_rep = [pauli_labels_loc for _ in range(int(np.log2(pdim)))]
        labels = [separator.join(map(str, x)) for x in product(*pauli_labels_rep)]
    else:
        std_labels_loc = ["0", "1"]
        std_labels_rep = [std_labels_loc for _ in range(int(np.log2(pdim)))]
        labels = [separator.join(map(str, x)) for x in product(*std_labels_rep)]

    return labels


def result_str_to_floats(result_str: str, err: str) -> Tuple[float, float]:
    """Converts formated string results from mgst to float (value, uncertainty) pairs

    Args:
        result_str: str
            The value of a result parameter formated as str
        err: str
            The error interval of the parameters

    Returns:
        value: float
            The parameter value as floar
        uncertainty: float
            A single uncertainty value
    """
    if err:
        value = float(result_str.split("[")[0])
        rest = result_str.split("[")[1].split(",")
        uncertainty = float(rest[1][:-1]) - float(rest[0])
        return value, uncertainty
    return float(result_str), np.NaN


def pandas_results_to_observations(
    dataset: xr.Dataset, df_g: DataFrame, df_o: DataFrame, identifier: BenchmarkObservationIdentifier
) -> List[BenchmarkObservation]:
    """Converts high level GST results from a pandas Dataframe to a simple observation dictionary

    Args:
    dataset: xarray.Dataset
        A dataset containing counts from the experiment and configurations
    qubit_layout: List[int]
        The list of qubits for the current GST experiment
    df_g: Pandas DataFrame
        The dataframe with properly formated gate results
    df_o: Pandas DataFrame
        The dataframe with properly formated non-gate results like SPAM error measures or fit quality.
    identifier: BenchmarkObservationIdentifier
        An identifier object for the current GST run

    Returns:
    observation_list: List[BenchmarkObservation]
        List of observations converted from the pandas dataframes
    """
    observation_list: list[BenchmarkObservation] = []
    err = dataset.attrs["bootstrap_samples"] > 0
    for idx, gate_label in enumerate(dataset.attrs["gate_labels"][identifier.string_identifier].values()):
        observation_list.extend(
            [
                BenchmarkObservation(
                    name=f"{name} - {gate_label}",
                    identifier=identifier,
                    value=result_str_to_floats(df_g[name].iloc[idx], err)[0],
                    uncertainty=result_str_to_floats(df_g[name].iloc[idx], err)[1],
                )
                for name in df_g.columns.tolist()
            ]
        )
    observation_list.extend(
        [
            BenchmarkObservation(
                name=f"{name}",
                identifier=identifier,
                value=result_str_to_floats(df_o[name].iloc[0], err)[0],
                uncertainty=result_str_to_floats(df_o[name].iloc[0], err)[1],
            )
            for name in df_o.columns.tolist()
        ]
    )
    return observation_list


def dataset_counts_to_mgst_format(dataset: xr.Dataset, qubit_layout: List[int]) -> ndarray:
    """Turns the dictionary of outcomes obtained from qiskit backend
        into the format which is used in mGST

    Args:
        dataset: xarray.Dataset
            A dataset containing counts from the experiment and configurations
        qubit_layout: List[int]
            The list of qubits for the current GST experiment

    Returns
    -------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence

    """
    num_qubits = len(qubit_layout)
    num_povm = dataset.attrs["num_povm"]
    y_list = []
    for run_index in range(dataset.attrs["num_circuits"]):
        # result = dataset[f"{qubit_layout}_counts_{run_index}"].to_dict()
        result = dataset[f"{qubit_layout}_state_{run_index}"].data.tolist()
        basis_dict = {entry: int("".join([entry[::-1][i] for i in range(num_qubits)][::-1]), 2) for entry in result}
        # Sort by index:
        basis_dict = dict(sorted(basis_dict.items(), key=lambda item: item[1]))

        counts_normalized = (
            dataset[f"{qubit_layout}_counts_{run_index}"] / dataset[f"{qubit_layout}_counts_{run_index}"].sum()
        )
        row = [counts_normalized.loc[key].data for key in basis_dict]
        # row = [result[key] for key in basis_dict]
        if len(row) < num_povm:
            missing_entries = list(np.arange(num_povm))
            for given_entry in basis_dict.values():
                missing_entries.remove(given_entry)
            for missing_entry in missing_entries:
                row.insert(missing_entry, 0)  # 0 measurement outcomes in not recorded entry
        y_list.append(row)
    y = np.array(y_list).T
    return y


def run_mGST_wrapper(
    dataset: xr.Dataset, y: ndarray
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """Wrapper function for mGST algorithm execution which prepares an initialization and sets the alg. parameters

    Args:
        dataset: xarray.Dataset
            A dataset containing counts from the experiment and configurations
        y: ndarray
            The circuit outcome probabilities as a num_povm x num_circuits array

    Returns:
        K : ndarray
            Kraus estimate array where each subarray along the first axis contains a set of Kraus operators.
            The second axis enumerates Kraus operators for a gate specified by the first axis.
        X : ndarray
            Superoperator estimate array where reconstructed CPT superoperators in
            standard basis are stacked along the first axis.
        E : ndarray
            Current POVM estimate
        rho : ndarray
            Current initial state estimate
        K_target : ndarray
            Target gate Kraus array where each subarray along the first axis contains a set of Kraus operators.
            The second axis enumerates Kraus operators for a gate specified by the first axis.
        X_target : ndarray
            Target gate superoperator estimate array where reconstructed CPT superoperators in
            standard basis are stacked along the first axis.
        E_target : ndarray
            Target POVM
        rho_target : ndarray
            Target initial state
    """

    K_target = qiskit_gate_to_operator(dataset.attrs["gate_set"])
    X_target = np.einsum("ijkl,ijnm -> iknlm", K_target, K_target.conj()).reshape(
        (dataset.attrs["num_gates"], dataset.attrs["pdim"] ** 2, dataset.attrs["pdim"] ** 2)
    )  # tensor of superoperators

    rho_target = (
        np.kron(additional_fns.basis(dataset.attrs["pdim"], 0).T.conj(), additional_fns.basis(dataset.attrs["pdim"], 0))
        .reshape(-1)
        .astype(np.complex128)
    )

    # Computational basis measurement:
    E_target = np.array(
        [
            np.kron(
                additional_fns.basis(dataset.attrs["pdim"], i).T.conj(), additional_fns.basis(dataset.attrs["pdim"], i)
            ).reshape(-1)
            for i in range(dataset.attrs["pdim"])
        ]
    ).astype(np.complex128)

    # Run mGST
    if dataset.attrs["from_init"]:
        K_init = additional_fns.perturbed_target_init(X_target, dataset.attrs["rank"])
        init_params = [K_init, E_target, rho_target]
    else:
        init_params = None

    K, X, E, rho, _ = algorithm.run_mGST(
        y,
        dataset.attrs["J"],
        dataset.attrs["seq_len_list"][-1],
        dataset.attrs["num_gates"],
        dataset.attrs["pdim"] ** 2,
        dataset.attrs["rank"],
        dataset.attrs["num_povm"],
        dataset.attrs["batch_size"],
        dataset.attrs["shots"],
        method=dataset.attrs["opt_method"],
        max_inits=dataset.attrs["max_inits"],
        max_iter=dataset.attrs["max_iterations"][0],
        final_iter=dataset.attrs["max_iterations"][1],
        threshold_multiplier=dataset.attrs["convergence_criteria"][0],
        target_rel_prec=dataset.attrs["convergence_criteria"][1],
        init=init_params,
        testing=dataset.attrs["testing"],
    )

    return K, X, E, rho, K_target, X_target, E_target, rho_target


def mgst_analysis(run: BenchmarkRunResult) -> BenchmarkAnalysisResult:
    """Analysis function for compressive GST

    Args:
        run: BenchmarkRunResult
            A BenchmarkRunResult instance storing the dataset
    Returns:
        result: BenchmarkAnalysisResult
            An BenchmarkAnalysisResult instance with the updated dataset, as well as plots and observations
    """
    dataset = run.dataset
    pdim = dataset.attrs["pdim"]
    plots = {}
    for i, qubit_layout in enumerate(dataset.attrs["qubit_layouts"]):
        identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier

        # Computing circuit outcome probabilities from counts
        y = dataset_counts_to_mgst_format(dataset, qubit_layout)

        # Main GST reconstruction
        start_timer = perf_counter()
        K, X, E, rho, K_target, X_target, E_target, rho_target = run_mGST_wrapper(dataset, y)
        main_gst_time = perf_counter() - start_timer

        # Gauge optimization
        start_timer = perf_counter()
        target_mdl = compatibility.arrays_to_pygsti_model(X_target, E_target, rho_target, basis="std")
        X_opt, E_opt, rho_opt = reporting.gauge_opt(X, E, rho, target_mdl, dataset.attrs[f"gauge_weights"])
        gauge_optimization_time = perf_counter() - start_timer

        # Quick report
        df_g, _ = reporting.quick_report(
            X_opt, E_opt, rho_opt, dataset.attrs["J"], y, target_mdl, dataset.attrs["gate_labels"][identifier]
        )

        # Gate set in the Pauli basis
        X_opt_pp, _, _ = compatibility.std2pp(X_opt, E_opt, rho_opt)
        X_target_pp, _, _ = compatibility.std2pp(X_target, E_target, rho_target)

        # Saving
        dataset.attrs["results_layout_" + identifier] = {
            "raw_Kraus_operators": K,
            "raw_gates": X,
            "raw_POVM": E.reshape((dataset.attrs["num_povm"], pdim, pdim)),
            "raw_state": rho.reshape((pdim, pdim)),
            "gauge_opt_gates": X_opt,
            "gauge_opt_gates_Pauli_basis": X_opt_pp,
            "gauge_opt_POVM": E_opt.reshape((dataset.attrs["num_povm"], pdim, pdim)),
            "gauge_opt_state": rho_opt.reshape((pdim, pdim)),
            "main_mGST_time": main_gst_time,
            "gauge_optimization_time": gauge_optimization_time,
        }

        ### Bootstrap
        if dataset.attrs["bootstrap_samples"] > 0:
            bootstrap_results = bootstrap_errors(dataset, y, K, X, E, rho, target_mdl, identifier)
            dataset.attrs["results_layout_" + identifier].update({"bootstrap_data": bootstrap_results})

        _, df_o_full = reporting.report(
            X_opt, E_opt, rho_opt, dataset.attrs["J"], y, target_mdl, dataset.attrs["gate_labels"][identifier]
        )
        df_o_final, fig_o = generate_non_gate_results(dataset, qubit_layout, df_o_full)

        ### Result table generation and full report
        if dataset.attrs["rank"] == 1:
            df_g_final, df_g_rotation, fig_g, fig_rotation = generate_unit_rank_gate_results(
                dataset, qubit_layout, df_g, X_opt, K_target
            )
            dataset.attrs["results_layout_" + identifier].update({"hamiltonian_parameters": df_g_rotation.to_dict()})
            plots[f"layout_{qubit_layout}_hamiltonian_parameters"] = fig_rotation
        else:
            df_g_final, df_g_evals, fig_g, fig_choi = generate_gate_results(
                dataset, qubit_layout, df_g, X_opt, E_opt, rho_opt
            )
            dataset.attrs["results_layout_" + identifier].update({"choi_evals": df_g_evals.to_dict()})
            plots[f"layout_{qubit_layout}_choi_eigenvalues"] = fig_choi
        plots[f"layout_{qubit_layout}_gate_metrics"] = fig_g
        plots[f"layout_{qubit_layout}_other_metrics"] = fig_o

        observation_list = pandas_results_to_observations(
            dataset, df_g_final, df_o_final, BenchmarkObservationIdentifier(qubit_layout)
        )

        dataset.attrs["results_layout_" + identifier].update(
            {"full_metrics": {"Gates": df_g_final.to_dict(), "Outcomes and SPAM": df_o_final.to_dict()}}
        )

        ### Process matrix plots
        pauli_labels = generate_basis_labels(pdim, basis="Pauli")
        std_labels = generate_basis_labels(pdim)

        figures = figure_gen.generate_gate_err_pdf(
            "",
            X_opt_pp,
            X_target_pp,
            basis_labels=pauli_labels,
            gate_labels=dataset.attrs["gate_labels"][identifier],
            return_fig=True,
        )
        for i, figure in enumerate(figures):
            plots[f"layout_{qubit_layout}_process_matrix_{i}"] = figure

        plots[f"layout_{qubit_layout}_SPAM_matrices_real"] = figure_gen.generate_spam_err_std_pdf(
            "",
            E_opt.real,
            rho_opt.real,
            E_target.real,
            rho_target.real,
            basis_labels=std_labels,
            title=f"Real part of state and measurement effects in the standard basis",
            return_fig=True,
        )
        plots[f"layout_{qubit_layout}_SPAM_matrices_imag"] = figure_gen.generate_spam_err_std_pdf(
            "",
            E_opt.imag,
            rho_opt.imag,
            E_target.imag,
            rho_target.imag,
            basis_labels=std_labels,
            title=f"Imaginary part of state and measurement effects in the standard basis",
            return_fig=True,
        )
        plt.close("all")

    return BenchmarkAnalysisResult(dataset=dataset, observations=observation_list, plots=plots)
