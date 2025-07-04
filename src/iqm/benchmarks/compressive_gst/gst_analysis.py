"""
Data analysis code for compressive gate set tomography
"""

import ast
from time import perf_counter
from typing import Any, List, Tuple, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from numpy import ndarray
import numpy as np
from pandas import DataFrame
from pygsti.models.model import Model
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import xarray as xr

from iqm.benchmarks.benchmark_definition import (
    BenchmarkAnalysisResult,
    BenchmarkObservation,
    BenchmarkObservationIdentifier,
    BenchmarkRunResult,
)
from iqm.benchmarks.logging_config import qcvv_logger
from mGST import additional_fns, algorithm, compatibility
from mGST.low_level_jit import contract
from mGST.qiskit_interface import qiskit_gate_to_operator
from mGST.reporting import figure_gen, reporting
import multiprocessing as mp
import psutil


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

    qcvv_logger.info(f"Analyzing bootstrap samples...")
    with logging_redirect_tqdm(loggers=[qcvv_logger]):
        for i in trange(dataset.attrs["bootstrap_samples"]):
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
                verbose_level=0,
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
    qubit_layout: List[int], df_o: DataFrame, bootstrap_results: Union[None, tuple[Any, Any, Any, Any, Any]] = None
) -> DataFrame:
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
    if bootstrap_results is not None:
        _, _, _, _, df_o_array = bootstrap_results
        df_o_array[df_o_array == -1] = np.nan
        percentiles_o_low, percentiles_o_high = np.nanpercentile(df_o_array, [2.5, 97.5], axis=0)
        df_o_final = DataFrame(
            {
                f"mean_tvd_estimate_data": reporting.number_to_str(
                    df_o.values[0, 1].copy(), [percentiles_o_high[0, 1], percentiles_o_low[0, 1]], precision=5
                ),
                f"mean_tvd_target_data": reporting.number_to_str(
                    df_o.values[0, 2].copy(), [percentiles_o_high[0, 2], percentiles_o_low[0, 2]], precision=5
                ),
                f"povm_diamond_distance": reporting.number_to_str(
                    df_o.values[0, 3].copy(), [percentiles_o_high[0, 3], percentiles_o_low[0, 3]], precision=5
                ),
                f"state_trace_distance": reporting.number_to_str(
                    df_o.values[0, 4].copy(), [percentiles_o_high[0, 4], percentiles_o_low[0, 4]], precision=5
                ),
            },
            index=[""],
        )
    else:
        df_o_final = DataFrame(
            {
                f"mean_tvd_estimate_data": reporting.number_to_str(df_o.values[0, 1].copy(), precision=5),
                f"mean_tvd_target_data": reporting.number_to_str(df_o.values[0, 2].copy(), precision=5),
                f"povm_diamond_distance": reporting.number_to_str(df_o.values[0, 3].copy(), precision=5),
                f"state_trace_distance": reporting.number_to_str(df_o.values[0, 4].copy(), precision=5),
            },
            index=[""],
        )
    return df_o_final


def generate_unit_rank_gate_results(
    dataset: xr.Dataset, qubit_layout: List[int], df_g: DataFrame, X_opt: ndarray, K_target: ndarray,
        bootstrap_results: Union[None, tuple[Any, Any, Any, Any, Any]] = None
) -> Tuple[DataFrame, DataFrame]:
    """
    Produces all result tables for Kraus rank 1 estimates

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

    """
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    if bootstrap_results is not None:
        X_array, E_array, rho_array, df_g_array, _ = bootstrap_results
        df_g_array[df_g_array == -1] = np.nan
        percentiles_g_low, percentiles_g_high = np.nanpercentile(df_g_array, [2.5, 97.5], axis=0)
        df_g_rotation, hamiltonian_params = reporting.generate_rotation_param_results(
            dataset, qubit_layout, X_opt, K_target, X_array, E_array, rho_array
        )

    else:
        df_g_rotation, hamiltonian_params = reporting.generate_rotation_param_results(
            dataset, qubit_layout, X_opt, K_target
        )

    # Store non-formated results in dictionary
    dataset.attrs[f"results_layout_{identifier}"]["hamiltonian_params"] = hamiltonian_params
    df_g_final = DataFrame(
        {
            r"average_gate_fidelity": [
                reporting.number_to_str(
                    df_g.values[i, 0],
                    (
                        [percentiles_g_high[i, 0], percentiles_g_low[i, 0]]
                        if bootstrap_results is not None
                        else None
                    ),
                    precision=5,
                )
                for i in range(len(dataset.attrs["gate_labels"][identifier]))
            ],
            r"diamond_distance": [
                reporting.number_to_str(
                    df_g.values[i, 1],
                    (
                        [percentiles_g_high[i, 1], percentiles_g_low[i, 1]]
                        if bootstrap_results is not None
                        else None
                    ),
                    precision=5,
                )
                for i in range(dataset.attrs["num_gates"])
            ],
        }
    )

    return df_g_final, df_g_rotation


def generate_gate_results(
    dataset: xr.Dataset,
    qubit_layout: List[int],
    df_g: DataFrame,
    X_opt: ndarray,
    E_opt: ndarray,
    rho_opt: ndarray,
    bootstrap_results: Union[None, tuple[Any, Any, Any, Any, Any]] = None,
    max_evals: int = 6
) -> Tuple[DataFrame, DataFrame]:
    """
    Produces all result tables for arbitrary Kraus rank estimates

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

    """
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
    n_evals = np.min([max_evals, dataset.attrs["pdim"] ** 2])
    X_opt_pp, _, _ = compatibility.std2pp(X_opt, E_opt, rho_opt)
    df_g_evals = reporting.generate_Choi_EV_table(X_opt, n_evals, dataset.attrs["gate_labels"][identifier])

    if bootstrap_results is not None:
        X_array, E_array, rho_array, df_g_array, _ = bootstrap_results
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
                r"average_gate_fidelity": [
                    reporting.number_to_str(
                        df_g.values[i, 0], [percentiles_g_high[i, 0], percentiles_g_low[i, 0]], precision=5
                    )
                    for i in range(dataset.attrs["num_gates"])
                ],
                r"diamond_distance": [
                    reporting.number_to_str(
                        df_g.values[i, 1], [percentiles_g_high[i, 1], percentiles_g_low[i, 1]], precision=5
                    )
                    for i in range(dataset.attrs["num_gates"])
                ],
                r"unitarity": [
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
                "average_gate_fidelity": [
                    reporting.number_to_str(df_g.values[i, 0].copy(), precision=5)
                    for i in range(len(dataset.attrs["gate_labels"][identifier]))
                ],
                "diamond_distance": [
                    reporting.number_to_str(df_g.values[i, 1].copy(), precision=5)
                    for i in range(len(dataset.attrs["gate_labels"][identifier]))
                ],
                "unitarity": [
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

    return df_g_final, df_g_evals_final


def result_str_to_floats(result_str: str, err: str) -> Tuple[float, float]:
    """Converts formated string results from mgst to float (value, uncertainty) pairs

    Args:
        result_str: str
            The value of a result parameter formated as str
        err: str
            The error interval of the parameters

    Returns:
        value: float
            The parameter value as float
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
    qubits = "__".join([f"QB{i+1}" for i in ast.literal_eval(identifier.string_identifier)])
    for idx, gate_label in enumerate(dataset.attrs["gate_labels"][identifier.string_identifier].values()):
        observation_list.extend(
            [
                BenchmarkObservation(
                    name=f"{name}_{gate_label}:crosstalk_components={qubits}",
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
        if dataset.attrs["parallel_execution"]:
            result_da = dataset[f"parallel_results_counts_{run_index}"].copy()
            bit_pos = dataset.attrs["qubit_layouts"].index(qubit_layout)
            # Create a new coordinate of bits at the position given by the qubit layout and reverse order
            new_coords = [
                coord[::-1][bit_pos * num_qubits : (bit_pos + 1) * num_qubits]
                for coord in result_da.coords[result_da.dims[0]].values
            ]
        else:
            result_da = dataset[f"{qubit_layout}_counts_{run_index}"].copy()
            # Reverse order since counts are stored in qiskit order (bottom to top in circuit diagram)
            new_coords = [coord[::-1] for coord in result_da.coords[result_da.dims[0]].values]
        result_da.coords["new_coord"] = (result_da.dims[0], new_coords)
        result_da = result_da.groupby("new_coord").sum()

        coord_strings = list(result_da.coords[result_da.dims[0]].values)
        # Translating from binary basis labels to integer POVM labels
        basis_dict = {entry: int(entry, 2) for entry in coord_strings}
        # Sort by index:
        basis_dict = dict(sorted(basis_dict.items(), key=lambda item: item[1]))

        counts_normalized = result_da / result_da.sum()
        row = [float(counts_normalized.loc[key].data) for key in basis_dict]
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
        verbose_level=dataset.attrs["verbose_level"],
    )

    return K, X, E, rho, K_target, X_target, E_target, rho_target

def process_layout(args):
    """Process a single qubit layout in parallel"""
    dataset, qubit_layout, pdim = args
    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier

    qcvv_logger.info(f"Running mGST analysis for layout {qubit_layout}")

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

    # Prepare results dict
    results_dict = {
        "raw_Kraus_operators": K,
        "raw_gates": X,
        "raw_POVM": E.reshape((dataset.attrs["num_povm"], pdim, pdim)),
        "raw_state": rho.reshape((pdim, pdim)),
        "gauge_opt_gates": X_opt,
        "gauge_opt_gates_Pauli_basis": X_opt_pp,
        "gauge_opt_POVM": E_opt.reshape((dataset.attrs["num_povm"], pdim, pdim)),
        "gauge_opt_state": rho_opt.reshape((pdim, pdim)),
        "target_gates": X_target,
        "target_gates_Pauli_basis": X_target_pp,
        "target_POVM": E_target.reshape((dataset.attrs["num_povm"], pdim, pdim)),
        "target_state": rho_target.reshape((pdim, pdim)),
        "main_mGST_time": main_gst_time,
        "gauge_optimization_time": gauge_optimization_time,
    }

    # Bootstrap
    bootstrap_results = None
    if dataset.attrs["bootstrap_samples"] > 0:
        bootstrap_results = bootstrap_errors(dataset, y, K, X, E, rho, target_mdl, identifier, parametric=True)
        results_dict.update({"bootstrap_data": bootstrap_results})

    _, df_o_full = reporting.report(
        X_opt, E_opt, rho_opt, dataset.attrs["J"], y, target_mdl, dataset.attrs["gate_labels"][identifier]
    )
    df_o_final = generate_non_gate_results(qubit_layout, df_o_full, bootstrap_results)

    # Result table generation and full report
    if dataset.attrs["rank"] == 1:
        df_g_final, _ = generate_unit_rank_gate_results(
            dataset, qubit_layout, df_g, X_opt, K_target, bootstrap_results
        )
    else:
        df_g_final, df_g_evals = generate_gate_results(dataset, qubit_layout, df_g, X_opt, E_opt, rho_opt, bootstrap_results)
        results_dict.update({"choi_evals": df_g_evals.to_dict()})

    layout_observations = pandas_results_to_observations(
        dataset, df_g_final, df_o_final, BenchmarkObservationIdentifier(qubit_layout)
    )

    results_dict.update(
        {"full_metrics": {"Gates": df_g_final.to_dict(), "Outcomes and SPAM": df_o_final.to_dict()}}
    )
    return qubit_layout, results_dict, layout_observations, df_g_final, df_o_final, df_g_evals


def process_plots(dataset, qubit_layout, results_dict, df_g_final, df_o_final, df_g_evals_final):
    layout_plots = {}
    # Process matrix plots
    pdim = dataset.attrs["pdim"]
    pauli_labels = figure_gen.generate_basis_labels(pdim, basis="Pauli")
    std_labels = figure_gen.generate_basis_labels(pdim)

    identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier

    fig_g = dataframe_to_figure(df_g_final, dataset.attrs["gate_labels"][identifier])
    fig_choi = dataframe_to_figure(df_g_evals_final, dataset.attrs["gate_labels"][identifier])
    fig_o = dataframe_to_figure(df_o_final, [""])  # dataframe_to_figure(df_o_final, [""])


    layout_plots[f"layout_{qubit_layout}_gate_metrics"] = fig_g
    layout_plots[f"layout_{qubit_layout}_other_metrics"] = fig_o
    figures = figure_gen.generate_gate_err_pdf(
        "",
        results_dict["gauge_opt_gates_Pauli_basis"],
        results_dict["target_gates_Pauli_basis"],
        basis_labels=pauli_labels,
        gate_labels=dataset.attrs["gate_labels"][identifier],
        return_fig=True,
    )
    for i, figure in enumerate(figures):
        layout_plots[f"layout_{qubit_layout}_process_matrix_{i}"] = figure

    layout_plots[f"layout_{qubit_layout}_SPAM_matrices_real"] = figure_gen.generate_spam_err_std_pdf(
        "",
        results_dict["gauge_opt_POVM"].reshape((-1,pdim**2)).real,
        results_dict["gauge_opt_state"].reshape(-1).real,
        results_dict["target_POVM"].reshape((-1,pdim**2)).real,
        results_dict["target_state"].reshape(-1).real,
        basis_labels=std_labels,
        title=f"Real part of state and measurement effects in the standard basis\n(red:<0; blue:>0)",
        return_fig=True,
    )
    layout_plots[f"layout_{qubit_layout}_SPAM_matrices_imag"] = figure_gen.generate_spam_err_std_pdf(
        "",
        results_dict["gauge_opt_POVM"].reshape((-1,pdim**2)).imag,
        results_dict["gauge_opt_state"].reshape(-1).imag,
        results_dict["target_POVM"].reshape((-1,pdim**2)).imag,
        results_dict["target_state"].reshape(-1).imag,
        basis_labels=std_labels,
        title=f"Imaginary part of state and measurement effects in the standard basis\n(red:<0; blue:>0)",
        return_fig=True,
    )
    plt.close("all")
    return layout_plots


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
    # observations = []

    # Use all but one physical core
    num_physical_cores = psutil.cpu_count(logical=False)
    num_workers = max(1, num_physical_cores - 1)

    args_list = [(dataset, qubit_layout, pdim)
                 for qubit_layout in dataset.attrs["qubit_layouts"]]

    # Execute in parallel
    # all_results = []
    # for args in args_list:
    #     all_results.append(process_layout(args))
    qcvv_logger.info(f"Using {num_workers} out of {num_physical_cores} physical cores")

    # Use Manager to handle shared data structures safely
    with mp.Manager() as manager:
        all_results = []
        # Create a shared counter to track completed tasks
        counter = manager.Value('i', 0)
        total_layouts = len(dataset.attrs["qubit_layouts"])

        # Define a callback function to update progress
        def update_progress(result):
            counter.value += 1
            qcvv_logger.info(f"Completed estimation for {counter.value}/{total_layouts} qubit layouts")

        # Prepare arguments for each process
        args_list = [(dataset, qubit_layout, pdim)
                     for qubit_layout in dataset.attrs["qubit_layouts"]]

        # Execute in parallel using apply_async with callback
        with mp.Pool(num_workers) as pool:
            results = [pool.apply_async(process_layout, args=(arg,), callback=update_progress) for arg in args_list]
            all_results = [res.get() for res in results]  # Wait for all results

    # Collect results
    observations_list = []
    df_g_list = []
    df_o_list = []
    df_g_evals_list = []

    for qubit_layout, results_dict, layout_observations, df_g_final, df_o_final, df_g_evals_final in all_results:
        identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
        # Update dataset with results
        dataset.attrs["results_layout_" + identifier] = results_dict
        # Collect observations and dataframes
        observations_list.extend(layout_observations)
        df_g_list.append(df_g_final)
        df_o_list.append(df_o_final)
        df_g_evals_list.append(df_g_evals_final)

    # Generate figures for each layout
    for i, qubit_layout in enumerate(dataset.attrs["qubit_layouts"]):
        identifier = BenchmarkObservationIdentifier(qubit_layout).string_identifier
        results_dict = dataset.attrs["results_layout_" + identifier]
        # Update plots
        qcvv_logger.info(f"Generating figures for layout {i+1}/{len(dataset.attrs['qubit_layouts'])}")
        layout_plots = process_plots(dataset, qubit_layout, results_dict, df_g_list[i], df_o_list[i], df_g_evals_list[i])
        for key, fig in layout_plots.items():
            plots[key] = fig

    # Generate additional figures for Hamiltonian parameters if rank is 1
    if dataset.attrs["rank"] == 1:
        qcvv_logger.info(f"Generating additional rank 1 figures for all layouts")
        hamiltonian_plots = figure_gen.generate_hamiltonian_visualizations(dataset)
        plots.update(hamiltonian_plots)
    plt.close("all")
    qcvv_logger.info("Analysis completed")

    return BenchmarkAnalysisResult(dataset=dataset, observations=observations_list, plots=plots)
