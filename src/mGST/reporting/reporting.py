"""
Generation of error measures and result tables
"""

from argparse import Namespace
import csv
import os

import numpy as np
import numpy.linalg as la
import pandas as pd
from pygsti.algorithms import gaugeopt_to_target
from pygsti.baseobjs import Basis
from pygsti.models import gaugegroup
from pygsti.report.reportables import entanglement_fidelity
from pygsti.tools import change_basis
from pygsti.tools.optools import compute_povm_map
from qiskit.quantum_info import SuperOp
from qiskit.quantum_info.operators.measures import diamond_norm
from scipy.linalg import logm, schur
from scipy.optimize import linear_sum_assignment, minimize

from mGST import additional_fns, algorithm, compatibility, low_level_jit


def min_spectral_distance(X1, X2):
    """Computes the average absolute distance between the eigenvlues of two matrices
    The matrices are first diagonalized, then the eigenvalues are matched such that the average
    distance of matched eigenvalue pairs is minimal.
    The big advantage of this distance metric is that it is gauge invariant and it can thus be used
    to if the reconstructed gates are similar to the target gates before any gauge optimization.

    Parameters
    ----------
    X1: numpy array
        The first matrix
    X2: numpy array
        The second matrix

    Returns
    -------
    dist: float
        The minimal distance
    """
    r = X1.shape[0]
    eigs = la.eig(X1)[0]
    eigs_t = la.eig(X2)[0]
    cost_matrix = np.array([[np.abs(eigs[i] - eigs_t[j]) for i in range(r)] for j in range(r)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    normalization = np.abs(eigs).sum()
    dist = cost_matrix[row_ind, col_ind].sum() / normalization
    return dist


def MVE_data(X, E, rho, J, y):
    """Mean varation error between measured outcomes and predicted outcomes from a gate set

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence

    Returns
    -------
    dist : float
        Mean variation error
    max_dist : float
        Maximal varaition error

    Notes:
        For each sequence the total variation error of the two probability distribution
        over the POVM elements is computed. Afterwards the meean over these total
        variation errors is returned.
    """
    m = y.shape[1]
    n_povm = y.shape[0]
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C = low_level_jit.contract(X, j)
        curr = 0
        for k in range(n_povm):
            y_model = E[k].conj() @ C @ rho
            curr += np.abs(y_model - y[k, i])
        curr = curr / 2
        dist += curr
        max_dist = max(max_dist, curr)
    return dist / m, max_dist


def gauge_opt(X, E, rho, target_mdl, weights):
    """Performs pyGSti gauge optimization to target model

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    target_mdl : pygsti model object
        A model containing the target gate set
    weights : dict[str: float]
        A dictionary with keys being gate labels or "spam" and corresponding values being the
        weight of each gate in the gauge optimization.
        Example for uniform weights: dict({"G%i"%i:1  for i in range(d)}, **{"spam":1})

    Returns
    -------
    X_opt, E_opt, rho_opt: Numpy arrays
        The gauge optimized gates and SPAM arrays
    """
    mdl = compatibility.arrays_to_pygsti_model(X, E, rho, basis="std")
    X_t, E_t, rho_t = compatibility.pygsti_model_to_arrays(target_mdl, basis="std")
    target_mdl = compatibility.arrays_to_pygsti_model(X_t, E_t, rho_t, basis="std")  # For consistent gate labels

    gauge_optimized_mdl = gaugeopt_to_target(
        mdl,
        target_mdl,
        gauge_group=gaugegroup.UnitaryGaugeGroup(target_mdl.state_space, basis="pp"),
        item_weights=weights,
    )
    return compatibility.pygsti_model_to_arrays(gauge_optimized_mdl, basis="std")


def report(X, E, rho, J, y, target_mdl, gate_labels):
    """Generation of pandas dataframes with gate and SPAM quality measures
    The resutls can be converted to .tex tables or other formats to be used for GST reports

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    target_mdl : pygsti model object
        A model containing the target gate set
    gate_labels : list[str]
        A list of names for the gates in X

    Returns
    -------
        df_g : Pandas DataFrame
            DataFrame of gate quality measures
        df_o : Pandas DataFrame
            DataFrame of all other quality/error measures
        s_g : Pandas DataFrame.style object
        s_o : Pandas DataFrame.style object
    """
    pdim = int(np.sqrt(rho.shape[0]))
    X_t, E_t, rho_t = compatibility.pygsti_model_to_arrays(target_mdl, basis="std")
    target_mdl = compatibility.arrays_to_pygsti_model(X_t, E_t, rho_t, basis="std")  # For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X, E, rho, basis="std")
    E_map = compute_povm_map(gauge_optimized_mdl, "Mdefault")
    E_map_t = compute_povm_map(target_mdl, "Mdefault")

    final_objf = low_level_jit.objf(X, E, rho, J, y)
    MVE = MVE_data(X, E, rho, J, y)[0]
    MVE_target = MVE_data(X_t, E_t, rho_t, J, y)[0]

    povm_dd = diamond_norm(SuperOp(E_map) - SuperOp(E_map_t)) / 2
    rho_td = la.norm(rho.reshape((pdim, pdim)) - rho_t.reshape((pdim, pdim)), ord="nuc") / 2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl, target_mdl, pdim, basis_string="pp")
    DD = [diamond_norm(SuperOp(X[i]) - SuperOp(X_t[i])) / 2 for i in range(len(X))]
    min_spectral_dists = [min_spectral_distance(X[i], X_t[i]) for i in range(X.shape[0])]

    df_g = pd.DataFrame({"F_avg": F_avg, "Diamond distances": DD, "Min. Spectral distances": min_spectral_dists})
    df_o = pd.DataFrame(
        {
            "Final cost function value": final_objf,
            "Mean total variation dist. to data": MVE,
            "Mean total variation dist. target to data": MVE_target,
            "POVM - Meas. map diamond dist.": povm_dd,
            "State - Trace dist.": rho_td,
        },
        index=[0],
    )
    df_g.rename(index=gate_labels, inplace=True)
    df_o.rename(index={0: ""}, inplace=True)

    return df_g, df_o


def quick_report(X, E, rho, J, y, target_mdl, gate_labels=None):
    """Generation of pandas dataframes with gate and SPAM quality measures
    The quick report is intended to check on a GST estimate with fast to compute measures
    (no diamond distance) to get a first picture and check whether mGST and the gauge optimization
    produce meaningful results.

    Parameters
    ----------
    X : numpy array
        Gate set
    E : numpy array
        POVM
    rho : numpy array
        Initial state
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    target_mdl : pygsti model object
        A model containing the target gate set
    gate_labels : list[str]
        A list of names for the gates in X

    Returns
    -------
        df_g : Pandas DataFrame
            DataFrame of gate quality measures
        df_o : Pandas DataFrame
            DataFrame of all other quality/error measures
        s_g : Pandas DataFrame.style object
        s_o : Pandas DataFrame.style object
    """
    pdim = int(np.sqrt(rho.shape[0]))
    d = X.shape[0]
    if not gate_labels:
        gate_labels = {i: f"Gate %i" % i for i in range(d)}

    X_t, E_t, rho_t = compatibility.pygsti_model_to_arrays(target_mdl, basis="std")
    target_mdl = compatibility.arrays_to_pygsti_model(X_t, E_t, rho_t, basis="std")  # For consistent gate labels

    gauge_optimized_mdl = compatibility.arrays_to_pygsti_model(X, E, rho, basis="std")

    final_objf = low_level_jit.objf(X, E, rho, J, y)
    MVE = MVE_data(X, E, rho, J, y)[0]
    MVE_target = MVE_data(X_t, E_t, rho_t, J, y)[0]

    E_map = compute_povm_map(gauge_optimized_mdl, "Mdefault")
    E_map_t = compute_povm_map(target_mdl, "Mdefault")
    povm_dd = diamond_norm(SuperOp(E_map) - SuperOp(E_map_t)) / 2

    rho_td = la.norm(rho.reshape((pdim, pdim)) - rho_t.reshape((pdim, pdim)), ord="nuc") / 2
    F_avg = compatibility.average_gate_fidelities(gauge_optimized_mdl, target_mdl, pdim, basis_string="pp")
    min_spectral_dists = [min_spectral_distance(X[i], X_t[i]) for i in range(X.shape[0])]

    df_g = pd.DataFrame({"F_avg": F_avg, "Min. Spectral distances": min_spectral_dists})
    df_o = pd.DataFrame(
        {
            "Final cost function": final_objf,
            "Mean TVD estimate-data": MVE,
            "Mean TVD target-data": MVE_target,
            "SPAM error:": rho_td + povm_dd,
        },
        index=[0],
    )
    df_g.rename(index=gate_labels, inplace=True)
    df_o.rename(index={0: ""}, inplace=True)

    return df_g, df_o


def compute_angles_axes(U_set, alternative_phase=False):
    """Takes the matrix logarithm of the given unitaries and returns the Hamiltonian parameters
    The parametrization is U = exp(-i pi H/2), i.e. H = i log(U)*2/pi

    Parameters
    ----------
    U_set: list[numpy array]
        A list contining unitary matrices
    alternative_phase: bool
        Whether an attempt should be made to report more intuitive rotations,
        for example a rotation of 3 pi/2 around the -X axis would be turned into
        a pi/2 rotation around the X-axis.
    Returns
    -------
    angles: list[float]
        The rotation angle on the Bloch sphere for all gates
    axes : list[numpy array]
        The normalized rotation axes in the Pauli basis for all gates
    pauli_coeffs : list[numpy array]
        The full list of Pauli basis coefficients of the Hamiltonian for all gates

    Notes: sqrt(pdim) factor is due to Pauli basis normalization
    """
    d = U_set.shape[0]
    pdim = U_set.shape[1]
    angles = []
    axes = []
    pp_vecs = []
    for i in range(d):
        H = 1j * logm(U_set[i])
        pp_vec = change_basis(H.reshape(-1), "std", "pp")
        original_phase = la.norm(pp_vec[1:]) * 2 / np.sqrt(pdim)
        if alternative_phase and (-np.min(pp_vec) > np.max(pp_vec)) and original_phase > np.pi:
            alt_phase = (-original_phase + 2 * np.pi) % (2 * np.pi)
            pp_vec = -pp_vec
        else:
            alt_phase = original_phase
        angles.append(alt_phase / np.pi)
        axes.append(pp_vec[1:] / la.norm(pp_vec[1:]))
        pp_vecs.append(pp_vec)
    pauli_coeffs = np.array(pp_vecs) / np.sqrt(pdim) / np.pi * 2
    return angles, axes, pauli_coeffs


def compute_sparsest_Pauli_Hamiltonian(U_set):
    """Takes the matrix logarithms of the given unitaries and returns sparsest Hamiltonian parameters in Pauli basis
    The parametrization is U = exp(-i pi H/2), i.e. H = i log(U)*2/pi.
    Different branches in the matrix logarithm lead to different Hamiltonians. This function optimizes over
    combinations of adding 2*pi to different eigenvalues, in order arrive at the branch with the Hamiltonian
    whose Pauli basis representation is the most sparse.


    Parameters
    ----------
    U_set  : list[numpy array]
        A list contining unitary matrices

    Returns
    -------
    pauli_coeffs : list[numpy array]
        The full list of Pauli basis coefficients of the Hamiltonian for all gates

    Notes: sqrt(pdim) factor is due to Pauli basis normalization
    """
    pdim = U_set.shape[1]
    pp_vecs = []

    for num, U in enumerate(U_set):
        # Schur decomposition finds the unitary diagonalization of a unitary matrix, which is not always returned by np.linalg.eig
        T, evecs = schur(U)
        evals = np.diag(T)
        Pauli_norms = []
        log_evals = np.log(evals)
        for i in range(2**pdim):
            bits = low_level_jit.local_basis(i, 2, pdim)
            evals_new = 1j * log_evals + 2 * np.pi * bits
            H_new = evecs @ np.diag(evals_new) @ evecs.T.conj()
            pp_vec = change_basis(H_new.reshape(-1), "std", "pp")
            Pauli_norms.append(np.linalg.norm(pp_vec, ord=1))
        opt_bits = low_level_jit.local_basis(np.argsort(Pauli_norms)[0], 2, pdim)
        evals_opt = 1j * log_evals + 2 * np.pi * opt_bits
        H_opt = evecs @ np.diag(evals_opt) @ evecs.T.conj()
        pp_vecs.append(change_basis(H_opt.reshape(-1), "std", "pp"))
    pauli_coeffs = np.array(pp_vecs) / np.sqrt(pdim) / np.pi * 2
    return pauli_coeffs


def phase_err(angle, U, U_t):
    """Computes norm between two input unitaries after a global phase is added to one of them

    Parameters
    ----------
    angle : float
        The global phase angle (in rad)
    U : numpy array
        The first unitary matrix
    U_t : numpy array
        The second unitary matrix (typically the a target gate)

    Returns
    -------
    norm : floar
        The norm of the difference
    """
    return la.norm(np.exp(1j * angle) * U - U_t)


def phase_opt(X, K_t):
    """Return rK = 1 gate set with global phase fitting matching to target gate set

    Parameters
    ----------
    X: 3D numpy array
        Array where CPT superoperators are stacked along the first axis.
        These should correspond to rK = 1 gates for the outputs to be meaningful.
    K_t: 4D numpy array
        Array of target gate Kraus operators
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.

    Returns
    -------
    K_opt: 4D numpy array
        Array of Kraus operators with mathed global phase
    """
    d = X.shape[0]
    r = X.shape[1]
    pdim = int(np.sqrt(r))
    K = additional_fns.Kraus_rep(X, d, pdim, 1).reshape((d, pdim, pdim))
    K_t = K_t.reshape(d, pdim, pdim)
    K_opt = np.zeros(K.shape).astype(complex)
    for i in range(d):
        angle_opt = minimize(phase_err, 1, bounds=[[-np.pi, np.pi]], args=(K[i], K_t[i])).x
        K_opt[i] = K[i] * np.exp(1j * angle_opt)
    return K_opt


def eff_depol_params_agf(X_opt_pp):
    """Computes the average gate fidelities to the completely depolarizing channel

    Parameters
    ----------
    X_opt_pp: 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    ent_fids : list[float]
        List of average gate fidelities wrt. the depol. channel corresponding to the gates in X_opt_pp.
    """
    r = X_opt_pp.shape[1]
    pdim = np.sqrt(r)
    ent_fids = []
    basis = Basis.cast("pp", r)
    K_depol = additional_fns.depol(int(np.sqrt(r)), 1)
    X_depol = np.einsum("jkl,jnm -> knlm", K_depol, K_depol.conj()).reshape(r, r)
    for i in range(X_opt_pp.shape[0]):
        ent_fids.append(entanglement_fidelity(X_opt_pp[i], change_basis(X_depol, "std", "pp"), basis))
    return (pdim * np.array(ent_fids) + 1) / (pdim + 1)


def unitarities(X_opt_pp):
    """Computes the unitarities of all gates in the gate set

    Parameters
    ----------
    X_opt_pp : 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    unitarities : list[float]
        List of unitarities for the gates in X_opt_pp.
    """
    # Definition: Proposition 1 of https://arxiv.org/pdf/1503.07865.pdf
    pdim = int(np.sqrt(X_opt_pp.shape[1]))
    E_u = X_opt_pp[:, 1:, 1:]  # unital submatrices
    unitarity_list = np.real(np.einsum("ijk, ijk -> i", E_u.conj(), E_u) / (pdim**2 - 1))
    return unitarity_list


# Spectrum of the Choi matrix
def generate_Choi_EV_table(X_opt, n_evals, gate_labels, filename=None):
    """Outputs a .tex document containing a table with the larges eigenvalues of the Choi matrix for each gate

    Parameters
    ----------
    X_opt : 3D numpy array
        Array where CPT superoperators in standard basis are stacked along the first axis.
    n_evals : int
        Number of eigenvalues to be returned
    gate_labels : list[int: str]
        The names of gates in the gate set
    filename :
        The file name of the output .tex file
    """
    d, r, _ = X_opt.shape
    pdim = int(np.sqrt(r))
    Choi_evals_result = np.zeros((d, r))
    X_choi = X_opt.reshape(d, pdim, pdim, pdim, pdim)
    X_choi = np.einsum("ijklm->iljmk", X_choi).reshape((d, pdim**2, pdim**2))
    for j in range(d):
        Choi_evals_result[j, :] = np.sort(np.abs(la.eig(X_choi[j])[0]))[::-1]
    Choi_evals_normalized = np.einsum("ij,i -> ij", Choi_evals_result, 1 / la.norm(Choi_evals_result, axis=1, ord=1))

    df_g_evals = pd.DataFrame(Choi_evals_normalized)
    df_g_evals.rename(index=gate_labels, inplace=True)
    if filename:
        df_g_evals.style.to_latex(
            filename + ".tex",
            column_format="c|*{%i}{c}" % n_evals,
            position_float="centering",
            hrules=True,
            caption="Eigenvalues of the Choi state",
            position="h!",
        )
    return df_g_evals


def dephasing_dist(prob_vec, X_pp):
    """Returns the distance between a given channel and a local dephasing channel with given probabilities

    Parameters
    ----------
    prob_vec : list[float]
        A list of dephasing probabilities
    X_pp : 3D numpy array
        Array where CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    norm : float
        The norm between the channel difference

    """
    X_deph = additional_fns.local_dephasing_pp(prob_vec)
    return la.norm(X_pp - X_deph)


def dephasing_probabilities_2q(X_opt_pp, X_ideal_pp):
    """Determines the local dephasing channel parameters which best describe the noise model
    Works for two qubit gates only

    Parameters
    ----------
    X_opt_pp : 3D numpy array
        Array where reconstructed CPT superoperators in Pauli basis are stacked along the first axis.
    X_opt_pp : 3D numpy array
        Array where target gate CPT superoperators in Pauli basis are stacked along the first axis.

    Returns
    -------
    dephasing_probs : list[float]
        The two best fit dephasing probabilities
    """
    dephasing_probs = []
    for i in range(X_opt_pp.shape[0]):
        dephasing_probs.append(minimize(dephasing_dist, [0.1, 0.1], args=X_opt_pp[i] @ la.inv(X_ideal_pp[i])).x)
    return dephasing_probs


def bootstrap_errors(K, X, E, rho, mGST_args, bootstrap_samples, weights, gate_labels, target_mdl, parametric=False):
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
    mGST_args : dict[str: misc]
        Arguments with which the run_mGST function was called
    bootstrap_samples : int
        Number of bootstrapping repretitions
    weights : dict[str: float]
        Gate weights used for gauge optimization
    gate_labels : list[int: str]
        The names of gates in the gate set
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
    ns = Namespace(**mGST_args)
    if parametric:
        y = np.real(
            np.array([[E[i].conj() @ low_level_jit.contract(X, j) @ rho for j in ns.J] for i in range(ns.n_povm)])
        )
    else:
        y = ns.y
    X_array = np.zeros((bootstrap_samples, *X.shape)).astype(complex)
    E_array = np.zeros((bootstrap_samples, *E.shape)).astype(complex)
    rho_array = np.zeros((bootstrap_samples, *rho.shape)).astype(complex)
    df_g_list = []
    df_o_list = []

    for i in range(bootstrap_samples):
        y_sampled = additional_fns.sampled_measurements(y, ns.meas_samples).copy()
        _, X_, E_, rho_, _ = algorithm.run_mGST(
            y_sampled,
            ns.J,
            ns.l,
            ns.d,
            ns.r,
            ns.rK,
            ns.n_povm,
            ns.bsize,
            ns.meas_samples,
            method=ns.method,
            max_inits=ns.max_inits,
            max_iter=0,
            final_iter=ns.final_iter,
            threshold_multiplier=ns.threshold_multiplier,
            target_rel_prec=ns.target_rel_prec,
            init=[K, E, rho],
            testing=False,
        )

        X_opt, E_opt, rho_opt = gauge_opt(X_, E_, rho_, target_mdl, weights)
        df_g, df_o = report(X_opt, E_opt, rho_opt, ns.J, y_sampled, target_mdl, gate_labels)
        df_g_list.append(df_g.values)
        df_o_list.append(df_o.values)

        X_opt_pp, E_opt_pp, rho_opt_pp = compatibility.std2pp(X_opt, E_opt, rho_opt)

        X_array[i, :] = X_opt_pp
        E_array[i, :] = E_opt_pp
        rho_array[i, :] = rho_opt_pp

    return (X_array, E_array, rho_array, np.array(df_g_list), np.array(df_o_list))


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
        basis_dict = {entry: int("".join([entry[::-1][i] for i in self.qubits][::-1]), 2) for entry in result}
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


def save_var_latex(key, value):
    """Saves variables in data file to be read in latex document
    Credit to https://stackoverflow.com/a/66620671

    Parameters
    ----------
    key : str
        The name of the variable
    value : misc
        The variable content, could be a string as in the name of the experiment, or the number of circuits run (int),
        or any other python variable that needs to be transfered
    """

    dict_var = {}

    file_path = os.path.join(os.getcwd(), "report/latex_vars.dat")

    try:
        with open(file_path, newline="", encoding="ASCII") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w", encoding="ASCII") as f:
        for key_, value_ in dict_var:
            f.write(f"{key_},{value_}\n")


def number_to_str(number, uncertainty=None, precision=3):
    """Formats a floating point number to a string with the given precision.

    Parameters:
    number (float): The floating point number to format.
    uncertainty (tuple): The upper and lower values of the confidence interval
    precision (int): The number of decimal places to include in the formatted string.

    Returns:
    str: The formatted floating point number as a string.
    """
    if uncertainty is None:
        return f"{number:.{precision}f}"

    return f"{number:.{precision}f} [{uncertainty[1]:.{precision}f},{uncertainty[0]:.{precision}f}]"
