"""
The main algorithm and functions that perform iteration steps
"""

from decimal import Decimal
import time
from warnings import warn

import numpy as np
import numpy.linalg as la
from scipy.linalg import eig, eigh
from scipy.optimize import minimize
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from iqm.benchmarks.logging_config import qcvv_logger
from mGST.additional_fns import batch, random_gs, transp
from mGST.low_level_jit import ddA_derivs, ddB_derivs, ddM, dK, dK_dMdM, objf
from mGST.optimization import (
    lineobjf_A_geodesic,
    lineobjf_B_geodesic,
    lineobjf_isom_geodesic,
    tangent_proj,
    update_A_geodesic,
    update_B_geodesic,
    update_K_geodesic,
)
from mGST.reporting.figure_gen import plot_objf


def A_SFN_riem_Hess(K, A, B, y, J, d, r, n_povm, lam=1e-3):
    """Riemannian saddle free Newton step on the POVM parametrization

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    A : numpy array
        Current POVM parametrization
    B : numpy array
        Current initial state parametrization
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3

    Returns
    -------
    A_new : numpy array
        Updated POVM parametrization
    """
    pdim = int(np.sqrt(r))
    n = n_povm * pdim
    nt = n_povm * r
    rho = (B @ B.T.conj()).reshape(-1)
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    P_T = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    Fyconjy = np.zeros((n_povm, r, n_povm, r)).astype(np.complex128)
    Fyy = np.zeros((n_povm, r, n_povm, r)).astype(np.complex128)

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    dA_, dMdM, dMconjdM, dconjdA = ddA_derivs(X, A, B, J, y, r, pdim, n_povm)

    # Second derivatives
    for i in range(n_povm):
        Fyconjy[i, :, i, :] = dMconjdM[i] + dconjdA[i]
        Fyy[i, :, i, :] = dMdM[i]

    # derivative
    Fy = dA_.reshape(n, pdim)
    Y = A.reshape(n, pdim)
    rGrad = Fy.conj() - Y @ Fy.T @ Y
    G = np.array([rGrad, rGrad.conj()]).reshape(-1)

    P = np.eye(n) - Y @ Y.T.conj()
    T = transp(n, pdim)

    # Hessian assembly
    H00 = (
        -(np.kron(Y, Y.T)) @ T @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - (np.kron(np.eye(n), Y.T @ Fy)) / 2
        - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
        - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.kron(Y, Y.T) @ T @ Fyconjy.reshape(nt, nt).T
        + (np.kron(Fy.conj(), Y.T) @ T) / 2
        + (np.kron(Y, Fy.T.conj()) @ T) / 2
    )

    H[0, :, 0, :] = H00
    H[0, :, 1, :] = H01
    H[1, :, 0, :] = H01.conj()
    H[1, :, 1, :] = H00.conj()

    P_T[0, :, 0, :] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
    P_T[0, :, 1, :] = -np.kron(Y, Y.T) @ T / 2
    P_T[1, :, 0, :] = P_T[0, :, 1, :].conj()
    P_T[1, :, 1, :] = P_T[0, :, 0, :].conj()

    H = H.reshape(2 * nt, 2 * nt) @ P_T.reshape(2 * nt, 2 * nt)

    # saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Disregarding gauge directions (zero eigenvalues of the Hessian)
    # inv_diag = evals.copy()
    # inv_diag[np.abs(evals)<1e-14] = 1
    # inv_diag[np.abs(evals)>1e-14] = np.abs(inv_diag[np.abs(evals)>1e-14]) + lam
    # H_abs_inv = U@np.diag(1/inv_diag)@U.T.conj()

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()

    Delta_A = ((H_abs_inv @ G)[:nt]).reshape(n, pdim)

    Delta = tangent_proj(A, Delta_A, 1, n_povm)[0]

    a = minimize(lineobjf_A_geodesic, 1e-9, args=(Delta, X, A, rho, J, y), method="COBYLA").x
    A_new = update_A_geodesic(A, Delta, a)
    return A_new


def B_SFN_riem_Hess(K, A, B, y, J, d, r, n_povm, lam=1e-3):
    """Riemannian saddle free Newton step on the initial state parametrization

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    A : numpy array
        Current POVM parametrization
    B : numpy array
        Current initial state parametrization
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3

    Returns
    -------
    B_new : numpy array
        Updated initial state parametrization
    """

    pdim = int(np.sqrt(r))
    n = r
    nt = r
    E = np.array([(A[i].T.conj() @ A[i]).reshape(-1) for i in range(n_povm)])
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    P_T = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    Fyconjy = np.zeros((r, r)).astype(np.complex128)
    Fyy = np.zeros((r, r)).astype(np.complex128)

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    dB_, dMdM, dMconjdM, dconjdB = ddB_derivs(X, A, B, J, y, r, pdim)

    # Second derivatives
    Fyconjy = dMconjdM + dconjdB
    Fyy = dMdM

    # derivative
    Fy = dB_.reshape(n)
    Y = B.reshape(n)
    rGrad = Fy.conj() - Y * (Fy.T @ Y)
    G = np.array([rGrad, rGrad.conj()]).reshape(-1)

    P = np.eye(n) - np.outer(Y, Y.T.conj())

    # Hessian assembly
    H00 = (
        -(np.outer(Y, Y.T)) @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - np.eye(n) * (Y.T @ Fy) / 2
        - np.outer(Y, Fy.T) / 2
        - P * (Fy.T.conj() @ Y.conj()) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.outer(Y, Y.T) @ Fyconjy.reshape(nt, nt).T
        + np.outer(Fy.conj(), Y.T) / 2
        + np.outer(Y, Fy.T.conj()) / 2
    )

    H[0, :, 0, :] = H00
    H[0, :, 1, :] = H01
    H[1, :, 0, :] = H01.conj()
    H[1, :, 1, :] = H00.conj()

    P_T[0, :, 0, :] = np.eye(nt) - np.outer(Y, Y.T.conj()) / 2
    P_T[0, :, 1, :] = -np.outer(Y, Y.T) / 2
    P_T[1, :, 0, :] = P_T[0, :, 1, :].conj()
    P_T[1, :, 1, :] = P_T[0, :, 0, :].conj()

    H = H.reshape(2 * nt, 2 * nt) @ P_T.reshape(2 * nt, 2 * nt)

    # saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Disregarding gauge directions (zero eigenvalues of the Hessian)
    # inv_diag = evals.copy()
    # inv_diag[np.abs(evals)<1e-14] = 1
    # inv_diag[np.abs(evals)>1e-14] = np.abs(inv_diag[np.abs(evals)>1e-14]) + lam
    # H_abs_inv = U@np.diag(1/inv_diag)@U.T.conj()

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()

    Delta = (H_abs_inv @ G)[:nt]
    # Projection onto tangent space
    Delta = Delta - Y * (Y.T.conj() @ Delta + Delta.T.conj() @ Y) / 2
    res = minimize(lineobjf_B_geodesic, 1e-9, args=(Delta, X, E, B, J, y), method="COBYLA", options={"maxiter": 20})
    a = res.x

    B_new = update_B_geodesic(B, Delta, a)
    return B_new


def gd(K, E, rho, y, J, d, r, rK, fixed_gates, ls="COBYLA"):
    """Do Riemannian gradient descent optimization step on gates

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize

    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations

    Notes:
        Gradient descent using the Riemannian gradient and updating along the geodesic.
        The step size is determined by minimizing the objective function in the step size parameter.
    """
    # setup
    pdim = int(np.sqrt(r))
    n = rK * pdim
    Delta = np.zeros((d, n, pdim)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))

    dK_ = dK(X, K, E, rho, J, y, d, r, rK)
    for k in np.where(~fixed_gates)[0]:
        # derivative
        Fy = dK_[k].reshape(n, pdim)
        Y = K[k].reshape(n, pdim)
        # Riem. gradient taken from conjugate derivative
        rGrad = Fy.conj() - Y @ Fy.T @ Y
        Delta[k] = rGrad

    Delta = tangent_proj(K, Delta, d, rK)
    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta, K, E, rho, J, y), method=ls, options={"maxiter": 200})
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)

    return K_new


def SFN_riem_Hess(K, E, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA", fixed_gates=None):
    """Riemannian saddle free Newton step on each gate individually

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize
    fixed_gates : List
        List of gate indices which are not optimized over and assumed as fixed

    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations
    """
    # setup
    pdim = int(np.sqrt(r))
    n = rK * pdim
    nt = rK * r
    H = np.zeros((2 * nt, 2 * nt)).astype(np.complex128)
    P_T = np.zeros((2 * nt, 2 * nt)).astype(np.complex128)
    Delta_K = np.zeros((d, rK, pdim, pdim)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    if not fixed_gates:
        fixed_gates = []

    # compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, d, r, rK)
    dd, dconjd = ddM(X, K, E, rho, J, y, d, r, rK)

    # Second derivatives
    Fyconjy = dM11.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((d, nt, d, nt))
    Fyy = dM10.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((d, nt, d, nt))

    for k in np.where(~fixed_gates)[0]:
        Fy = dK_[k].reshape(n, pdim)
        Y = K[k].reshape(n, pdim)
        # riemannian gradient, taken from conjugate derivative
        rGrad = Fy.conj() - Y @ Fy.T @ Y
        G = np.array([rGrad, rGrad.conj()]).reshape(-1)

        P = np.eye(n) - Y @ Y.T.conj()
        T = transp(n, pdim)

        # Riemannian Hessian with correction terms
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        H[:nt, :nt] = H00
        H[:nt, nt:] = H01
        H[nt:, :nt] = H[:nt, nt:].conj()
        H[nt:, nt:] = H[:nt, :nt].conj()

        # Tangent space projection
        P_T[:nt, :nt] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
        P_T[:nt, nt:] = -np.kron(Y, Y.T) @ T / 2
        P_T[nt:, :nt] = P_T[:nt, nt:].conj()
        P_T[nt:, nt:] = P_T[:nt, :nt].conj()

        H = H @ P_T

        # saddle free newton method
        evals, S = eig(H)

        H_abs_inv = S @ np.diag(1 / (np.abs(evals) + lam)) @ la.inv(S)
        Delta_K[k] = ((H_abs_inv @ G)[:nt]).reshape(rK, pdim, pdim)

    Delta = tangent_proj(K, Delta_K, d, rK)

    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta, K, E, rho, J, y), method=ls, options={"maxiter": 200})
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)

    return K_new


def SFN_riem_Hess_full(K, E, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA"):
    """Riemannian saddle free Newton step on product manifold of all gates

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize

    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations
    """
    pdim = int(np.sqrt(r))
    n = rK * pdim
    nt = rK * r
    H = np.zeros((2, d, nt, 2, d, nt)).astype(np.complex128)
    P_T = np.zeros((2, d, nt, 2, d, nt)).astype(np.complex128)
    G = np.zeros((2, d, nt)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))

    # compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, d, r, rK)
    dd, dconjd = ddM(X, K, E, rho, J, y, d, r, rK)

    # Second derivatives
    Fyconjy = dM11.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((d, nt, d, nt))
    Fyy = dM10.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((d, nt, d, nt))

    for k in range(d):
        Fy = dK_[k].reshape((n, pdim))
        Y = K[k].reshape((n, pdim))
        rGrad = Fy.conj() - Y @ Fy.T @ Y

        G[0, k, :] = rGrad.reshape(-1)
        G[1, k, :] = rGrad.conj().reshape(-1)

        P = np.eye(n) - Y @ Y.T.conj()
        T = transp(n, pdim)
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        # Riemannian Hessian with correction terms
        H[0, k, :, 0, k, :] = H00
        H[0, k, :, 1, k, :] = H01
        H[1, k, :, 0, k, :] = H01.conj()
        H[1, k, :, 1, k, :] = H00.conj()

        # Tangent space projection
        P_T[0, k, :, 0, k, :] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
        P_T[0, k, :, 1, k, :] = -np.kron(Y, Y.T) @ T / 2
        P_T[1, k, :, 0, k, :] = P_T[0, k, :, 1, k, :].conj()
        P_T[1, k, :, 1, k, :] = P_T[0, k, :, 0, k, :].conj()

        for k2 in range(d):
            if k2 != k:
                Yk2 = K[k2].reshape(n, pdim)
                H[0, k2, :, 0, k, :] = Fyconjy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyy[k, :, k2, :].T
                H[0, k2, :, 1, k, :] = Fyy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyconjy[k, :, k2, :].T
                H[1, k2, :, 0, k, :] = H[0, k2, :, 1, k, :].conj()
                H[1, k2, :, 1, k, :] = H[0, k2, :, 0, k, :].conj()

    H = H.reshape(2 * d * nt, -1) @ P_T.reshape((2 * d * nt, -1))

    # application of saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()
    Delta_K = ((H_abs_inv @ G.reshape(-1))[: d * nt]).reshape((d, rK, pdim, pdim))

    # Delta_K is already in tangent space but not to sufficient numerical accuracy
    Delta = tangent_proj(K, Delta_K, d, rK)
    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta, K, E, rho, J, y), method=ls, options={"maxiter": 20})
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)
    return K_new


def optimize(y, J, d, r, rK, n_povm, method, K, rho, A, B, fixed_elements):
    """Full gate set optimization update alternating on E, K and rho

    Parameters
    ----------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    method : {"SFN", "GD"}
        Optimization method, Default: "SFN"
    K : numpy array
        Current estimates of Kraus operators
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    A : numpy array
        Current POVM parametrization
    B : numpy array
        Current initial state parametrization

    Returns
    -------
    K_new : numpy array
        Updated estimates of Kraus operators
    X_new : numpy array
        Updated estimates of superoperatos corresponding to K_new
    E_new : numpy array
        Updated POVM estimate
    rho_new : numpy array
        Updated initial state estimate
    A_new : numpy array
        Updated POVM parametrization
    B_new : numpy array
        Updated initial state parametrization
    """
    if "E" in fixed_elements:
        A_new = A
        E_new = np.array([(A_new[i].T.conj() @ A_new[i]).reshape(-1) for i in range(n_povm)])
    else:
        A_new = A_SFN_riem_Hess(K, A, B, y, J, d, r, n_povm)
        E_new = np.array([(A_new[i].T.conj() @ A_new[i]).reshape(-1) for i in range(n_povm)])

    if any(((f"G%i" % i in fixed_elements) for i in range(d))):
        fixed_gates = np.array([(f"G%i" % i in fixed_elements) for i in range(d)])
        if method == "SFN":
            K_new = SFN_riem_Hess(K, E_new, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA", fixed_gates=fixed_gates)
        else:
            K_new = gd(K, E_new, rho, y, J, d, r, rK, ls="COBYLA", fixed_gates=fixed_gates)
    else:
        if method == "SFN":
            K_new = SFN_riem_Hess_full(K, E_new, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA")
        else:
            fixed_gates = np.array([(f"G%i" % i in fixed_elements) for i in range(d)])
            K_new = gd(K, E_new, rho, y, J, d, r, rK, fixed_gates=fixed_gates, ls="COBYLA")

    if "rho" in fixed_elements:
        rho_new = rho
        B_new = B
    else:
        B_new = B_SFN_riem_Hess(K_new, A_new, B, y, J, d, r, n_povm, lam=1e-3)
        rho_new = (B_new @ B_new.T.conj()).reshape(-1)
    X_new = np.einsum("ijkl,ijnm -> iknlm", K_new, K_new.conj()).reshape((d, r, r))
    return K_new, X_new, E_new, rho_new, A_new, B_new


def run_mGST(
    *args,
    method="SFN",
    max_inits=10,
    max_iter=200,
    final_iter=70,
    target_rel_prec=1e-4,
    threshold_multiplier=3,
    fixed_elements=None,
    init=None,
    testing=False,
):  # pylint: disable=too-many-branches
    """Main mGST routine

    Parameters
    ----------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    bsize : int
        Size of the batch (number of sequences)
    meas_samples : int
        Number of samples taken per gate sequence to obtain measurement array y
    method : {"SFN", "GD"}
        Optimization method, Default: "SFN"
    max_iter : int
        Maximum number of iterations on batches; Default: 200
    final_iter : int
        Maximum number of iterations on full data set; Default: 70
    target_rel_prec : float
        Target precision relative to stopping value at which the final iteration loop breaks
    init : [ , , ]
        List of 3 numpy arrays in the format [X,E,rho], that can be used as an initialization;
        If no initialization is given a random initialization is used

    Returns
    -------
    K : numpy array
        Updated estimates of Kraus operators
    X : numpy array
        Updated estimates of superoperatos corresponding to K_new
    E : numpy array
        Updated POVM estimate
    rho : numpy array
        Updated initial state estimate
    res_list : list
        Collected objective function values after each iteration
    """
    y, J, _, d, r, rK, n_povm, bsize, meas_samples = args
    t0 = time.time()
    pdim = int(np.sqrt(r))
    # stopping criterion (Factor 3 can be increased if model mismatch is high)
    delta = threshold_multiplier * (1 - y.reshape(-1)) @ y.reshape(-1) / len(J) / n_povm / meas_samples

    if not fixed_elements:
        fixed_elements = []

    if any(((f"G%i" % i in fixed_elements) for i in range(d))) and method == "SFN":
        warn(
            f"The SFN method with fixed gates is currently only implemented via \n"
            f"iterative updates over individual gates and might lead to a slower converges \n"
            f"compared to the default SFN method.",
            stacklevel=2,
        )

    success = False
    qcvv_logger.info(f"Starting mGST optimization...")

    if init:
        K, E = (init[0], init[1])
        # offset small negative eigenvalues for stability
        rho = init[2] + 1e-14 * np.eye(pdim).reshape(-1)
        max_inits = 0
    else:
        K, _, E, rho = random_gs(d, r, rK, n_povm)

    A = np.array([la.cholesky(E[k].reshape(pdim, pdim) + 1e-14 * np.eye(pdim)).T.conj() for k in range(n_povm)])
    B = la.cholesky(rho.reshape(pdim, pdim))
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    res_list = [objf(X, E, rho, J, y)]

    for i in range(max_inits):
        with logging_redirect_tqdm(loggers=[qcvv_logger]):
            for _ in trange(max_iter):
                yb, Jb = batch(y, J, bsize)
                K, X, E, rho, A, B = optimize(yb, Jb, d, r, rK, n_povm, method, K, rho, A, B, fixed_elements)
                res_list.append(objf(X, E, rho, J, y))
                if res_list[-1] < delta:
                    qcvv_logger.info(f"Batch optimization successful, improving estimate over full data....")
                    success = True
                    break
        if testing:
            plot_objf(res_list, delta, f"Objective function for batch optimization")
        if success:
            break
        qcvv_logger.info(f"Run ", i, f"failed, trying new initialization...")

    if not success and max_inits > 0:
        qcvv_logger.info(f"Success threshold not reached, attempting optimization over full data set...")
    with logging_redirect_tqdm(loggers=[qcvv_logger]):
        for _ in trange(final_iter):
            K, X, E, rho, A, B = optimize(y, J, d, r, rK, n_povm, method, K, rho, A, B, fixed_elements)
            res_list.append(objf(X, E, rho, J, y))
            if np.abs(res_list[-2] - res_list[-1]) < delta * target_rel_prec:
                break
    if testing:
        plot_objf(res_list, delta, f"Objective function over batches and full data")
    if success or (res_list[-1] < delta):
        qcvv_logger.info(f"Convergence criterion satisfied")
    else:
        qcvv_logger.info(
            f"Convergence criterion not satisfied,inspect results and consider increasing max_iter or using new initializations.",
        )
    qcvv_logger.info(
        f"Final objective {Decimal(res_list[-1]):.2e} in time {(time.time() - t0):.2f}s",
    )
    return K, X, E, rho, res_list
