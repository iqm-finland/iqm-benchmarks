"""
Functions related to optimization on manifolds
"""

import numpy as np
from scipy.linalg import eigh

from mGST.low_level_jit import ddM, dK_dMdM, objf


def eigy_expm(A):
    """Custom Matrix exponential using the eigendecomposition of numpy.linalg

    Parameters
    ----------
    A : numpy array
        Matrix to be exponentiated

    Returns
    -------
    M: numpy array
        Matrix exponential of A
    """
    vals, vects = np.linalg.eig(A)
    return np.einsum("...ik, ...k, ...kj -> ...ij", vects, np.exp(vals), np.linalg.inv(vects))


def tangent_proj(K, Z, d, rK):
    """Projection onto the local tangent space

    Parameters
    ----------
    K : numpy array
        Current position
    Z : numpy array
        Element of the ambient space to be projected onto the tangent space at K
    d : int
        Number of matrices which are projected (i.e. number of gates in the gate set)
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension

    Returns
    -------
    G: numpy array
        Projection of Z onto the tangent space of the Stiefel manifold at position K

    Notes:
        The projection is done with respect to the canonical metrik.
    """
    pdim = K.shape[2]
    n = pdim * rK
    K = K.reshape(d, n, pdim)
    Z = Z.reshape(d, n, pdim)
    G = np.ascontiguousarray(np.zeros((d, n, pdim)).astype(np.complex128))
    for i in range(d):
        G[i] += Z[i] - (K[i] @ K[i].T.conj() @ Z[i] + K[i] @ Z[i].T.conj() @ K[i]) / 2
    return G


def update_K_geodesic(K, H, a):
    """Compute a new point on the geodesic

    G = np.ascontiguousarray(np.zeros((d,n,pdim)).astype(np.complex128))

    Compute a new point on the geodesic

    Parameters
    ----------
    K : numpy array
        Current position
    H : numpy array
        Element of the tangent space at K and local direction of the geodesic
    a : float
        Geodesic curve parameter

    Returns
    -------
    K_new: numpy array
        New position given by K_new = K(a) with K(a) being a geodesic with K(0) = K, [d/dt K](0) = H
    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    n = pdim * rK
    K = K.reshape(d, n, pdim)
    K_new = K.copy()
    AR_mat = np.zeros((2 * pdim, 2 * pdim)).astype(np.complex128)
    for i in range(d):
        Q, R = np.linalg.qr((np.eye(n) - K[i] @ K[i].T.conj()) @ H[i])
        AR_mat[:pdim, :pdim] = K[i].T.conj() @ H[i]
        AR_mat[pdim:, :pdim] = R
        AR_mat[:pdim, pdim:] = -R.T.conj()
        MN = eigy_expm(-a * AR_mat) @ np.eye(2 * pdim, pdim)
        K_new[i] = K[i] @ MN[:pdim, :] + Q @ MN[pdim:, :]
    return K_new.reshape(d, rK, pdim, pdim)


def lineobjf_isom_geodesic(a, H, K, E, rho, J, y):
    """Compute objective function at position on geodesic

    Parameters
    ----------
    a : float
        Geodesic curve parameter
    H : numpy array
        Element of the tangent space at K and local direction of the geodesic
    K : numpy array
        Current position
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        The columns contain the outcome probabilities for different povm elements

    Returns
    -------
    f(a): float
        Objective function value at new position along the geodesic
    """
    d = K.shape[0]
    pdim = K.shape[2]
    r = pdim**2
    K_test = update_K_geodesic(K, H, a)
    X_test = np.einsum("ijkl,ijnm -> iknlm", K_test, K_test.conj()).reshape((d, r, r))
    return objf(X_test, E, rho, J, y)


def update_A_geodesic(A, H, a):
    """Compute a new point on the geodesic for the POVM parametrization

    Parameters
    ----------
    A : numpy array
        Current position
    H : numpy array
        Element of the tangent space at A and local direction of the geodesic
    a : float
        Geodesic curve parameter

    Returns
    -------
    A_new: numpy array
        New position given by A_new = A(a) with A(a) being a geodesic with A(0) = A, [d/dt A](0) = H
    """
    n_povm = A.shape[0]
    pdim = A.shape[1]
    n = pdim * n_povm
    A = A.reshape(n, pdim)
    H = H.reshape(n, pdim)
    A_new = A.copy()
    AR_mat = np.zeros((2 * pdim, 2 * pdim)).astype(np.complex128)
    Q, R = np.linalg.qr((np.eye(n) - A @ A.T.conj()) @ H)
    AR_mat[:pdim, :pdim] = A.T.conj() @ H
    AR_mat[pdim:, :pdim] = R
    AR_mat[:pdim, pdim:] = -R.T.conj()
    MN = eigy_expm(-a * AR_mat) @ np.eye(2 * pdim, pdim)
    A_new = A @ MN[:pdim, :] + Q @ MN[pdim:, :]
    return A_new.reshape(n_povm, pdim, pdim)


def update_B_geodesic(B, H, a):
    """Compute a new point on the geodesic for the initial state parametrization

    Parameters
    ----------
    B : numpy array
        Current initial state parametrization
    H : numpy array
        Element of the tangent space at B and local direction of the geodesic
    a : float
        Geodesic curve parameter

    Returns
    -------
    A_new: numpy array
        New position given by B_new = B(a) with B(a) being a geodesic with B(0) = B,
        [d/dt B](t=0) = H
    """
    pdim = B.shape[0]
    n = pdim**2
    B = B.reshape(n)
    H = H.reshape(n)
    B_temp = B.copy()
    AR_mat = np.zeros((2, 2)).astype(np.complex128)
    R = np.linalg.norm((np.eye(n) - np.outer(B, B.T.conj())) @ H)
    Q = ((np.eye(n) - np.outer(B, B.T.conj())) @ H) / R
    AR_mat[0, 0] = B.T.conj() @ H
    AR_mat[1, 0] = R
    AR_mat[0, 1] = -R.T.conj()
    MN = eigy_expm(-a * AR_mat) @ np.array([1, 0])
    B_temp = B * MN[0] + Q * MN[1]
    return B_temp.reshape(pdim, pdim)


def lineobjf_A_geodesic(a, H, X, A, rho, J, y):
    """Compute objective function at position on geodesic for POVM parametrization

    Parameters
    ----------
    a : float
        Geodesic curve parameter
    H : numpy array
        Element of the tangent space at A and local direction of the geodesic
    X : numpy array
        Current gate estimate
    A : numpy array
        Current position
    rho : numpy array
        Current initial state estimate
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        The columns contain the outcome probabilities for different povm elements

    Returns
    -------
    f(a): float
        Objective function value at new position along the geodesic
    """
    n_povm = A.shape[0]
    A_test = update_A_geodesic(A, H, a)
    E_test = np.array([(A_test[i].T.conj() @ A_test[i]).reshape(-1) for i in range(n_povm)])
    return objf(X, E_test, rho, J, y)


def lineobjf_B_geodesic(a, H, X, E, B, J, y):
    """Compute objective function at position on geodesic for the initial state parametrization

    Parameters
    ----------
    a : float
        Geodesic curve parameter
    H : numpy array
        Element of the tangent space at B and local direction of the geodesic
    X : numpy array
        Current gate estimate
    E : numpy array
        Current POVM estimate
    B : numpy array
        Current initial state parametrization
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        The columns contain the outcome probabilities for different povm elements

    Returns
    -------
    f(a): float
        Objective function value at new position along the geodesic
    """
    B_test = update_B_geodesic(B, H, a)
    rho_test = (B_test @ B_test.T.conj()).reshape(-1)
    return objf(X, E, rho_test, J, y)


def lineobjf_A_B(a, v, delta_v, X, C, y, J, argument):
    """Compute objective function at translated position

    Parameters
    ----------
    a : float
        Step size
    v : numpy array
        Current position
    delta_v : numpy array
        Step direction
    X : numpy array
        Current gate estimate
    C : numpy array
        Current initial state/POVM estimate that is not updated
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        The columns contain the outcome probabilities for different povm elements
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    argument: string
        Takes the following options: "E" or "rho", and indicates which gate set
        component is optimized over

    Returns
    -------
    f(a): float
        Objective function value at new position v + a*delta_v

    Notes
    -------
    This function is used for the line search with linear updates v_new = v + a*delta_v,
    where v can be either the POVM estimate or the state estimate.
    """
    v_test = v - a * delta_v
    if argument == "rho":
        rho_test = (v_test @ v_test.T.conj()).reshape(-1)
        return objf(X, C, rho_test, J, y)
    if argument == "E":
        E_test = (v_test @ v_test.T.conj()).reshape(-1)
        return objf(X, E_test, C, J, y)
    raise ValueError("The <argument> variable takes either E or rho")


def Hess_evals(K, E, rho, y, J):
    """Compute eigenvalues of the euclidean Hessian

    Parameters
    ----------
    K : numpy array
        Current position
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        The columns contain the outcome probabilities for different povm elements
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence

    Returns
    -------
    evals: 1D numpy array
        Eigenvalues of the euclidean Hessian for the Kraus operators at position (K,E,rho)
    """
    d = K.shape[0]
    rK = K.shape[1]
    pdim = K.shape[2]
    r = pdim**2
    n = d * rK * r
    H = np.zeros((2 * n, 2 * n)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    _, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, d, r, rK)
    dd, dconjd = ddM(X, K, E, rho, J, y, d, r, rK)

    A00 = dM11.reshape(n, n) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape(n, n)
    A10 = dM10.reshape(n, n) + np.einsum("ijklmnop->ikmojlnp", dd).reshape(n, n)
    A11 = A00.conj()
    A01 = A10.conj()

    H[:n, :n] = A00
    H[:n, n:] = A01
    H[n:, :n] = A10
    H[n:, n:] = A11

    evals, _ = eigh(H)
    return evals
