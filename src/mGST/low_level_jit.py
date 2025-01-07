"""
All functions compiled with numba, such as tensor contractions, derivatives...
"""

import os

from numba import njit, prange
import numpy as np


def kill_files(folder):
    """Delete all files in the specified folder.

    This function iterates over all files in the given folder and attempts to delete them.
    If a file cannot be deleted, an error message is printed.

    Parameters
    ----------
    folder : str
        The path to the folder containing the files to be deleted.
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except OSError as e:
            print(f"Failed on filepath: {file_path}. Error: {e}")


def kill_numba_cache():
    """Delete all __pycache__ folders in the project directory tree.

    This function iterates through the project directory tree, looking for __pycache__ folders.
    When found, it attempts to delete all files within the __pycache__ folders using the
    `kill_files` function. If the files cannot be deleted, an error message is printed.
    """
    root_folder = os.path.realpath(__file__ + "/../../")
    for root, dirnames, _ in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except OSError as e:
                    print(f"Failed on {root}. Error: {e}")


@njit(cache=True)
def local_basis(x, b, length):
    """Convert a base-10 integer to an integer in a specified base with a fixed number of digits.

    This function takes an integer `x` in base-10 and converts it to base `b`.
    The result is returned as an array of length `length` with leading zeros.

    Parameters
    ----------
    x : int
        The input number in base-10 to be converted.
    b : int
        The target base to convert the input number to.
    length : int
        The number of output digits in the target base representation.

    Returns
    -------
    numpy.ndarray
        A numpy array of integers representing the base-`b` digits of the converted number,
        with leading zeros if necessary. The length of the array is `length`.
    """
    r = np.zeros(length).astype(np.int32)
    k = 1
    while x > 0:
        r[-k] = x % b
        x //= b
        k += 1
    return r


@njit(cache=True)
def contract(X, j_vec):
    """Contract a sequence of matrices in the given order.

    This function computes the product of a sequence of matrices specified by
    the indices in `j_vec`. The result is the contracted product of the matrices
    in the given order.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
    j_vec : numpy.ndarray
        A 1D array of indices specifying the order in which to contract the matrices in X.

    Returns
    -------
    numpy.ndarray
        The contracted product of the matrices specified by the indices in `j_vec`.
    """
    j_vec = j_vec[j_vec >= 0]
    res = np.eye(X[0].shape[0])
    res = res.astype(np.complex128)
    for j in j_vec:
        res = res.dot(X[j])
    return res


@njit(cache=True, fastmath=True)
def objf(X, E, rho, J, y):
    """Calculate the objective function value for matrices, POVM elements, and target values.

    This function computes the objective function value based on input matrices X, POVM elements E,
    density matrix rho, and target values y.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing the input matrices, of shape (n_matrices, n_rows, n_columns).
    E : numpy.ndarray
        A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the density matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the objective function will be evaluated.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.

    Returns
    -------
    float
        The objective function value for the given set of matrices, POVM elements,
        and target values, normalized by m and n_povm.
    """
    m = len(J)
    n_povm = y.shape[0]
    objf_ = 0
    for i in prange(m):  # pylint: disable=not-an-iterable
        j = J[i][J[i] >= 0]
        C = contract(X, j)
        for o in range(n_povm):
            objf_ += abs(E[o].conj() @ C @ rho - y[o, i]) ** 2
    return objf_ / m / n_povm


@njit(cache=True)
def MVE_lower(X_true, E_true, rho_true, X, E, rho, J, n_povm):
    """Compute the lower bound of the mean value error (MVE) between true and estimated parameters.

    This function calculates the lower bound of the MVE between the true parameters (X_true,
    E_true, rho_true) and the estimated parameters (X, E, rho) based on the provided J indices.

    Parameters
    ----------
    X_true : numpy.ndarray
        A 3D array containing true input matrices, of shape (n_matrices, n_rows, n_columns).
    E_true : numpy.ndarray
        A 2D array representing the true POVM elements, of shape (n_povm, r).
    rho_true : numpy.ndarray
        A 1D array representing the true density matrix.
    X : numpy.ndarray
        A 3D array containing estimated input matrices, of shape (n_matrices, n_rows, n_columns).
    E : numpy.ndarray
        A 2D array representing the estimated POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the estimated density matrix.
    J : numpy.ndarray
        A 2D array representing indices for which the objective function will be evaluated.
    n_povm : int
        The number of POVM elements.

    Returns
    -------
    tuple of float
        A tuple containing the lower bound of the mean value error and the maximum distance.
    """
    m = len(J)
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true, j)
        C = contract(X, j)
        curr = 0
        for k in range(n_povm):
            y_t = E_true[k].conj() @ C_t @ rho_true
            y = E[k].conj() @ C @ rho
            curr += np.abs(y_t - y)
        curr = curr / 2
        dist += curr
        max_dist = max(max_dist, curr)
    return dist / m, max_dist


@njit(cache=True)
def Mp_norm_lower(X_true, E_true, rho_true, X, E, rho, J, n_povm, p):
    """Compute the Mp-norm lower bound of the distance between true and estimated parameters.

    This function calculates the lower bound of the Mp-norm between the true parameters (X_true,
    E_true, rho_true) and the estimated parameters (X, E, rho) based on the provided J indices.

    Parameters
    ----------
    X_true : numpy.ndarray
        A 3D array containing true input matrices, of shape (n_matrices, n_rows, n_columns).
    E_true : numpy.ndarray
        A 2D array representing the true POVM elements, of shape (n_povm, r).
    rho_true : numpy.ndarray
        A 1D array representing the true density matrix.
    X : numpy.ndarray
        A 3D array containing estimated input matrices, of shape (n_matrices, n_rows, n_columns).
    E : numpy.ndarray
        A 2D array representing the estimated POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the estimated density matrix.
    J : numpy.ndarray
        A 2D array representing indices for which the objective function will be evaluated.
    n_povm : int
        The number of POVM elements.
    p : float
        The order of the Mp-norm (p > 0).

    Returns
    -------
    tuple of float
        A tuple containing the Mp-norm lower bound and the maximum distance.
    """
    m = len(J)
    dist: float = 0
    max_dist: float = 0
    curr: float = 0
    for i in range(m):
        j = J[i]
        C_t = contract(X_true, j)
        C = contract(X, j)
        for k in range(n_povm):
            y_t = E_true[k].conj() @ C_t @ rho_true
            y = E[k].conj() @ C @ rho
            dist += np.abs(y_t - y) ** p
        max_dist = max(max_dist, curr)
    return dist ** (1 / p) / m / n_povm, max_dist ** (1 / p)


@njit(cache=True)
def dK(X, K, E, rho, J, y, d, r, rK):
    """Compute the derivative of the objective function with respect to the Kraus tensor K.

    This function calculates the derivative of the Kraus operator K, based on the
    input matrices X, E, and rho, as well as the isometry condition.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix X, of shape (pdim, pdim).
    K : numpy.ndarray
        The Kraus operator K, reshaped to (d, rK, -1).
    E : numpy.ndarray
        A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the density matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the derivatives will be computed.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    d : int
        The number of Kraus operators.
    r : int
        The rank of the problem.
    rK : int
        The number of rows in the reshaped Kraus operator K.

    Returns
    -------
    numpy.ndarray
        The derivative objective function with respect to the Kraus tensor K,
        reshaped to (d, rK, pdim, pdim), and scaled by 2/m/n_povm.
    """
    K = K.reshape(d, rK, -1)
    pdim = int(np.sqrt(r))
    n_povm = y.shape[0]
    dK_ = np.zeros((d, rK, r))
    dK_ = np.ascontiguousarray(dK_.astype(np.complex128))
    m = len(J)
    for k in range(d):
        for n in range(m):
            j = J[n][J[n] >= 0]
            for i, j_curr in enumerate(j):
                if j_curr == k:
                    for o in range(n_povm):
                        L = E[o].conj() @ contract(X, j[:i])
                        R = contract(X, j[i + 1 :]) @ rho
                        D_ind = L @ X[k] @ R - y[o, n]
                        dK_[k] += D_ind * K[k].conj() @ np.kron(L.reshape(pdim, pdim).T, R.reshape(pdim, pdim).T)
    return dK_.reshape(d, rK, pdim, pdim) * 2 / m / n_povm


@njit(cache=True)
def dK_dMdM(X, K, E, rho, J, y, d, r, rK):
    """Compute the derivatives of the objective function with respect to K and the
    product of derivatives of the measurement map with respect to K.

    This function calculates the derivatives of K, dM10, and dM11 based on the input matrices X,
    matrix K, POVM elements E, density matrix rho, and target values y.

    Parameters
    ----------
    X : numpy.ndarray
        A 3D array containing input matrices, of shape (n_matrices, n_rows, n_columns).
    K : numpy.ndarray
        A 1D array representing the matrix K.
    E : numpy.ndarray
        A 2D array representing the POVM elements, of shape (n_povm, r).
    rho : numpy.ndarray
        A 1D array representing the density matrix.
    J : numpy.ndarray
        A 2D array representing indices for which the objective function will be evaluated.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing target values.
    d : int
        The number of dimensions for the matrix K.
    r : int
        The number of rows for the matrix K.
    rK : int
        The number of columns for the matrix K.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the derivatives of K, dM10, and dM11, each of which is a numpy.ndarray.
    """
    K = K.reshape(d, rK, -1)
    pdim = int(np.sqrt(r))
    n = d * rK * r
    n_povm = y.shape[0]
    dK_ = np.zeros((d, rK, r)).astype(np.complex128)
    dM11 = np.zeros(n**2).astype(np.complex128)
    dM10 = np.zeros(n**2).astype(np.complex128)
    m = len(J)
    for n in range(m):
        j = J[n][J[n] >= 0]
        dM = np.ascontiguousarray(np.zeros((n_povm, d, rK, r)).astype(np.complex128))
        for i, _ in enumerate(j):
            k = j[i]
            C = contract(X, j[:i])
            R = contract(X, j[i + 1 :]) @ rho
            for o in range(n_povm):
                L = E[o].conj() @ C
                D_ind = L @ X[k] @ R - y[o, n]
                dM_loc = K[k].conj() @ np.kron(L.reshape((pdim, pdim)).T, R.reshape((pdim, pdim)).T)
                dM[o, k, :, :] += dM_loc
                dK_[k] += D_ind * dM_loc
        for o in range(n_povm):
            dM11 += np.kron(dM[o].conj().reshape(-1), dM[o].reshape(-1))
            dM10 += np.kron(dM[o].reshape(-1), dM[o].reshape(-1))
    return (dK_.reshape((d, rK, pdim, pdim)) * 2 / m / n_povm, 2 * dM10 / m / n_povm, 2 * dM11 / m / n_povm)


@njit(cache=True, parallel=False)
def ddM(X, K, E, rho, J, y, d, r, rK):
    """Compute the second derivative of the objective function with respect to the Kraus tensor K.

    This function calculates the second derivative of the objective function for a given
    set of input parameters.

    Parameters
    ----------
    X : ndarray
        Array of input matrices.
    K : ndarray
        Array of Kraus operators.
    E : ndarray
        Array of measurement operators.
    rho : ndarray
        Array of quantum states.
    J : ndarray
        Array of indices corresponding to the sequence of operations.
    y : ndarray
        Array of observed probabilities.
    d : int
        Number of Kraus operators.
    r : int
        Dimension of the local basis.
    rK : int
        Number of rows in the Kraus operator matrix.

    Returns
    -------
    ddK : ndarray, shape (d, d, rK, rK, pdim, pdim, pdim, pdim)
        Second derivative of the objective function with respect to matrix elements, reshaped
        for easier manipulation.
    dconjdK : ndarray, shape (d, d, rK, rK, pdim, pdim, pdim, pdim)
        Conjugate of the second derivative of the objective function with respect to matrix
        elements, reshaped for easier manipulation.
    """
    # pylint: disable=too-many-branches, too-many-nested-blocks
    pdim = int(np.sqrt(r))
    n_povm = y.shape[0]
    ddK = np.zeros((d**2, rK**2, r, r))
    ddK = np.ascontiguousarray(ddK.astype(np.complex128))
    dconjdK = np.zeros((d**2, rK**2, r, r))
    dconjdK = np.ascontiguousarray(ddK.astype(np.complex128))
    m = len(J)
    for k in range(d**2):
        k1, k2 = local_basis(k, d, 2)
        for n in range(m):
            j = J[n][J[n] >= 0]
            for i1, j_1 in enumerate(j):
                if j_1 == k1:
                    for i2, j_2 in enumerate(j):
                        if j_2 == k2:
                            L0 = contract(X, j[: min(i1, i2)])
                            C = contract(X, j[min(i1, i2) + 1 : max(i1, i2)]).reshape(pdim, pdim, pdim, pdim)
                            R = contract(X, j[max(i1, i2) + 1 :]) @ rho
                            for o in range(n_povm):
                                L = E[o].conj() @ L0
                                if i1 == i2:
                                    D_ind = L @ X[k1] @ R - y[o, n]
                                elif i1 < i2:
                                    D_ind = L @ X[k1] @ C.reshape(r, r) @ X[k2] @ R - y[o, n]
                                else:
                                    D_ind = L @ X[k2] @ C.reshape(r, r) @ X[k1] @ R - y[o, n]

                                ddK_loc = np.zeros((rK**2, r, r)).astype(np.complex128)
                                dconjdK_loc = np.zeros((rK**2, r, r)).astype(np.complex128)
                                for rk1 in range(rK):
                                    for rk2 in range(rK):
                                        if i1 < i2:
                                            ddK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k1, rk1].conj(),
                                                R.reshape(pdim, pdim) @ K[k2, rk2].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose(1, 3, 0, 2)).reshape(r, r)

                                            ddK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                ddK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(0, 3, 2, 1)
                                            ).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k1, rk1].conj(),
                                                R.reshape(pdim, pdim).T @ K[k2, rk2].T,
                                            ) @ np.ascontiguousarray(C.transpose(1, 2, 3, 0)).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                dconjdK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(0, 2, 3, 1)
                                            ).reshape(r, r)

                                        elif i1 == i2:
                                            dconjdK_loc[rk1 * rK + rk2] = np.outer(L, R)

                                        elif i1 > i2:
                                            ddK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim) @ K[k2, rk2].conj(),
                                                R.reshape(pdim, pdim) @ K[k1, rk1].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose(1, 3, 0, 2)).reshape(r, r)

                                            ddK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                ddK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(3, 0, 1, 2)
                                            ).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.kron(
                                                L.reshape(pdim, pdim).T @ K[k2, rk2],
                                                R.reshape(pdim, pdim) @ K[k1, rk1].T.conj(),
                                            ) @ np.ascontiguousarray(C.transpose((0, 3, 2, 1))).reshape(r, r)

                                            dconjdK_loc[rk1 * rK + rk2] = np.ascontiguousarray(
                                                dconjdK_loc[rk1 * rK + rk2]
                                                .reshape(pdim, pdim, pdim, pdim)
                                                .transpose(2, 0, 1, 3)
                                            ).reshape(r, r)

                                ddK[k1 * d + k2] += D_ind * ddK_loc
                                dconjdK[k1 * d + k2] += D_ind * dconjdK_loc
    return (
        ddK.reshape(d, d, rK, rK, pdim, pdim, pdim, pdim) * 2 / m / n_povm,
        dconjdK.reshape(d, d, rK, rK, pdim, pdim, pdim, pdim) * 2 / m / n_povm,
    )


@njit(parallel=True, cache=True)
def dA(X, A, B, J, y, r, pdim, n_povm):
    """Compute the derivative of to the objective function with respect to the POVM tensor A

    This function calculates the gradient of A for a given set of input parameters.

    Parameters
    ----------
    X : ndarray
        Array of input matrices.
    A : ndarray
        Array of measurement operators.
    B : ndarray
        Array of quantum states.
    J : ndarray
        Array of indices corresponding to the sequence of operations.
    y : ndarray
        Array of observed probabilities.
    r : int
        Number of elements in each measurement operator.
    pdim : int
        Dimension of the density matrices.
    n_povm : int
        Number of measurement operators.

    Returns
    -------
    dA : ndarray
        Derivative of the objective function with respect to A.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dA_ = np.zeros((n_povm, pdim, pdim))
    dA_ = dA_.astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        jE = J[n][J[n] >= 0][0]
        j = J[n][J[n] >= 0][1:]
        inner_deriv = contract(X, j) @ rho
        D_ind = E[jE].conj().dot(inner_deriv) - y[n]
        dA_[jE] += D_ind * A[jE] @ inner_deriv.reshape(pdim, pdim).T.conj()
    return dA_


@njit(parallel=True, cache=True)
def dB(X, A, B, J, y, pdim):
    """Compute the derivative of the objective function with respect to the state tensor B.

    Parameters
    ----------
    X : ndarray
        Array of input matrices.
    A : ndarray
        Array of measurement operators.
    B : ndarray
        Array of quantum states.
    J : ndarray
        Array of indices corresponding to the sequence of operations.
    y : ndarray
        Array of observed probabilities.
    pdim : int
        Dimension of the density matrices.

    Returns
    -------
    dB : ndarray
        Derivative of the objective function with respect to the state tensor B.
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = (A.T.conj() @ A).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dB_ = np.zeros((pdim, pdim))
    dB_ = dB_.astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        jE = J[n][J[n] >= 0][0]
        j = J[n][J[n] >= 0][1:]
        inner_deriv = E[jE].conj().dot(contract(X, j))
        D_ind = inner_deriv.dot(rho) - y[n]
        dB_ += D_ind * inner_deriv.reshape(pdim, pdim).conj() @ B
    return dB_


@njit(parallel=True, cache=True)
def ddA_derivs(X, A, B, J, y, r, pdim, n_povm):
    """Calculate all nonzero terms of the second derivatives with respect to the POVM tensor A.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix X, of shape (pdim, pdim).
    A : numpy.ndarray
        A 3D array of shape (n_povm, pdim, pdim) representing the POVM elements.
    B : numpy.ndarray
        A 2D array of shape (pdim, pdim) representing the isometry matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the derivatives will be computed.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    r : int
        The rank of the problem.
    pdim : int
        The dimension of the input matrices A and B.
    n_povm : int
        The number of POVM elements.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the computed derivatives:
        - dA: The derivative w.r.t. A
        of shape (n_povm, pdim, pdim).
        - dMdM: The product of the measurement map derivatives dM and dM, of shape (n_povm, r, r).
        - dMconjdM: The product of the conjugate of dM and dM, of shape (n_povm, r, r).
        - dconjdA: The product of the conjugate of dA, of shape (n_povm, r, r).
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dA_ = np.zeros((n_povm, pdim, pdim)).astype(np.complex128)
    dMdM = np.zeros((n_povm, r, r)).astype(np.complex128)
    dMconjdM = np.zeros((n_povm, r, r)).astype(np.complex128)
    dconjdA = np.zeros((n_povm, r, r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        j = J[n][J[n] >= 0]
        R = contract(X, j) @ rho
        for o in range(n_povm):
            D_ind = E[o].conj() @ R - y[o, n]
            dM = A[o].conj() @ R.reshape(pdim, pdim).T
            dMdM[o] += np.outer(dM, dM)
            dMconjdM[o] += np.outer(dM.conj(), dM)
            dA_[o] += D_ind * dM
            dconjdA[o] += D_ind * np.kron(np.eye(pdim).astype(np.complex128), R.reshape(pdim, pdim).T)
    return dA_ * 2 / m / n_povm, dMdM * 2 / m / n_povm, dMconjdM * 2 / m / n_povm, dconjdA * 2 / m / n_povm


@njit(parallel=True, cache=True)
def ddB_derivs(X, A, B, J, y, r, pdim):
    """Calculate all nonzero terms of the second derivative with respect to the state tensor B.

    Parameters
    ----------
    X : numpy.ndarray
        The input matrix X, of shape (pdim, pdim).
    A : numpy.ndarray
        A 3D array of shape (n_povm, pdim, pdim) representing the POVM elements.
    B : numpy.ndarray
        A 2D array of shape (pdim, pdim) representing the isometry matrix.
    J : numpy.ndarray
        A 2D array representing the indices for which the derivatives will be computed.
    y : numpy.ndarray
        A 2D array of shape (n_povm, len(J)) containing the target values.
    r : int
        The rank of the problem.
    pdim : int
        The dimension of the input matrices A and B.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing the computed derivatives:
        - dB: The derivative w.r.t. B, of shape (pdim, pdim).
        - dMdM: The product of the derivatives dM and dM, of shape (r, r).
        - dMconjdM: The product of the conjugate of dM and dM, of shape (r, r).
        - dconjdB: The mixed second derivative of by dB and dB*, of shape (r, r).
    """
    n_povm = A.shape[0]
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    E = np.zeros((n_povm, r)).astype(np.complex128)
    for k in range(n_povm):
        E[k] = (A[k].T.conj() @ A[k]).reshape(-1)
    rho = (B @ B.T.conj()).reshape(-1)
    dB_ = np.zeros((pdim, pdim)).astype(np.complex128)
    dM = np.zeros((pdim, pdim)).astype(np.complex128)
    dMdM = np.zeros((r, r)).astype(np.complex128)
    dMconjdM = np.zeros((r, r)).astype(np.complex128)
    dconjdB = np.zeros((r, r)).astype(np.complex128)
    m = len(J)
    for n in prange(m):  # pylint: disable=not-an-iterable
        j = J[n][J[n] >= 0]
        C = contract(X, j)
        for o in range(n_povm):
            L = E[o].conj() @ C
            D_ind = L @ rho - y[o, n]

            dM = L.reshape(pdim, pdim) @ B.conj()
            dMdM += np.outer(dM, dM)
            dMconjdM += np.outer(dM.conj(), dM)

            dB_ += D_ind * dM
            dconjdB += D_ind * np.kron(L.reshape(pdim, pdim), np.eye(pdim).astype(np.complex128))
    return dB_ * 2 / m / n_povm, dMdM * 2 / m / n_povm, dMconjdM * 2 / m / n_povm, dconjdB.T * 2 / m / n_povm
