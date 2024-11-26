"""
Interface to pyGSTi
"""

import numpy as np
from pygsti.baseobjs import Basis, Label
from pygsti.circuits import create_lsgst_circuits
from pygsti.models.modelconstruction import create_explicit_model_from_expressions
from pygsti.report.reportables import entanglement_fidelity
from pygsti.tools import change_basis


def pygsti_model_to_arrays(model, basis="pp"):
    """Turn the gate set of a pygsti model into numpy arrays used by mGST.

    Parameters
    ----------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism

    basis : {'pp','std'}
        The basis in which the output gate set will be given. It can be either
        'pp' (Pauli basis) or 'std' (standard basis)

    Returns
    -------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2)
    rho : numpy array
        Initial state vector of shape (dimension^2)
    """
    X = []
    op_Labels = list(model.__dict__["operations"].keys())
    effect_Labels = list(model["Mdefault"].keys())
    E = np.array([model["Mdefault"][label].to_dense().reshape(-1) for label in effect_Labels])
    rho = model["rho0"].to_dense().reshape(-1)
    for op_Label in op_Labels:
        X.append(model[op_Label].to_dense())
    if basis == "pp":
        return (np.array(X).astype(np.complex128), E.astype(np.complex128), rho.astype(np.complex128))
    if basis == "std":
        return pp2std(np.array(X), E, rho)
    raise ValueError(f"Wrong basis input, use either std or pp")


def average_gate_fidelities(model1, model2, pdim, basis_string="pp"):
    """Return the average gate fidelities between gates of two pygsti models.

    Parameters
    ----------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    model2 : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    pdim : int
        physical dimension
    basis : {'pp','std'}
        The basis in which the input models are gíven. It can be either
        'pp' (Pauli basis) or 'std' (standard basis, Default)

    Returns
    -------
    fidelities : 1D numpy array
        Array containing the average gate fidelities for all gates
    """
    ent_fids = []
    basis = Basis.cast(basis_string, pdim**2)
    labels1 = list(model1.__dict__["operations"].keys())
    labels2 = list(model2.__dict__["operations"].keys())

    for i, label in enumerate(labels1):
        ent_fids.append(float(entanglement_fidelity(model1[label], model2[labels2[i]], basis)))
    fidelities = (np.array(ent_fids) * pdim + 1) / (pdim + 1)
    return fidelities


def model_agfs(model, pdim):
    """Return the average gate fidelities between gates of two pygsti models.

    Parameters
    ----------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    pdim : int
        physical dimension

    Returns
    -------
    fidelities : 1D numpy array
        Average gate fidelities between all different gates of the input model
    """
    ent_fids = []
    basis = Basis.cast("pp", pdim**2)
    labels = list(model.__dict__["operations"].keys())
    for i, _ in enumerate(labels):
        for j, _ in enumerate(labels):
            if j > i:
                ent_fids.append(float(entanglement_fidelity(model[labels[i]], model[labels[j]], basis)))
    fidelities = (np.array(ent_fids) * pdim + 1) / (pdim + 1)
    return fidelities


def arrays_to_pygsti_model(X, E, rho, basis="std"):
    """Turns a gate set given by numpy arrays into a pygsti model

    pygsti model is by default in Pauli-basis

    Parameters
    ----------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2)
    rho : numpy array
        Initial state vector of shape (dimension^2)
    basis : {'pp','std'}
        The basis in which the INPUT gate set is given. It can be either
        'pp' (Pauli basis) or 'std' (standard basis)

    Returns
    -------
    model : pygsti ExplicitOpModel object
        Contains the model parameters in the Pauli transfer matrix formalism
    """
    d = X.shape[0]
    pdim = int(np.sqrt(len(rho)))
    Id = np.zeros(pdim**2)
    Id[0] = 1
    effect_label_str = [f"%i" % k for k in range(E.shape[0])]
    if basis == "std":
        X, E, rho = std2pp(X, E, rho)
    mdl_out = create_explicit_model_from_expressions(
        list(range(int(np.log(pdim) / np.log(2)))),
        [Label(f"G%i" % i) for i in range(d)],
        [f":".join([f"I(%i)" % i for i in range(int(np.log(pdim) / np.log(2)))]) for length in range(d)],
        effect_labels=effect_label_str,
    )
    mdl_out["rho0"] = np.real(rho)
    mdl_out["Mdefault"].from_vector(np.real(E).reshape(-1))
    for i in range(d):
        mdl_out[Label(f"G%i" % i)] = np.real(X[i])
    return mdl_out


def std2pp(X, E, rho):
    """Basis change of an mGST model from the standard basis to the Pauli basis

    Parameters
    ----------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
        in standard basis
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in standard basis
    rho : numpy array
        Initial state vector of shape (dimension^2) in standard basis

    Returns
    -------
    Xpp : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
        in Pauli basis
    Epp : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in Pauli basis
    rhopp : numpy array
        Initial state vector of shape (dimension^2) in Pauli basis
    """
    Xpp = np.array([np.array(change_basis(X[i], "std", "pp")) for i in range(X.shape[0])])
    Epp = np.array([np.array(change_basis(E[i], "std", "pp")) for i in range(E.shape[0])])
    return (Xpp.astype(np.complex128), Epp.astype(np.complex128), change_basis(rho, "std", "pp").astype(np.complex128))


def pp2std(X, E, rho):
    """Basis change of an mGST model from the Pauli basis to the standard basis.

    Parameters
    ----------
    X : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
        in Pauli basis
    E : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in Pauli basis
    rho : numpy array
        Initial state vector of shape (dimension^2) in Pauli basis

    Returns
    -------
    Xstd : numpy array
        Gate set tensor of shape (Number of Gates, Kraus rank, dimension^2, dimension^2)
        in standard basis
    Estd : numpy array
        POVM matrix of shape (#POVM elements, dimension^2) in standard basis
    rhostd : numpy array
        Initial state vector of shape (dimension^2) in standard
    """
    Xstd = np.array([np.array(change_basis(X[i], "pp", "std")) for i in range(X.shape[0])])
    Estd = np.array([np.array(change_basis(E[i], "pp", "std")) for i in range(E.shape[0])])
    return (
        Xstd.astype(np.complex128),
        Estd.astype(np.complex128),
        change_basis(rho, "pp", "std").astype(np.complex128),
    )


def pygstiExp_to_list(model, max_germ_len):
    """Takes the sequences of a pyGSTi model and turns them into a sequence list for mGST.

    Parameters
    ----------
    model : pygsti ExplicitOpModel object
        The model containing the gate set parameters in the Pauli transfer matrix formalism.
    max_germ_len : int
        The maximum number of germ repetitions in the pyGSTi circuit design.

    Returns
    -------
    J_GST : numpy array, shape (num_experiments, max_sequence_length)
        A 2D array where each row contains the gate indices of a gate sequence. Each gate index
        corresponds to a specific gate in the model's target gate set.
    """
    prep_fiducials = model.prep_fiducials()
    meas_fiducials = model.meas_fiducials()
    germs = model.germs()
    maxLengths = [max_germ_len]
    listOfExperiments = create_lsgst_circuits(model.target_model(), prep_fiducials, meas_fiducials, germs, maxLengths)
    op_Labels = list(model.target_model().__dict__["operations"].keys())
    exp_list = []
    max_length = max((len(x.to_pythonstr(op_Labels)) for x in listOfExperiments))
    for i, exp_current in enumerate(listOfExperiments):
        exp = exp_current.to_pythonstr(op_Labels)
        gate_numbers = [ord(x) - 97 for x in exp.lower()]
        gate_numbers = np.pad(gate_numbers, (0, max_length - len(gate_numbers)), "constant", constant_values=-1)
        exp_list.append(gate_numbers)

    J_GST = np.array([[int(exp_list[i][j]) for j in range(max_length)] for i in range(len(exp_list))])
    return J_GST
