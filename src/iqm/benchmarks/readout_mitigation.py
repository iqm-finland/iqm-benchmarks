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
"""M3 modification for readout mitigation at IQM QPU's."""
import logging
from math import ceil
import threading
from typing import Any, Dict, Iterable, List
import warnings

import mthree
from mthree.circuits import _marg_meas_states, _tensor_meas_states, balanced_cal_circuits, balanced_cal_strings
from mthree.classes import QuasiCollection
from mthree.exceptions import M3Error
from mthree.mitigation import _job_thread
from mthree.utils import final_measurement_mapping
from qiskit import transpile  # pylint: disable = no-name-in-module
from qiskit.providers import Backend, BackendV1, BackendV2

from iqm.benchmarks.logging_config import qcvv_logger
from iqm.benchmarks.utils import get_iqm_backend, timeit
from iqm.qiskit_iqm import IQMCircuit as QuantumCircuit
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase


# The code here is close to the original M3 code licenced under Apache 2 as well (https://github.com/Qiskit/qiskit-addon-mthree/blob/main/LICENSE.txt).
# We deactivate this to not break functionality.
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-branches, too-many-statements,
# pylint: disable=attribute-defined-outside-init, too-few-public-methods, access-member-before-definition
class M3IQM(mthree.M3Mitigation):
    """M3 readout mitigation class modified to work with IQM devices."""

    def cals_from_system(
        self,
        qubits=None,
        shots=None,
        method=None,
        initial_reset=False,
        rep_delay=None,
        cals_file=None,
        async_cal=False,
        cal_id=None,
    ):
        """Grab calibration data from system.

        Parameters:
            qubits (array_like): Qubits over which to correct calibration data. Default is all.
            shots (int): Number of shots per circuit. min(1e4, max_shots).
            method (str): Type of calibration, 'balanced' (default for hardware),
                         'independent' (default for simulators), or 'marginal'.
            initial_reset (bool): Use resets at beginning of calibration circuits, default=False.
            rep_delay (float): Delay between circuits on IBM Quantum backends.
            cals_file (str): Output path to write JSON calibration data to.
            async_cal (bool): Do calibration async in a separate thread, default is False.

        Raises:
            M3Error: Called while a calibration currently in progress.
        """
        if self._thread:
            raise M3Error("Calibration currently in progress.")
        if qubits is None:
            qubits = range(self.num_qubits)
            # Remove faulty qubits if any
            if any(self.system_info["inoperable_qubits"]):
                qubits = list(
                    filter(
                        lambda item: item not in self.system_info["inoperable_qubits"],
                        list(range(self.num_qubits)),
                    )
                )
                warnings.warn(
                    f"Backend reporting inoperable qubits. Skipping calibrations for: %s",
                    self.system_info["inoperable_qubits"],
                )

        if method is None:
            method = "balanced"
            # if self.system_info["simulator"]:
            #     method = "independent"
        self.cal_method = method
        self.rep_delay = rep_delay
        self.cals_file = cals_file
        self.cal_timestamp = None
        self._grab_additional_cals(
            qubits,
            shots=shots,
            method=method,
            rep_delay=rep_delay,
            initial_reset=initial_reset,
            async_cal=async_cal,
            cal_id=cal_id,
        )

    def _grab_additional_cals(
        self,
        qubits,
        shots=None,
        method="balanced",
        rep_delay=None,
        initial_reset=False,
        async_cal=False,
        cal_id=None,
    ):
        """Grab missing calibration data from backend.

        Parameters:
            qubits (array_like): List of measured qubits.
            shots (int): Number of shots to take, min(1e4, max_shots).
            method (str): Type of calibration, 'balanced' (default), 'independent', or 'marginal'.
            rep_delay (float): Delay between circuits on IBM Quantum backends.
            initial_reset (bool): Use resets at beginning of calibration circuits, default=False.
            async_cal (bool): Do calibration async in a separate thread, default is False.

        Raises:
            M3Error: Backend not set.
            M3Error: Faulty qubits found.
        """
        if self.system is None:
            raise M3Error("System is not set.  Use 'cals_from_file'.")
        if self.single_qubit_cals is None:
            self.single_qubit_cals = [None] * self.num_qubits
        if self.cal_shots is None:
            if shots is None:
                shots = min(self.system_info["max_shots"], 10000)
            self.cal_shots = shots
        if self.rep_delay is None:
            self.rep_delay = rep_delay

        if method not in ["independent", "balanced", "marginal"]:
            raise M3Error(
                f"Invalid calibration method: {method}. Valid methods are 'independent', 'balanced', or 'marginal'."
            )

        if isinstance(qubits, dict):
            # Assuming passed a mapping
            qubits = list(set(qubits.values()))
        elif isinstance(qubits, list):
            # Check if passed a list of mappings
            if isinstance(qubits[0], dict):
                # Assuming list of mappings, need to get unique elements
                _qubits = []
                for item in qubits:
                    _qubits.extend(list(set(item.values())))
                qubits = list(set(_qubits))

        # Do check for inoperable qubits here
        inoperable_overlap = list(set(qubits) & set(self.system_info["inoperable_qubits"]))
        if any(inoperable_overlap):
            raise M3Error(f"Attempting to calibrate inoperable qubits: {inoperable_overlap}")

        num_cal_qubits = len(qubits)
        cal_strings = []
        # shots is needed here because balanced cals will use a value
        # different from cal_shots
        shots = self.cal_shots
        if method == "marginal":
            trans_qcs = _marg_meas_states(qubits, self.num_qubits, initial_reset=initial_reset)
        elif method == "balanced":
            cal_strings = balanced_cal_strings(num_cal_qubits)
            trans_qcs = balanced_cal_circuits(cal_strings, qubits, self.num_qubits, initial_reset=initial_reset)
            shots = self.cal_shots // num_cal_qubits
            if self.cal_shots / num_cal_qubits != shots:
                shots += 1
            self._balanced_shots = shots * num_cal_qubits
        # Independent
        else:
            trans_qcs = []
            for qubit in qubits:
                trans_qcs.extend(_tensor_meas_states(qubit, self.num_qubits, initial_reset=initial_reset))

        num_circs = len(trans_qcs)
        # check for max number of circuits per job
        if isinstance(self.system, BackendV1):
            # Aer simulator has no 'max_experiments'
            max_circuits = getattr(self.system.configuration(), "max_experiments", 300)
        elif isinstance(self.system, BackendV2):
            max_circuits = self.system.max_circuits
            # Needed for https://github.com/Qiskit/qiskit-terra/issues/9947
            if max_circuits is None:
                max_circuits = 300
        else:
            raise M3Error("Unknown backend type")
        # Determine the number of jobs required
        num_jobs = ceil(num_circs / max_circuits)
        # Get the slice length
        circ_slice = ceil(num_circs / num_jobs)
        circs_list = [trans_qcs[kk * circ_slice : (kk + 1) * circ_slice] for kk in range(num_jobs - 1)] + [
            trans_qcs[(num_jobs - 1) * circ_slice :]
        ]
        # Do job submission here
        # This Backend check is here for Qiskit direct access.  Should be removed later.
        jobs = []
        if not isinstance(self.system, Backend):
            for circs in circs_list:
                transpiled_circuit = transpile(circs, self.system, optimization_level=0)
                _job = self.system.run(transpiled_circuit, shots=shots, rep_delay=self.rep_delay)
                jobs.append(_job)

        # *****************************************
        else:  # That's what IQM backend do!
            # *****************************************
            for circs in circs_list:
                if cal_id is None:
                    _job = self.system.run(circs, shots=shots, rep_delay=self.rep_delay)
                else:
                    _job = self.system.run(
                        circs,
                        shots=shots,
                        rep_delay=self.rep_delay,
                        calibration_set_id=cal_id,
                    )
                jobs.append(_job)
                qcvv_logger.info(f"REM: {len(circs)} calibration circuits to be executed!")

        # Execute job and cal building in new thread.
        self._job_error = None
        if async_cal:
            thread = threading.Thread(
                target=_job_thread,
                args=(jobs, self, qubits, num_cal_qubits, cal_strings),
            )
            self._thread = thread
            self._thread.start()
        else:
            _job_thread(jobs, self, qubits, num_cal_qubits, cal_strings)


def readout_error_m3(counts: Dict[str, float], mit: M3IQM, qubits: Iterable) -> Dict[str, float]:
    """Counts processor using M3IQM for readout error mitigation.

    The qubits argument can be either a dictionary coming from `mthree.utils.final_measurement_mapping`
    or an array like the initial layout coming from [backend.qubit_name_to_index(name)]

    returns a dictionary of quasiprobabilties.

    NOTE: we could also pass a list of input counts and then this would return a list of quasiprobabilities.
    This would not work out of the box for us since we need the annotations of either Dict or List (not Union).
    """
    return mit.apply_correction(counts, qubits)


@timeit
def apply_readout_error_mitigation(
    backend_arg: str | IQMBackendBase,
    transpiled_circuits: List[QuantumCircuit],
    counts: List[Dict[str, int]],
    mit_shots: int = 1000,
) -> List[tuple[Any, Any]] | List[tuple[QuasiCollection, list]] | List[QuasiCollection]:
    """
    Args:
        backend_arg (str | IQMBackendBase): the backend to calibrate an M3 mitigator against.
        transpiled_circuits (List[QuantumCircuit]): the list of transpiled quantum circuits.
        counts (List[Dict[str, int]]): the measurement counts corresponding to the circuits.
        mit_shots (int): number of shots per circuit.
    Returns:
        tuple[Any, Any] | tuple[QuasiCollection, list] | QuasiCollection: a list of dictionaries with REM-corrected quasiprobabilities for each outcome.
    """
    # M3IQM uses mthree.mitigation, which for some reason displays way too many INFO messages
    logging.getLogger().setLevel(logging.WARN)
    if isinstance(backend_arg, str):
        backend = get_iqm_backend(backend_arg)
    else:
        backend = backend_arg

    # Initialize with the given system and get calibration data
    qubits_rem = [final_measurement_mapping(c) for c in transpiled_circuits]

    mit = M3IQM(backend)
    mit.cals_from_system(qubits_rem, shots=mit_shots)
    # Apply the REM correction to the given measured counts
    rem_quasidistro = [mit.apply_correction(c, q) for c, q in zip(counts, qubits_rem)]
    logging.getLogger().setLevel(logging.INFO)

    return rem_quasidistro
