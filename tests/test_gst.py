"""Tests for compressive GST benchmark"""

from unittest.mock import patch
import multiprocessing as mp
import numpy as np

from iqm.benchmarks.compressive_gst.compressive_gst import CompressiveGST, GSTConfiguration
from mGST.additional_fns import is_positive
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
from qiskit.circuit import QuantumCircuit


class TestGST:
    backend = IQMFakeApollo()

    @patch('matplotlib.pyplot.figure')
    def test_1q(self, mock_fig):
        mp.set_start_method("spawn", force=True)
        # Testing minimal gate context
        gate_context = [QuantumCircuit(self.backend.num_qubits) for _ in range(3)]
        gate_context[0].h(0)
        gate_context[1].s(0)
        # config
        qubit_layouts = [[4], [1]]
        minimal_1Q_config = GSTConfiguration(
            qubit_layouts=[[4], [1]],
            gate_set="1QXYI",
            gate_context=gate_context,
            num_circuits=50,
            shots=100,
            rank=4,
            bootstrap_samples=2,
            max_iterations=[10, 10],
            parallel_execution=True,
        )
        benchmark = CompressiveGST(self.backend, minimal_1Q_config)
        benchmark.run()
        result = benchmark.analyze()

        for layout in qubit_layouts:
            #Check if all metrics are between 0 and 1 (currently all metrics like fidelity and diamond dist are normalized)
            assert all([0 < metric.value < 1 for metric in result.observations])

            # Check if outcomes satisfy physicality constraints
            X = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_state"]
            E = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_POVM"]
            is_positive(X,E,rho) # raises error if any gate set element is not physical

            X = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_state"]
            E = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_POVM"]
            is_positive(X,E,rho) # raises error if any gate set element is not physical

            # Check if gates in the Pauli basis have real entries
            gauge_opt_gates_pp = result.dataset.attrs[f"results_layout_{str(layout)}"]['gauge_opt_gates_Pauli_basis']
            assert np.all(np.imag(gauge_opt_gates_pp.reshape(-1)) < 1e-10)

            # Check if GST estimate fits data better than target gate set
            tvd_estimate = float(result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"]['mean_tvd_estimate_data'][""].split(" ")[0])
            tvd_target = float(result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"]['mean_tvd_target_data'][""].split(" ")[0])
            assert tvd_estimate < tvd_target

        mock_fig.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_2q(self, mock_fig):
        mp.set_start_method("spawn", force=True)
        qubit_layouts = [[2, 3]]
        minimal_2Q_GST = GSTConfiguration(
            qubit_layouts=qubit_layouts,
            gate_set="2QXYICZ",
            num_circuits=100,
            shots=100,
            rank=1,
            bootstrap_samples=2,
            max_iterations=[10, 10],
        )
        benchmark = CompressiveGST(self.backend, minimal_2Q_GST)
        benchmark.run()
        result = benchmark.analyze()

        for layout in qubit_layouts:
            #Check if all metrics are between 0 and 1 (currently all metrics like fidelity and diamond dist are normalized)
            assert all([0 < metric.value < 1 for metric in result.observations])

            # Check if outcomes satisfy physicality constraints
            X = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_state"]
            E = result.dataset.attrs[f"results_layout_{str(layout)}"]["raw_POVM"]
            is_positive(X,E,rho) # raises error if any gate set element is not physical

            X = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_gates"]
            rho = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_state"]
            E = result.dataset.attrs[f"results_layout_{str(layout)}"]["gauge_opt_POVM"]
            is_positive(X,E,rho) # raises error if any gate set element is not physical

            # Check if gates in the Pauli basis have real entries
            gauge_opt_gates_pp = result.dataset.attrs[f"results_layout_{str(layout)}"]['gauge_opt_gates_Pauli_basis']
            assert np.all(np.imag(gauge_opt_gates_pp.reshape(-1)) < 1e-10)

            # Check if GST estimate fits data better than target gate set
            tvd_estimate = float(result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"]['mean_tvd_estimate_data'][""].split(" ")[0])
            tvd_target = float(result.dataset.attrs[f"results_layout_{str(layout)}"]["full_metrics"]["Outcomes and SPAM"]['mean_tvd_target_data'][""].split(" ")[0])
            assert tvd_estimate < tvd_target

        mock_fig.assert_called()


class TestGSTDeneb(TestGST):
    backend = IQMFakeDeneb()
