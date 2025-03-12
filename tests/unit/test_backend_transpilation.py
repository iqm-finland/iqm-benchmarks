from qiskit import qpy
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from iqm.qiskit_iqm.fake_backends.fake_deneb import IQMFakeDeneb
from iqm.benchmarks.utils import perform_backend_transpilation, set_coupling_map

# Work in progress transpilation test
# class TestTranspilationCrystal:
#     backend = IQMFakeApollo()
#
#     def test_transpilation(self):
#         with open('reference_data/circuit_ghz.qpy', 'rb') as fd:
#             qc_list = qpy.load(fd)
#         with open('reference_data/circuit_ghz_transp.qpy', 'rb') as fd:
#             qc_list_transp = qpy.load(fd)
#
#         # Assert that the lists have the same length
#         assert len(qc_list) == len(qc_list_transp)
#
#         qubit_layout = [1, 3, 4, 5]
#         qiskit_optim_level = 3
#         optimize_sqg = True
#
#         fixed_coupling_map = set_coupling_map(qubit_layout, self.backend, "fixed")
#
#         # Manually transpile the circuits and compare with loaded transpiled circuits
#         qc_list_transp_test, _ = perform_backend_transpilation(
#             qc_list,
#             self.backend,
#             qubit_layout,
#             fixed_coupling_map,
#             qiskit_optim_level=qiskit_optim_level,
#             optimize_sqg=optimize_sqg,
#         )
#         for i, _ in enumerate(qc_list):
#             assert qc_list_transp_test[i].data == qc_list_transp[i].data
#
#
#
# class TestTranspilationDeneb(TestTranspilationCrystal):
#     backend = IQMFakeDeneb()
