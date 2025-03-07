"""
Error Per Layered Gate (EPLG).
"""
from time import strftime
import xarray as xr
from iqm.qiskit_iqm.iqm_backend import IQMBackendBase

from iqm.benchmarks import Benchmark, BenchmarkCircuit
from iqm.benchmarks.benchmark import BenchmarkConfigurationBase


class EPLG(Benchmark):
    """EPLG estimates the layer fidelity of native 2Q gate layers"""

    name: str = "EPLG"

    def __init__(self, backend_arg: IQMBackendBase | str, configuration: "EPLGConfiguration"):
        """Construct the EPLG class

        Args:
            backend_arg (IQMBackendBase | str): _description_
            configuration (MirrorRBConfiguration): _description_
        """
        super().__init__(backend_arg, configuration)
        # EXPERIMENT
        self.backend_configuration_name = backend_arg if isinstance(backend_arg, str) else backend_arg.name
        self.session_timestamp = strftime("%Y%m%d-%H%M%S")
        self.execution_timestamp = ""

        # Initialize the variable to contain the circuits for each layout
        self.untranspiled_circuits = BenchmarkCircuit("untranspiled_circuits")
        self.transpiled_circuits = BenchmarkCircuit("transpiled_circuits")

        self.chain_length = configuration.chain_length

    def add_all_meta_to_dataset(self, dataset: xr.Dataset):
        """Adds all configuration metadata and circuits to the dataset variable

        Args:
            dataset (xr.Dataset): The xarray dataset
        """
        dataset.attrs["session_timestamp"] = self.session_timestamp
        dataset.attrs["execution_timestamp"] = self.execution_timestamp
        dataset.attrs["backend"] = self.backend
        dataset.attrs["backend_configuration_name"] = self.backend_configuration_name
        dataset.attrs["backend_name"] = self.backend.name

        for key, value in self.configuration:
            if key == "benchmark":  # Avoid saving the class object
                dataset.attrs[key] = value.name
            else:
                dataset.attrs[key] = value
        # Defined outside configuration - if any

    def execute(self, backend: IQMBackendBase) -> xr.Dataset:
        """Execute the EPLG Benchmark"""
        raise NotImplementedError


class EPLGConfiguration(BenchmarkConfigurationBase):
    """EPLG Configuration

    Attributes:
        chain_length (int): Length of the linear chain of 2Q gates to consider.
        chain_path_samples (int): Number of chain path samples to consider.

    """
    chain_length: int
    chain_path_samples: int
