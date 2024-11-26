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
"""Module to initialize logger.

Should be used in every module to output information/warnings.
"""

import logging


# Configure the logger to log messages to the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Create the logger
qcvv_logger = logging.getLogger(__name__)

# Adjust logging level for Qiskit logger
qiskit_logger = logging.getLogger("qiskit")
qiskit_logger.setLevel(logging.WARNING)
