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
"""
Quantum Volume reflects the deepest circuit a given number of qubits can execute with meaningful results.

Circuit Layer Operations per Second (CLOPS) reflects the speed at which parametrized quantum circuits can be executed
 (CLOPS_v corresponding to QV circuits, CLOPS_h to square, parallel-gate layered, circuits)
"""

from . import clops, quantum_volume
