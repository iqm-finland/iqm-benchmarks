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
Compressive GST is a self-consistent characterization method which constructs the process matrices for a set of gates,
as well as full parametrizations of an initial state and a POVM.
Low rank compression of the process matrix is used to reduce measurement and post-processing overhead.
"""

from . import compressive_gst, gst_analysis
