# Copyright 2026 FlagOS Contributors
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

import torch
from torch.quasirandom import SobolEngine

from .performance_utils import Benchmark


def test_perf_underscore_sobol_engine_draw():
    def underscore_sobol_engine_draw_args(n, dimension, dtype):
        eng = SobolEngine(dimension=dimension, scramble=False)
        quasi = eng.quasi.cuda()
        sobolstate = eng.sobolstate.cuda()
        num_generated = 0
        return (quasi, n, sobolstate, dimension, num_generated), {"dtype": dtype}

    bench = Benchmark(
        op_name="underscore_sobol_engine_draw",
        arg_func=underscore_sobol_engine_draw_args,
        dtypes=[torch.float32, torch.float64],
        shape_args=[
            (100, 2),
            (1000, 5),
            (5000, 10),
            (10000, 3),
            (10000, 10),
            (50000, 5),
            (100000, 3),
        ],
    )
    bench.run()
