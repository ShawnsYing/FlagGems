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

import pytest
import torch
from torch.quasirandom import SobolEngine

from . import base


@pytest.mark.underscore_sobol_engine_draw
def test_perf_underscore_sobol_engine_draw():
    def sobol_draw_setup(n: int, dimension: int):
        eng = SobolEngine(dimension=dimension, scramble=False)
        quasi = eng.quasi.cuda()
        sobolstate = eng.sobolstate.cuda()
        num_generated = 0
        dtype = torch.float32
        return quasi, n, sobolstate, dimension, num_generated, dtype

    bench = base.Benchmark(
        op_name="underscore_sobol_engine_draw",
        torch_op=torch._sobol_engine_draw,
        arg_func=sobol_draw_setup,
        dtypes=[torch.float32],
        kwargs_func=lambda n, dimension: {"dtype": torch.float32},
    )
    bench.run([(100, 2), (1000, 3), (10000, 5), (100000, 3)])
