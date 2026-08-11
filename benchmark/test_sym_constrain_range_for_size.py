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

from . import base, consts

# Benchmark cases for sym_constrain_range_for_size - scalar constraint operator
SYM_CONSTRAIN_RANGE_FOR_SIZE_CASES = [
    (5, 0, 100),
    (10, 1, 50),
    (100, 10, 1000),
    (1000, 100, 10000),
    (7, 3, 20),
]


class SymConstrainRangeForSizeBenchmark(base.Benchmark):
    """Custom benchmark for sym_constrain_range_for_size - void operator with scalar validation."""

    def set_shapes(self, shape_file_path=None):
        self.cases = SYM_CONSTRAIN_RANGE_FOR_SIZE_CASES

    def get_input_iter(self, cur_dtype):
        # This operator doesn't use dtype, but we maintain compatibility with benchmark framework
        for size, min_val, max_val in self.cases:
            yield (size, min_val, max_val)


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size():
    bench = SymConstrainRangeForSizeBenchmark(
        op_name="sym_constrain_range_for_size",
        torch_op=lambda size, min_val, max_val: torch.ops.aten.sym_constrain_range_for_size(
            size, min=min_val, max=max_val
        ),
        dtypes=consts.FLOAT_DTYPES[:1],  # Only run once, dtype irrelevant for scalar op
    )
    bench.run()
