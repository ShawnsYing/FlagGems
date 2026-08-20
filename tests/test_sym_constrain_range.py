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

import flag_gems

# Scalar bounds for sym_constrain_range - covering min-only, max-only, both,
# unconstrained, and negative ranges. sym_constrain_range operates on a scalar
# symbolic size (not a tensor), so shapes/dtypes do not apply.
SYM_CONSTRAIN_RANGE_CASES = [
    # (size, min, max)
    (10, 0, 100),
    (50, 0, None),
    (50, None, 100),
    (42, None, None),
    (-5, -10, 0),
]


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_val, max_val", SYM_CONSTRAIN_RANGE_CASES)
def test_sym_constrain_range(size, min_val, max_val):
    """Test sym_constrain_range operator against the ATen reference.

    sym_constrain_range is a no-op compiler hint that validates a scalar bound
    and returns nothing (void), so we assert the FlagGems implementation matches
    ATen: both accept in-range values and return None.
    """
    ref_out = torch.ops.aten.sym_constrain_range(size, min=min_val, max=max_val)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.sym_constrain_range(size, min=min_val, max=max_val)

    assert res_out is None
    assert ref_out is None


@pytest.mark.sym_constrain_range
def test_sym_constrain_range_out_of_bounds():
    """sym_constrain_range must raise when the value violates the bounds."""
    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            torch.ops.aten.sym_constrain_range(5, min=10, max=100)
        with pytest.raises(RuntimeError):
            torch.ops.aten.sym_constrain_range(200, min=0, max=100)
