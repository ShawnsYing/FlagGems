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


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_valid():
    """Test sym_constrain_range_for_size with valid ranges."""
    # Valid cases: size values with appropriate min/max bounds
    # Note: This is a void operator that only validates constraints

    # Case 1: min >= 0 (default), max unspecified
    with flag_gems.use_gems():
        torch.ops.aten.sym_constrain_range_for_size(5)

    # Case 2: explicit min=0, max=100
    with flag_gems.use_gems():
        torch.ops.aten.sym_constrain_range_for_size(10, min=0, max=100)

    # Case 3: min=1, max > 2 (satisfies max > 2 constraint)
    with flag_gems.use_gems():
        torch.ops.aten.sym_constrain_range_for_size(7, min=1, max=10)

    # Case 4: edge case - size at boundary
    with flag_gems.use_gems():
        torch.ops.aten.sym_constrain_range_for_size(3, min=0, max=5)

    # Case 5: large size value
    with flag_gems.use_gems():
        torch.ops.aten.sym_constrain_range_for_size(1000, min=100, max=2000)


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_negative():
    """Test sym_constrain_range_for_size rejects negative size."""
    # Size must be >= 0 (implicit min=0 default for size semantics)
    with pytest.raises(RuntimeError, match="[Cc]onstrain|[Rr]ange"):
        with flag_gems.use_gems():
            torch.ops.aten.sym_constrain_range_for_size(-1)


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_max_too_small():
    """Test sym_constrain_range_for_size rejects max <= 2."""
    # When max is specified and <= 2, should raise error
    with pytest.raises(RuntimeError, match="[Mm]ax.*must be greater than 2"):
        with flag_gems.use_gems():
            torch.ops.aten.sym_constrain_range_for_size(1, max=2)

    with pytest.raises(RuntimeError, match="[Mm]ax.*must be greater than 2"):
        with flag_gems.use_gems():
            torch.ops.aten.sym_constrain_range_for_size(0, max=1)
