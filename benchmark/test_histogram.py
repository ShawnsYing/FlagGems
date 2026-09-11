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


@pytest.mark.histogram
@pytest.mark.histogram_bin_ct
@pytest.mark.parametrize(
    "size, bins",
    [
        (1024 * 1024, 100),
        (1024 * 1024, 256),
        (2 * 1024 * 1024, 100),
        (2 * 1024 * 1024, 256),
        (4 * 1024 * 1024, 100),
        (4 * 1024 * 1024, 256),
    ],
)
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_perf_histogram_bin_ct(size, bins, dtype):
    def histogram_torch(inp):
        # torch.histogram is CPU-only, move to CPU for reference
        return torch.histogram(inp.cpu(), bins=bins)

    def histogram_gems(inp):
        import flag_gems

        return flag_gems.histogram_bin_ct(inp, bins=bins)

    inp = torch.randn(size, dtype=dtype, device="cuda")
    base.run_benchmark(
        histogram_gems,
        histogram_torch,
        (inp,),
        f"histogram.bin_ct-{dtype}-{size}-{bins}bins",
    )


@pytest.mark.histogram
@pytest.mark.histogram_bins_tensor
@pytest.mark.parametrize(
    "size, bins",
    [
        (1024 * 1024, 100),
        (2 * 1024 * 1024, 100),
        (4 * 1024 * 1024, 100),
    ],
)
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_perf_histogram_bins_tensor(size, bins, dtype):
    bin_edges = torch.linspace(0.0, 1.0, bins + 1, dtype=dtype, device="cuda")

    def histogram_torch(inp):
        return torch.histogram(inp.cpu(), bins=bin_edges.cpu())

    def histogram_gems(inp):
        import flag_gems

        return flag_gems.histogram_bins_tensor(inp, bins=bin_edges)

    inp = torch.rand(size, dtype=dtype, device="cuda")
    base.run_benchmark(
        histogram_gems,
        histogram_torch,
        (inp,),
        f"histogram.bins_tensor-{dtype}-{size}-{bins}bins",
    )
