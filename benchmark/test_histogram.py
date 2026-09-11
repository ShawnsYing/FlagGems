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

from . import base, consts


def _histogram_bin_ct_input_fn(shape, dtype, device):
    """Generate input for histogram.bin_ct benchmark."""
    inp = torch.randn(shape, dtype=dtype, device=device)
    bins = 100
    # histogram.bin_ct(inp, bins=bins)
    yield inp, bins


def _histogram_bins_tensor_input_fn(shape, dtype, device):
    """Generate input for histogram.bins_tensor benchmark."""
    inp = torch.rand(shape, dtype=dtype, device=device)
    bins = torch.linspace(0.0, 1.0, 101, dtype=dtype, device=device)
    # histogram.bins_tensor(inp, bins=bins)
    yield inp, bins


@consts.perf
def test_perf_histogram_bin_ct():
    bench = base.GenericBenchmark(
        input_fn=_histogram_bin_ct_input_fn,
        op_name="histogram.bin_ct",
        torch_op=lambda inp, bins: torch.histogram(inp.cpu(), bins=bins),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@consts.perf
def test_perf_histogram_bins_tensor():
    bench = base.GenericBenchmark(
        input_fn=_histogram_bins_tensor_input_fn,
        op_name="histogram.bins_tensor",
        torch_op=lambda inp, bins: torch.histogram(inp.cpu(), bins=bins.cpu()),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
