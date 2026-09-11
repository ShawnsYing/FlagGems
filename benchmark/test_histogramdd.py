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


def _input_fn(shape, dtype, device):
    # histogramdd expects 2D input (N, D), but benchmark may pass 1D or other shapes
    # Reshape to 2D if needed
    if len(shape) == 1:
        # 1D shape: treat as N points in 2D space
        N = shape[0]
        D = 2
        inp = torch.randn(N, D, dtype=dtype, device=device)
    elif len(shape) == 2:
        # Already 2D: use as-is
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        # 3D or higher: flatten to 2D
        N = shape[0] * shape[1]
        D = 2
        inp = torch.randn(N, D, dtype=dtype, device=device)

    # Default: 5 bins per dimension
    bins = [5] * inp.shape[1]
    yield inp, {"bins": bins}


@pytest.mark.histogramdd
def test_histogramdd():
    """
    Benchmark histogramdd operator.

    Note: Native torch.histogramdd has no CUDA implementation and only runs on CPU.
    This benchmark compares GPU Triton kernel (gems) against CPU native reference,
    which is not a fair device-to-device comparison but shows performance capability.
    """
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="histogramdd",
        torch_op=torch.histogramdd,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
