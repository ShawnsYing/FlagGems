# Copyright 2026, The FlagOS Contributors.
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

import flag_gems

from .performance_utils import GenericBenchmark


def test_histogramdd():
    """
    Benchmark histogramdd operator.

    Note: Native torch.histogramdd has no CUDA implementation and only runs on CPU.
    This benchmark compares GPU Triton kernel (gems) against CPU native reference,
    which is not a fair device-to-device comparison. The speedup numbers reflect
    GPU-vs-CPU performance, not kernel optimization quality.
    """

    def input_fn(shape, dtype, device):
        # shape is expected to be a tuple (N, D)
        inp = torch.randn(shape, dtype=dtype, device=device)
        # Default: 5 bins per dimension
        D = shape[1]
        bins = [5] * D
        yield inp, bins, None, None, False

    bench = GenericBenchmark(
        input_fn=input_fn,
        op_name="histogramdd",
        torch_op=lambda inp, bins, range, weight, density: torch.histogramdd(
            inp.cpu(), bins=bins, range=range, weight=weight, density=density
        ),
        dtypes=[torch.float32, torch.float64],
        device=flag_gems.device,
    )
    bench.set_shapes(
        [
            # (N, D) shapes - various point counts and dimensions
            (1000, 2),
            (5000, 2),
            (10000, 2),
            (1000, 3),
            (5000, 3),
            (1000, 4),
            (5000, 4),
        ]
    )
    bench.run()
