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

from . import profile_runner

HISTOGRAM_BENCH_SHAPES = [(1048576,), (4096, 1024), (2048, 2048)]
HISTOGRAM_BENCH_BINS = [100, 256]


@profile_runner.benchmark(
    "histogram.bin_ct",
    [
        (shape, bins)
        for shape in HISTOGRAM_BENCH_SHAPES
        for bins in HISTOGRAM_BENCH_BINS
    ],
)
def histogram_bin_ct_bench(shape, bins):
    inp = torch.randn(shape, dtype=torch.float32, device="cuda")
    yield inp, bins


@profile_runner.benchmark(
    "histogram.bins_tensor",
    [((1048576,), 100), ((4096, 1024), 256)],
)
def histogram_bins_tensor_bench(shape, num_bins):
    inp = torch.randn(shape, dtype=torch.float32, device="cuda")
    edges = torch.linspace(-3.0, 3.0, num_bins + 1, dtype=torch.float32, device="cuda")
    yield inp, edges
