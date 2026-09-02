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

import pytest
import torch

from flag_gems.testing import GenericBenchmark


class Benchmark(GenericBenchmark):
    """Benchmark for _sobol_engine_ff_ operator."""

    def set_more_shapes(self):
        """Define benchmark shapes for different dimensions and n values."""
        # Small dimensions, various n values
        self.shapes = [
            (2, 10, 0),
            (3, 100, 0),
            (5, 1000, 0),
            (8, 256, 512),
            (16, 1024, 0),
            (32, 2048, 1024),
            (64, 4096, 2048),
            (128, 8192, 4096),
            (256, 16384, 8192),
            (512, 1024, 0),
            (1024, 1024, 0),
        ]

    def set_input_tensor(self):
        """Generate input tensors for benchmark."""
        dimension, n, num_generated = self.shape
        MAXBIT = 30

        self.quasi = torch.zeros(dimension, dtype=torch.long, device=self.device)
        self.sobolstate = torch.randint(
            0, 2**30, (dimension, MAXBIT), dtype=torch.long, device=self.device
        )
        self.n = n
        self.dimension = dimension
        self.num_generated = num_generated

    def numpy_op_impl(self):
        """Numpy implementation not applicable for this operator."""
        pass

    def eager_op_impl(self):
        """PyTorch eager implementation."""
        quasi_copy = self.quasi.clone()
        torch._sobol_engine_ff_(
            quasi_copy, self.n, self.sobolstate, self.dimension, self.num_generated
        )
        return quasi_copy

    def inductor_op_impl(self):
        """PyTorch inductor implementation."""
        return self.eager_op_impl()

    def triton_op_impl(self):
        """Flag-Gems Triton implementation."""
        import flag_gems

        quasi_copy = self.quasi.clone()
        return flag_gems.ops._sobol_engine_ff_(
            quasi_copy, self.n, self.sobolstate, self.dimension, self.num_generated
        )


@pytest.mark.sobol_engine_ff_
@pytest.mark.skipif(
    torch.cuda.get_device_properties(0).total_memory < 32 * 1024**3,
    reason="Skip on devices with less than 32GB memory (tsingmicro A100)",
)
def test_benchmark(benchmark):
    """Pytest wrapper for benchmark."""
    benchmark(Benchmark())
