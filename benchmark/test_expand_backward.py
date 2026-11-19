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
from flag_gems.ops.expand_backward import ExpandFunction

from . import base, consts


def _expand_input_fn(shape, dtype, device):
    """Yield ``(input, size)`` pairs for an expand forward.

    ``shape`` is the forward input shape; the expanded gradient shape is
    derived by broadcasting every size-1 dim out to 8 (and prepending one
    leading dim), which exercises the three reduction layouts of
    ``ExpandBackward0``: prepended leading dims, a broadcast middle dim and a
    broadcast trailing dim.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad_shape = list(shape)
    grad_shape = [8] + grad_shape
    grad_shape = [s if s != 1 else 8 for s in grad_shape]
    yield inp, grad_shape


def _torch_expand(inp, size):
    return inp.expand(tuple(size))


class ExpandBackwardBenchmark(base.GenericBenchmark):
    # The backward keeps the input, the expanded gradient and the reduced
    # gradient live at once. The 2**30-element "from perf" default shapes
    # would expand 8x to 32 GiB per tensor, so the sweep uses shapes that stay
    # in the few-GiB range while still covering the three reduction layouts
    # (prepended leading dim, broadcast middle dim, broadcast trailing dim).
    SHAPES = [
        (64, 64),
        (1024, 1024),
        (64, 512, 512),
    ]

    def set_shapes(self, shape_file_path=None):
        # Bypass the shared shape YAML: ``set_shapes`` walks the class MRO and
        # would otherwise pick up the catch-all ``Benchmark:`` entry, which
        # carries the billion-element shapes this op cannot afford.
        self.shapes = list(self.SHAPES)


@pytest.mark.expand_backward
def test_expand_backward():
    # ``aten::expand`` keeps its own ExpandBackward0 node (a sum_to_size), so
    # drive the gradient through the FlagGems autograd bridge explicitly --
    # otherwise this would time the native sum chain, not this kernel.
    bench = ExpandBackwardBenchmark(
        op_name="expand_backward",
        torch_op=_torch_expand,
        gems_op=ExpandFunction.apply,
        input_fn=_expand_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
