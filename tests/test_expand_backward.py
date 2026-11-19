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
from flag_gems.ops.expand_backward import ExpandFunction, expand_backward

from . import accuracy_utils as utils

# (input shape, expand target size) pairs.  They cover every branch of the
# sum-to-shape reduction: a prepended leading dim, a broadcast middle dim, a
# broadcast trailing dim, a full reduce, an un-broadcast (identity) expand, a
# multi-run reduce, and the -1 "keep this dim" spelling.
EXPAND_CASES = [
    ((8,), (4, 8)),
    ((1, 8), (4, 8)),
    ((2, 1, 8), (2, 5, 8)),
    ((5,), (2, 3, 5)),
    ((1, 1, 1), (3, 4, 5)),
    ((2, 3), (2, 3)),
    ((1, 4096), (1024, 4096)),
    ((512, 1), (512, 8192)),
    ((1, 1, 4096), (16, 128, 4096)),
    ((8, 1, 32, 1), (8, 16, 32, 64)),
    ((2, 1, 3, 1, 4), (2, 5, 3, 6, 4)),
    ((5, 7, 9), (-1, 7, -1)),
]


def _expand_backward_reference(grad_output, input):
    """Native autograd chain: ATen's ``ExpandBackward0`` node (a sum_to_size)."""
    inp = input.detach().clone().requires_grad_(True)
    out = inp.expand(tuple(grad_output.shape))
    (grad_input,) = torch.autograd.grad(out, (inp,), (grad_output,))
    return grad_input


@pytest.mark.expand_backward
@pytest.mark.parametrize("input_shape,size", EXPAND_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_backward(input_shape, size, dtype):
    torch.manual_seed(0)
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    expanded = inp.expand(tuple(size))
    grad_output = torch.randn(expanded.shape, dtype=dtype, device=flag_gems.device)

    ref_grad = _expand_backward_reference(grad_output, inp)
    res_grad = expand_backward(grad_output, inp)

    assert res_grad.shape == inp.shape
    utils.gems_assert_close(
        res_grad, ref_grad, dtype, atol=1e-2 if dtype != torch.float32 else 1e-4
    )


@pytest.mark.expand_backward
@pytest.mark.parametrize("input_shape,size", EXPAND_CASES[:6])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_backward_autograd(input_shape, size, dtype):
    """``aten::expand`` keeps ATen's own ExpandBackward0 node, so the gradient
    must be driven through the FlagGems autograd bridge explicitly."""
    torch.manual_seed(0)
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    out = ExpandFunction.apply(inp, tuple(size))
    assert out.requires_grad and out.grad_fn is not None
    assert tuple(out.shape) == tuple(inp.expand(tuple(size)).shape)

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = _expand_backward_reference(grad_output, inp.detach())
    utils.gems_assert_close(
        res_grad, ref_grad, dtype, atol=1e-2 if dtype != torch.float32 else 1e-4
    )


@pytest.mark.expand_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_backward_non_contiguous(dtype):
    torch.manual_seed(0)
    inp = torch.randn((2, 1, 8), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn((2, 5, 8), dtype=dtype, device=flag_gems.device)
    assert not grad_output.transpose(0, 1).contiguous().transpose(0, 1).is_contiguous()

    ref_grad = _expand_backward_reference(grad_output, inp)
    res_grad = expand_backward(
        grad_output.transpose(0, 1).contiguous().transpose(0, 1), inp
    )

    utils.gems_assert_close(
        res_grad, ref_grad, dtype, atol=1e-2 if dtype != torch.float32 else 1e-4
    )


@pytest.mark.expand_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_backward_empty(dtype):
    inp = torch.empty((0, 4), dtype=dtype, device=flag_gems.device)
    grad_output = torch.empty((0, 4), dtype=dtype, device=flag_gems.device)
    res_grad = expand_backward(grad_output, inp)
    assert res_grad.shape == inp.shape


@pytest.mark.expand_backward
def test_expand_backward_dispatch():
    """The bridge is reachable from the dispatched FlagGems ``expand`` forward."""
    torch.manual_seed(0)
    size = (4, 8)
    inp = torch.randn((1, 8), dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn(size, dtype=torch.float32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_output, True)
    (ref_grad_in,) = torch.autograd.grad(ref_inp.expand(size), (ref_inp,), (ref_grad,))

    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["expand"])
        x = inp.clone().requires_grad_(True)
        out = ExpandFunction.apply(x, size)
        (res_grad,) = torch.autograd.grad(out, (x,), (grad_output,))
    finally:
        lib._destroy()

    utils.gems_assert_close(res_grad, ref_grad_in, torch.float32)
