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

import logging
from typing import List, Union

import torch

logger = logging.getLogger(__name__)


def dsplit(input: torch.Tensor, indices_or_sections: Union[int, List[int]]):
    """Split a tensor along the third axis (depth-wise).

    ``torch.dsplit`` is a view operation: every returned chunk shares storage
    with the input. We therefore compute the split boundaries along ``dim=2``
    and materialize each chunk as a zero-copy view via ``torch.as_strided``
    (advancing the storage offset and shrinking the depth dimension), rather
    than delegating to PyTorch computation operators. No custom Triton kernel
    is needed.
    """
    logger.debug("GEMS DSPLIT")

    # dsplit requires at least 3 dimensions and always splits along dim=2.
    dim = 2
    assert input.ndim >= 3, f"dsplit requires a tensor with 3+ dims, got {input.ndim}"
    dim_size = input.shape[dim]

    # Compute (start, length) of each chunk along the depth dimension.
    if isinstance(indices_or_sections, int):
        # Integer sections: split into equal parts (must divide evenly).
        n = indices_or_sections
        assert (
            dim_size % n == 0
        ), f"dsplit: dim size {dim_size} not divisible by sections {n}"
        section = dim_size // n
        bounds = [(i * section, section) for i in range(n)]
    else:
        # Explicit indices: chunks are [0:i0], [i0:i1], ..., [i_{k-1}:dim_size].
        bounds = []
        prev = 0
        for idx in indices_or_sections:
            idx = min(idx, dim_size)
            bounds.append((prev, idx - prev))
            prev = idx
        bounds.append((prev, dim_size - prev))

    size = list(input.shape)
    stride = input.stride()
    base_offset = input.storage_offset()
    depth_stride = stride[dim]

    out = []
    for start, length in bounds:
        chunk_size = list(size)
        chunk_size[dim] = length
        out.append(
            torch.as_strided(
                input, chunk_size, stride, base_offset + start * depth_stride
            )
        )
    return tuple(out)
