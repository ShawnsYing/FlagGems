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

logger = logging.getLogger(__name__)


def sym_constrain_range_for_size(size, *, min=None, max=None):
    """Constrain a symbolic size to [min, max] range.

    This operator is used in torch.compile symbolic shape analysis to assert
    that an unbacked symbolic integer representing a size falls within the
    specified range. Unlike sym_constrain_range, this variant enforces that
    sizes are non-negative (min defaults to 0) and max must be > 2.

    Args:
        size: Scalar value to constrain (must be non-negative)
        min: Minimum allowed value (inclusive, defaults to 0)
        max: Maximum allowed value (inclusive, must be > 2 if specified)

    Returns:
        None (void operator)

    Raises:
        RuntimeError: If max <= 2, or if size is outside [min, max] range
    """
    logger.debug("GEMS SYM_CONSTRAIN_RANGE_FOR_SIZE")

    # Apply defaults: min=0 for size semantics
    if min is None:
        min = 0
    if max is None:
        max = 9223372036854775807  # int64_max

    # Validate max constraint (for_size specific)
    if max <= 2:
        raise RuntimeError(
            f"Max value to constrain_range_for_size must be greater than 2. got: {max}"
        )

    # Validate range
    if size < min or size > max:
        raise RuntimeError(f"Invalid value range for {size} between [{min}, {max}].")

    return None
