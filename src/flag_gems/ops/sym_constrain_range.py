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


def sym_constrain_range(size, *, min=None, max=None):
    """Constrain a symbolic integer to [min, max] range.

    This operator is used in torch.compile symbolic shape analysis to assert
    that an unbacked symbolic integer (e.g., tensor size) falls within the
    specified range. Raises RuntimeError if the constraint is violated.

    Args:
        size: Scalar value to constrain
        min: Minimum allowed value (inclusive, optional)
        max: Maximum allowed value (inclusive, optional)

    Returns:
        None (void operator)

    Raises:
        RuntimeError: If size is outside [min, max] range
    """
    logger.debug("GEMS SYM_CONSTRAIN_RANGE")

    # Validate constraints
    if min is not None and size < min:
        raise RuntimeError(f"Invalid value range for {size} between [{min}, {max}].")
    if max is not None and size > max:
        raise RuntimeError(f"Invalid value range for {size} between [{min}, {max}].")

    return None
