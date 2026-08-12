# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses


@dataclasses.dataclass(frozen=True)
class SpecForwardTimings:
    target_ms: float
    draft_ms: float | None
    num_tokens: int
    num_reqs: int
    num_spec_tokens: int
