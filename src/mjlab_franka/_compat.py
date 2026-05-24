"""Apply project-wide environment workarounds before importing torch/mjlab.

Currently this only disables ``torch.jit`` to avoid a segfault that crashes
``torch.jit.script`` in ``mjlab/utils/lab_api/math.py`` on this machine's
torch+CUDA combination. The decorator falls back to plain Python execution
when ``PYTORCH_JIT=0``.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")

# Belt-and-suspenders: if torch was already imported before this module loaded
# (e.g. from a long-running notebook), the env var alone is too late — flip the
# in-process JIT flags directly so the mjlab matrix_from_quat decorator falls
# back to plain Python and doesn't segfault.
if "torch" in sys.modules:
    import torch  # noqa: E402

    try:
        torch.jit._state.disable()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
    except Exception:
        pass

# Load project-local .env (WANDB_API_KEY, WANDB_ENTITY, ...) before anything
# imports wandb.
from mjlab_franka import _envfile  # noqa: E402, F401
