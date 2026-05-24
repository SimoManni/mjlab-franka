"""Test infrastructure: set the PYTORCH_JIT workaround before mjlab imports."""

import os

os.environ.setdefault("PYTORCH_JIT", "0")
