"""Make the ``src`` layout importable and keep CPU inference single-threaded.

Two threads on this box are slower than one for these tiny tensors (thread
start-up dominates the actual convolution), so the suite pins one thread to keep
the whole run well under a minute.
"""

import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
