#!/usr/bin/env python3
"""bonsai_chat.py — the documented entry point. Kept as a shim on purpose.

v0.6.0 split the single 4,754-line client into the ``bonsai_chat`` package so that the
transport, capability, streaming, reasoning, conversation, tool, rendering,
diagnostics and CLI layers can each be tested and reused in isolation. The command
``python3 bonsai_chat.py …`` was documented in every release note and in the
deployment cell's printed instructions, so it must keep working from a bare checkout
with no install step — hence this file.

See ``python3 bonsai_chat.py --help`` for usage, or ``bonsai_chat/__init__.py`` for
the importable surface.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bonsai_chat.cli import main  # noqa: E402
from bonsai_chat._meta import VERSION  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
