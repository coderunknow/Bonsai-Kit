"""Constants shared by every module in the package.

Kept out of ``__init__.py`` so submodules can read them without importing the
package (which would be circular).
"""

VERSION = '0.6.0'
DEFAULT_MODEL = 'ternary-bonsai-2-27b'

# How the streaming renderer decides to flush: write immediately when enough text has
# piled up, otherwise wait at most this long. Flushing per token costs a write(2) per
# token (measured: several hundred syscalls/second on a fast stream); coalescing keeps
# the output visually real-time at a small fraction of the syscall count.
FLUSH_MIN_CHARS = 24
FLUSH_MAX_DELAY = 0.033          # ~30 Hz — imperceptible, and far below terminal refresh cost

# Streaming output that has to survive a torn connection: a partial answer is reported
# as partial, never silently promoted to a finished assistant turn.
MAX_SSE_EVENT_BYTES = 4 * 1024 * 1024
