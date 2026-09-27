"""Live terminal renderer with coalesced flushing."""

from __future__ import annotations

import sys
import time

from ._meta import FLUSH_MAX_DELAY, FLUSH_MIN_CHARS
from .markdown import MarkdownWriter

class LiveRenderer:
    """Streams reasoning + answer to the terminal, rendering Markdown as blocks complete.

    Output is coalesced. Flushing once per token turns a 40 tok/s stream into 40 write(2)
    calls per second *plus* a terminal repaint each, which is measurably more expensive
    than the generation itself on a busy notebook. Instead we buffer and flush when either
    FLUSH_MIN_CHARS have accumulated or FLUSH_MAX_DELAY has passed — real-time to the eye,
    a small fraction of the syscalls.
    """

    def __init__(self, style, markdown=True, highlight=False, out=None,
                 reasoning_display='compact', coalesce=True):
        self.style = style
        self.out = out if out is not None else sys.stdout
        self.markdown = markdown
        self.writer = MarkdownWriter(style, highlight=highlight) if markdown else None
        self.reasoning_display = reasoning_display
        self.coalesce = coalesce
        self._reason_open = False
        self._buf = []
        self._buf_len = 0
        self._last_flush = time.monotonic()
        self.reasoning_chars = 0
        self.answer_chars = 0

    # ------------------------------------------------------------------
    def write(self, text, force=False):
        """Buffered write; flushes on size, on age, or when forced."""
        if not text:
            return
        self._buf.append(text)
        self._buf_len += len(text)
        now = time.monotonic()
        if (force or not self.coalesce or self._buf_len >= FLUSH_MIN_CHARS or
                now - self._last_flush >= FLUSH_MAX_DELAY):
            self.flush()

    def flush(self):
        if not self._buf:
            self._last_flush = time.monotonic()
            return
        try:
            self.out.write(''.join(self._buf))
            self.out.flush()
        except (BrokenPipeError, ValueError):
            pass                      # reader went away; nothing useful left to do
        self._buf = []
        self._buf_len = 0
        self._last_flush = time.monotonic()

    # ------------------------------------------------------------------
    def notice(self, text):
        self.flush()
        self.write(self.style.dim('· ' + text) + '\n', force=True)

    def reasoning(self, text):
        self.reasoning_chars += len(text)
        if self.reasoning_display == 'hidden':
            return
        if not self._reason_open:
            self._reason_open = True
            if self.reasoning_display == 'full':
                self.write(self.style.dim('✻ thinking …\n'))
        if self.reasoning_display == 'full':
            self.write(self.style.paint(text, 2))
        else:
            # 'compact' shows a one-line indicator instead of the trace; the token count
            # is printed with the turn's stats. That keeps the answer area clean on the
            # long thinking traces this model produces by default.
            self.reasoning_progress(f'({self.reasoning_chars} chars)')

    def reasoning_progress(self, text=''):
        """One-line compact indicator, redrawn in place when the terminal allows it."""
        if self.reasoning_display != 'compact':
            return
        if not self._reason_open:
            self._reason_open = True
        label = f'✻ thinking {text}'.rstrip()
        if getattr(self.out, 'isatty', lambda: False)():
            self.write('\r\033[K' + self.style.dim(label), force=True)
        elif not self._compact_line_open:
            self._compact_line_open = True
            self.write(self.style.dim('✻ thinking …'), force=True)

    _compact_line_open = False

    def delta(self, text):
        if self._reason_open:
            if self.reasoning_display == 'compact' and getattr(self.out, 'isatty', lambda: False)():
                self.write('\r\033[K', force=True)
            else:
                self.write('\n\n' if self.reasoning_display == 'full' else '\n', force=True)
            self._reason_open = False
            self._compact_line_open = False
        self.answer_chars += len(text)
        if self.writer:
            self.write(self.writer.feed(text))
        else:
            self.write(text)

    def tool(self, name, args, result):
        shown = (result or '').strip().replace('\n', ' ')
        if len(shown) > 240:
            shown = shown[:240] + ' …'
        self.write('\n' + self.style.yellow('⚙ ' + name) + self.style.dim(' ' + str(args)[:160]) +
                   '\n' + self.style.dim('  ↳ ' + shown) + '\n\n', force=True)

    def error(self, text):
        self.flush()
        self.write(self.style.red('✗ ' + text) + '\n', force=True)

    def finish(self):
        if self._reason_open:
            if self.reasoning_display == 'compact' and getattr(self.out, 'isatty', lambda: False)():
                self.write('\r\033[K', force=True)
            elif self.reasoning_display == 'full':
                self.write('\n', force=True)
            self._reason_open = False
            self._compact_line_open = False
        if self.writer:
            self.write(self.writer.finish())
        self.write('\n', force=True)
        self.flush()

    def reset_turn(self):
        self.reasoning_chars = 0
        self.answer_chars = 0

def format_stats(stats, style, context=None):
    """One-line or multi-line timing summary. Only measured values are printed.

    Anything the server did not report is printed as `n/a`; a rate derived from wall-clock
    time is labelled `~` so it is never mistaken for a decode rate.
    """
    def num(v, fmt='{:.1f}'):
        return fmt.format(v) if v else 'n/a'
    parts = []
    parts.append(f'ttft {num(stats.ttft * 1000 if stats.ttft else None, "{:.0f}")} ms'
                 if stats.ttft else 'ttft n/a')
    if stats.prompt_tokens is not None:
        parts.append(f'{stats.prompt_tokens} tok in')
    if stats.completion_tokens is not None:
        parts.append(f'{stats.completion_tokens} tok out')
    if stats.reasoning_tokens:
        parts.append(f'{stats.reasoning_tokens} reasoning')
    if stats.total_tokens:
        parts.append(f'{stats.total_tokens} total')
    if stats.prompt_tokens_per_s:
        parts.append(f'prefill {stats.prompt_tokens_per_s:.0f} tok/s')
    if stats.tokens_per_s:
        parts.append(f'decode {stats.tokens_per_s:.1f} tok/s')
    elif stats.rate():
        parts.append(f'~{stats.rate():.1f} tok/s wall')
    parts.append(f'{stats.elapsed:.1f}s')
    if stats.tool_calls:
        parts.append(f'{stats.tool_calls} tool call(s)')
    if stats.rounds > 1:
        parts.append(f'{stats.rounds} rounds')
    util = stats.context_utilization()
    if util is not None:
        parts.append(f'context {util:.0f}%')
    if stats.cancelled:
        parts.append('cancelled')
    if stats.interrupted:
        parts.append('interrupted: ' + stats.interrupted)
    return ' | '.join(parts)

def format_stats_block(stats, style):
    """/stats-style table, one measured value per line."""
    rows = [('time to first token', f'{stats.ttft * 1000:.0f} ms' if stats.ttft else 'n/a'),
            ('prompt tokens', stats.prompt_tokens if stats.prompt_tokens is not None else 'n/a'),
            ('completion tokens',
             stats.completion_tokens if stats.completion_tokens is not None else 'n/a'),
            ('reasoning tokens', stats.reasoning_tokens or 'n/a'),
            ('total tokens', stats.total_tokens if stats.total_tokens is not None else 'n/a'),
            ('prompt speed', f'{stats.prompt_tokens_per_s:.1f} tok/s'
             if stats.prompt_tokens_per_s else 'n/a'),
            ('generation speed', f'{stats.tokens_per_s:.2f} tok/s'
             if stats.tokens_per_s else 'n/a'),
            ('total elapsed', f'{stats.elapsed:.2f} s'),
            ('tool rounds', stats.rounds),
            ('context used',
             f'{stats.context_used} / {stats.context_window} '
             f'({stats.context_utilization():.0f}%)' if stats.context_utilization() is not None
             else 'n/a')]
    width = max(len(k) for k, _ in rows)
    return '\n'.join(style.dim(f'  {k.ljust(width)}  {v}') for k, v in rows)
