"""Terminal styling and ANSI-aware text measurement."""

from __future__ import annotations

import os
import re
import shutil
import sys

class Style:
    """ANSI styling that degrades to nothing when it would be noise."""

    def __init__(self, stream=None, force_color=None, width=None):
        self.stream = stream if stream is not None else sys.stdout
        if force_color is None:
            env = os.environ.get('NO_COLOR')
            force_color = bool(getattr(self.stream, 'isatty', lambda: False)()) and not env
        self.color = force_color
        term_width = width or shutil.get_terminal_size((100, 24)).columns
        self.width = max(40, min(int(term_width), 120))

    def paint(self, text, *codes):
        if not self.color or not codes:
            return text
        return '\033[' + ';'.join(str(c) for c in codes) + 'm' + text + '\033[0m'

    def dim(self, t):
        return self.paint(t, 2)

    def bold(self, t):
        return self.paint(t, 1)

    def cyan(self, t):
        return self.paint(t, 36)

    def green(self, t):
        return self.paint(t, 32)

    def yellow(self, t):
        return self.paint(t, 33)

    def red(self, t):
        return self.paint(t, 31)

    def magenta(self, t):
        return self.paint(t, 35)

    def blue(self, t):
        return self.paint(t, 34)

ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)

def visible_len(text: str) -> int:
    return len(strip_ansi(text))

def wrap_ansi(text: str, width: int, indent: str = '') -> str:
    """Word-wrap a string that may contain ANSI escapes (escapes never count toward width)."""
    if width <= 4:
        width = 4
    out, line, line_len, in_escape, esc = [], [], 0, False, []
    pending_word, pending_len = [], 0

    def flush_word():
        nonlocal pending_word, pending_len, line, line_len
        if not pending_word:
            return
        if line_len + pending_len > width and line:
            out.append(indent + ''.join(line).rstrip())
            line, line_len = [], 0
        line.extend(pending_word)
        line_len += pending_len
        pending_word, pending_len = [], 0

    for ch in text:
        if in_escape:
            esc.append(ch)
            if ch.isalpha():
                pending_word.append(''.join(esc))
                in_escape, esc = False, []
            continue
        if ch == '\033':
            in_escape, esc = True, [ch]
            continue
        if ch == '\n':
            flush_word()
            out.append(indent + ''.join(line).rstrip())
            line, line_len = [], 0
            continue
        if ch == ' ':
            flush_word()
            if line:
                line.append(' ')
                line_len += 1
            continue
        pending_word.append(ch)
        pending_len += 1
    flush_word()
    out.append(indent + ''.join(line).rstrip())
    return '\n'.join(out)
