"""Terminal Markdown renderer."""

from __future__ import annotations

import re

from ._optional import _PygTerm, _pyg_lexer, _pygments_highlight, pygments
from .style import strip_ansi, visible_len, wrap_ansi

INLINE_RE = re.compile(
    r'(?P<code>`+)(?P<code_body>.+?)(?P=code)'
    r'|\*\*\*(?P<bi>.+?)\*\*\*'
    r'|\*\*(?P<bold>.+?)\*\*'
    r'|__(?P<bold_u>.+?)__'
    r'|(?<!\w)\*(?P<em>[^*\n]+?)\*(?!\w)'
    r'|(?<!\w)_(?P<em_u>[^_\n]+?)_(?!\w)'
    r'|~~(?P<strike>.+?)~~'
    r'|!\[(?P<img_alt>[^\]]*)\]\((?P<img_url>[^)\s]+)[^)]*\)'
    r'|\[(?P<link_text>[^\]]*)\]\((?P<link_url>[^)\s]+)[^)]*\)'
    r'|<(?P<autolink>https?://[^>\s]+)>'
)

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$')

HR_RE = re.compile(r'^\s{0,3}([-*_])\s*(?:\1\s*){2,}$')

LIST_RE = re.compile(r'^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$')

QUOTE_RE = re.compile(r'^\s{0,3}>\s?(.*)$')

TABLE_ROW_RE = re.compile(r'^\s*\|?.*\|.*$')

TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')

def html_unescape(text: str) -> str:
    for a, b in (('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"'), ('&#39;', "'"),
                 ('&nbsp;', ' '), ('&amp;', '&')):
        text = text.replace(a, b)
    return text

def highlight_code(code: str, lang: str) -> str:
    """Pygments-highlighted code when available; the input untouched otherwise."""
    if pygments is None:
        return code
    try:
        lexer = _pyg_lexer(lang or 'text', stripnl=False)
    except Exception:
        try:
            lexer = _pyg_lexer('text')
        except Exception:
            return code
    try:
        out = _pygments_highlight(code, lexer, _PygTerm(style='monokai'))
    except Exception:
        return code
    return out.rstrip('\n')

class MarkdownWriter:
    """Renders Markdown to terminal text, incrementally.

    `feed()` takes any chunk of raw markdown (a streamed delta or a whole document) and
    returns whatever is safe to print now; `finish()` flushes the tail. Feeding a complete
    document in one call and printing `finish()` is exactly how `--no-stream` renders, so
    both modes agree.
    """

    def __init__(self, style, highlight=False):
        self.style = style
        self.highlight = highlight and pygments is not None
        self.width = style.width
        self.buf = ''
        self.block = None          # None | 'para' | 'quote' | 'list' | 'table'
        self.pending = []
        self.in_fence = False
        self.fence = ''
        self.fence_lang = ''
        self.fence_lines = []
        self.out = []

    # ------------------------------------------------------------------
    def feed(self, text):
        self.buf += text
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            self._line(line.rstrip('\r'))
        return self._drain()

    def finish(self):
        if self.buf:
            self._line(self.buf.rstrip('\r'))
            self.buf = ''
        self._close_fence()
        self._flush_block()
        return self._drain()

    def render(self, text):
        return self.feed(text) + self.finish()

    def _drain(self):
        out = ''.join(self.out)
        self.out = []
        return out

    def _emit(self, text):
        self.out.append(text if text.endswith('\n') else text + '\n')

    # ------------------------------------------------------------------
    def _line(self, line):
        if self.in_fence:
            if line.strip().startswith(self.fence) and line.strip() == self.fence.strip():
                self._close_fence()
            else:
                self.fence_lines.append(line)
            return
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            self._flush_block()
            self.in_fence = True
            self.fence = stripped[:3]
            self.fence_lang = stripped[3:].strip().split()[0] if len(stripped) > 3 else ''
            self.fence_lines = []
            self._emit(self.style.dim('┌─' + ('─ ' + self.fence_lang + ' ' if self.fence_lang else '─')
                                     + '─' * max(3, min(40, self.width - 12))))
            return
        if not stripped:
            self._flush_block()
            return
        m = HEADING_RE.match(line)
        if m:
            self._flush_block()
            level = len(m.group(1))
            body = self._inline(m.group(2))
            if level == 1:
                self._emit(self.style.paint(body, 1, 36))
                self._emit(self.style.dim('─' * min(self.width, max(visible_len(strip_ansi(body)), 3))))
            else:
                color = {2: 36, 3: 32, 4: 33, 5: 35, 6: 2}.get(level, 36)
                self._emit(self.style.paint(body, 1, color))
            return
        if HR_RE.match(line):
            self._flush_block()
            self._emit(self.style.dim('─' * min(self.width, 60)))
            return
        if TABLE_ROW_RE.match(line) and '|' in stripped:
            if self.block in (None, 'table'):
                self.block = 'table'
                self.pending.append(line)
                return
            self._flush_block()
            self.block = 'table'
            self.pending.append(line)
            return
        m = LIST_RE.match(line)
        if m:
            if self.block != 'list':
                self._flush_block()
                self.block = 'list'
            self.pending.append(line)
            return
        m = QUOTE_RE.match(line)
        if m:
            if self.block != 'quote':
                self._flush_block()
                self.block = 'quote'
            self.pending.append(m.group(1))
            return
        if self.block not in (None, 'para'):
            self._flush_block()
        self.block = 'para'
        self.pending.append(line)

    def _close_fence(self):
        if not self.in_fence:
            return
        code = '\n'.join(self.fence_lines)
        if self.highlight:
            self._emit(highlight_code(code, self.fence_lang))
        else:
            for ln in self.fence_lines:
                self._emit(self.style.paint(ln, 38, 5, 222) if ln.strip() else '')
        self._emit(self.style.dim('└' + '─' * min(self.width, max(4, min(42, self.width - 10)))))
        self.in_fence = False
        self.fence_lang = ''
        self.fence_lines = []

    # ------------------------------------------------------------------
    def _flush_block(self):
        if not self.pending:
            self.block = None
            return
        if self.block == 'para':
            text = ' '.join(x.strip() for x in self.pending)
            self._emit(wrap_ansi(self._inline(text), self.width))
            self._emit('')
        elif self.block == 'quote':
            body = ' '.join(x.strip() for x in self.pending)
            rendered = wrap_ansi(self._inline(body), self.width - 2, indent='  ')
            for ln in rendered.split('\n'):
                self._emit(self.style.dim('│') + ln)
            self._emit('')
        elif self.block == 'list':
            self._emit_list()
        elif self.block == 'table':
            self._emit_table()
        self.pending = []
        self.block = None

    def _emit_list(self):
        items = []
        for line in self.pending:
            m = LIST_RE.match(line)
            if m:
                indent = len(m.group(1).expandtabs(4))
                marker = m.group(2)
                body = m.group(3)
                items.append([indent, marker, body])
            elif items:
                items[-1][2] += ' ' + line.strip()
        for indent, marker, body in items:
            depth = indent // 2
            pad = '  ' * depth
            bullet = self.style.cyan(marker) if not marker[0].isdigit() else self.style.cyan(marker)
            m = re.match(r'^\[( |x|X)\]\s*(.*)$', body)
            if m:
                box = self.style.green('[x]') if m.group(1).lower() == 'x' else '[ ]'
                body = box + ' ' + m.group(2)
            rendered = wrap_ansi(self._inline(body), self.width - len(pad) - 3, indent=pad + '   ')
            first, *rest = rendered.split('\n')
            self._emit(pad + bullet + ' ' + first.lstrip())
            for r in rest:
                self._emit(r)
        self._emit('')

    def _emit_table(self):
        rows = []
        for line in self.pending:
            if TABLE_SEP_RE.match(line):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            rows.append(cells)
        rows = [r for r in rows if r]
        if not rows:
            return
        ncols = max(len(r) for r in rows)
        rows = [r + [''] * (ncols - len(r)) for r in rows]
        rendered = [[self._inline(c) for c in r] for r in rows]
        budget = max(12, self.width - (3 * ncols + 1))
        widths = [max(3, min(max(visible_len(c) for c in col), budget // ncols))
                  for col in zip(*rendered)]
        sep = self.style.dim('├' + '┼'.join('─' * (w + 2) for w in widths) + '┤')
        lines = [self.style.dim('┌' + '┬'.join('─' * (w + 2) for w in widths) + '┐')]
        for i, row in enumerate(rendered):
            cells = []
            for cell, w in zip(row, widths):
                cells.append(wrap_ansi(cell, w).split('\n'))
            height = max(len(c) for c in cells)
            for h in range(height):
                parts = []
                for cell, w in zip(cells, widths):
                    txt = cell[h] if h < len(cell) else ''
                    parts.append(' ' + txt + ' ' * (w - visible_len(txt)) + ' ')
                lines.append(self.style.dim('│') + self.style.dim('│').join(parts) + self.style.dim('│'))
            if i == 0 and len(rendered) > 1:
                lines.append(sep)
            elif i < len(rendered) - 1:
                lines.append(self.style.dim('├' + '┼'.join('─' * (w + 2) for w in widths) + '┤'))
        lines.append(self.style.dim('└' + '┴'.join('─' * (w + 2) for w in widths) + '┘'))
        for ln in lines:
            self._emit(ln)
        self._emit('')

    # ------------------------------------------------------------------
    def _inline(self, text):
        text = html_unescape(text)
        out = []
        pos = 0
        for m in INLINE_RE.finditer(text):
            out.append(text[pos:m.start()])
            g = m.groupdict()
            if g.get('code_body') is not None:
                out.append(self.style.paint(' ' + g['code_body'] + ' ', 38, 5, 222))
            elif g.get('bi'):
                out.append(self.style.paint(g['bi'], 1, 3))
            elif g.get('bold') or g.get('bold_u'):
                out.append(self.style.bold(g.get('bold') or g.get('bold_u')))
            elif g.get('em') or g.get('em_u'):
                out.append(self.style.paint(g.get('em') or g.get('em_u'), 3))
            elif g.get('strike'):
                out.append(self.style.paint(g['strike'], 9))
            elif g.get('img_url') is not None:
                alt = g.get('img_alt') or 'image'
                out.append(self.style.paint('🖼 ' + alt, 35) + self.style.dim(' (' + g['img_url'] + ')'))
            elif g.get('link_url') is not None:
                label = g.get('link_text') or g['link_url']
                shown = self.style.paint(label, 4, 34)
                if label != g['link_url']:
                    shown += self.style.dim(' (' + g['link_url'] + ')')
                out.append(shown)
            elif g.get('autolink'):
                out.append(self.style.paint(g['autolink'], 4, 34))
            pos = m.end()
        out.append(text[pos:])
        return ''.join(out)

def render_markdown(text, style, highlight=False):
    return MarkdownWriter(style, highlight=highlight).render(text)
