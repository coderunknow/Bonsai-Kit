"""Export a conversation to JSON or to a single self-contained HTML file.

The HTML has no external resources — no CDN, no web fonts, no scripts from anywhere —
because a transcript is private and a file that phones home when opened is not a
transcript, it is a leak. Content is escaped, and the only markup produced is from the
conversation's own fenced code blocks.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from ._meta import VERSION


def _text(message):
    body = message.get('content')
    if isinstance(body, list):
        parts = []
        for part in body:
            if isinstance(part, dict):
                parts.append(part.get('text') or f"[{part.get('type', 'part')}]")
        body = '\n'.join(parts)
    return body if isinstance(body, str) else ''


def export_json(conversation, path, title='Bonsai session', extra=None):
    """Machine-readable export: one object, schema-stable, everything measured kept."""
    payload = {
        'schema': 1,
        'title': title,
        'generator': f'Bonsai-Kit {VERSION}',
        'system': conversation.system_text,
        'messages': list(conversation.messages),
    }
    if extra:
        payload.update(extra)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
_CSS = """
:root { color-scheme: light dark; }
body {
  margin: 0 auto; padding: 2rem 1.25rem 4rem; max-width: 46rem;
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
h1 { font-size: 1.35rem; margin: 0 0 .25rem; }
.meta { color: #6b7280; font-size: .82rem; margin-bottom: 2rem; }
.msg { margin: 0 0 1.4rem; padding: .8rem 1rem; border-radius: 10px; }
.who { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
       color: #6b7280; margin-bottom: .35rem; }
.user { background: #eef2ff; }
.assistant { background: #f3f4f6; }
.tool { background: #fffbeb; font-size: .88rem; }
.system { background: #f9fafb; border: 1px dashed #d1d5db; }
pre { background: #111827; color: #f9fafb; padding: .8rem .9rem; border-radius: 8px;
      overflow-x: auto; font-size: .85rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
p code { background: rgba(0,0,0,.06); padding: .1rem .3rem; border-radius: 4px; }
.tool pre { background: rgba(0,0,0,.04); color: inherit; border: 1px solid #e5e7eb; }
footer { color: #9ca3af; font-size: .78rem; margin-top: 3rem; }
"""


def _render_markdownish(text):
    """Just enough Markdown for a transcript: paragraphs and fenced code blocks.

    Deliberately not a full Markdown implementation — the interactive UI already has
    one, and duplicating it here would be a second renderer to keep in step.
    """
    out = []
    buf = []
    in_fence = False
    lang = ''
    for line in (text or '').splitlines():
        if line.startswith('```'):
            if in_fence:
                body = html.escape('\n'.join(buf))
                out.append(f'<pre><code data-lang="{html.escape(lang)}">{body}</code></pre>')
                buf = []
                in_fence = False
            else:
                if buf:
                    out.append('<p>' + html.escape('\n'.join(buf)).replace('\n', '<br>') + '</p>')
                    buf = []
                in_fence = True
                lang = line[3:].strip()
            continue
        if in_fence:
            buf.append(line)
        elif line.strip():
            buf.append(line)
        else:
            if buf:
                out.append('<p>' + html.escape('\n'.join(buf)).replace('\n', '<br>') + '</p>')
                buf = []
    if in_fence and buf:                       # unterminated fence: still show the code
        out.append(f'<pre><code>{html.escape(chr(10).join(buf))}</code></pre>')
    elif buf:
        out.append('<p>' + html.escape('\n'.join(buf)).replace('\n', '<br>') + '</p>')
    return '\n'.join(out) or '<p><em>(no content)</em></p>'


def export_html(conversation, path, title='Bonsai session', extra=None):
    """Write one self-contained HTML file. No external requests are made when opened."""
    parts = [f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>{html.escape(title)}</title>',
             f'<style>{_CSS}</style></head><body>',
             f'<h1>{html.escape(title)}</h1>',
             f'<div class="meta">Exported from Bonsai-Kit {html.escape(VERSION)}'
             f' &middot; {len(conversation.messages)} message(s)</div>']
    for message in conversation.messages:
        role = message.get('role', 'user')
        cls = role if role in ('user', 'assistant', 'tool', 'system') else 'system'
        who = {'user': 'You', 'assistant': 'Bonsai', 'tool': 'Tool result',
               'system': 'System'}.get(cls, role)
        parts.append(f'<div class="msg {cls}"><div class="who">{html.escape(who)}</div>')
        if role == 'assistant' and message.get('tool_calls'):
            calls = json.dumps(message['tool_calls'], ensure_ascii=False, indent=2)
            parts.append('<pre><code>' + html.escape(calls) + '</code></pre>')
        if role == 'tool':
            label = message.get('name') or message.get('tool_call_id') or 'tool'
            parts.append(f'<div class="who">{html.escape(str(label))}</div>')
        parts.append(_render_markdownish(_text(message)))
        parts.append('</div>')
    if extra:
        facts = json.dumps(extra, indent=2, ensure_ascii=False, default=str)
        parts.append('<h2>Session facts</h2><pre><code>' + html.escape(facts) +
                     '</code></pre>')
    parts.append('<footer>Generated locally by Bonsai-Kit. This file loads no external '
                 'resources.</footer></body></html>')
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts), encoding='utf-8')
    return path


def export(conversation, path, fmt=None, title='Bonsai session', extra=None):
    """Dispatch on the file extension, so `--export out.html` just works."""
    path = Path(path).expanduser()
    fmt = (fmt or path.suffix.lstrip('.') or 'json').lower()
    if fmt in ('html', 'htm'):
        return export_html(conversation, path, title=title, extra=extra)
    if fmt == 'json':
        return export_json(conversation, path, title=title, extra=extra)
    raise ValueError(f'unknown export format {fmt!r} (use .json or .html)')
