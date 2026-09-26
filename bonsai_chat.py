#!/usr/bin/env python3
"""
bonsai_chat.py — a real chat client for a Ternary Bonsai 2 27B OpenAI-compatible endpoint.

The deployment cell (colab_kaggle_cell.py) prints a base URL + bearer key and a three-line
`openai` SDK snippet. This is what you actually want to sit in front of that API:

  * an interactive multi-turn chat loop (persistent history, slash commands, /undo, /retry)
  * streaming output with the model's reasoning ("thinking") shown separately
  * Markdown rendering in the terminal (headings, lists, tables, quotes, fenced code —
    with Pygments highlighting when it happens to be installed)
  * image handling: attach files with /image; if the server has a vision projector the
    pixels are sent, if it is text-only (the default deployment) the client extracts real
    facts about the file — format, dimensions, Pillow stats, optional OCR — and sends those
  * native tool calling: a registry of sandboxed tools the model can call, with an
    approve/deny flow, parallel calls, and an automatic tool-result loop

Standard library only. `rich`, `pillow`, `pygments` and `pytesseract` are used when they are
importable and silently skipped when they are not, so the file runs unchanged on Colab,
Kaggle, or a bare python3.

Usage:
    export BONSAI_BASE_URL=https://<tunnel-host>/v1
    export BONSAI_API_KEY=<key printed by the cell>
    python3 bonsai_chat.py                       # interactive
    python3 bonsai_chat.py -p "why is C++ fast"  # one-shot
    cat notes.txt | python3 bonsai_chat.py -p "summarise"
    python3 bonsai_chat.py --selftest            # protocol self-test against a local stub
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import ssl
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

VERSION = '0.3.0'
DEFAULT_MODEL = 'ternary-bonsai-2-27b'

# Optional integrations — every one of these is a nicety, never a requirement.
try:  # pragma: no cover - depends on environment
    from PIL import Image as _PILImage  # type: ignore
except Exception:  # pragma: no cover
    _PILImage = None

try:  # pragma: no cover
    import pygments  # type: ignore
    from pygments import highlight as _pygments_highlight  # type: ignore
    from pygments.formatters import Terminal256Formatter as _PygTerm  # type: ignore
    from pygments.lexers import get_lexer_by_name as _pyg_lexer  # type: ignore
except Exception:  # pragma: no cover
    pygments = None

try:  # pragma: no cover
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


def optional_features() -> dict:
    """What the environment offers on top of the standard library."""
    return {'pillow': _PILImage is not None, 'pygments': pygments is not None,
            'pytesseract': pytesseract is not None,
            'tesseract': shutil.which('tesseract') is not None}


# ======================================================================
# Errors
# ======================================================================
class BonsaiError(RuntimeError):
    """Base error for anything the client itself reports."""


class BonsaiAPIError(BonsaiError):
    """The endpoint answered with an HTTP error status."""

    def __init__(self, status, message, url=''):
        self.status = status
        self.message = message
        self.url = url
        super().__init__(f'HTTP {status}: {message}')

    def hint(self):
        s = (self.message or '').lower()
        if self.status in (401, 403):
            return ('authentication rejected — check BONSAI_API_KEY against the key the '
                    'deployment cell printed (it is printed exactly once).')
        if self.status == 404:
            return 'unknown path or model id — check BONSAI_BASE_URL ends in /v1 and the model name.'
        if self.status == 400 and 'context' in s:
            return 'the request exceeds the server context window — lower /max-tokens or /reset.'
        if self.status == 400:
            return 'the server rejected the payload; try /effort none or disable tools.'
        if self.status == 429:
            return 'rate limited or the single serving slot is busy — wait and retry.'
        if self.status >= 500:
            return 'server-side error; the notebook may be out of VRAM or restarting.'
        return ''


class CancelledByUser(Exception):
    """Raised when Ctrl-C interrupts the current turn (the session survives)."""


# ======================================================================
# Terminal styling
# ======================================================================
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


# ======================================================================
# HTTP transport (OpenAI-compatible, llama.cpp aware)
# ======================================================================
def normalize_base_url(base_url: str) -> str:
    """Accept `https://host`, `https://host/v1`, or a trailing slash; return `https://host/v1`."""
    url = (base_url or '').strip().rstrip('/')
    if not url:
        raise BonsaiError('no base URL — set BONSAI_BASE_URL or pass --base-url')
    if not re.match(r'^https?://', url):
        url = 'http://' + url
    if not url.endswith('/v1'):
        url += '/v1'
    return url


def iter_sse_events(resp):
    """Yield the payload of each `data:` event from a streaming HTTP response.

    Handles multi-line data fields, CRLF, and SSE comment keep-alives.
    """
    data_lines = []
    for raw in resp:
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if line == '':
            if data_lines:
                yield '\n'.join(data_lines)
                data_lines = []
            continue
        if line.startswith(':'):
            continue
        if line.startswith('data:'):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield '\n'.join(data_lines)


def accumulate_tool_calls(delta_calls, acc):
    """Fold streamed tool-call fragments into complete OpenAI-shaped tool calls."""
    for tc in delta_calls or []:
        idx = tc.get('index', 0)
        slot = acc.setdefault(idx, {'id': None, 'type': 'function',
                                    'function': {'name': '', 'arguments': ''}})
        if tc.get('id'):
            slot['id'] = tc['id']
        fn = tc.get('function') or {}
        if fn.get('name'):
            slot['function']['name'] += fn['name']
        args = fn.get('arguments')
        if args:
            slot['function']['arguments'] += args if isinstance(args, str) else json.dumps(args)
    return acc


class BonsaiClient:
    """Minimal OpenAI-compatible client: chat (streaming + not), models, props, tokenize."""

    def __init__(self, base_url, api_key, model=DEFAULT_MODEL, timeout=600,
                 connect_timeout=30, retries=3, opener=None, log=None):
        self.base_url = normalize_base_url(base_url)
        # llama.cpp serves OpenAI routes under /v1 but its own routes (/props, /tokenize,
        # /health) at the server root — the deployment cell does the same split.
        self.root_url = self.base_url[:-3] if self.base_url.endswith('/v1') else self.base_url
        self.api_key = api_key or ''
        self.model = model
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.retries = max(1, retries)
        self.opener = opener  # injectable for tests
        self.log = log or (lambda *a, **k: None)
        self.supports_reasoning_effort = True
        self.supports_tools = True
        self._vision = None
        self._context = None

    # ------------------------------------------------------------------
    def _headers(self, stream=False):
        h = {'Content-Type': 'application/json',
             'Accept': 'text/event-stream' if stream else 'application/json',
             'User-Agent': f'bonsai-chat/{VERSION}'}
        if self.api_key:
            h['Authorization'] = 'Bearer ' + self.api_key
        return h

    def _open(self, req, timeout):
        if self.opener is not None:
            return self.opener.open(req, timeout=timeout)
        ctx = ssl.create_default_context()
        return urllib.request.urlopen(req, timeout=timeout, context=ctx)

    def request(self, path, payload=None, stream=False, timeout=None, root=False):
        """One HTTP call with retry/backoff. Returns the response object (caller closes).

        `root=True` targets a llama.cpp-native route (/props, /tokenize, /health) instead of
        an OpenAI-compatible one under /v1.
        """
        url = (self.root_url if root else self.base_url) + path
        data = json.dumps(payload).encode() if payload is not None else None
        last = None
        for attempt in range(1, self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=self._headers(stream),
                                         method='POST' if data is not None else 'GET')
            try:
                return self._open(req, timeout or (self.timeout if stream else self.connect_timeout))
            except urllib.error.HTTPError as e:
                body = b''
                try:
                    body = e.read() or b''
                except Exception:
                    pass
                try:
                    parsed = json.loads(body.decode('utf-8', 'replace'))
                    msg = parsed.get('error')
                    if isinstance(msg, dict):
                        msg = msg.get('message') or json.dumps(msg)
                    msg = str(msg or body[:400].decode('utf-8', 'replace'))
                except Exception:
                    msg = body[:400].decode('utf-8', 'replace') or e.reason or ''
                if e.code in (429, 502, 503, 504) and attempt < self.retries:
                    last = BonsaiAPIError(e.code, msg, url)
                    wait = min(2 ** attempt, 8)
                    self.log(f'transient HTTP {e.code}, retrying in {wait}s')
                    time.sleep(wait)
                    continue
                raise BonsaiAPIError(e.code, msg, url) from None
            except urllib.error.URLError as e:
                last = e
                reason = str(getattr(e, 'reason', e))
                if attempt < self.retries:
                    wait = min(2 ** attempt, 8)
                    self.log(f'network error ({reason}), retrying in {wait}s')
                    time.sleep(wait)
                    continue
                raise BonsaiError(
                    f'cannot reach {url}: {reason}. Is the notebook runtime still alive? '
                    'Quick Tunnel URLs die with the session — rerun the deployment cell and '
                    'use the new URL.') from None
            except (socket.timeout, TimeoutError) as e:
                last = e
                if attempt < self.retries:
                    wait = min(2 ** attempt, 8)
                    self.log(f'timeout, retrying in {wait}s')
                    time.sleep(wait)
                    continue
                raise BonsaiError(f'request to {url} timed out after '
                                  f'{timeout or self.connect_timeout}s') from None
        raise BonsaiError(f'request to {url} failed: {last}')

    def get_json(self, path, timeout=None, root=False):
        with self.request(path, timeout=timeout, root=root) as r:
            return json.loads(r.read().decode('utf-8', 'replace'))

    # ------------------------------------------------------------------
    def models(self):
        return self.get_json('/models')

    def model_ids(self):
        try:
            return [m.get('id') for m in (self.models().get('data') or []) if m.get('id')]
        except BonsaiError:
            return []

    def props(self):
        """llama.cpp /props — context size, slot state. Not part of the OpenAI spec."""
        try:
            return self.get_json('/props', timeout=15, root=True)
        except BonsaiError:
            return None

    def health(self):
        """llama.cpp /health -> the parsed body, or None when unreachable."""
        try:
            return self.get_json('/health', timeout=15, root=True)
        except BonsaiError:
            return None

    def tokenize(self, text):
        """Exact server-side token count, or None if the endpoint is unavailable."""
        try:
            with self.request('/tokenize', {'content': text}, timeout=30, root=True) as r:
                data = json.loads(r.read().decode('utf-8', 'replace'))
            tokens = data.get('tokens')
            return len(tokens) if isinstance(tokens, list) else None
        except BonsaiError:
            return None

    def context_window(self, default=8192):
        """Server context size in tokens, probed once from /props and then cached."""
        if self._context:
            return self._context
        props = self.props() or {}
        found = None
        sub = props.get('default_generation_settings') or {}
        for source in (sub, props):
            for k in ('n_ctx', 'n_ctx_total'):
                if isinstance(source.get(k), int) and source[k] > 0:
                    found = source[k]
                    break
            if found:
                break
        self._context = found or default
        return self._context

    # ------------------------------------------------------------------
    def _post_chat(self, payload, stream, timeout):
        """POST /chat/completions, dropping fields the server build rejects."""
        payload = dict(payload)
        if payload.get('reasoning_effort') is None:
            payload.pop('reasoning_effort', None)
        if not self.supports_reasoning_effort:
            payload.pop('reasoning_effort', None)
        if not payload.get('tools'):
            payload.pop('tools', None)
            payload.pop('tool_choice', None)
        if not self.supports_tools:
            payload.pop('tools', None)
            payload.pop('tool_choice', None)
        try:
            return self.request('/chat/completions', payload, stream=stream, timeout=timeout)
        except BonsaiAPIError as e:
            if e.status != 400:
                raise
            msg = (e.message or '').lower()
            if 'reasoning_effort' in msg and 'reasoning_effort' in payload:
                self.supports_reasoning_effort = False
                self.log('server rejected reasoning_effort — continuing without it')
                payload.pop('reasoning_effort', None)
                return self.request('/chat/completions', payload, stream=stream, timeout=timeout)
            if ('tool' in msg or 'jinja' in msg) and 'tools' in payload:
                self.supports_tools = False
                self.log('server rejected tools — continuing without tool calling')
                payload.pop('tools', None)
                payload.pop('tool_choice', None)
                return self.request('/chat/completions', payload, stream=stream, timeout=timeout)
            raise

    def chat(self, messages, tools=None, **params):
        """Non-streaming chat completion -> the raw JSON response dict."""
        payload = {'model': self.model, 'messages': messages, 'stream': False}
        payload.update({k: v for k, v in params.items() if v is not None})
        if tools:
            payload['tools'] = tools
            payload.setdefault('tool_choice', 'auto')
        with self._post_chat(payload, False, params.get('timeout')) as r:
            return json.loads(r.read().decode('utf-8', 'replace'))

    def stream_chat(self, messages, tools=None, **params):
        """Streaming chat completion. Yields normalized events; see module docs."""
        payload = {'model': self.model, 'messages': messages, 'stream': True,
                   'stream_options': {'include_usage': True}}
        payload.update({k: v for k, v in params.items() if v is not None})
        if tools:
            payload['tools'] = tools
            payload.setdefault('tool_choice', 'auto')
        resp = self._post_chat(payload, True, params.get('timeout'))
        return self._event_stream(resp)

    def _event_stream(self, resp):
        text, reason = [], []
        tool_acc = {}
        finish, usage, timings = None, None, None
        started = time.monotonic()
        first_token_at = None
        with resp:
            for payload in iter_sse_events(resp):
                if payload.strip() == '[DONE]':
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                if ev.get('usage'):
                    usage = ev['usage']
                if ev.get('timings'):
                    timings = ev['timings']
                choices = ev.get('choices') or []
                if not choices:
                    continue
                ch = choices[0]
                if ch.get('finish_reason'):
                    finish = ch['finish_reason']
                delta = ch.get('delta') or {}
                accumulate_tool_calls(delta.get('tool_calls'), tool_acc)
                rc = delta.get('reasoning_content') or delta.get('reasoning')
                if rc:
                    reason.append(rc)
                    first_token_at = first_token_at or time.monotonic()
                    yield {'kind': 'reasoning', 'text': rc}
                ct = delta.get('content')
                if ct:
                    text.append(ct)
                    first_token_at = first_token_at or time.monotonic()
                    yield {'kind': 'delta', 'text': ct}
        tool_calls = [tool_acc[k] for k in sorted(tool_acc)]
        message = {'role': 'assistant', 'content': ''.join(text)}
        if ''.join(reason):
            message['reasoning_content'] = ''.join(reason)
        if tool_calls:
            message['tool_calls'] = tool_calls
            message['content'] = message['content'] or None
        yield {'kind': 'done', 'message': message, 'tool_calls': tool_calls,
               'finish_reason': finish, 'usage': usage, 'timings': timings,
               'elapsed': time.monotonic() - started,
               'ttft': (first_token_at - started) if first_token_at else None}

    # ------------------------------------------------------------------
    def probe_vision(self, force=None):
        """Can this server actually see pixels? Probed once with a 1x1 PNG, then cached.

        The default deployment is deliberately text-only (no mmproj/vision projector), so
        this normally returns False and the client sends extracted image facts instead.
        """
        if force is not None:
            self._vision = bool(force)
            return self._vision
        if self._vision is not None:
            return self._vision
        modalities = (self.props() or {}).get('modalities')
        if isinstance(modalities, dict) and isinstance(modalities.get('vision'), bool):
            self._vision = modalities['vision']
            self.log(f'/props reports vision={self._vision}')
            return self._vision
        png_1x1 = ('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA'
                   'hKmMIQAAAABJRU5ErkJggg==')
        try:
            self.chat([{'role': 'user', 'content': [
                {'type': 'text', 'text': 'Reply with the single word: ok'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + png_1x1}}]}],
                max_tokens=8)
            self._vision = True
        except BonsaiAPIError as e:
            self._vision = False
            self.log(f'vision probe returned HTTP {e.status} — treating server as text-only')
        except BonsaiError as e:
            self.log(f'vision probe failed: {e}')
            self._vision = False
        return self._vision


# ======================================================================
# Token accounting
# ======================================================================
def estimate_tokens(text) -> int:
    """Rough token estimate used when /tokenize is unavailable."""
    if isinstance(text, list):
        return sum(estimate_tokens(part.get('text', '')) for part in text if isinstance(part, dict))
    if not text:
        return 0
    s = str(text)
    cjk = len(re.findall(r'[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]', s))
    return max(1, int(len(s) / 4) + cjk)


class TokenCounter:
    """Uses the server's own tokenizer when the endpoint exists, else a heuristic.

    Counts are memoized: budget trimming re-measures the same history on every turn and
    each measurement is a round trip to /tokenize.
    """

    CACHE_LIMIT = 1024

    def __init__(self, client=None):
        self.client = client
        self._server = None
        self._cache = {}

    def count(self, text) -> int:
        if isinstance(text, str):
            cached = self._cache.get(text)
            if cached is not None:
                return cached
        n = self._count_uncached(text)
        if isinstance(text, str):
            if len(self._cache) >= self.CACHE_LIMIT:
                self._cache.clear()
            self._cache[text] = n
        return n

    def using_server(self):
        """True once a real /tokenize round trip has succeeded."""
        return self._server is True

    def _count_uncached(self, text) -> int:
        if self._server is False:
            return estimate_tokens(text)
        if self.client is not None and isinstance(text, str) and text:
            n = self.client.tokenize(text)
            if n is None:
                self._server = False
            else:
                self._server = True
                return n
        return estimate_tokens(text)

    def messages(self, messages) -> int:
        total = 0
        for m in messages:
            total += self.count(m.get('content')) + 4
            for tc in m.get('tool_calls') or []:
                total += self.count((tc.get('function') or {}).get('arguments') or '') + 8
        return total


# ======================================================================
# Markdown rendering
# ======================================================================
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


# ======================================================================
# Image handling
# ======================================================================
PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def sniff_format(data: bytes):
    """Identify a raster format from magic bytes (no third-party library needed)."""
    if data.startswith(PNG_MAGIC):
        return 'png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'webp'
    if data[:2] == b'BM':
        return 'bmp'
    return None


def png_dimensions(data):
    if len(data) < 26:
        return None
    w, h = struct.unpack('>II', data[16:24])
    depth, ctype = data[24], data[25]
    colortypes = {0: 'grayscale', 2: 'rgb', 3: 'indexed', 4: 'grayscale+alpha', 6: 'rgba'}
    return w, h, {'bit_depth': depth, 'color_type': colortypes.get(ctype, ctype)}


def jpeg_dimensions(data):
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack('>HH', data[i + 5:i + 9])
            return w, h, {'components': data[i + 9] if i + 9 < n else None}
        i += 2 + seg_len
    return None


def gif_dimensions(data):
    if len(data) < 10:
        return None
    w, h = struct.unpack('<HH', data[6:10])
    return w, h, {}


def bmp_dimensions(data):
    if len(data) < 26:
        return None
    w, h = struct.unpack('<ii', data[18:26])
    return abs(w), abs(h), {}


def webp_dimensions(data):
    if len(data) < 30:
        return None
    fourcc = data[12:16]
    try:
        if fourcc == b'VP8X':
            w = int.from_bytes(data[24:27], 'little') + 1
            h = int.from_bytes(data[27:30], 'little') + 1
            return w, h, {'variant': 'extended'}
        if fourcc == b'VP8L':
            b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
            w = ((b1 & 0x3F) << 8 | b0) + 1
            h = ((b3 & 0x0F) << 10 | b2 << 2 | (b1 & 0xC0) >> 6) + 1
            return w, h, {'variant': 'lossless'}
        if fourcc == b'VP8 ':
            w = struct.unpack('<H', data[26:28])[0] & 0x3FFF
            h = struct.unpack('<H', data[28:30])[0] & 0x3FFF
            return w, h, {'variant': 'lossy'}
    except (struct.error, IndexError):
        return None
    return None


def header_info(data: bytes):
    """Format + pixel dimensions straight from the file header, or (None, None, {})."""
    fmt = sniff_format(data)
    parsers = {'png': png_dimensions, 'jpeg': jpeg_dimensions, 'gif': gif_dimensions,
               'bmp': bmp_dimensions, 'webp': webp_dimensions}
    parser = parsers.get(fmt)
    if not parser:
        return None, None, {}
    res = parser(data)
    if not res:
        return fmt, None, {}
    w, h, extra = res
    return fmt, (w, h), extra


@dataclass
class ImageAttachment:
    path: Path
    fmt: str = ''
    mime: str = 'application/octet-stream'
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    header_extra: dict = field(default_factory=dict)
    pillow: dict = field(default_factory=dict)
    ocr_text: str = ''
    data_url: str = ''
    payload_bytes: int = 0
    notes: list = field(default_factory=list)

    @property
    def display_name(self):
        return self.path.name

    def aspect(self):
        if not self.width or not self.height:
            return ''
        from math import gcd
        g = gcd(self.width, self.height) or 1
        w, h = self.width // g, self.height // g
        if max(w, h) > 40:
            return f'{self.width / self.height:.2f}:1'
        return f'{w}:{h}'

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path, max_side=1024, use_ocr=False, encode=True):
        p = Path(path).expanduser()
        if not p.is_file():
            raise BonsaiError(f'image not found: {p}')
        data = p.read_bytes()
        fmt, dims, extra = header_info(data)
        if fmt is None:
            mime = mimetypes.guess_type(p.name)[0] or ''
            if not mime.startswith('image/'):
                raise BonsaiError(
                    f'{p.name}: not a recognised image (no PNG/JPEG/GIF/WebP/BMP magic bytes, '
                    f'mime={mime or "unknown"}).')
            fmt = mime.split('/', 1)[1]
        mime = {'jpg': 'image/jpeg'}.get(fmt, 'image/' + fmt)
        att = cls(path=p, fmt=fmt, mime=mime, size_bytes=len(data), header_extra=extra)
        if dims:
            att.width, att.height = dims
        att.pillow = cls._pillow_stats(p)
        if att.pillow.get('size') and not att.width:
            att.width, att.height = att.pillow['size']
        if use_ocr:
            att.ocr_text = cls._ocr(p)
        if encode:
            att.data_url, att.payload_bytes = cls._encode(p, data, att.width, att.height, max_side)
        return att

    @staticmethod
    def _pillow_stats(path):
        if _PILImage is None:
            return {}
        try:
            with _PILImage.open(path) as im:
                info = {'size': im.size, 'mode': im.mode, 'format': im.format}
                try:
                    orient = im.getexif().get(274)
                    if orient:
                        info['exif_orientation'] = orient
                except Exception:
                    pass
                try:
                    small = im.convert('RGB').resize((16, 16))
                    px = list(small.getdata())
                    n = len(px) or 1
                    mean = tuple(sum(c[i] for c in px) // n for i in range(3))
                    info['mean_rgb'] = '#%02x%02x%02x' % mean
                    quant = im.convert('RGB').quantize(colors=4, method=_PILImage.Quantize.MEDIANCUT)
                    palette = [c[:3] for c in quant.getpalette()[:12]] if quant.getpalette() else []
                    info['dominant_rgb'] = ['#%02x%02x%02x' % tuple(c) for c in palette[:4] if c]
                except Exception:
                    pass
                return info
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def _ocr(path):
        if pytesseract is None or not shutil.which('tesseract'):
            return ''
        if _PILImage is None:
            return ''
        try:
            with _PILImage.open(path) as im:
                text = pytesseract.image_to_string(im.convert('RGB'))
            return re.sub(r'\n{3,}', '\n\n', text).strip()[:2000]
        except Exception as e:
            return f'[OCR failed: {e}]'

    @staticmethod
    def _encode(path, data, width, height, max_side):
        """Base64 data URL, downscaled when Pillow is available and the image is big."""
        if _PILImage is not None and max_side and (max(width, height) > max_side or len(data) > 1_500_000):
            try:
                import io
                with _PILImage.open(path) as im:
                    im = im.convert('RGB') if im.mode in ('P', 'LA', 'RGBA', 'CMYK') else im
                    im.thumbnail((max_side, max_side))
                    buf = io.BytesIO()
                    im.save(buf, format='JPEG', quality=85, optimize=True)
                    payload = buf.getvalue()
                if len(payload) < len(data):
                    b64 = base64.b64encode(payload).decode()
                    return 'data:image/jpeg;base64,' + b64, len(payload)
            except Exception:
                pass
        try:
            b64 = base64.b64encode(data).decode()
        except (binascii.Error, ValueError) as e:
            raise BonsaiError(f'could not encode {path}: {e}') from None
        fmt = sniff_format(data) or 'jpeg'
        mime = 'image/jpeg' if fmt == 'jpg' else 'image/' + fmt
        return f'data:{mime};base64,' + b64, len(data)

    # ------------------------------------------------------------------
    def message_part(self):
        return {'type': 'image_url', 'image_url': {'url': self.data_url}}

    def text_card(self, vision):
        """What the model is actually told about the file.

        With a vision projector the pixels go through too, so this is a short caption.
        Without one (the default text-only deployment) this card is the *only* thing the
        model receives, so it carries every fact the client could measure.
        """
        lines = [f'[Image attached: {self.display_name}]']
        dims = f'{self.width}x{self.height} px' if self.width else 'dimensions unknown'
        extra = ''
        if self.header_extra:
            bits = self.header_extra.get('bit_depth')
            ct = self.header_extra.get('color_type')
            variant = self.header_extra.get('variant')
            extra = ', '.join(x for x in (
                f'{bits}-bit/channel' if bits else '',
                str(ct) if ct else '',
                f'{variant} variant' if variant else '') if x)
        lines.append(f'- file: {self.fmt.upper()}, {dims}'
                     + (f', aspect {self.aspect()}' if self.aspect() else '')
                     + f', {self.size_bytes / 1024:.1f} KiB on disk'
                     + (f' ({extra})' if extra else ''))
        if self.pillow:
            if 'error' in self.pillow:
                lines.append(f'- Pillow could not decode it: {self.pillow["error"]}')
            else:
                bits = []
                if self.pillow.get('mode'):
                    bits.append('mode ' + str(self.pillow['mode']))
                if self.pillow.get('mean_rgb'):
                    bits.append('mean colour ' + str(self.pillow['mean_rgb']))
                if self.pillow.get('dominant_rgb'):
                    bits.append('dominant ' + ', '.join(self.pillow['dominant_rgb']))
                if self.pillow.get('exif_orientation'):
                    bits.append('EXIF orientation ' + str(self.pillow['exif_orientation']))
                if bits:
                    lines.append('- decoded (Pillow): ' + '; '.join(bits))
        else:
            lines.append('- Pillow is not installed here, so no pixel statistics were extracted '
                         '(pip install pillow for mean/dominant colours).')
        if self.ocr_text:
            lines.append('- OCR text (tesseract):\n' + textwrap.indent(self.ocr_text, '    '))
        elif pytesseract is None or not shutil.which('tesseract'):
            lines.append('- OCR: unavailable (needs pytesseract + the tesseract binary; rerun with --ocr).')
        if vision:
            lines.append('- the pixels are attached to this message; the server has a vision projector.')
        else:
            lines.append('- NOTE: this server is TEXT-ONLY (no vision projector), so the model cannot '
                         'see the image. The measurements above are everything it has.')
        return '\n'.join(lines)


def load_attachments(paths, max_side=1024, use_ocr=False):
    return [ImageAttachment.load(p, max_side=max_side, use_ocr=use_ocr) for p in paths]


def build_user_message(text, attachments, vision):
    """Compose the user message: multimodal parts when the server can see, text card otherwise."""
    if not attachments:
        return {'role': 'user', 'content': text or ''}
    if vision:
        parts = [{'type': 'text', 'text': text or 'Describe this image.'}]
        parts += [a.message_part() for a in attachments]
        return {'role': 'user', 'content': parts}
    cards = '\n\n'.join(a.text_card(vision=False) for a in attachments)
    body = (text or 'What can you tell me about this file?')
    return {'role': 'user', 'content': body + '\n\n' + cards}


# ======================================================================
# Tools the model can call
# ======================================================================
class ToolError(BonsaiError):
    """A tool refused or failed; the message goes back to the model so it can recover."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: object
    risk: str = 'safe'          # safe | write | dangerous
    enabled: bool = True

    def schema(self):
        return {'type': 'function',
                'function': {'name': self.name, 'description': self.description,
                             'parameters': self.parameters}}


@dataclass
class ToolContext:
    sandbox: Path
    approve: object = None            # fn(name, args, risk) -> bool
    notify: object = None             # fn(text) -> None
    http_timeout: int = 30
    auto_approve: bool = False
    max_output: int = 8000

    def say(self, text):
        if self.notify:
            self.notify(text)

    def ask(self, name, args, risk):
        if self.auto_approve:
            return True
        if self.approve is None:
            return risk == 'safe'
        return bool(self.approve(name, args, risk))

    def inside(self, path):
        """Resolve a path and keep it inside the sandbox."""
        p = Path(str(path)).expanduser()
        p = p if p.is_absolute() else (self.sandbox / p)
        try:
            resolved = p.resolve()
            root = self.sandbox.resolve()
        except OSError as e:
            raise ToolError(f'cannot resolve path {path}: {e}') from None
        if resolved != root and root not in resolved.parents:
            raise ToolError(f'{path} is outside the sandbox ({root}); refusing.')
        return resolved

    def truncate(self, text):
        text = str(text)
        if len(text) <= self.max_output:
            return text
        return (text[:self.max_output] +
                f'\n... [truncated {len(text) - self.max_output} chars — narrow the request]')


_MATH_FUNCS = {
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
    'sqrt': __import__('math').sqrt, 'floor': __import__('math').floor,
    'ceil': __import__('math').ceil, 'log': __import__('math').log,
    'log2': __import__('math').log2, 'log10': __import__('math').log10,
    'exp': __import__('math').exp, 'sin': __import__('math').sin, 'cos': __import__('math').cos,
    'tan': __import__('math').tan, 'atan2': __import__('math').atan2, 'pi': __import__('math').pi,
    'e': __import__('math').e, 'radians': __import__('math').radians,
    'degrees': __import__('math').degrees, 'hypot': __import__('math').hypot,
}


def _safe_eval(node):
    import ast as _ast
    import operator as _op
    binops = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.Div: _op.truediv,
              _ast.FloorDiv: _op.floordiv, _ast.Mod: _op.mod, _ast.Pow: _op.pow}
    unops = {_ast.UAdd: _op.pos, _ast.USub: _op.neg}
    if isinstance(node, _ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, _ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError('only numbers are allowed in expressions')
    if isinstance(node, _ast.BinOp) and type(node.op) in binops:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, _ast.Pow) and abs(right) > 64:
            raise ToolError('exponent too large')
        return binops[type(node.op)](left, right)
    if isinstance(node, _ast.UnaryOp) and type(node.op) in unops:
        return unops[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
        fn = _MATH_FUNCS.get(node.func.id)
        if fn is None or not callable(fn):
            raise ToolError(f'function {node.func.id!r} is not available')
        return fn(*[_safe_eval(a) for a in node.args])
    if isinstance(node, _ast.Name):
        val = _MATH_FUNCS.get(node.id)
        if val is None:
            raise ToolError(f'unknown name {node.id!r}')
        return val
    raise ToolError('unsupported expression syntax')


class ToolRegistry:
    """Built-in tools + a decorator for adding your own.

    Everything the model can touch is confined to `sandbox` and anything that writes or
    executes goes through the approval callback first.
    """

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.tools = {}
        self._register_builtins()

    # ------------------------------------------------------------------
    def tool(self, name, description, parameters, risk='safe'):
        def deco(fn):
            self.tools[name] = Tool(name, description, parameters, fn, risk=risk)
            return fn
        return deco

    def add(self, tool: Tool):
        self.tools[tool.name] = tool

    def _register_builtins(self):
        ctx = self.ctx

        @self.tool('read_file',
                   'Read a text file inside the sandbox and return its contents.',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string', 'description': 'File path, relative to the sandbox'},
                                   'max_lines': {'type': 'integer', 'description': 'Maximum lines to return (default 200)'}},
                    'required': ['path']})
        def read_file(args):
            path = ctx.inside(args['path'])
            if not path.is_file():
                raise ToolError(f'no such file: {args["path"]}')
            max_lines = int(args.get('max_lines') or 200)
            try:
                text = path.read_text(errors='replace')
            except OSError as e:
                raise ToolError(f'cannot read {args["path"]}: {e}') from None
            lines = text.splitlines()
            head = lines[:max_lines]
            out = '\n'.join(head)
            if len(lines) > max_lines:
                out += f'\n... [{len(lines) - max_lines} more lines; raise max_lines to see them]'
            return ctx.truncate(out or '(empty file)')

        @self.tool('list_dir',
                   'List the entries of a directory inside the sandbox with sizes.',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string', 'description': 'Directory (default: sandbox root)'}},
                    'required': []})
        def list_dir(args):
            path = ctx.inside(args.get('path') or '.')
            if not path.is_dir():
                raise ToolError(f'not a directory: {args.get("path")}')
            rows = []
            for entry in sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))[:200]:
                if entry.is_dir():
                    rows.append(f'{entry.name}/')
                else:
                    rows.append(f'{entry.name}  ({entry.stat().st_size} bytes)')
            return ctx.truncate('\n'.join(rows) or '(empty directory)')

        @self.tool('search_text',
                   'Search file contents in the sandbox with a regular expression and return matching lines.',
                   {'type': 'object',
                    'properties': {'pattern': {'type': 'string', 'description': 'Regular expression'},
                                   'path': {'type': 'string', 'description': 'Directory or file to search (default: sandbox root)'},
                                   'glob': {'type': 'string', 'description': 'Filename glob, e.g. *.py (default *)'},
                                   'max_results': {'type': 'integer', 'description': 'Maximum matches (default 25)'}},
                    'required': ['pattern']})
        def search_text(args):
            root = ctx.inside(args.get('path') or '.')
            try:
                rx = re.compile(args['pattern'], re.IGNORECASE)
            except re.error as e:
                raise ToolError(f'invalid regular expression: {e}') from None
            limit = int(args.get('max_results') or 25)
            glob = args.get('glob') or '*'
            files = [root] if root.is_file() else sorted(root.rglob(glob))
            hits, scanned = [], 0
            for f in files:
                if not f.is_file() or f.stat().st_size > 2_000_000:
                    continue
                scanned += 1
                if scanned > 2000:
                    hits.append('... [stopped after scanning 2000 files]')
                    break
                try:
                    for n, line in enumerate(f.read_text(errors='replace').splitlines(), 1):
                        if rx.search(line):
                            try:
                                rel = f.relative_to(ctx.sandbox.resolve())
                            except ValueError:
                                rel = f
                            hits.append(f'{rel}:{n}: {line.strip()[:200]}')
                            if len(hits) >= limit:
                                return ctx.truncate('\n'.join(hits))
                except OSError:
                    continue
            return ctx.truncate('\n'.join(hits) or 'no matches')

        @self.tool('write_file',
                   'Write text to a file inside the sandbox (creates or overwrites).',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                    'required': ['path', 'content']},
                   risk='write')
        def write_file(args):
            path = ctx.inside(args['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get('content', '')))
            return f'wrote {path.stat().st_size} bytes to {args["path"]}'

        @self.tool('run_shell',
                   'Run a shell command in the sandbox and return combined stdout/stderr. Needs approval.',
                   {'type': 'object',
                    'properties': {'command': {'type': 'string'},
                                   'timeout': {'type': 'integer', 'description': 'Seconds (default 60)'}},
                    'required': ['command']},
                   risk='dangerous')
        def run_shell(args):
            cmd = str(args['command'])
            timeout = min(int(args.get('timeout') or 60), 300)
            try:
                proc = subprocess.run(cmd, shell=True, cwd=str(ctx.sandbox), capture_output=True,
                                      text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise ToolError(f'command timed out after {timeout}s') from None
            out = (proc.stdout or '') + (('\n[stderr]\n' + proc.stderr) if proc.stderr else '')
            return ctx.truncate(out.strip() + f'\n[exit code {proc.returncode}]')

        @self.tool('http_get',
                   'Fetch a URL and return the beginning of the response body. Needs approval.',
                   {'type': 'object',
                    'properties': {'url': {'type': 'string'},
                                   'max_bytes': {'type': 'integer', 'description': 'Default 8000'}},
                    'required': ['url']},
                   risk='dangerous')
        def http_get(args):
            url = str(args['url'])
            if not re.match(r'^https?://', url):
                raise ToolError('only http(s) URLs are allowed')
            limit = int(args.get('max_bytes') or 8000)
            req = urllib.request.Request(url, headers={'User-Agent': f'bonsai-chat/{VERSION}'})
            try:
                with urllib.request.urlopen(req, timeout=ctx.http_timeout) as r:
                    body = r.read(limit + 1)
                    ctype = r.headers.get('Content-Type', '')
            except Exception as e:
                raise ToolError(f'request failed: {e}') from None
            text = body[:limit].decode('utf-8', 'replace')
            more = f'\n... [truncated to {limit} bytes]' if len(body) > limit else ''
            return ctx.truncate(f'[content-type: {ctype}]\n{text}{more}')

        @self.tool('calculator',
                   'Evaluate an arithmetic expression. Supports + - * / // % ** and '
                   'sqrt, log, log2, log10, exp, sin, cos, tan, atan2, hypot, floor, ceil, '
                   'abs, round, min, max, sum, pow, pi, e.',
                   {'type': 'object',
                    'properties': {'expression': {'type': 'string', 'description': 'e.g. 2*(3+4)/sqrt(2)'}},
                    'required': ['expression']})
        def calculator(args):
            import ast as _ast
            expr = str(args['expression'])
            if len(expr) > 500:
                raise ToolError('expression too long')
            try:
                tree = _ast.parse(expr, mode='eval')
            except SyntaxError as e:
                raise ToolError(f'cannot parse expression: {e.msg}') from None
            value = _safe_eval(tree)
            return f'{expr} = {value}'

        @self.tool('current_time',
                   'Current date and time (UTC by default).',
                   {'type': 'object',
                    'properties': {'utc_offset_hours': {'type': 'number', 'description': 'Optional offset from UTC'}},
                    'required': []})
        def current_time(args):
            import datetime as _dt
            offset = args.get('utc_offset_hours')
            if offset is None:
                now = _dt.datetime.now(_dt.timezone.utc)
                label = 'UTC'
            else:
                tz = _dt.timezone(_dt.timedelta(hours=float(offset)))
                now = _dt.datetime.now(tz)
                label = f'UTC{float(offset):+g}'
            return now.strftime(f'%Y-%m-%d %H:%M:%S {label} (%A)')

        @self.tool('image_inspect',
                   'Measure an image file: format, pixel dimensions, Pillow colour statistics '
                   'and (when tesseract is installed) OCR text. Use this when the user mentions '
                   'an image and the server cannot see it.',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string'},
                                   'ocr': {'type': 'boolean', 'description': 'Attempt OCR if available'}},
                    'required': ['path']})
        def image_inspect(args):
            path = ctx.inside(args['path'])
            att = ImageAttachment.load(path, encode=False, use_ocr=bool(args.get('ocr')))
            return ctx.truncate(att.text_card(vision=False))

    # ------------------------------------------------------------------
    def schemas(self):
        return [t.schema() for t in self.tools.values() if t.enabled]

    def names(self):
        return sorted(self.tools)

    def set_enabled(self, name, enabled):
        if name not in self.tools:
            raise ToolError(f'unknown tool: {name}')
        self.tools[name].enabled = bool(enabled)

    def parse_arguments(self, raw):
        """Model-supplied arguments -> dict, with a repairable error message on bad JSON."""
        if raw in (None, ''):
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            raise ToolError(f'arguments were not valid JSON ({e}); send them as a JSON object') from None
        if not isinstance(parsed, dict):
            raise ToolError('arguments must be a JSON object')
        return parsed

    def execute(self, name, raw_args):
        """Run one tool call; always returns a string for the model (errors included)."""
        tool = self.tools.get(name)
        if tool is None:
            return f'ERROR: unknown tool {name!r}. Available: {", ".join(self.names())}'
        if not tool.enabled:
            return f'ERROR: tool {name!r} is disabled in this session.'
        try:
            args = self.parse_arguments(raw_args)
        except ToolError as e:
            return f'ERROR: {e}'
        if tool.risk != 'safe' and not self.ctx.ask(name, args, tool.risk):
            return (f'DENIED: the user declined to run {name} with {json.dumps(args)[:300]}. '
                    'Do not retry it; answer with what you already know.')
        self.ctx.say(f'⚙ {name} {json.dumps(args, ensure_ascii=False)[:200]}')
        try:
            result = tool.func(args)
        except ToolError as e:
            return f'ERROR: {e}'
        except Exception as e:
            return f'ERROR: {type(e).__name__}: {e}'
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


# ======================================================================
# Conversation state
# ======================================================================
DEFAULT_SYSTEM = (
    'You are Bonsai, a precise coding and reasoning assistant served from a single-slot '
    'llama.cpp endpoint. Answer in the language the user writes in. Use Markdown: fenced '
    'code blocks with a language tag, tables for comparisons, and short paragraphs. When a '
    'tool would give a better answer than guessing, call it; report tool output faithfully '
    'and say plainly when you do not know.')

WIRE_ROLES = ('system', 'user', 'assistant', 'tool')


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 2048
    effort: str = 'medium'       # none|low|medium|high -> reasoning_effort
    stream: bool = True
    use_tools: bool = True
    markdown: bool = True
    highlight: bool = True
    max_tool_rounds: int = 6
    context_budget: float = 0.75

    def api_params(self):
        params = {'temperature': self.temperature, 'top_p': self.top_p,
                  'max_tokens': self.max_tokens}
        if self.effort and self.effort != 'none':
            params['reasoning_effort'] = self.effort
        return params


class Conversation:
    """Message history with token-budget trimming that never orphans a tool result."""

    def __init__(self, system=DEFAULT_SYSTEM, counter=None):
        self.counter = counter or TokenCounter()
        self.messages = []
        self.system_text = None
        if system:
            self.set_system(system)

    # ------------------------------------------------------------------
    def set_system(self, text):
        self.system_text = text
        if self.messages and self.messages[0]['role'] == 'system':
            if text:
                self.messages[0] = {'role': 'system', 'content': text}
            else:
                self.messages.pop(0)
        elif text:
            self.messages.insert(0, {'role': 'system', 'content': text})

    def add(self, message):
        self.messages.append(message)
        return message

    def add_user(self, content):
        return self.add({'role': 'user', 'content': content})

    def add_assistant(self, message):
        wire = {'role': 'assistant', 'content': message.get('content')}
        if message.get('tool_calls'):
            wire['tool_calls'] = message['tool_calls']
            wire['content'] = message.get('content')
        if wire['content'] is None and not wire.get('tool_calls'):
            wire['content'] = ''
        return self.add(wire)

    def add_tool_result(self, tool_call_id, content, name=None):
        msg = {'role': 'tool', 'tool_call_id': tool_call_id, 'content': content}
        if name:
            msg['name'] = name
        return self.add(msg)

    def last_user_index(self):
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]['role'] == 'user':
                return i
        return None

    def wire(self):
        """Messages in exactly the shape the API expects."""
        out = []
        for m in self.messages:
            if m['role'] not in WIRE_ROLES:
                continue
            if m['role'] == 'tool':
                out.append({'role': 'tool', 'tool_call_id': m.get('tool_call_id'),
                            'content': m.get('content', '')})
                continue
            item = {'role': m['role'], 'content': m.get('content', '')}
            if m.get('tool_calls'):
                item['tool_calls'] = m['tool_calls']
            out.append(item)
        return out

    def tokens(self):
        return self.counter.messages(self.wire())

    def trim(self, budget):
        """Drop the oldest complete turns until the history fits `budget` tokens.

        Removal happens per turn group (a user message plus the assistant/tool messages that
        answer it), so a `tool` message can never survive without the `tool_calls` it belongs
        to — that combination makes llama.cpp reject the whole request.
        """
        dropped = 0
        while self.tokens() > budget and len(self.messages) > 2:
            start = 1 if (self.messages and self.messages[0]['role'] == 'system') else 0
            end = start + 1
            while end < len(self.messages) and self.messages[end]['role'] != 'user':
                end += 1
            if end >= len(self.messages):
                break
            del self.messages[start:end]
            dropped += end - start
        return dropped

    def drop_last_turn(self):
        """Remove the newest user turn and everything after it (for /undo and /retry)."""
        i = self.last_user_index()
        if i is None:
            return 0
        removed = len(self.messages) - i
        del self.messages[i:]
        return removed

    # ------------------------------------------------------------------
    def save(self, path):
        path = Path(path).expanduser()
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            for m in self.messages:
                fh.write(json.dumps(m, ensure_ascii=False) + '\n')
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path, counter=None):
        path = Path(path).expanduser()
        conv = cls(system=None, counter=counter)
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get('role') == 'system' and not conv.messages:
                conv.set_system(msg.get('content') or '')
            else:
                conv.messages.append(msg)
        return conv

    def export_markdown(self, path, title='Bonsai session'):
        lines = [f'# {title}', '']
        for m in self.messages:
            role = m['role']
            body = m.get('content')
            if isinstance(body, list):
                body = '\n'.join(p.get('text', f"[{p.get('type')}]") for p in body if isinstance(p, dict))
            body = body or ''
            if role == 'system':
                lines += ['## System', '', '```', body, '```', '']
            elif role == 'user':
                lines += ['## You', '', body, '']
            elif role == 'assistant':
                lines += ['## Bonsai', '']
                if m.get('tool_calls'):
                    lines += ['```json',
                              json.dumps(m['tool_calls'], ensure_ascii=False, indent=2),
                              '```', '']
                if body:
                    lines += [body, '']
            elif role == 'tool':
                lines += [f'### tool result — `{m.get("name") or m.get("tool_call_id")}`', '',
                          '```', body[:4000], '```', '']
        path = Path(path).expanduser()
        path.write_text('\n'.join(lines), encoding='utf-8')
        return path


def mask_key(key):
    if not key:
        return '(no key)'
    if len(key) <= 12:
        return key[:2] + '…'
    return key[:6] + '…' + key[-4:]


@dataclass
class TurnResult:
    text: str = ''
    reasoning: str = ''
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ''
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    elapsed: float = 0.0
    ttft: float = 0.0
    rounds: int = 0
    cancelled: bool = False

    @property
    def prompt_tokens(self):
        return (self.usage or {}).get('prompt_tokens') or 0

    @property
    def completion_tokens(self):
        return (self.usage or {}).get('completion_tokens') or 0

    def rate(self):
        t = (self.timings or {}).get('tokens_per_second')
        if t:
            return float(t)
        decode = (self.timings or {}).get('predicted_ms')
        n = (self.timings or {}).get('predicted_n') or self.completion_tokens
        if decode and n:
            return float(n) / (float(decode) / 1000.0)
        if self.elapsed and self.completion_tokens:
            return self.completion_tokens / self.elapsed
        return 0.0


class Agent:
    """Drives one turn: stream the answer, run any tool calls, feed results back, repeat."""

    def __init__(self, client, settings, registry=None, counter=None,
                 on_delta=None, on_reasoning=None, on_tool=None, on_notice=None):
        self.client = client
        self.settings = settings
        self.registry = registry
        self.counter = counter or TokenCounter(client)
        self.on_delta = on_delta or (lambda t: None)
        self.on_reasoning = on_reasoning or (lambda t: None)
        self.on_tool = on_tool or (lambda name, args, result: None)
        self.on_notice = on_notice or (lambda t: None)

    # ------------------------------------------------------------------
    def run(self, conversation, text='', attachments=None, regenerate=False):
        settings = self.settings
        vision = False
        if attachments:
            vision = self.client.probe_vision()
            if vision:
                self.on_notice(f'attaching {len(attachments)} image(s) as pixels '
                               '(server has a vision projector)')
            else:
                self.on_notice('server is text-only — sending measured image facts instead of pixels')
            message = build_user_message(text, attachments, vision)
        else:
            message = {'role': 'user', 'content': text or ''}
        if not regenerate:
            conversation.add(message)

        tools = self.registry.schemas() if (self.registry and settings.use_tools) else None
        result = TurnResult()
        cancelled = False
        for round_no in range(1, settings.max_tool_rounds + 1):
            result.rounds = round_no
            try:
                assistant, usage, timings, finish, elapsed, ttft, cancelled = self._one_request(
                    conversation.wire(), tools)
            except CancelledByUser:
                cancelled = True
                assistant = {'role': 'assistant', 'content': result.text or '[cancelled]'}
                usage, timings, finish, elapsed, ttft = {}, {}, 'cancelled', 0.0, None
            result.usage = usage or result.usage
            result.timings = timings or result.timings
            result.finish_reason = finish or result.finish_reason
            result.elapsed += elapsed or 0.0
            result.ttft = result.ttft or ttft or 0.0
            conversation.add_assistant(assistant)
            if assistant.get('reasoning_content'):
                result.reasoning = ((result.reasoning + '\n') if result.reasoning else '') + \
                    assistant['reasoning_content']
            calls = assistant.get('tool_calls') or []
            if not calls:
                result.text = assistant.get('content') or ''
                break
            result.tool_calls.extend(calls)
            if round_no >= settings.max_tool_rounds:
                self.on_notice(f'stopped after {round_no} tool rounds (raise --max-tool-rounds)')
                result.text = assistant.get('content') or ''
                break
            for call in calls:
                fn = call.get('function') or {}
                name = fn.get('name') or '?'
                answer = self.registry.execute(name, fn.get('arguments')) if self.registry else \
                    'ERROR: tools are disabled'
                conversation.add_tool_result(call.get('id') or f'call_{round_no}', answer, name)
                self.on_tool(name, fn.get('arguments') or '', answer)
            if cancelled:
                break
        result.cancelled = cancelled
        return result

    def _one_request(self, messages, tools):
        settings = self.settings
        params = settings.api_params()
        if not settings.stream:
            resp = self.client.chat(messages, tools=tools, **params)
            choice = (resp.get('choices') or [{}])[0]
            msg = choice.get('message') or {}
            usage = resp.get('usage') or {}
            timings = resp.get('timings') or {}
            finish = choice.get('finish_reason') or ''
            text = msg.get('content') or ''
            if text:
                self.on_delta(text)
            return msg, usage, timings, finish, 0.0, None, False
        text_parts, reason_parts = [], []
        cancelled = False
        stream = self.client.stream_chat(messages, tools=tools, **params)
        try:
            for ev in stream:
                if ev['kind'] == 'delta':
                    text_parts.append(ev['text'])
                    self.on_delta(ev['text'])
                elif ev['kind'] == 'reasoning':
                    reason_parts.append(ev['text'])
                    self.on_reasoning(ev['text'])
                elif ev['kind'] == 'done':
                    return (ev['message'], ev.get('usage'), ev.get('timings'),
                            ev.get('finish_reason'), ev.get('elapsed'), ev.get('ttft'), False)
        except KeyboardInterrupt:
            cancelled = True
            try:
                stream.close()
            except Exception:
                pass
            raise CancelledByUser() from None
        msg = {'role': 'assistant', 'content': ''.join(text_parts)}
        return msg, None, None, None, 0.0, None, cancelled


# ======================================================================
# Live output
# ======================================================================
class LiveRenderer:
    """Streams reasoning + answer to the terminal, rendering Markdown as blocks complete."""

    def __init__(self, style, markdown=True, highlight=False, out=None):
        self.style = style
        self.out = out if out is not None else sys.stdout
        self.markdown = markdown
        self.writer = MarkdownWriter(style, highlight=highlight) if markdown else None
        self._reason_open = False

    def write(self, text):
        self.out.write(text)
        self.out.flush()

    def notice(self, text):
        self.write(self.style.dim('· ' + text) + '\n')

    def reasoning(self, text):
        if not self._reason_open:
            self.write(self.style.dim('✻ thinking …\n'))
            self._reason_open = True
        self.write(self.style.paint(text, 2))

    def delta(self, text):
        if self._reason_open:
            self.write('\n\n')
            self._reason_open = False
        if self.writer:
            self.write(self.writer.feed(text))
        else:
            self.write(text)

    def tool(self, name, args, result):
        shown = (result or '').strip().replace('\n', ' ')
        if len(shown) > 240:
            shown = shown[:240] + ' …'
        self.write('\n' + self.style.yellow('⚙ ' + name) + self.style.dim(' ' + str(args)[:160]) +
                   '\n' + self.style.dim('  ↳ ' + shown) + '\n\n')

    def finish(self):
        if self._reason_open:
            self.write('\n')
            self._reason_open = False
        if self.writer:
            self.write(self.writer.finish())
        self.write('\n')


HELP_TEXT = """\
Commands:
  /help                     this help
  /system <text>            replace the system prompt (empty text clears it)
  /reset                    start a fresh conversation
  /undo                     drop the last exchange
  /retry                    regenerate the previous answer
  /image <path> [...]       attach image(s) to the next message
  /ocr on|off               run tesseract OCR on attached images (needs pytesseract)
  /tools                    list tools; /tools <name> on|off toggles one; /tools off all
  /model [id]               show or change the model id
  /temp <0..2>  /topp <0..1>  /max-tokens <n>  /effort none|low|medium|high
  /stream on|off            toggle streaming
  /markdown on|off          toggle terminal Markdown rendering
  /vision on|off|probe      override or re-probe whether the server can see images
  /context                  token budget, context window, message count
  /history                  show the conversation so far
  /usage                    cumulative tokens and timings for this session
  /save [file]  /load <file>  /export [file.md]
  /quit                     leave (Ctrl-D also works; Ctrl-C cancels just this turn)

Anything else is sent to the model. End a line with a backslash to keep typing.
"""


class ChatApp:
    """The interactive loop: prompt -> slash command or model turn -> rendered answer."""

    def __init__(self, client, settings, style, registry, conversation, renderer=None,
                 session_path=None, use_ocr=False, image_max_side=1024,
                 input_fn=None, out=None, auto_approve=False, quiet=False):
        self.client = client
        self.settings = settings
        self.style = style
        self.registry = registry
        self.conversation = conversation
        self.out = out if out is not None else sys.stdout
        self.renderer = renderer or LiveRenderer(style, markdown=settings.markdown,
                                                 highlight=settings.highlight, out=self.out)
        self.session_path = session_path
        self.use_ocr = use_ocr
        self.image_max_side = image_max_side
        self.input_fn = input_fn or input
        self.auto_approve = auto_approve
        self.quiet = quiet
        self.pending_images = []
        self.totals = {'turns': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                       'tool_calls': 0, 'seconds': 0.0}
        self.agent = Agent(client, settings, registry=registry,
                           counter=conversation.counter,
                           on_delta=self.renderer.delta,
                           on_reasoning=self.renderer.reasoning,
                           on_tool=self.renderer.tool,
                           on_notice=self.renderer.notice)

    # ------------------------------------------------------------------
    def say(self, text=''):
        self.out.write(text + '\n')
        self.out.flush()

    def banner(self):
        s = self.style
        feats = optional_features()
        extras = ', '.join(k for k, v in feats.items() if v) or 'none (standard library only)'
        self.say(s.bold(s.cyan('Ternary Bonsai 2 27B — chat client v' + VERSION)))
        self.say(s.dim(f'endpoint {self.client.base_url}  key {mask_key(self.client.api_key)}'))
        models = self.client.model_ids()
        self.say(s.dim(f'model {self.client.model}  '
                       f'(served: {", ".join(models) if models else "unknown"})'))
        self.say(s.dim(f'context {self.client.context_window()} tokens, budget '
                       f'{int(self.client.context_window() * self.settings.context_budget)} | '
                       f'tools {"on: " + ", ".join(self.registry.names()) if self.settings.use_tools else "off"}'))
        self.say(s.dim(f'optional extras: {extras} | /help for commands'))
        self.say()

    # ------------------------------------------------------------------
    def approve(self, name, args, risk):
        """Interactive gate for tools that write or execute."""
        if self.auto_approve:
            return True
        self.say()
        self.say(self.style.yellow(f'⚠ the model wants to run {name} [{risk}]'))
        self.say(self.style.dim('  args: ' + json.dumps(args, ensure_ascii=False)[:600]))
        while True:
            try:
                answer = self.input_fn(self.style.bold('  allow? [y]es / [n]o / [a]lways: ')).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.say()
                return False
            if answer in ('y', 'yes'):
                return True
            if answer in ('n', 'no', ''):
                return False
            if answer in ('a', 'always'):
                self.auto_approve = True
                return True
            self.say(self.style.dim('  please answer y, n or a'))

    # ------------------------------------------------------------------
    def show_cards(self, attachments):
        """Print what the model will be told about each attached file."""
        for a in attachments:
            self.say(self.style.dim(a.text_card(self.client.probe_vision())))
        if attachments:
            self.say()

    def send(self, text, attachments=None, regenerate=False):
        attachments = attachments or []
        budget = int(self.client.context_window() * self.settings.context_budget)
        dropped = self.conversation.trim(budget)
        if dropped:
            self.renderer.notice(f'trimmed {dropped} oldest messages to fit the context budget')
        try:
            result = self.agent.run(self.conversation, text=text, attachments=attachments,
                                    regenerate=regenerate)
        except BonsaiError as e:
            self.renderer.finish()
            self.say(self.style.red('✗ ' + str(e)))
            hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
            if hint:
                self.say(self.style.dim('  hint: ' + hint))
            return None
        self.renderer.finish()
        self.totals['turns'] += 1
        self.totals['prompt_tokens'] += result.prompt_tokens
        self.totals['completion_tokens'] += result.completion_tokens
        self.totals['tool_calls'] += len(result.tool_calls)
        self.totals['seconds'] += result.elapsed or 0.0
        self.say(self.style.dim(
            f'{result.completion_tokens} tok out / {result.prompt_tokens} in'
            + (f' | {result.rate():.1f} tok/s' if result.rate() else '')
            + (f' | ttft {result.ttft * 1000:.0f} ms' if result.ttft else '')
            + (f' | {len(result.tool_calls)} tool call(s)' if result.tool_calls else '')
            + f' | {result.elapsed:.1f}s' + (' | cancelled' if result.cancelled else '')))
        self.say()
        if self.session_path:
            try:
                self.conversation.save(self.session_path)
            except OSError as e:
                self.say(self.style.dim(f'(could not autosave session: {e})'))
        return result

    # ------------------------------------------------------------------
    def command(self, line):
        """Handle one slash command. Returns False to exit the loop."""
        parts = line[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ''
        arg = parts[1].strip() if len(parts) > 1 else ''
        s = self.style
        try:
            if cmd in ('q', 'quit', 'exit'):
                return False
            if cmd == 'help':
                self.say(HELP_TEXT)
            elif cmd == 'system':
                self.conversation.set_system(arg)
                self.say(s.dim('system prompt updated' if arg else 'system prompt cleared'))
            elif cmd == 'reset':
                system = self.conversation.system_text
                self.conversation = Conversation(system=system, counter=self.conversation.counter)
                self.agent = Agent(self.client, self.settings, registry=self.registry,
                                   counter=self.conversation.counter,
                                   on_delta=self.renderer.delta, on_reasoning=self.renderer.reasoning,
                                   on_tool=self.renderer.tool, on_notice=self.renderer.notice)
                self.say(s.dim('conversation cleared'))
            elif cmd == 'undo':
                n = self.conversation.drop_last_turn()
                self.say(s.dim(f'removed {n} message(s)' if n else 'nothing to undo'))
            elif cmd == 'retry':
                i = self.conversation.last_user_index()
                if i is None:
                    self.say(s.dim('nothing to retry yet'))
                else:
                    text = self.conversation.messages[i].get('content')
                    if isinstance(text, list):
                        text = ' '.join(p.get('text', '') for p in text if isinstance(p, dict))
                    self.conversation.drop_last_turn()
                    self.say(s.dim('regenerating …'))
                    self.send(text or '')
            elif cmd == 'image':
                if not arg:
                    self.say(s.dim('queued: ' + ', '.join(a.display_name for a in self.pending_images)
                                   if self.pending_images else 'usage: /image <path> [more paths]'))
                else:
                    loaded = load_attachments(arg.split(), max_side=self.image_max_side,
                                              use_ocr=self.use_ocr)
                    self.pending_images.extend(loaded)
                    for a in loaded:
                        self.say(s.dim(a.text_card(self.client.probe_vision())))
            elif cmd == 'ocr':
                self.use_ocr = arg.lower() in ('on', '1', 'true', 'yes')
                self.say(s.dim('OCR ' + ('on' if self.use_ocr else 'off') +
                               ('' if (pytesseract and shutil.which('tesseract')) or not self.use_ocr
                                else ' — but pytesseract/tesseract is not installed')))
            elif cmd == 'tools':
                self._tools_command(arg)
            elif cmd == 'model':
                if arg:
                    self.client.model = arg
                    self.settings.model = arg
                    self.say(s.dim('model set to ' + arg))
                else:
                    self.say(s.dim(f'model {self.client.model}; served: '
                                   f'{", ".join(self.client.model_ids()) or "unknown"}'))
            elif cmd == 'temp':
                self.settings.temperature = float(arg)
                self.say(s.dim(f'temperature {self.settings.temperature}'))
            elif cmd == 'topp':
                self.settings.top_p = float(arg)
                self.say(s.dim(f'top_p {self.settings.top_p}'))
            elif cmd in ('max-tokens', 'maxtokens'):
                self.settings.max_tokens = int(arg)
                self.say(s.dim(f'max_tokens {self.settings.max_tokens}'))
            elif cmd == 'effort':
                self.settings.effort = (arg or 'medium').lower()
                self.say(s.dim(f'reasoning effort {self.settings.effort}'))
            elif cmd == 'stream':
                self.settings.stream = arg.lower() in ('on', '1', 'true', 'yes')
                self.say(s.dim('streaming ' + ('on' if self.settings.stream else 'off')))
            elif cmd == 'markdown':
                self.settings.markdown = arg.lower() in ('on', '1', 'true', 'yes')
                self.renderer.markdown = self.settings.markdown
                self.renderer.writer = (MarkdownWriter(self.style, highlight=self.settings.highlight)
                                        if self.settings.markdown else None)
                self.say(s.dim('markdown ' + ('on' if self.settings.markdown else 'off')))
            elif cmd == 'vision':
                if arg.lower() == 'probe':
                    self.client._vision = None
                    self.say(s.dim('vision: ' + str(self.client.probe_vision())))
                elif arg:
                    self.client.probe_vision(force=arg.lower() in ('on', '1', 'true', 'yes'))
                    self.say(s.dim('vision forced ' + str(self.client._vision)))
                else:
                    self.say(s.dim('vision: ' + str(self.client.probe_vision())))
            elif cmd == 'context':
                self.say(s.dim(f'context window {self.client.context_window()} | '
                               f'history {self.conversation.tokens()} tokens over '
                               f'{len(self.conversation.messages)} messages | '
                               f'budget {int(self.client.context_window() * self.settings.context_budget)}'))
            elif cmd == 'history':
                for m in self.conversation.messages:
                    body = m.get('content')
                    if isinstance(body, list):
                        body = ' '.join(p.get('text', '') for p in body if isinstance(p, dict))
                    body = (body or '').replace('\n', ' ')
                    self.say(s.dim(f'[{m["role"]}] ') + body[:300])
            elif cmd == 'usage':
                t = self.totals
                self.say(s.dim(f'{t["turns"]} turn(s) | {t["completion_tokens"]} tokens out, '
                               f'{t["prompt_tokens"]} in | {t["tool_calls"]} tool call(s) | '
                               f'{t["seconds"]:.1f}s of model time'))
            elif cmd == 'save':
                path = self.conversation.save(arg or self.session_path or 'bonsai-session.jsonl')
                self.say(s.dim('saved ' + str(path)))
            elif cmd == 'load':
                if not arg:
                    self.say(s.dim('usage: /load <file.jsonl>'))
                else:
                    self.conversation = Conversation.load(arg, counter=self.conversation.counter)
                    self.agent = Agent(self.client, self.settings, registry=self.registry,
                                       counter=self.conversation.counter,
                                       on_delta=self.renderer.delta, on_reasoning=self.renderer.reasoning,
                                       on_tool=self.renderer.tool, on_notice=self.renderer.notice)
                    self.say(s.dim(f'loaded {len(self.conversation.messages)} messages from {arg}'))
            elif cmd == 'export':
                path = self.conversation.export_markdown(arg or 'bonsai-session.md')
                self.say(s.dim('exported ' + str(path)))
            else:
                self.say(s.red(f'unknown command /{cmd} — try /help'))
        except (ValueError, BonsaiError, OSError, json.JSONDecodeError) as e:
            self.say(s.red(f'{cmd} failed: {e}'))
        return True

    def _tools_command(self, arg):
        s = self.style
        if not arg:
            for name in self.registry.names():
                tool = self.registry.tools[name]
                state = s.green('on ') if tool.enabled else s.dim('off')
                self.say(f'  {state} {name:<14} {s.dim(tool.risk):<12} '
                         f'{tool.description.splitlines()[0][:70]}')
            return
        bits = arg.split()
        if bits[0].lower() == 'off' and len(bits) == 1:
            self.settings.use_tools = False
            self.say(s.dim('tool calling disabled for this session'))
            return
        if bits[0].lower() == 'on' and len(bits) == 1:
            self.settings.use_tools = True
            self.say(s.dim('tool calling enabled'))
            return
        if len(bits) == 2:
            self.registry.set_enabled(bits[0], bits[1].lower() in ('on', '1', 'true', 'yes'))
            self.say(s.dim(f'{bits[0]} -> {bits[1]}'))
        else:
            self.say(s.dim('usage: /tools | /tools <name> on|off | /tools on | /tools off'))

    # ------------------------------------------------------------------
    def run(self):
        s = self.style
        self.banner()
        if self.pending_images:
            pass
        while True:
            try:
                line = self._read_input()
            except EOFError:
                self.say()
                break
            except KeyboardInterrupt:
                self.say(s.dim('\n(interrupted — /quit to leave)'))
                continue
            if line is None:
                break
            if not line:
                continue
            if line.startswith('/'):
                if not self.command(line):
                    break
                continue
            attachments, self.pending_images = self.pending_images, []
            self.send(line, attachments=attachments)
        return 0

    def _read_input(self):
        prompt = self.style.bold(self.style.cyan('you › '))
        chunks = []
        while True:
            raw = self.input_fn(prompt if not chunks else self.style.dim('… '))
            if raw is None:
                return None
            if raw.endswith('\\'):
                chunks.append(raw[:-1])
                continue
            chunks.append(raw)
            line = '\n'.join(chunks).strip()
            return line


# ======================================================================
# CLI
# ======================================================================
def build_parser():
    p = argparse.ArgumentParser(
        prog='bonsai_chat.py',
        description='Chat client for a Ternary Bonsai 2 27B OpenAI-compatible endpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Environment: BONSAI_BASE_URL, BONSAI_API_KEY, BONSAI_MODEL, BONSAI_SESSION.\n'
               'The deployment cell (colab_kaggle_cell.py) prints the base URL and key.')
    p.add_argument('--base-url', default=os.environ.get('BONSAI_BASE_URL'),
                   help='e.g. https://<tunnel-host>/v1')
    p.add_argument('--api-key', default=os.environ.get('BONSAI_API_KEY'),
                   help='bearer key printed by the deployment cell')
    p.add_argument('--model', default=os.environ.get('BONSAI_MODEL', DEFAULT_MODEL))
    p.add_argument('-p', '--prompt', help='one-shot prompt instead of the interactive loop')
    p.add_argument('--image', action='append', default=[], help='attach an image (repeatable)')
    p.add_argument('--ocr', action='store_true', help='OCR attached images with tesseract')
    p.add_argument('--image-max-side', type=int, default=1024,
                   help='downscale attached images to this many pixels (default 1024)')
    p.add_argument('--system', default=None, help='system prompt text')
    p.add_argument('--system-file', default=None, help='read the system prompt from a file')
    p.add_argument('--session', default=os.environ.get('BONSAI_SESSION'),
                   help='autosave/load the conversation to this .jsonl file')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--effort', default='medium', choices=['none', 'low', 'medium', 'high'],
                   help='reasoning effort sent as reasoning_effort')
    p.add_argument('--no-stream', action='store_true', help='request one complete answer')
    p.add_argument('--no-markdown', action='store_true', help='print model output verbatim')
    p.add_argument('--no-highlight', action='store_true', help='no Pygments highlighting in code blocks')
    p.add_argument('--no-tools', action='store_true', help='disable tool calling')
    p.add_argument('--tools', help='comma-separated tool names to enable (default: all)')
    p.add_argument('--max-tool-rounds', type=int, default=6)
    p.add_argument('--sandbox', default=None, help='directory tools may touch (default: cwd)')
    p.add_argument('--auto-approve', action='store_true',
                   help='let the model write files and run commands without asking')
    p.add_argument('--timeout', type=int, default=600, help='streaming timeout in seconds')
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--context', type=int, default=None, help='override the context window in tokens')
    p.add_argument('--plain', action='store_true', help='disable colour')
    p.add_argument('--color', action='store_true',
                   help='force colour even when stdout is not a terminal')
    p.add_argument('--width', type=int, default=None, help='render width (default: terminal width)')
    p.add_argument('--json', action='store_true', help='one-shot mode: print the raw result as JSON')
    p.add_argument('--quiet', action='store_true', help='suppress the banner')
    p.add_argument('--selftest', action='store_true',
                   help='run the client against a local protocol stub and exit')
    p.add_argument('--doctor', action='store_true',
                   help='diagnose a real endpoint (health, models, context, auth, chat, '
                        'streaming, tool calling, vision) and exit')
    p.add_argument('--mock', action='store_true',
                   help='run against the bundled protocol stub (scripted replies, no model, '
                        'no GPU) to try the client without a deployment')
    p.add_argument('--version', action='version', version='bonsai_chat ' + VERSION)
    return p


def make_png(path, width=4, height=3, rgb=(200, 100, 50)):
    """Write a tiny valid PNG (used by --selftest; no third-party dependency)."""
    import zlib

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    row = b'\x00' + bytes(rgb) * width
    idat = zlib.compress(row * height)
    Path(path).write_bytes(PNG_MAGIC + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) +
                           chunk(b'IEND', b''))
    return Path(path)


def run_selftest(style=None, out=None):
    """Exercise the real client code paths against the offline protocol stub."""
    import tempfile
    from mock_bonsai_server import DEFAULT_KEY, MockBonsaiServer

    style = style or Style(force_color=False)
    out = out if out is not None else sys.stdout
    results = []

    def check(name, ok, detail=''):
        results.append(bool(ok))
        out.write(f'{"PASS" if ok else "FAIL"}  {name}' + (f' — {detail}' if detail else '') + '\n')
        out.flush()

    server = MockBonsaiServer().start()
    tmp = tempfile.TemporaryDirectory()
    try:
        root = Path(tmp.name)
        client = BonsaiClient(server.base_url, DEFAULT_KEY, model=DEFAULT_MODEL, retries=1)
        settings = Settings(effort='medium', max_tokens=512)

        check('models endpoint lists the alias', DEFAULT_MODEL in client.model_ids(),
              ', '.join(client.model_ids()))
        check('/props reports the context window', client.context_window() == 8192,
              f'{client.context_window()} tokens')
        counter = TokenCounter(client)
        n = counter.count('hello world, this is a tokenisation probe')
        check('/tokenize counts server-side', n > 0, f'{n} tokens')

        ctx = ToolContext(sandbox=root, notify=lambda t: None, auto_approve=True)
        registry = ToolRegistry(ctx)
        registry.add(Tool('get_weather', 'Current weather for a city',
                          {'type': 'object', 'properties': {'city': {'type': 'string'}},
                           'required': ['city']},
                          func=lambda a: '21°C and sunny in ' + str(a.get('city'))))
        conv = Conversation(system=DEFAULT_SYSTEM, counter=counter)
        app = ChatApp(client, settings, style, registry, conv,
                      session_path=root / 'session.jsonl', quiet=True,
                      input_fn=lambda *_: '', out=out, auto_approve=True)
        app.renderer = LiveRenderer(style, markdown=False, out=out)
        app.agent = Agent(client, settings, registry=registry, counter=counter,
                          on_delta=app.renderer.delta, on_reasoning=app.renderer.reasoning,
                          on_tool=lambda *a: None, on_notice=app.renderer.notice)

        answer = app.send('Explain why C++ can be fast in 3 sentences.')
        check('streaming chat produced an answer', bool(answer and answer.text.strip()),
              f'{len(answer.text)} chars' if answer else 'no result')
        check('streaming reported usage', bool(answer and answer.completion_tokens),
              f'{answer.completion_tokens} completion tokens' if answer else '')
        check('reasoning deltas arrived', bool(answer and answer.reasoning),
              f'{len(answer.reasoning)} chars of thinking' if answer else '')

        conv2 = Conversation(system=DEFAULT_SYSTEM, counter=counter)
        app2 = ChatApp(client, settings, style, registry, conv2, out=out, auto_approve=True)
        app2.renderer = LiveRenderer(style, markdown=False, out=out)
        app2.agent = Agent(client, settings, registry=registry, counter=counter,
                           on_delta=app2.renderer.delta, on_reasoning=app2.renderer.reasoning,
                           on_tool=lambda *a: None, on_notice=app2.renderer.notice)
        tooled = app2.send('What is the weather in Lisbon right now? Call the tool.')
        roles = [m['role'] for m in conv2.messages]
        check('model requested a tool', bool(tooled and tooled.tool_calls),
              ', '.join(tc['function']['name'] for tc in (tooled.tool_calls if tooled else [])))
        check('tool result was fed back', 'tool' in roles, ' -> '.join(roles))
        check('tool loop finished with an answer',
              bool(tooled and tooled.text.strip() and tooled.rounds >= 2),
              f'{tooled.rounds if tooled else 0} round(s)')

        png = make_png(root / 'chart.png')
        attachment = ImageAttachment.load(png, use_ocr=False)
        check('image header parsed without Pillow', attachment.width == 4 and attachment.height == 3,
              f'{attachment.fmt} {attachment.width}x{attachment.height}')
        check('vision probe detects a text-only server', client.probe_vision() is False)
        card = attachment.text_card(vision=False)
        check('text-only image card states the limits', 'TEXT-ONLY' in card and '4x3 px' in card)
        msg = build_user_message('what is in this image?', [attachment], vision=False)
        check('image facts are sent to the model', 'Image attached' in msg['content'])

        rendered = render_markdown('## Title\n\n- one\n- two\n\n```python\nprint(1)\n```\n',
                                   Style(force_color=False))
        check('markdown renderer emits blocks', '┌─' in rendered and 'Title' in rendered and
              'print(1)' in rendered)
        table = render_markdown('| a | b |\n| --- | --- |\n| 1 | 2 |\n', Style(force_color=False))
        check('markdown renderer draws tables', '┼' in table and '│' in table)

        trimmed = Conversation(system=DEFAULT_SYSTEM, counter=TokenCounter())
        for i in range(40):
            trimmed.add_user('question ' + str(i) + ' ' + ('x' * 400))
            trimmed.add_assistant({'role': 'assistant', 'content': 'answer ' + str(i) + ' ' + ('y' * 400)})
        before = len(trimmed.messages)
        trimmed.trim(400)
        wired = trimmed.wire()
        orphans = [i for i, m in enumerate(wired) if m['role'] == 'tool']
        check('history trimming keeps the budget', trimmed.tokens() <= 400,
              f'{before} -> {len(trimmed.messages)} messages, {trimmed.tokens()} tokens')
        check('trimming never orphans tool results', not orphans)

        bad = BonsaiClient(server.base_url, 'wrong-key', model=DEFAULT_MODEL, retries=1)
        try:
            bad.models()
            check('bad key is rejected', False, 'no error raised')
        except BonsaiAPIError as e:
            check('bad key is rejected', e.status == 401 and bool(e.hint()),
                  f'HTTP {e.status}')
        try:
            dead = BonsaiClient('http://127.0.0.1:1/v1', 'x', retries=1, timeout=2)
            dead.chat([{'role': 'user', 'content': 'hi'}])
            check('dead endpoint explains itself', False, 'no error raised')
        except BonsaiError as e:
            check('dead endpoint explains itself', 'cannot reach' in str(e))
        doctor_out = io.StringIO()
        doctor_code = run_doctor(client, style=style, out=doctor_out)
        check('--doctor passes against the stub endpoint', doctor_code == 0,
              doctor_out.getvalue().strip().splitlines()[-2] if doctor_out.getvalue() else '')
    finally:
        server.stop()
        tmp.cleanup()

    passed = sum(results)
    out.write(f'\n{passed}/{len(results)} self-test checks passed\n')
    out.flush()
    return 0 if passed == len(results) else 1


def run_doctor(client, style=None, out=None):
    """Live diagnostics against a real deployment: is this endpoint actually usable?"""
    style = style or Style(force_color=False)
    out = out if out is not None else sys.stdout
    results = []

    def check(name, state, detail=''):
        results.append(state)
        mark = {'PASS': style.green('PASS'), 'FAIL': style.red('FAIL'),
                'SKIP': style.yellow('SKIP')}[state]
        out.write(f'{mark}  {name}' + (style.dim(' — ' + detail) if detail else '') + '\n')
        out.flush()

    out.write(style.bold(f'Diagnosing {client.base_url} '
                         f'(key {mask_key(client.api_key)}, model {client.model})\n\n'))

    try:
        health = client.get_json('/health', timeout=15, root=True)
        check('/health', 'PASS' if isinstance(health, dict) else 'FAIL', json.dumps(health)[:80])
    except BonsaiError as e:
        check('/health', 'FAIL', str(e))
        out.write(style.red('\nThe endpoint is not reachable — nothing else can be tested. '
                            'If this was a Colab/Kaggle deployment, the runtime (and its '
                            'Quick Tunnel URL) is gone: rerun colab_kaggle_cell.py and use the '
                            'new base URL and key.\n'))
        out.flush()
        return 1

    try:
        ids = client.model_ids()
        check('/v1/models', 'PASS' if client.model in ids else 'FAIL', ', '.join(ids) or 'empty')
        if client.model not in ids:
            out.write(style.yellow(f'  note: --model {client.model} is not what the server '
                                   'reports; llama.cpp serves the loaded model anyway.\n'))
    except BonsaiError as e:
        check('/v1/models', 'FAIL', str(e))

    props = client.props()
    ctx = client.context_window()
    check('/props context window', 'PASS' if props else 'SKIP',
          f'{ctx} tokens' + (f', slots {props.get("total_slots")}' if props else ''))

    counter = TokenCounter(client)
    n = counter.count('doctor probe: tokenise this sentence')
    check('/tokenize', 'PASS' if counter.using_server() else 'SKIP',
          f'{n} tokens for a 36-char string' if counter.using_server()
          else 'unavailable — the client will estimate tokens (len/4) instead')

    try:
        wrong = BonsaiClient(client.base_url, 'deliberately-wrong-key', model=client.model,
                             retries=1)
        wrong.models()
        check('bearer auth enforced', 'FAIL', 'a wrong key was accepted')
    except BonsaiAPIError as e:
        check('bearer auth enforced', 'PASS' if e.status in (401, 403) else 'FAIL',
              f'HTTP {e.status}')
    except BonsaiError as e:
        check('bearer auth enforced', 'SKIP', str(e))

    try:
        reply = client.chat([{'role': 'user', 'content': 'Reply with the single word: ok'}],
                            max_tokens=16)
        text = ((reply.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        check('chat completion', 'PASS' if text.strip() else 'FAIL', repr(text[:40]))
    except BonsaiError as e:
        check('chat completion', 'FAIL', str(e))
        hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
        if hint:
            out.write(style.dim('  hint: ' + hint) + '\n')

    try:
        chunks, got = 0, []
        for ev in client.stream_chat([{'role': 'user', 'content': 'Count from 1 to 5.'}],
                                     max_tokens=64):
            if ev['kind'] in ('delta', 'reasoning'):
                chunks += 1
                got.append(ev['text'])
        check('streaming (SSE)', 'PASS' if chunks >= 3 else 'FAIL', f'{chunks} chunks')
    except BonsaiError as e:
        check('streaming (SSE)', 'FAIL', str(e))

    tools = [{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Get the current weather for a city',
        'parameters': {'type': 'object',
                       'properties': {'city': {'type': 'string', 'description': 'City name'}},
                       'required': ['city']}}}]
    try:
        reply = client.chat([{'role': 'user',
                              'content': 'What is the weather in Lisbon right now? '
                                         'Answer only by calling the provided tool.'}],
                            tools=tools, max_tokens=512)
        msg = (reply.get('choices') or [{}])[0].get('message') or {}
        finish = (reply.get('choices') or [{}])[0].get('finish_reason')
        if msg.get('tool_calls') or finish == 'tool_calls':
            names = ', '.join((tc.get('function') or {}).get('name', '?')
                              for tc in msg.get('tool_calls') or [])
            check('native tool calling', 'PASS', names)
        else:
            check('native tool calling', 'FAIL',
                  f'no tool_calls (finish_reason={finish}); the server may lack --jinja')
    except BonsaiAPIError as e:
        check('native tool calling', 'SKIP', f'HTTP {e.status}: {e.message[:80]}')
    except BonsaiError as e:
        check('native tool calling', 'SKIP', str(e))

    if client.supports_tools is False:
        out.write(style.dim('  note: the client already fell back to no-tool mode for this '
                            'endpoint.\n'))
    if client.supports_reasoning_effort is False:
        out.write(style.dim('  note: reasoning_effort is not accepted by this build.\n'))

    vision = client.probe_vision()
    check('vision (image input)', 'PASS' if vision else 'SKIP',
          'pixels can be sent' if vision else
          'text-only build — the client sends measured image facts instead')

    passed = results.count('PASS')
    failed = results.count('FAIL')
    out.write(f'\n{passed} passed, {failed} failed, {results.count("SKIP")} skipped\n')
    out.write(style.green('Endpoint is usable.' if not failed else
                          'Endpoint has failures — see above.') + '\n')
    out.flush()
    return 0 if not failed else 1


def make_app(client, settings, style, registry=None, conversation=None, sandbox=None,
             session_path=None, use_ocr=False, image_max_side=1024, auto_approve=False,
             quiet=False, input_fn=None, out=None):
    """Assemble a ChatApp exactly the way the CLI does (tests and --selftest use this too)."""
    out = out if out is not None else sys.stdout
    sandbox = Path(sandbox).expanduser().resolve() if sandbox else Path.cwd().resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    registry = registry or ToolRegistry(ToolContext(sandbox=sandbox, auto_approve=auto_approve))
    conversation = conversation or Conversation(counter=TokenCounter(client))
    app = ChatApp(client, settings, style, registry, conversation,
                  session_path=Path(session_path).expanduser() if session_path else None,
                  use_ocr=use_ocr, image_max_side=image_max_side,
                  input_fn=input_fn, out=out, auto_approve=auto_approve, quiet=quiet)
    registry.ctx.approve = app.approve
    registry.ctx.notify = app.renderer.notice
    return app


def main(argv=None):
    args = build_parser().parse_args(argv)
    style = Style(force_color=False if args.plain else (True if args.color else None),
                  width=args.width)

    if args.selftest:
        return run_selftest(style=style)

    mock_server = None
    if args.mock:
        from mock_bonsai_server import DEFAULT_KEY as MOCK_KEY, MockBonsaiServer
        mock_server = MockBonsaiServer().start()
        args.base_url = args.base_url or mock_server.base_url
        args.api_key = args.api_key or MOCK_KEY
        sys.stderr.write(f'mock endpoint {args.base_url} — scripted replies, '
                         'this is a protocol stub and not the model\n')

    if not args.base_url:
        sys.stderr.write(
            'error: no endpoint. Export BONSAI_BASE_URL (the deployment cell prints it as '
            '"Base URL: https://<host>/v1") or pass --base-url. '
            'Try --selftest to exercise the client offline.\n')
        return 2

    explicit_system = args.system is not None or bool(args.system_file)
    system = None
    if args.system is not None:
        system = args.system
    elif args.system_file:
        system = Path(args.system_file).expanduser().read_text(encoding='utf-8')

    settings = Settings(model=args.model, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, effort=args.effort,
                        stream=not args.no_stream, use_tools=not args.no_tools,
                        markdown=not args.no_markdown, highlight=not args.no_highlight,
                        max_tool_rounds=args.max_tool_rounds)

    client = BonsaiClient(args.base_url, args.api_key, model=args.model,
                          timeout=args.timeout, retries=args.retries)
    if args.doctor:
        return run_doctor(client, style=style)
    if args.context:
        client.context_window = (lambda override=args.context: override)

    counter = TokenCounter(client)
    resumed = bool(args.session) and Path(args.session).expanduser().is_file()
    if resumed:
        conversation = Conversation.load(args.session, counter=counter)
        if explicit_system:
            conversation.set_system(system or '')
    else:
        conversation = Conversation(system=system if explicit_system else DEFAULT_SYSTEM,
                                    counter=counter)

    registry = ToolRegistry(ToolContext(
        sandbox=Path(args.sandbox).expanduser().resolve() if args.sandbox else Path.cwd().resolve(),
        auto_approve=args.auto_approve))
    if args.tools:
        wanted = {t.strip() for t in args.tools.split(',') if t.strip()}
        for name in registry.names():
            registry.set_enabled(name, name in wanted)
        unknown = wanted - set(registry.names())
        if unknown:
            sys.stderr.write(f'warning: unknown tool(s): {", ".join(sorted(unknown))}\n')

    app = make_app(client, settings, style, registry=registry, conversation=conversation,
                   session_path=args.session, use_ocr=args.ocr,
                   image_max_side=args.image_max_side, auto_approve=args.auto_approve,
                   quiet=args.quiet)
    if args.quiet:
        app.banner = lambda: None

    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            prompt = piped

    try:
        attachments = load_attachments(args.image, max_side=args.image_max_side,
                                       use_ocr=args.ocr) if args.image else []
    except BonsaiError as e:
        sys.stderr.write(f'error: {e}\n')
        return 2
    if attachments:
        app.show_cards(attachments)

    if args.json:
        app.renderer = LiveRenderer(style, markdown=False, out=io.StringIO())
        app.agent = Agent(client, settings, registry=registry, counter=counter,
                          on_delta=lambda t: None, on_reasoning=lambda t: None,
                          on_tool=lambda *a: None, on_notice=lambda t: None)

    try:
        if prompt is not None:
            result = app.send(prompt, attachments=attachments)
            if result is None:
                return 1
            if args.json:
                print(json.dumps({'model': client.model, 'text': result.text,
                                  'reasoning': result.reasoning,
                                  'tool_calls': result.tool_calls,
                                  'finish_reason': result.finish_reason,
                                  'usage': result.usage, 'timings': result.timings,
                                  'elapsed': result.elapsed, 'ttft': result.ttft,
                                  'rounds': result.rounds}, ensure_ascii=False, indent=2))
            return 0
        return app.run()
    except KeyboardInterrupt:
        sys.stderr.write('\ninterrupted\n')
        return 130
    except BonsaiError as e:
        sys.stderr.write(f'error: {e}\n')
        hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
        if hint:
            sys.stderr.write(f'hint: {hint}\n')
        return 1
    finally:
        if mock_server is not None:
            mock_server.stop()


if __name__ == '__main__':
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
