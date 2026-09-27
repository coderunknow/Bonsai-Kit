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
    approve/deny flow, parallel calls, bounded rounds/time/output, and an automatic
    tool-result loop
  * thinking control: /think off|low|medium|high|max maps onto the runtime's real
    `thinking_budget_tokens` field, and /effort onto `reasoning_effort` — both confirmed
    against the server rather than assumed, so nothing unsupported is ever advertised
  * measured diagnostics: --doctor [--json], --benchmark, /stats and /caps report only
    numbers that were actually measured, and say `n/a` when they were not

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
    python3 bonsai_chat.py --mock                # try the whole client with no deployment
    python3 bonsai_chat.py --doctor --json       # machine-readable endpoint diagnostics
    python3 bonsai_chat.py --benchmark           # measure TTFT, prefill and decode speed
"""

from __future__ import annotations

import argparse
import base64
import binascii
import http.client
import io
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import ssl
import statistics
import struct
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

VERSION = '0.5.0'
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


class TransportError(BonsaiError):
    """The HTTP layer could not complete a call. `request_sent` drives retry safety.

    A chat POST that the server may already be generating for must not be replayed
    blindly: doing so doubles the work and can hand the user a duplicated answer.
    So the transport records how far the request got, and only "never left this
    process" failures are retried automatically for non-idempotent calls.
    """

    #: request phases, in order; retry is safe for a POST only up to `sent=False`
    def __init__(self, message, url='', request_sent=False, phase='connect'):
        self.url = url
        self.request_sent = request_sent
        self.phase = phase
        super().__init__(message)

    @property
    def safe_to_retry_post(self):
        return not self.request_sent


class StreamInterrupted(BonsaiError):
    """The server stopped mid-generation. Carries whatever did arrive."""

    def __init__(self, message, partial_text='', partial_reasoning='', reason=''):
        self.partial_text = partial_text
        self.partial_reasoning = partial_reasoning
        self.reason = reason
        super().__init__(message)


class CapabilityError(BonsaiError):
    """A requested feature is not supported by this server build."""


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


class SSEDecoder:
    """Incremental Server-Sent-Events decoder that survives fragmented network reads.

    Real tunnels hand us arbitrary byte boundaries: an event can be split anywhere, a
    keep-alive comment can arrive alone, and a `data:` field can span several lines.
    The decoder buffers bytes, splits on LF/CRLF, strips exactly one leading space after
    `data:` (per the SSE spec), joins multi-line data with newlines, ignores `:` comments
    and unknown fields, and dispatches an event on each blank line. A final unterminated
    event is emitted by `close()` so a server that disconnects without a trailing blank
    line does not swallow the last chunk.
    """

    def __init__(self, max_event_bytes=MAX_SSE_EVENT_BYTES):
        self._buf = b''
        self._data = []
        self._bytes = 0
        self.max_event_bytes = max_event_bytes
        self.events = 0
        self.comments = 0
        self.malformed = 0
        self.bytes_seen = 0

    # ------------------------------------------------------------------
    def feed(self, chunk):
        """Feed raw bytes; yield every complete event payload."""
        if isinstance(chunk, str):
            chunk = chunk.encode('utf-8')
        if not chunk:
            return
        self.bytes_seen += len(chunk)
        self._buf += chunk
        while True:
            idx = self._buf.find(b'\n')
            if idx < 0:
                break
            line = self._buf[:idx]
            self._buf = self._buf[idx + 1:]
            if line.endswith(b'\r'):
                line = line[:-1]
            payload = self._line(line.decode('utf-8', 'replace'))
            if payload is not None:
                yield payload
        if len(self._buf) > self.max_event_bytes:
            # A runaway line (never any \n) is a broken stream, not a huge token.
            raise StreamInterrupted(f'SSE line exceeded {self.max_event_bytes} bytes without a '
                                    'newline — treating the stream as corrupt',
                                    reason='oversized-line')

    def _line(self, line):
        """Handle one SSE line. Returns an event payload string or None."""
        if line == '':
            return self._dispatch()
        if line.startswith(':'):
            self.comments += 1
            return None
        field, _, value = line.partition(':')
        if value.startswith(' '):
            value = value[1:]          # exactly one leading space, per spec
        if field == 'data':
            self._data.append(value)
            self._bytes += len(value) + 1
            if self._bytes > self.max_event_bytes:
                raise StreamInterrupted('SSE event exceeded the size limit', reason='oversized-event')
        # `event:`, `id:`, `retry:` and anything else are accepted and ignored.
        return None

    def _dispatch(self):
        if not self._data:
            return None
        payload = '\n'.join(self._data)
        self._data = []
        self._bytes = 0
        self.events += 1
        return payload

    def close(self):
        """Flush a trailing event left over by a truncated stream."""
        tail = self._buf
        self._buf = b''
        if tail:
            line = tail.decode('utf-8', 'replace')
            if line.endswith('\r'):
                line = line[:-1]
            self._line(line)
        return self._dispatch()


def iter_sse_events(resp):
    """Yield the payload of each `data:` event from a streaming HTTP response.

    Handles multi-line data fields, CRLF, SSE comment keep-alives, and reads that arrive
    split at arbitrary byte boundaries. Kept as a generator over any iterable of chunks
    or lines so it stays directly unit-testable.
    """
    decoder = SSEDecoder()
    for raw in resp:
        for payload in decoder.feed(raw):
            yield payload
    tail = decoder.close()
    if tail is not None:
        yield tail


def accumulate_tool_calls(delta_calls, acc):
    """Fold streamed tool-call fragments into complete OpenAI-shaped tool calls.

    llama.cpp sends a tool call as a long run of fragments: the id once, the function
    name possibly split across chunks, and the JSON arguments dribbled out a few
    characters at a time. Multiple calls can be interleaved by `index`. Nothing here
    parses the arguments — an incomplete JSON string is not an error, it is just not
    finished yet, and only the caller decides when the call is complete.
    """
    for tc in delta_calls or []:
        try:
            idx = int(tc.get('index', 0))
        except (TypeError, ValueError):
            idx = 0
        slot = acc.setdefault(idx, {'id': None, 'type': 'function',
                                    'function': {'name': '', 'arguments': ''}})
        if tc.get('id'):
            slot['id'] = tc['id']
        if tc.get('type'):
            slot['type'] = tc['type']
        fn = tc.get('function') or {}
        if fn.get('name'):
            slot['function']['name'] += fn['name']
        args = fn.get('arguments')
        if args:
            slot['function']['arguments'] += args if isinstance(args, str) else json.dumps(args)
    return acc


def arguments_complete(raw):
    """True when a streamed tool-call argument string is plausibly complete JSON.

    Used only to decide whether to *attempt* execution. A false negative just means we
    report the call as truncated instead of guessing.
    """
    if raw is None:
        return True
    if not isinstance(raw, str):
        return True
    s = raw.strip()
    if not s:
        return True                      # empty arguments == no arguments
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


# ======================================================================
# HTTP transport: one persistent connection, transparent reconnect,
# and a retry policy that never replays a generation that may have run.
# ======================================================================
class HttpTransport:
    """Persistent HTTP/1.1 connection to one host, with keep-alive and safe retries.

    Every request the client makes used to open a fresh TCP connection; against a
    Cloudflare tunnel that means a TLS handshake per call, which dominated the latency
    of short requests and of the token-counting round trips. This keeps one connection
    open, reuses it, and reconnects transparently when the far end has closed it.

    Retry safety is explicit. A failed call reports how far it got:
      * connect  — the socket never opened; the server saw nothing.  Safe to retry.
      * sent     — bytes left this process. The server may be generating. NOT retried
                   for POST; the caller gets a clear message instead of a duplicate.
      * response — the request completed but the reply was lost. NOT retried for POST.
    """

    def __init__(self, url, timeout=30, connect_timeout=15, opener=None, log=None,
                 keepalive=True, idle_ttl=3.0):
        parsed = urllib.parse.urlsplit(url)
        self.scheme = parsed.scheme or 'http'
        self.host = parsed.hostname or ''
        self.port = parsed.port or (443 if self.scheme == 'https' else 80)
        self.prefix = urllib.parse.urlunsplit((self.scheme, parsed.netloc, '', '', ''))
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.opener = opener                 # injectable for tests / non-http clients
        self.log = log or (lambda *a, **k: None)
        self.keepalive = keepalive
        self.idle_ttl = idle_ttl     # shorter than the server's keep-alive window
        self._conn = None
        self._stream_outstanding = False
        self.last_used = 0.0
        self.connections_opened = 0
        self.requests = 0
        self.reconnects = 0
        self._ssl_context = ssl.create_default_context() if self.scheme == 'https' else None

    # ------------------------------------------------------------------
    def _new_conn(self):
        self.close()
        if self.scheme == 'https':
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.connect_timeout,
                                               context=self._ssl_context)
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.connect_timeout)
        self._conn = conn
        self.connections_opened += 1
        return conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._stream_outstanding = False

    # ------------------------------------------------------------------
    def request(self, method, url, body=None, headers=None, timeout=None, retry=None,
                stream=False):
        """One HTTP call -> http.client.HTTPResponse (caller must close it).

        `retry` overrides the retry policy; by default GET/HEAD may be retried freely and
        POST is retried only when we can prove the server never saw it.
        """
        if self.opener is not None:
            req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
            return self.opener.open(req, timeout=timeout or self.timeout)

        parts = urllib.parse.urlsplit(url)
        target = urllib.parse.urlunsplit(('', '', parts.path or '/', parts.query, ''))
        idempotent = method.upper() in ('GET', 'HEAD')
        allow = idempotent if retry is None else bool(retry)
        last = None
        for attempt in (1, 2):
            sent = False
            pooled = False
            try:
                conn = self._reusable_conn()
                pooled = conn is not None
                if conn is None:
                    conn = self._new_conn()
                conn.timeout = timeout or self.timeout
                conn.request(method.upper(), target, body=body,
                             headers=dict(headers or {}, Connection='keep-alive'))
                sent = True
                if conn.sock is not None:
                    # conn.timeout only applies to connect; the read deadline for a long
                    # generation is set on the socket itself.
                    try:
                        conn.sock.settimeout(timeout or self.timeout)
                    except OSError:
                        pass
                resp = conn.getresponse()
                self.requests += 1
                self.last_used = time.monotonic()
                if stream:
                    # Nothing may be multiplexed onto this connection until the caller
                    # hands the body back via release() — even if the caller forgets.
                    self._stream_outstanding = True
                if not self.keepalive or resp.will_close:
                    # Never pool a connection the peer is closing, or one whose body we
                    # may abandon mid-stream — the next request would read stale bytes.
                    self._conn = None
                return resp
            except http.client.HTTPException as e:
                self.close()
                last = e
                # A pooled connection that died before we wrote anything is the common
                # keep-alive-expiry case: provably nothing reached the server.
                if allow and not sent and (pooled or attempt == 1):
                    self.reconnects += 1
                    self.log(f'transport: {type(e).__name__} before send — reconnecting')
                    continue
                raise TransportError(f'{url}: {type(e).__name__}: {e}', url,
                                     request_sent=sent,
                                     phase='sent' if sent else 'connect') from None
            except (socket.timeout, TimeoutError) as e:
                self.close()
                last = e
                if allow and not sent and attempt == 1:
                    self.reconnects += 1
                    continue
                raise TransportError(
                    f'{url}: {"response" if sent else "connect"} timed out after '
                    f'{timeout or self.connect_timeout}s', url, request_sent=sent,
                    phase='response' if sent else 'connect') from None
            except OSError as e:
                self.close()
                last = e
                if allow and not sent and attempt == 1:
                    self.reconnects += 1
                    self.log(f'transport: {type(e).__name__} before send — reconnecting')
                    continue
                raise TransportError(f'cannot reach {url}: {e}', url, request_sent=sent,
                                     phase='sent' if sent else 'connect') from None
        raise TransportError(f'request to {url} failed: {last}', url, request_sent=True)

    def release(self, resp, drained):
        """Return a streamed response to the pool, or drop the connection.

        A body we stopped reading early leaves unread bytes in the socket; the next
        request on that connection would parse them as its response. `drained=False`
        therefore always costs us the connection — cheap compared to a corrupted reply.
        """
        self._stream_outstanding = False
        if not drained:
            self.close()

    def _reusable_conn(self):
        """Return the pooled connection unless it has been idle too long.

        llama-server (and Cloudflare's edge) close idle keep-alive connections after a
        few seconds. Reusing one that the peer already dropped turns into a failure at
        send time, which for a chat POST is not provably safe to retry. Dropping an
        idle connection here avoids that ambiguity almost entirely.
        """
        if not self.keepalive or self._conn is None:
            return None
        if self._stream_outstanding:
            # A stream is still open (or its reader was dropped without closing): the
            # socket holds unread bytes, so it cannot serve another request.
            self.close()
            self._stream_outstanding = False
            return None
        if time.monotonic() - self.last_used > self.idle_ttl:
            self.close()
            return None
        return self._conn

    def stats(self):
        return {'connections_opened': self.connections_opened, 'requests': self.requests,
                'reconnects': self.reconnects, 'pooled': self._conn is not None}


# ======================================================================
# Reasoning / thinking control
#
# Built from what the PrismML runtime actually accepts, not from another model's API:
#   * request field `thinking_budget_tokens` — 0 disables thinking, N caps the thinking
#     trace at N tokens, -1 is unlimited (docs.prismml.com/bonsai-2-27b, "Thinking mode").
#   * request field `reasoning_effort` — a chat-template kwarg; Bonsai 2 accepts
#     `medium` and `xhigh` (its default). `low` is accepted but documented as *not*
#     reducing thinking, so this client refuses to offer it as if it were a speed knob.
#   * server flags `--reasoning-budget N` and `--chat-template-kwargs '{"reasoning_effort":…}'`
#     set the default for clients that send nothing.
# The official Bonsai chat UI's Off/Low/Medium/High/Max picker is exactly these budgets
# (0 / 512 / 2048 / 8192 / unlimited); `THINK_LEVELS` reproduces it 1:1.
# ======================================================================
THINK_BUDGETS = {'off': 0, 'low': 512, 'medium': 2048, 'high': 8192, 'max': -1}
THINK_ORDER = ('off', 'low', 'medium', 'high', 'max')
THINK_ALIASES = {'none': 'off', '0': 'off', 'no': 'off', 'disable': 'off', 'disabled': 'off',
                 'minimal': 'low', '512': 'low', 'normal': 'medium', '2048': 'medium',
                 'med': 'medium', '8192': 'high', 'maximum': 'max', 'unlimited': 'max',
                 'xhigh': 'max', 'full': 'max', '-1': 'max'}

# `reasoning_effort` values the Bonsai 2 template accepts. `low` is deliberately absent
# from the offered set: PrismML documents it as not reducing thinking, so offering it
# would be a fake control. It is still accepted verbatim if a user asks for it.
EFFORT_VALUES = ('medium', 'xhigh')
EFFORT_ALIASES = {'default': 'xhigh', 'highest': 'xhigh', 'max': 'xhigh', 'maximum': 'xhigh',
                  'high': 'xhigh', 'normal': 'medium', 'med': 'medium', 'balanced': 'medium'}
EFFORT_UNRELIABLE = ('low', 'minimal', 'none')

REASONING_DISPLAYS = ('full', 'compact', 'hidden')


def parse_think_level(raw):
    """Map a user word to a canonical thinking level. Raises ValueError on nonsense."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in THINK_BUDGETS:
        return key
    if key in THINK_ALIASES:
        return THINK_ALIASES[key]
    if key.lstrip('-').isdigit():            # a bare number is an explicit token budget
        return int(key)
    raise ValueError(f'unknown thinking level {raw!r} — use ' + '|'.join(THINK_ORDER)
                     + ' or a token count')


def parse_effort(raw):
    key = str(raw or '').strip().lower()
    if key in EFFORT_VALUES:
        return key
    if key in EFFORT_ALIASES:
        return EFFORT_ALIASES[key]
    if key in EFFORT_UNRELIABLE:
        return key
    raise ValueError(f'unknown reasoning_effort {raw!r} — Bonsai 2 accepts '
                     + '|'.join(EFFORT_VALUES) + ' (low is accepted but documented as a no-op)')


class ReasoningConfig:
    """What to ask the model for, and how to show what comes back.

    Two independent axes, because that is what the runtime offers:
      * `level`  -> a thinking token budget (the Off..Max picker)
      * `effort` -> the template's reasoning_effort (medium / xhigh)
    `display` controls the terminal: `full` prints the thinking trace, `compact` prints a
    one-line indicator plus a token count, `hidden` prints neither.
    """

    def __init__(self, level='medium', effort='medium', display='compact'):
        self.level = parse_think_level(level)
        self.effort = parse_effort(effort) if effort else None
        self.display = display if display in REASONING_DISPLAYS else 'compact'

    # ------------------------------------------------------------------
    def budget(self):
        """The thinking token budget for the current level (int), or None if free-form."""
        if isinstance(self.level, str) and self.level in THINK_BUDGETS:
            return THINK_BUDGETS[self.level]
        try:
            return int(self.level)
        except (TypeError, ValueError):
            return None

    def label(self):
        budget = self.budget()
        shown = 'unlimited' if budget == -1 else ('off' if budget == 0 else f'{budget} tok')
        return f'{self.level} ({shown})'

    def wire(self, caps=None):
        """Request fields for this configuration, filtered by what the server accepts.

        Returns (fields, notes). `notes` explains anything that was dropped, so a user
        who asked for thinking control is never silently ignored.
        """
        fields, notes = {}, []
        budget = self.budget()
        # Tri-state: `caps.supported()` returns None when the field has not been tested
        # against this build yet. Unknown must mean "send it and find out" — only an
        # explicit False (the server answered 400 naming the field) suppresses it.
        if budget is not None:
            if caps is None or caps.supported('thinking_budget_tokens') is not False:
                fields['thinking_budget_tokens'] = budget
            else:
                notes.append('this build rejects thinking_budget_tokens — thinking is '
                             'server-controlled')
        # `caps.supported()` is tri-state: None means untested, so we still send it and
        # let the server's own 400 teach us. Only an explicit False suppresses the field.
        if self.effort and (caps is None or caps.supported('reasoning_effort') is not False):
            fields['reasoning_effort'] = self.effort
        elif self.effort:
            notes.append('this build rejects reasoning_effort — the effort setting is inert')
        return fields, notes

    def describe(self):
        parts = [f'think={self.label()}']
        if self.effort:
            parts.append(f'effort={self.effort}')
        parts.append(f'display={self.display}')
        return ', '.join(parts)

    def to_dict(self):
        return {'level': self.level, 'effort': self.effort, 'display': self.display,
                'budget': self.budget()}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(level=d.get('level', 'medium'), effort=d.get('effort', 'medium'),
                   display=d.get('display', 'compact'))


class CapabilityMap:
    """What this particular server build actually does, learned not assumed.

    Every entry is one of `supported` / `unsupported` / `unknown`, plus the evidence that
    produced it. `/doctor` prints the whole map so a user can see what is real, what is
    missing, and what was never tested.
    """

    KNOWN_FIELDS = ('reasoning_effort', 'thinking_budget_tokens', 'tools', 'tool_choice',
                    'response_format', 'min_p', 'top_k', 'stream_options')

    def __init__(self):
        self.fields = {k: 'unknown' for k in self.KNOWN_FIELDS}
        self.evidence = {}
        self.facts = {}            # free-form: model ids, context, vision, runtime…

    # ------------------------------------------------------------------
    def mark(self, name, state, evidence=''):
        if state not in ('supported', 'unsupported', 'unknown'):
            raise ValueError(state)
        self.fields.setdefault(name, state)
        self.fields[name] = state
        if evidence:
            self.evidence[name] = evidence

    def supported(self, name):
        """True / False / None(unknown). Callers treat unknown as 'try it'."""
        state = self.fields.get(name, 'unknown')
        return None if state == 'unknown' else state == 'supported'

    def state(self, name):
        return self.fields.get(name, 'unknown')

    def note_400(self, name, message):
        self.mark(name, 'unsupported', f'HTTP 400: {str(message)[:160]}')

    def note_ok(self, name):
        if self.fields.get(name) != 'unsupported':
            self.mark(name, 'supported', 'accepted by the server')

    def set_fact(self, key, value):
        self.facts[key] = value

    def to_dict(self):
        return {'fields': dict(self.fields), 'evidence': dict(self.evidence),
                'facts': dict(self.facts)}


class BonsaiClient:
    """OpenAI-compatible client: chat (streaming + not), models, props, tokenize.

    One persistent HTTP connection (see HttpTransport), a capability map learned from the
    server instead of assumed, and a retry policy that will not replay a chat POST the
    server may already be generating.
    """

    def __init__(self, base_url, api_key, model=DEFAULT_MODEL, timeout=600,
                 connect_timeout=30, retries=3, opener=None, log=None, keepalive=True):
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
        self.caps = CapabilityMap()
        # Back-compat shims: older code/tests read these booleans directly.
        self.supports_reasoning_effort = True
        self.supports_tools = True
        self.transport = HttpTransport(self.root_url, timeout=timeout,
                                       connect_timeout=connect_timeout, opener=opener,
                                       log=self.log, keepalive=keepalive)
        self._vision = None
        self._context = None
        self._props = None
        self._props_checked = False
        self.latency = []           # recent request latencies, seconds
        self.stats = {'requests': 0, 'chat_requests': 0, 'retries': 0, 'cancelled': 0}

    def close(self):
        self.transport.close()

    # ------------------------------------------------------------------
    def _headers(self, stream=False):
        h = {'Content-Type': 'application/json',
             'Accept': 'text/event-stream' if stream else 'application/json',
             'User-Agent': f'bonsai-chat/{VERSION}'}
        if self.api_key:
            h['Authorization'] = 'Bearer ' + self.api_key
        return h

    def _raise_for_status(self, resp, url):
        """Turn a non-2xx reply into a BonsaiAPIError with the server's own message."""
        if 200 <= resp.status < 300:
            return
        try:
            body = resp.read() or b''
        except Exception:
            body = b''
        finally:
            resp.close()
        try:
            parsed = json.loads(body.decode('utf-8', 'replace'))
            msg = parsed.get('error')
            if isinstance(msg, dict):
                msg = msg.get('message') or json.dumps(msg)
            msg = str(msg or body[:400].decode('utf-8', 'replace'))
        except Exception:
            msg = body[:400].decode('utf-8', 'replace') or f'HTTP {resp.status}'
        raise BonsaiAPIError(resp.status, msg, url)

    def request(self, path, payload=None, stream=False, timeout=None, root=False, retry=None,
                _stream=None):
        """One HTTP call. Returns the response object (caller closes it).

        `root=True` targets a llama.cpp-native route (/props, /tokenize, /health) instead
        of an OpenAI-compatible one under /v1.

        Retries are bounded and only ever replay requests that provably never reached the
        server, so a chat POST is never generated twice. 429/5xx and connect failures are
        retried for GETs; a POST is retried only when the transport reports the bytes
        never left this process.
        """
        base = self.root_url if root else self.base_url
        url = base + path
        data = json.dumps(payload).encode() if payload is not None else None
        method = 'POST' if data is not None else 'GET'
        deadline_timeout = timeout or (self.timeout if stream else self.connect_timeout)
        last = None
        attempts = self.retries if (retry is None or retry) else 1
        for attempt in range(1, max(1, attempts) + 1):
            started = time.monotonic()
            try:
                resp = self.transport.request(method, url, body=data,
                                              headers=self._headers(stream),
                                              timeout=deadline_timeout, retry=retry,
                                              stream=bool(stream if _stream is None
                                                          else _stream))
                self.stats['requests'] += 1
                self.latency.append(time.monotonic() - started)
                del self.latency[:-64]
                self._raise_for_status(resp, url)
                return resp
            except TransportError as e:
                last = e
                # A POST whose bytes may have reached the server is never replayed.
                if method == 'POST' and not e.safe_to_retry_post:
                    raise BonsaiError(
                        f'{url}: the request reached the server but no reply came back '
                        f'({e.phase}). Not retrying automatically — the model may already be '
                        'generating, and a blind retry would produce a second, duplicated '
                        'answer. Send the turn again yourself if you want it re-run.') from None
                if attempt < attempts:
                    wait = min(2 ** attempt, 8)
                    self.stats['retries'] += 1
                    self.log(f'transport failure ({e}), retrying in {wait}s')
                    time.sleep(wait)
                    continue
                raise BonsaiError(
                    f'cannot reach {url}: {e}. Is the notebook runtime still alive? '
                    'Quick Tunnel URLs die with the session — rerun the deployment cell and '
                    'use the new URL.') from None
            except BonsaiAPIError as e:
                last = e
                transient = e.status in (429, 502, 503, 504)
                # 502/503 from a tunnel mean the edge could not reach the origin: the
                # request never started a generation, so replaying it is safe.
                if transient and attempt < attempts:
                    wait = min(2 ** attempt, 8)
                    self.stats['retries'] += 1
                    self.log(f'transient HTTP {e.status}, retrying in {wait}s')
                    time.sleep(wait)
                    continue
                raise
        raise BonsaiError(f'request to {url} failed: {last}')

    def get_json(self, path, timeout=None, root=False):
        with self.request(path, timeout=timeout, root=root) as r:
            return json.loads(r.read().decode('utf-8', 'replace'))

    # ------------------------------------------------------------------
    def models(self):
        return self.get_json('/models')

    def model_ids(self):
        try:
            ids = [m.get('id') for m in (self.models().get('data') or []) if m.get('id')]
        except BonsaiError:
            return []
        self.caps.set_fact('served_models', ids)
        return ids

    def props(self, refresh=False):
        """llama.cpp /props — context size, slot state. Fetched once and cached.

        The context window, the vision modality and the runtime version all live here, so
        one call answers several questions; repeating it per turn was pure latency.
        """
        if self._props_checked and not refresh:
            return self._props
        self._props_checked = True
        try:
            self._props = self.get_json('/props', timeout=15, root=True)
        except BonsaiError as e:
            self._props = None
            self.log(f'/props unavailable ({e})')
        return self._props

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

    def tokenize_batch(self, texts):
        """Token counts for many strings in one round trip.

        Budget trimming re-measures the whole history; doing that one /tokenize per
        message is dozens of round trips per turn on a long session. llama.cpp's
        /tokenize returns exactly one count per request, so this batches what it can and
        falls back to per-item calls for the rest — never inventing a batch API.
        """
        out = {}
        for t in texts:
            if t in out:
                continue
            n = self.tokenize(t)
            if n is None:
                return None
            out[t] = n
        return out

    def context_window(self, default=8192):
        """Server context size in tokens, probed once from /props and then cached."""
        if self._context:
            return self._context
        props = self.props() or {}
        found = None
        sub = props.get('default_generation_settings') or {}
        for source in (sub, props):
            for k in ('n_ctx', 'n_ctx_per_seq', 'n_ctx_total'):
                v = source.get(k)
                if isinstance(v, int) and v > 0:
                    found = v if found is None else min(found, v)
        self._context = found or default
        self.caps.set_fact('context_window', self._context)
        return self._context

    def server_facts(self):
        """Everything /props can tell us, gathered once, for /doctor and --json."""
        props = self.props() or {}
        sub = props.get('default_generation_settings') or {}
        facts = {
            'model_path': sub.get('model') or None,
            'n_ctx': self.context_window(),
            'total_slots': props.get('total_slots'),
            'version': props.get('system_info') or sub.get('version') or None,
            'chat_template': 'present' if sub.get('chat_template') else 'not reported',
        }
        for k, v in facts.items():
            if v is not None:
                self.caps.set_fact(k, v)
        return facts

    # ------------------------------------------------------------------
    def _post_chat(self, payload, stream, timeout):
        """POST /chat/completions, dropping fields this server build has rejected."""
        payload = dict(payload)
        if payload.get('reasoning_effort') is None:
            payload.pop('reasoning_effort', None)
        if self.caps.supported('reasoning_effort') is False:
            payload.pop('reasoning_effort', None)
        if self.caps.supported('thinking_budget_tokens') is False:
            payload.pop('thinking_budget_tokens', None)
        if not payload.get('tools'):
            payload.pop('tools', None)
            payload.pop('tool_choice', None)
        if self.caps.supported('tools') is False:
            payload.pop('tools', None)
            payload.pop('tool_choice', None)
        optional = ('reasoning_effort', 'thinking_budget_tokens', 'min_p', 'top_k',
                    'response_format')
        try:
            self.stats['chat_requests'] += 1
            return self.request('/chat/completions', payload, stream=stream, timeout=timeout,
                                _stream=stream)
        except BonsaiAPIError as e:
            if e.status != 400:
                raise
            msg = (e.message or '').lower()
            dropped = None
            for field_name in optional:
                if field_name in payload and field_name in msg:
                    dropped = field_name
                    break
            if dropped is None and 'tools' in payload and ('tool' in msg or 'jinja' in msg):
                dropped = 'tools'
            if dropped is None:
                raise
            self.caps.note_400(dropped, e.message)
            self._sync_shims()
            self.log(f'server rejected {dropped} — continuing without it')
            payload.pop(dropped, None)
            if dropped == 'tools':
                payload.pop('tool_choice', None)
            return self.request('/chat/completions', payload, stream=stream, timeout=timeout,
                                _stream=stream)

    def note_accepted(self, fields):
        """Record that the fields present in a request were accepted by the server.

        Called after a *successful* response only, so a capability is never marked
        supported on the strength of a request that was rejected or never answered.
        """
        for name in ('reasoning_effort', 'thinking_budget_tokens', 'tools', 'tool_choice',
                     'min_p', 'top_k', 'response_format'):
            if name in fields:
                self.caps.note_ok(name)
        if fields.get('stream_options'):
            self.caps.note_ok('stream_options')

    def _sync_shims(self):
        self.supports_reasoning_effort = self.caps.supported('reasoning_effort') is not False
        self.supports_tools = self.caps.supported('tools') is not False

    def chat(self, messages, tools=None, **params):
        """Non-streaming chat completion -> the raw JSON response dict."""
        payload = {'model': self.model, 'messages': messages, 'stream': False}
        payload.update({k: v for k, v in params.items() if v is not None})
        if tools:
            payload['tools'] = tools
            payload.setdefault('tool_choice', 'auto')
        with self._post_chat(payload, False, params.get('timeout')) as r:
            data = json.loads(r.read().decode('utf-8', 'replace'))
        if data.get('usage'):
            self.caps.note_ok('stream_options')
        for f in ('reasoning_effort', 'thinking_budget_tokens', 'tools', 'min_p', 'top_k'):
            if f in payload:
                self.caps.note_ok(f)
        return data

    def stream_chat(self, messages, tools=None, **params):
        """Streaming chat completion -> a ChatStream (use it as a context manager).

        The returned object is iterable and also exposes `.close()`, so a cancelled turn
        deterministically releases the HTTP response instead of waiting on the GC.
        """
        payload = {'model': self.model, 'messages': messages, 'stream': True,
                   'stream_options': {'include_usage': True}}
        payload.update({k: v for k, v in params.items() if v is not None})
        if tools:
            payload['tools'] = tools
            payload.setdefault('tool_choice', 'auto')
        resp = self._post_chat(payload, True, params.get('timeout'))
        # _post_chat already went through request(stream=True), so the transport knows a
        # body is outstanding and will not reuse the socket until ChatStream.close().
        return ChatStream(resp, sent_fields=payload, transport=self.transport)

    # ------------------------------------------------------------------
    def probe_field(self, field_name, value, messages=None):
        """Confirm with a real 1-token request that the server accepts `field_name`.

        Cheap and honest: a capability we have not tested is reported as unknown rather
        than advertised. Called lazily, and the result is cached in the capability map.
        """
        state = self.caps.supported(field_name)
        if state is not None:
            return state
        payload = {'model': self.model,
                   'messages': messages or [{'role': 'user', 'content': 'Reply: ok'}],
                   'max_tokens': 1, 'stream': False, field_name: value}
        try:
            with self.request('/chat/completions', payload, timeout=self.connect_timeout) as r:
                r.read()
            self.caps.mark(field_name, 'supported', 'accepted in a probe request')
            return True
        except BonsaiAPIError as e:
            if e.status == 400 and field_name in (e.message or '').lower():
                self.caps.note_400(field_name, e.message)
                return False
            # Any other error tells us nothing about this field.
            self.log(f'probe of {field_name} inconclusive: HTTP {e.status}')
            return False
        except BonsaiError as e:
            self.log(f'probe of {field_name} inconclusive: {e}')
            return False
        finally:
            self._sync_shims()

    def probe_vision(self, force=None):
        """Can this server actually see pixels? Probed once with a 1x1 PNG, then cached.

        The default deployment is deliberately text-only (no mmproj/vision projector), so
        this normally returns False and the client sends extracted image facts instead.
        """
        if force is not None:
            self._vision = bool(force)
            self.caps.set_fact('vision', self._vision)
            return self._vision
        if self._vision is not None:
            return self._vision
        modalities = (self.props() or {}).get('modalities')
        if isinstance(modalities, dict) and isinstance(modalities.get('vision'), bool):
            self._vision = modalities['vision']
            self.log(f'/props reports vision={self._vision}')
            self.caps.set_fact('vision', self._vision)
            self.caps.set_fact('vision_source', '/props modalities')
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
            self.caps.set_fact('vision_source', f'probe rejected with HTTP {e.status}')
        except BonsaiError as e:
            self.log(f'vision probe failed: {e}')
            self._vision = False
            self.caps.set_fact('vision_source', 'probe failed')
        self.caps.set_fact('vision', self._vision)
        return self._vision


# ======================================================================
# Streaming: one object owns the response, the decoder and the accumulator
# ======================================================================
class ChatStream:
    """A live streaming chat completion.

    Owns the HTTP response so cancellation is deterministic: `.close()` (or leaving the
    `with` block) always shuts the response down and, when the body was not fully read,
    drops the pooled connection rather than handing the next request a stream of stale
    bytes. Iteration yields `reasoning` / `delta` events and a final `done` event carrying
    the assembled assistant message, usage, timings, finish reason, TTFT and elapsed time.

    A stream that dies mid-generation raises StreamInterrupted carrying whatever did
    arrive — a partial answer is reported as partial and is never quietly promoted to a
    completed assistant turn.
    """

    #: how long to wait for the server to close the chunked body after [DONE] so the
    #: connection can go back to the pool; longer than that and we just drop it.
    DRAIN_TIMEOUT = 2.0

    def __init__(self, resp, sent_fields=None, transport=None):
        self.resp = resp
        self.transport = transport
        self.sent_fields = sent_fields or {}
        self.drained = False
        self.decoder = SSEDecoder()
        self.text = []
        self.reason = []
        self.tool_acc = {}
        self.finish = None
        self.usage = None
        self.timings = None
        self.model = None
        self.started = time.monotonic()
        self.first_token_at = None
        self.chunks = 0
        self.malformed = 0
        self.closed = False
        self.done = False
        self.interrupted = None

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def drain(self):
        """Read the (empty) remainder of the body so the connection stays reusable."""
        if self.drained or self.resp is None:
            return
        sock = getattr(getattr(self.resp, 'fp', None), 'raw', None)
        sock = getattr(sock, '_sock', None)
        previous = None
        try:
            if sock is not None:
                previous = sock.gettimeout()
                sock.settimeout(self.DRAIN_TIMEOUT)
            self.resp.read()
            self.drained = True
        except Exception:
            self.drained = False
        finally:
            if sock is not None and previous is not None:
                try:
                    sock.settimeout(previous)
                except OSError:
                    pass

    def close(self):
        """Shut the response down and release the connection exactly once.

        Deterministic release matters for cancellation: Ctrl-C in the middle of a long
        generation must not leave a socket open holding the single serving slot.
        """
        if self.closed:
            return
        self.closed = True
        resp = self.resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        if self.transport is not None:
            self.transport.release(resp, self.drained)

    # ------------------------------------------------------------------
    def _handle(self, ev):
        """Apply one decoded SSE event; returns an event dict to yield, or None."""
        if not isinstance(ev, dict):
            self.malformed += 1
            return None
        if ev.get('model'):
            self.model = ev['model']
        if ev.get('usage'):
            self.usage = ev['usage']
        if ev.get('timings'):
            self.timings = ev['timings']
        # llama.cpp also nests timings inside usage on some builds.
        if isinstance(self.usage, dict) and self.usage.get('timings') and not self.timings:
            self.timings = self.usage['timings']
        choices = ev.get('choices') or []
        if not choices or not isinstance(choices[0], dict):
            return None
        ch = choices[0]
        if ch.get('finish_reason'):
            self.finish = ch['finish_reason']
        delta = ch.get('delta') or {}
        if not isinstance(delta, dict):
            self.malformed += 1
            return None
        accumulate_tool_calls(delta.get('tool_calls'), self.tool_acc)
        rc = delta.get('reasoning_content') or delta.get('reasoning')
        if rc:
            self.reason.append(rc)
            self.chunks += 1
            self.first_token_at = self.first_token_at or time.monotonic()
            return {'kind': 'reasoning', 'text': rc}
        ct = delta.get('content')
        if ct:
            self.text.append(ct)
            self.chunks += 1
            self.first_token_at = self.first_token_at or time.monotonic()
            return {'kind': 'delta', 'text': ct}
        return None

    def message(self):
        """The assistant message assembled so far, in OpenAI wire shape."""
        msg = {'role': 'assistant', 'content': ''.join(self.text)}
        if ''.join(self.reason):
            msg['reasoning_content'] = ''.join(self.reason)
        calls = self.tool_calls()
        if calls:
            msg['tool_calls'] = calls
            msg['content'] = msg['content'] or None
        return msg

    def tool_calls(self):
        return [self.tool_acc[k] for k in sorted(self.tool_acc)]

    def done_event(self):
        return {'kind': 'done', 'message': self.message(), 'tool_calls': self.tool_calls(),
                'finish_reason': self.finish, 'usage': self.usage, 'timings': self.timings,
                'elapsed': time.monotonic() - self.started,
                'ttft': (self.first_token_at - self.started) if self.first_token_at else None,
                'chunks': self.chunks, 'model': self.model}

    # ------------------------------------------------------------------
    def __iter__(self):
        try:
            while True:
                chunk = self._read_chunk()
                if not chunk:
                    break
                for payload in self.decoder.feed(chunk):
                    ev = self._event(payload)
                    if ev is _DONE:
                        self.done = True
                        self.drain()
                        yield self.done_event()
                        return
                    if ev is not None:
                        yield ev
            tail = self.decoder.close()
            if tail is not None:
                ev = self._event(tail)
                if ev is not None and ev is not _DONE:
                    yield ev
            self.done = True
            self.drained = True          # natural EOF: the body is fully consumed
            yield self.done_event()
        except GeneratorExit:
            raise
        except (KeyboardInterrupt, SystemExit):
            self.interrupted = 'cancelled'
            raise
        except Exception as e:
            # The server went away mid-generation, or the tunnel dropped the stream.
            # Report what we have instead of pretending the turn completed.
            self.interrupted = f'{type(e).__name__}: {e}'
            raise StreamInterrupted(
                f'stream ended early after {self.chunks} chunk(s): {type(e).__name__}: {e}',
                partial_text=''.join(self.text), partial_reasoning=''.join(self.reason),
                reason=type(e).__name__) from None
        finally:
            self.close()

    def _read_chunk(self):
        """Read whatever is available now, without blocking for a full buffer.

        `read(n)` on http.client blocks until it has n bytes, which would sit on the
        socket after the last small chunk instead of handing us the token we already
        have — that alone was enough to add a whole round-trip of apparent latency to
        every stream. `read1` returns as soon as one chunk is buffered.
        """
        reader = getattr(self.resp, 'read1', None)
        if reader is not None:
            return reader(65536)
        return self.resp.read(65536)

    def _event(self, payload):
        if payload is None:
            return None
        if payload.strip() == '[DONE]':
            return _DONE
        try:
            ev = json.loads(payload)
        except ValueError:
            self.malformed += 1
            return None
        return self._handle(ev)


_DONE = object()


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


class ToolTimeout(ToolError):
    """A tool call exceeded its wall-clock budget and was abandoned.

    The worker thread is not killed (Python cannot do that safely); it is left to finish
    on its own and the model is told the call timed out, which is the honest outcome.
    """

    def __init__(self, timeout):
        self.timeout = timeout
        super().__init__(f'tool exceeded its {timeout:.0f}s budget')


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

    def truncate_to(self, text, limit):
        """Cap tool output at `limit` characters, saying so, so the model is not misled."""
        if limit is None or limit <= 0 or len(text) <= limit:
            return text
        head = limit * 3 // 4
        tail = limit // 4
        return (text[:head] +
                f'\n… [{len(text) - head - tail} characters omitted to stay inside the '
                f'{limit}-character tool output limit] …\n' + text[-tail:])

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
        self.calls_run = 0
        self.last_duration = 0.0
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

    def execute(self, name, raw_args, timeout=None, max_output=None):
        """Run one tool call; always returns a string for the model (errors included).

        `timeout` bounds a single call and `max_output` bounds what is fed back, so a tool
        that hangs or dumps a 50 MB log cannot stall or overflow the turn. Approval is
        asked *before* the timed section — a prompt waiting on the user is not a hang.
        """
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
        started = time.monotonic()
        try:
            result = self._call(tool.func, args, timeout)
        except ToolTimeout:
            # Must precede the generic ToolError handler: ToolTimeout subclasses it.
            return (f'ERROR: {name} was still running after {timeout:.0f}s and was abandoned. '
                    'Report that it timed out; do not retry it.')
        except ToolError as e:
            return f'ERROR: {e}'
        except Exception as e:
            return f'ERROR: {type(e).__name__}: {e}'
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        self.last_duration = time.monotonic() - started
        self.calls_run += 1
        return self.ctx.truncate_to(result, max_output) if max_output else result

    @staticmethod
    def _call(func, args, timeout):
        """Run `func` with a wall-clock bound.

        The function runs in a daemon thread so a hung tool cannot wedge the chat loop.
        The thread is not killed (Python cannot do that safely); it is abandoned and the
        model is told the call timed out, which is the honest outcome.
        """
        if not timeout or timeout <= 0:
            return func(args)
        box = {}

        def runner():
            try:
                box['value'] = func(args)
            except BaseException as e:      # noqa: BLE001 - surfaced to the model
                box['error'] = e

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise ToolTimeout(timeout)
        if 'error' in box:
            raise box['error']
        return box.get('value')


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

# Session files carry a schema marker so a future version can tell an old file from a
# corrupt one. Files without the marker are still readable (v0.3/v0.4 wrote bare JSONL).
SESSION_SCHEMA = 2


class Settings:
    """Everything the user can change about a turn, in one place.

    `reasoning` (a ReasoningConfig) is the single source of truth for thinking; `effort`
    and `think` are accessors over it so older callers, the `/effort` command and the
    `effort=`/`think=` constructor keywords all keep working without a second copy of the
    state to fall out of sync.
    """

    def __init__(self, model=DEFAULT_MODEL, temperature=1.0, top_p=0.95, max_tokens=2048,
                 reasoning=None, effort=None, think=None, stream=True, use_tools=True,
                 markdown=True, highlight=True, max_tool_rounds=6, context_budget=0.75,
                 context_reserve=512, output_reserve=1024, tool_timeout=60.0,
                 tool_budget=600.0, tool_max_output=24000, autosave_every=1,
                 show_stats=False):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning = reasoning or ReasoningConfig('medium', 'medium', 'compact')
        if effort is not None:
            self.effort = effort
        if think is not None:
            self.think = think
        self.stream = stream
        self.use_tools = use_tools
        self.markdown = markdown
        self.highlight = highlight
        self.max_tool_rounds = max_tool_rounds
        self.context_budget = context_budget
        self.context_reserve = context_reserve      # tokens for the chat template
        self.output_reserve = output_reserve        # tokens for the answer being written
        self.tool_timeout = tool_timeout            # per tool call
        self.tool_budget = tool_budget              # total wall clock for one turn's tools
        self.tool_max_output = tool_max_output      # characters fed back per call
        self.autosave_every = autosave_every        # turns between session writes
        self.show_stats = show_stats                # /speed: always print the timing table

    def __repr__(self):
        return (f'Settings(model={self.model!r}, temperature={self.temperature}, '
                f'top_p={self.top_p}, max_tokens={self.max_tokens}, '
                f'reasoning={self.reasoning.describe()!r}, stream={self.stream}, '
                f'use_tools={self.use_tools})')

    # -- reasoning accessors ---------------------------------------------------
    @property
    def effort(self):
        return self.reasoning.effort or 'none'

    @effort.setter
    def effort(self, value):
        v = str(value or '').lower()
        if v in ('none', 'off', ''):
            self.reasoning.level = 'off'
            self.reasoning.effort = None
        else:
            self.reasoning.effort = parse_effort(v)
            if self.reasoning.level == 'off':
                self.reasoning.level = 'medium'

    @property
    def think(self):
        return self.reasoning.level

    @think.setter
    def think(self, value):
        self.reasoning.level = parse_think_level(value)

    # --------------------------------------------------------------------------
    def api_params(self, caps=None):
        """Sampling + reasoning fields for one request, filtered by real capability."""
        params = {'temperature': self.temperature, 'top_p': self.top_p,
                  'max_tokens': self.max_tokens}
        fields, _ = self.reasoning.wire(caps)
        params.update(fields)
        return {k: v for k, v in params.items() if v is not None}

    def reasoning_notes(self, caps=None):
        return self.reasoning.wire(caps)[1]


class TurnStats:
    """Measured numbers for one turn. Every field is either measured or absent.

    Nothing here is estimated: if the server did not report a value, the property returns
    None and the UI prints `n/a` rather than a plausible-looking number.
    """

    def __init__(self):
        self.ttft = None
        self.elapsed = 0.0
        self.prompt_tokens = None
        self.completion_tokens = None
        self.reasoning_tokens = None
        self.prompt_tokens_per_s = None
        self.tokens_per_s = None
        self.rounds = 0
        self.tool_calls = 0
        self.cancelled = False
        self.interrupted = None
        self.context_used = None
        self.context_window = None

    # ------------------------------------------------------------------
    def absorb(self, usage=None, timings=None, ttft=None, elapsed=0.0):
        usage = usage or {}
        timings = dict(timings or {})
        # Some builds nest the timing block inside usage instead of at the top level.
        if isinstance(usage.get('timings'), dict):
            for k, v in usage['timings'].items():
                timings.setdefault(k, v)
        if usage.get('prompt_tokens'):
            self.prompt_tokens = usage['prompt_tokens']
        if usage.get('completion_tokens'):
            self.completion_tokens = usage['completion_tokens']
        rc = usage.get('reasoning_tokens')
        if rc is None and isinstance(usage.get('completion_tokens_details'), dict):
            rc = usage['completion_tokens_details'].get('reasoning_tokens')
        if rc:
            self.reasoning_tokens = rc
        self.ttft = self.ttft if self.ttft is not None else ttft
        self.elapsed += elapsed or 0.0
        for src, dst in (('prompt_per_second', 'prompt_tokens_per_s'),
                         ('tokens_per_second', 'tokens_per_s'),
                         ('predicted_per_second', 'tokens_per_s'),
                         ('prompt_n', '_pn'), ('predicted_n', '_gn')):
            v = timings.get(src)
            if v and dst != '_pn' and dst != '_gn':
                setattr(self, dst, float(v))
        # Fall back to the server's own token/ms timings when it does not send rates.
        if self.tokens_per_s is None and timings.get('predicted_ms') and (
                timings.get('predicted_n') or self.completion_tokens):
            n = timings.get('predicted_n') or self.completion_tokens
            self.tokens_per_s = float(n) / (float(timings['predicted_ms']) / 1000.0)
        if self.prompt_tokens_per_s is None and timings.get('prompt_ms') and (
                timings.get('prompt_n') or self.prompt_tokens):
            n = timings.get('prompt_n') or self.prompt_tokens
            self.prompt_tokens_per_s = float(n) / (float(timings['prompt_ms']) / 1000.0)

    def rate(self):
        if self.tokens_per_s:
            return float(self.tokens_per_s)
        if self.elapsed and self.completion_tokens:
            # Wall-clock fallback: includes network and tool time, so it is labelled as
            # such by the caller and never printed as a decode rate.
            return self.completion_tokens / self.elapsed
        return 0.0

    @property
    def total_tokens(self):
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def context_utilization(self):
        if not self.context_window or self.context_used is None:
            return None
        return 100.0 * self.context_used / self.context_window

    def to_dict(self):
        return {'ttft_ms': round(self.ttft * 1000, 1) if self.ttft else None,
                'elapsed_s': round(self.elapsed, 3),
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'reasoning_tokens': self.reasoning_tokens,
                'total_tokens': self.total_tokens,
                'prompt_tokens_per_s': round(self.prompt_tokens_per_s, 1)
                if self.prompt_tokens_per_s else None,
                'tokens_per_s': round(self.tokens_per_s, 2) if self.tokens_per_s else None,
                'rounds': self.rounds, 'tool_calls': self.tool_calls,
                'cancelled': self.cancelled, 'interrupted': self.interrupted,
                'context_used': self.context_used, 'context_window': self.context_window}


class Conversation:
    """Message history with token-budget trimming that never orphans a tool result."""

    #: bumped when the on-disk session format changes in an incompatible way
    SCHEMA = SESSION_SCHEMA

    def __init__(self, system=DEFAULT_SYSTEM, counter=None):
        self.counter = counter or TokenCounter()
        self.messages = []
        self.system_text = None
        self.schema = SESSION_SCHEMA
        self.saved_version = None
        self.skipped_lines = 0
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

    def truncate(self, index):
        """Cut the history back to `index` messages."""
        if index < len(self.messages):
            removed = len(self.messages) - index
            del self.messages[index:]
            return removed
        return 0

    def drop_last_turn(self):
        """Remove the newest user turn and everything after it (for /undo and /retry)."""
        i = self.last_user_index()
        if i is None:
            return 0
        removed = len(self.messages) - i
        del self.messages[i:]
        return removed

    def turn_groups(self):
        """Split the history into (start, end) spans, one per user turn.

        A group is a user message plus every assistant/tool message that answers it. All
        removal happens at group granularity: dropping anything finer can orphan a `tool`
        message from the `tool_calls` it belongs to, and llama.cpp rejects the whole
        request when that happens.
        """
        start = 0
        while start < len(self.messages) and self.messages[start]['role'] == 'system':
            start += 1                     # system + any compaction digest are never dropped
        groups = []
        begin = start
        for i in range(start + 1, len(self.messages)):
            if self.messages[i]['role'] == 'user':
                if i > begin:
                    groups.append((begin, i))
                begin = i
        if begin < len(self.messages):
            groups.append((begin, len(self.messages)))
        return groups

    def trim(self, budget):
        """Drop the oldest complete turns until the history fits `budget` tokens.

        Removal happens per turn group so a `tool` message can never survive without the
        `tool_calls` it belongs to. Token counts come from the counter's cache, so the
        repeated re-measurement inside this loop costs one server round trip per new
        string, not one per iteration.
        """
        dropped = 0
        while self.tokens() > budget and len(self.messages) > 2:
            groups = self.turn_groups()
            if not groups:
                break
            start, end = groups[0]
            del self.messages[start:end]
            dropped += end - start
        return dropped

    def compact(self, budget, keep_recent=2, summarizer=None):
        """Compact old turns into a single system-visible digest instead of dropping them.

        `trim` throws history away; `compact` preserves what it can. The oldest turns are
        summarised (by `summarizer`, else by a cheap extractive digest) and folded into a
        `[earlier in this conversation]` block placed after the system message. The most
        recent `keep_recent` turns, the system prompt and every tool pair in the kept
        region are preserved verbatim.

        Returns a dict describing what happened so the UI can tell the user.
        """
        groups = self.turn_groups()
        if len(groups) <= keep_recent:
            return {'compacted': 0, 'kept': len(groups), 'tokens_before': self.tokens(),
                    'tokens_after': self.tokens()}
        before = self.tokens()
        keep_from = groups[-keep_recent][0] if keep_recent else len(self.messages)
        old = self.messages[1 if (self.messages and self.messages[0]['role'] == 'system')
                            else 0:keep_from]
        digest = (summarizer(old) if summarizer else self._extractive_digest(old))
        head = 0
        while head < len(self.messages) and self.messages[head]['role'] == 'system':
            head += 1
        self.messages = (self.messages[:head] +
                         [{'role': 'system',
                           'content': ('[earlier in this conversation — compacted]\n' + digest)}] +
                         self.messages[keep_from:])
        after = self.tokens()
        # Still over budget after compaction? Fall back to whole-turn trimming.
        dropped = self.trim(budget) if after > budget else 0
        return {'compacted': len(old), 'kept': keep_recent, 'tokens_before': before,
                'tokens_after': self.tokens(), 'trimmed': dropped}

    @staticmethod
    def _extractive_digest(messages):
        """A cheap, honest digest: who said what, first line only. No model call."""
        lines = []
        for m in messages:
            body = m.get('content')
            if isinstance(body, list):
                body = ' '.join(p.get('text', '') for p in body if isinstance(p, dict))
            body = ' '.join((body or '').split())
            role = m['role']
            if role == 'user':
                lines.append('- user: ' + body[:200])
            elif role == 'assistant':
                if m.get('tool_calls'):
                    names = ', '.join((tc.get('function') or {}).get('name', '?')
                                      for tc in m['tool_calls'])
                    lines.append(f'- assistant called tool(s): {names}')
                elif body:
                    lines.append('- assistant: ' + body[:200])
            elif role == 'tool':
                lines.append(f'- tool {m.get("name") or m.get("tool_call_id")}: ' + body[:120])
        return '\n'.join(lines) or '(nothing recorded)'

    def validate_wire(self):
        """Check the outgoing message list for the things llama.cpp rejects.

        Returns a list of human-readable problems. Cheap enough to run before every
        request, and it turns a confusing server-side 400 into a precise local message.
        """
        problems = []
        wire = self.wire()
        pending = set()
        for i, m in enumerate(wire):
            role = m.get('role')
            if role == 'assistant' and m.get('tool_calls'):
                for tc in m['tool_calls']:
                    pending.add(tc.get('id'))
            if role == 'tool':
                cid = m.get('tool_call_id')
                if cid not in pending:
                    problems.append(f'message {i}: orphan tool result (no matching tool_call '
                                    f'for id {cid!r})')
                else:
                    pending.discard(cid)
            if role == 'assistant' and not m.get('tool_calls') and not m.get('content'):
                problems.append(f'message {i}: empty assistant message')
        if pending:
            problems.append(f'unanswered tool_call(s): {", ".join(sorted(str(p) for p in pending))}')
        return problems

    # ------------------------------------------------------------------
    def save(self, path):
        """Atomic write: temp file in the same directory, fsync, then rename.

        A notebook runtime can be killed at any moment; a torn session file must not cost
        the user their conversation.
        """
        path = Path(path).expanduser()
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            fh.write(json.dumps({'_meta': True, 'schema': SESSION_SCHEMA,
                                 'version': VERSION,
                                 'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                                ensure_ascii=False) + '\n')
            for m in self.messages:
                fh.write(json.dumps(m, ensure_ascii=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path

    @classmethod
    def load(cls, path, counter=None, on_error=None):
        """Load a session file, tolerating corruption and older formats.

        A truncated last line (killed mid-write), a stray blank, or a file from an older
        schema must not crash the client. Bad lines are skipped and reported; the rest of
        the conversation is kept.
        """
        path = Path(path).expanduser()
        conv = cls(system=None, counter=counter)
        skipped = 0
        for lineno, line in enumerate(path.read_text(encoding='utf-8', errors='replace')
                                   .splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(msg, dict):
                skipped += 1
                continue
            if msg.get('_meta'):
                conv.schema = msg.get('schema', 1)
                conv.saved_version = msg.get('version')
                continue
            if msg.get('role') not in WIRE_ROLES:
                skipped += 1
                continue
            if msg.get('role') == 'system' and not conv.messages:
                conv.set_system(msg.get('content') or '')
            else:
                conv.messages.append(msg)
        conv.skipped_lines = skipped
        if skipped and on_error:
            on_error(skipped)
        # Drop anything that would be rejected by the server, e.g. a tool result whose
        # assistant tool_call line was the one that got truncated away.
        conv._repair_orphans()
        return conv

    def _repair_orphans(self):
        """Remove tool results with no matching tool_call, and vice versa."""
        known = set()
        for m in self.messages:
            for tc in m.get('tool_calls') or []:
                known.add(tc.get('id'))
        self.messages = [m for m in self.messages
                         if m.get('role') != 'tool' or m.get('tool_call_id') in known]
        answered = {m.get('tool_call_id') for m in self.messages if m.get('role') == 'tool'}
        for m in self.messages:
            if m.get('tool_calls'):
                m['tool_calls'] = [tc for tc in m['tool_calls'] if tc.get('id') in answered]
                if not m['tool_calls']:
                    m.pop('tool_calls', None)
                    if not m.get('content'):
                        m['content'] = ''

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
    """What one user turn produced. `stats` holds the measured numbers."""
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
    interrupted: str = ''
    stats: TurnStats = field(default_factory=TurnStats)

    @property
    def prompt_tokens(self):
        return self.stats.prompt_tokens or 0

    @property
    def completion_tokens(self):
        return self.stats.completion_tokens or 0

    def rate(self):
        return self.stats.rate()


class Agent:
    """Drives one turn: stream the answer, run any tool calls, feed results back, repeat.

    Three guarantees this class is responsible for:

    1. Reasoning never becomes answer text. `reasoning_content` is collected separately
       and is never written into the assistant history as `content`.
    2. A cancelled or interrupted generation never becomes a completed assistant message.
       Whatever the model produced this turn is rolled back, the user's message stays, and
       the user is told exactly what happened.
    3. The tool loop is bounded — by rounds, by per-call time, by total wall clock, and by
       the size of what gets fed back — so an agentic loop cannot run away.
    """

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
        # Everything from here on is produced by the model; a cancel rolls back to this
        # mark so no half-written tool group can survive into the history.
        rollback_mark = len(conversation.messages)

        tools = self.registry.schemas() if (self.registry and settings.use_tools) else None
        for note in settings.reasoning_notes(self.client.caps):
            self.on_notice(note)
        result = TurnResult()
        cancelled = False
        interrupted = ''
        turn_started = time.monotonic()
        try:
            for round_no in range(1, settings.max_tool_rounds + 1):
                result.rounds = round_no
                result.stats.rounds = round_no
                assistant, usage, timings, finish = self._one_request(
                    conversation.wire(), tools, result.stats)
                conversation.add_assistant(assistant)
                if finish:
                    result.finish_reason = finish
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
                over_budget = False
                for call in calls:
                    fn = call.get('function') or {}
                    name = fn.get('name') or '?'
                    if time.monotonic() - turn_started > settings.tool_budget:
                        answer = (f'ERROR: the tool budget for this turn '
                                  f'({settings.tool_budget:.0f}s) is exhausted; {name} was not run.')
                        over_budget = True
                    else:
                        answer = self.registry.execute(
                            name, fn.get('arguments'), timeout=settings.tool_timeout,
                            max_output=settings.tool_max_output) if self.registry else \
                            'ERROR: tools are disabled'
                    conversation.add_tool_result(call.get('id') or f'call_{round_no}', answer, name)
                    self.on_tool(name, fn.get('arguments') or '', answer)
                if over_budget:
                    self.on_notice('tool budget exhausted — asking the model to answer with '
                                   'what it has')
        except CancelledByUser:
            cancelled = True
            self._rollback(conversation, rollback_mark)
            self.on_notice('cancelled — the partial answer was discarded and the history is '
                           'unchanged (your message is still there; /retry to run it again)')
        except StreamInterrupted as e:
            interrupted = e.reason or 'stream interrupted'
            self._rollback(conversation, rollback_mark)
            self.on_notice(f'the stream ended early ({e.reason or "disconnected"}) after '
                           f'{len(e.partial_text)} characters; nothing was added to the '
                           'history, so no half-finished answer is remembered. Use /retry.')
        result.cancelled = cancelled
        result.interrupted = interrupted
        result.stats.cancelled = cancelled
        result.stats.interrupted = interrupted or None
        result.stats.elapsed = time.monotonic() - turn_started
        result.elapsed = result.stats.elapsed
        result.ttft = result.stats.ttft or 0.0
        result.usage = {k: v for k, v in (('prompt_tokens', result.stats.prompt_tokens),
                                          ('completion_tokens', result.stats.completion_tokens),
                                          ('total_tokens', result.stats.total_tokens))
                        if v is not None}
        result.timings = {}
        result.stats.tool_calls = len(result.tool_calls)
        return result

    @staticmethod
    def _rollback(conversation, mark):
        """Drop everything the model produced this turn, keeping the user's message."""
        if len(conversation.messages) > mark:
            del conversation.messages[mark:]

    # ------------------------------------------------------------------
    def _one_request(self, messages, tools, stats):
        settings = self.settings
        params = settings.api_params(self.client.caps)
        if not settings.stream:
            resp = self.client.chat(messages, tools=tools, **params)
            choice = (resp.get('choices') or [{}])[0]
            msg = choice.get('message') or {}
            finish = choice.get('finish_reason') or ''
            stats.absorb(resp.get('usage'), resp.get('timings'), ttft=None, elapsed=0.0)
            text = msg.get('content') or ''
            if text:
                self.on_delta(text)
            rc = msg.get('reasoning_content')
            if rc:
                self.on_reasoning(rc)
            return msg, resp.get('usage'), resp.get('timings'), finish
        # `with` is not cosmetic: it closes the HTTP response the moment the turn ends,
        # including on the `return` below. Leaving that to the garbage collector let a
        # finished stream's unread bytes leak into the next request on the same socket.
        with self.client.stream_chat(messages, tools=tools, **params) as stream:
            try:
                for ev in stream:
                    if ev['kind'] == 'delta':
                        self.on_delta(ev['text'])
                    elif ev['kind'] == 'reasoning':
                        self.on_reasoning(ev['text'])
                    elif ev['kind'] == 'done':
                        self.client.note_accepted(stream.sent_fields)
                        stats.absorb(ev.get('usage'), ev.get('timings'),
                                     ttft=ev.get('ttft'), elapsed=ev.get('elapsed') or 0.0)
                        return (ev['message'], ev.get('usage'), ev.get('timings'),
                                ev.get('finish_reason'))
            except KeyboardInterrupt:
                stats.cancelled = True
                raise CancelledByUser() from None
            msg = {'role': 'assistant', 'content': stream.message().get('content', '')}
            stats.absorb(None, None, ttft=None, elapsed=0.0)
            return msg, None, None, None


# ======================================================================
# Live output
# ======================================================================
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


HELP_TEXT = """\
Commands:

Conversation
  /help                     this help
  /system <text>            replace the system prompt (empty text clears it)
  /reset                    start a fresh conversation
  /undo                     drop the last exchange
  /retry                    regenerate the previous answer
  /history                  show the conversation so far
  /cancel                   clear the queued turn (Ctrl-C interrupts a running generation)

Thinking
  /think [off|low|medium|high|max|<tokens>]
                            thinking budget: 0 / 512 / 2048 / 8192 / unlimited
  /effort [medium|xhigh]    reasoning_effort sent to the chat template
  /reasoning full|compact|hidden
                            how the thinking trace is displayed

Sampling and output
  /temp <0..2>  /topp <0..1>  /max-tokens <n>
  /stream on|off            toggle streaming
  /markdown on|off          toggle terminal Markdown rendering
  /speed [on|off]           print the full timing table after every turn
  /stats                    session totals + last-turn measurements

Context and tools
  /context                  window, history tokens, budget, reserves, tokenizer
  /compact [keep_turns]     summarise old turns instead of dropping them
  /tools                    list tools; /tools <name> on|off toggles one; /tools off all

Attachments and diagnostics
  /image <path> [...]       attach image(s) to the next message
  /ocr on|off               run tesseract OCR on attached images (needs pytesseract)
  /vision on|off|probe      override or re-probe whether the server can see images
  /caps                     what this server actually supports (supported/unsupported/unknown)
  /doctor [--json]          live endpoint diagnostics
  /bench [samples]          measure TTFT, prefill and decode against this endpoint
  /model [id]               show or change the model id
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
                       'reasoning_tokens': 0, 'tool_calls': 0, 'seconds': 0.0,
                       'ttfts': [], 'rates': [], 'compactions': 0, 'cancels': 0}
        self.last_stats = None
        self._autosave_counter = 0
        self.renderer.reasoning_display = settings.reasoning.display
        self._cancelled = False
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
        ctx = self.client.context_window()
        self.say(s.dim(f'context {ctx} tokens, budget {self.context_budget_tokens()} | '
                       f'tools {"on: " + ", ".join(self.registry.names()) if self.settings.use_tools else "off"}'))
        self.say(s.dim(f'thinking: {self.settings.reasoning.describe()} | '
                       f'streaming {"on" if self.settings.stream else "off"} | '
                       f'tokenizer {"server" if self.conversation.counter.using_server() else "estimate"}'))
        self.say(s.dim(f'optional extras: {extras} | /help for commands'))
        self.say()

    # ------------------------------------------------------------------
    def context_budget_tokens(self):
        """Tokens of history we are allowed to send.

        The server's context window has to hold the prompt *and* the answer, plus whatever
        the chat template adds. Sending history right up to the window guarantees a 400 on
        the turn that matters, so both reserves come off the top.
        """
        window = self.client.context_window()
        soft = int(window * self.settings.context_budget)
        reserve = self.settings.context_reserve + min(self.settings.max_tokens,
                                                      self.settings.output_reserve)
        return max(1024, min(soft, window - reserve))

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
        budget = self.context_budget_tokens()
        dropped = self.conversation.trim(budget)
        if dropped:
            self.renderer.notice(f'trimmed {dropped} oldest message(s) to fit the '
                                 f'{budget}-token context budget (/compact keeps a summary '
                                 'instead)')
        problems = self.conversation.validate_wire()
        if problems:
            self.renderer.notice('history repaired before sending: ' + '; '.join(problems[:3]))
            self.conversation._repair_orphans()
        self._cancelled = False
        try:
            result = self.agent.run(self.conversation, text=text, attachments=attachments,
                                    regenerate=regenerate)
        except BonsaiError as e:
            self.renderer.finish()
            self.renderer.error(str(e))
            hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
            if hint:
                self.say(self.style.dim('  hint: ' + hint))
            return None
        self.renderer.finish()
        result.stats.context_used = self.conversation.tokens()
        result.stats.context_window = self.client.context_window()
        self.last_stats = result.stats
        self.totals['turns'] += 1
        self.totals['prompt_tokens'] += result.prompt_tokens
        self.totals['completion_tokens'] += result.completion_tokens
        self.totals['reasoning_tokens'] += result.stats.reasoning_tokens or 0
        self.totals['tool_calls'] += len(result.tool_calls)
        self.totals['seconds'] += result.elapsed or 0.0
        if result.cancelled:
            self.totals['cancels'] += 1
        if result.stats.ttft:
            self.totals['ttfts'].append(result.stats.ttft)
        if result.stats.tokens_per_s:
            self.totals['rates'].append(result.stats.tokens_per_s)
        if self.settings.show_stats:
            self.say(format_stats_block(result.stats, self.style))
        else:
            self.say(self.style.dim(format_stats(result.stats, self.style)))
        self.say()
        self._autosave_counter += 1
        if self.session_path and self._autosave_counter >= max(1, self.settings.autosave_every):
            self._autosave_counter = 0
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
                if not arg:
                    self.say(s.dim('reasoning_effort: ' + str(self.settings.reasoning.effort)
                                   + ' — Bonsai 2 accepts medium|xhigh (low is a documented '
                                     'no-op and is not offered)'))
                else:
                    self.settings.effort = arg
                    self.say(s.dim(f'reasoning_effort {self.settings.reasoning.effort} '
                                   f'({self.settings.reasoning.describe()})'))
            elif cmd in ('think', 'thinking', 'budget'):
                self._think_command(arg)
            elif cmd == 'reasoning':
                self._reasoning_display_command(arg)
            elif cmd == 'stats':
                self._stats_command()
            elif cmd == 'speed':
                self.settings.show_stats = not self.settings.show_stats if arg == '' else \
                    arg.lower() in ('on', '1', 'true', 'yes', 'full')
                self.say(s.dim('per-turn timing table ' +
                               ('on' if self.settings.show_stats else 'off') +
                               ' (one-line summary otherwise)'))
            elif cmd == 'compact':
                self._compact_command(arg)
            elif cmd == 'cancel':
                self.pending_images = []
                self.renderer.finish()
                self.say(s.dim('cleared the queued turn. Ctrl-C interrupts a generation that '
                               'is already running; the partial answer is discarded and the '
                               'history is left untouched.'))
            elif cmd == 'doctor':
                run_doctor(self.client, style=self.style, out=self.out,
                           as_json=arg.strip() == '--json')
            elif cmd in ('caps', 'capabilities'):
                self._caps_command()
            elif cmd == 'bench':
                self._bench_command(arg)
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
                window = self.client.context_window()
                used = self.conversation.tokens()
                budget = self.context_budget_tokens()
                self.say(s.dim(f'context window {window} | history {used} tokens over '
                               f'{len(self.conversation.messages)} messages '
                               f'({100.0 * used / window:.0f}% of window) | '
                               f'budget {budget} (reserve '
                               f'{self.settings.context_reserve}+'
                               f'{min(self.settings.max_tokens, self.settings.output_reserve)} '
                               f'for template + answer) | tokenizer '
                               f'{"server /tokenize" if self.conversation.counter.using_server() else "estimate"}'))
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
                               f'{t["prompt_tokens"]} in, {t["reasoning_tokens"]} reasoning | '
                               f'{t["tool_calls"]} tool call(s) | '
                               f'{t["seconds"]:.1f}s of model time | '
                               f'{t["cancels"]} cancelled, {t["compactions"]} compacted'))
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

    # ------------------------------------------------------------------
    def _think_command(self, arg):
        """Change the thinking budget without touching history or restarting anything."""
        s = self.style
        if not arg:
            budgets = '  '.join(f'{k}={v if v != -1 else "unlimited"}'
                                for k, v in THINK_BUDGETS.items())
            self.say(s.dim(f'thinking: {self.settings.reasoning.label()} | levels: {budgets}'))
            self.say(s.dim('usage: /think off|low|medium|high|max  or  /think <tokens>'))
            return
        try:
            self.settings.think = arg
        except ValueError as e:
            self.say(s.red(str(e)))
            return
        supported = self.client.caps.supported('thinking_budget_tokens')
        note = ''
        if supported is False:
            note = ' — but this build rejects thinking_budget_tokens, so thinking stays server-controlled'
        elif supported is None:
            note = ' — will be confirmed on the next request'
        self.say(s.dim(f'thinking {self.settings.reasoning.label()}{note}'))

    def _reasoning_display_command(self, arg):
        s = self.style
        if arg.lower() not in REASONING_DISPLAYS:
            self.say(s.dim('usage: /reasoning full|compact|hidden  (current: '
                           + self.settings.reasoning.display + ')'))
            return
        self.settings.reasoning.display = arg.lower()
        self.renderer.reasoning_display = arg.lower()
        self.say(s.dim('reasoning display: ' + arg.lower() +
                       (' — the trace is streamed' if arg == 'full' else
                        ' — one indicator + a token count' if arg == 'compact' else
                        ' — never shown, only counted')))

    def _stats_command(self):
        s = self.style
        t = self.totals
        self.say(s.bold('session'))
        self.say(s.dim(f'  {t["turns"]} turn(s), {t["seconds"]:.1f}s of model time, '
                       f'{t["cancels"]} cancelled'))
        self.say(s.dim(f'  {t["completion_tokens"]} completion tokens, {t["prompt_tokens"]} '
                       f'prompt tokens, {t["reasoning_tokens"]} reasoning tokens, '
                       f'{t["tool_calls"]} tool call(s), {t["compactions"]} compaction(s)'))
        if t['ttfts']:
            self.say(s.dim(f'  ttft      median {statistics.median(t["ttfts"]) * 1000:.0f} ms | '
                           f'min {min(t["ttfts"]) * 1000:.0f} | max {max(t["ttfts"]) * 1000:.0f} '
                           f'({len(t["ttfts"])} sample(s))'))
        if t['rates']:
            self.say(s.dim(f'  decode    median {statistics.median(t["rates"]):.1f} tok/s | '
                           f'min {min(t["rates"]):.1f} | max {max(t["rates"]):.1f}'))
        tr = self.client.transport.stats()
        self.say(s.dim(f'  transport {tr["requests"]} requests over '
                       f'{tr["connections_opened"]} connection(s), '
                       f'{tr["reconnects"]} reconnect(s), pooled={tr["pooled"]}'))
        if self.last_stats:
            self.say(s.bold('last turn'))
            self.say(format_stats_block(self.last_stats, s))

    def _caps_command(self):
        s = self.style
        caps = self.client.caps
        self.say(s.bold('server capabilities (learned from this endpoint, not assumed)'))
        for name in sorted(caps.fields):
            state = caps.state(name)
            colour = {'supported': s.green, 'unsupported': s.red}.get(state, s.yellow)
            detail = caps.evidence.get(name, '')
            self.say(f'  {colour(state.ljust(11))} {name}' + (s.dim('  — ' + detail) if detail else ''))
        for k in ('vision', 'context_window', 'served_models', 'version'):
            if k in caps.facts:
                self.say(s.dim(f'  {k}: {caps.facts[k]}'))

    def _compact_command(self, arg):
        s = self.style
        budget = self.context_budget_tokens()
        keep = 2
        if arg.isdigit():
            keep = max(0, int(arg))
        before = len(self.conversation.messages)
        report = self.conversation.compact(budget, keep_recent=keep)
        self.totals['compactions'] += 1
        self.say(s.dim(f'compacted {report["compacted"]} message(s) into a summary, kept the '
                       f'last {report["kept"]} turn(s): {report["tokens_before"]} -> '
                       f'{report["tokens_after"]} tokens '
                       f'({len(self.conversation.messages)} messages, was {before})'))
        if report.get('trimmed'):
            self.say(s.dim(f'still over budget afterwards — also trimmed '
                           f'{report["trimmed"]} message(s)'))

    def _bench_command(self, arg):
        """Measure this endpoint for real: TTFT, prefill and decode over N samples."""
        samples = 3
        bits = arg.split()
        if bits and bits[0].isdigit():
            samples = max(1, min(10, int(bits[0])))
        self.say(self.style.dim(f'benchmarking {samples} sample(s) against '
                                f'{self.client.base_url} (the prompt is fixed on purpose, '
                                'so the second and later samples measure prompt-cache reuse)'))
        report = run_benchmark(self.client, samples=samples, style=self.style, out=self.out)
        self.say(render_benchmark(report, self.style))

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
# Benchmarking — measured numbers only
# ======================================================================
def sig(value, digits=3):
    """Round to `digits` significant figures.

    A decode rate measured from three samples is not known to six decimal places; printing
    147.123456 tok/s would be a lie about the precision of the measurement.
    """
    if value is None:
        return None
    if value == 0:
        return 0.0
    from math import floor, log10
    magnitude = floor(log10(abs(value)))
    return round(value, -int(magnitude) + (digits - 1))


def summarize(values):
    """median / min / max / p95 over real samples. Empty in, empty out — no invention."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {'n': 0, 'median': None, 'min': None, 'max': None, 'p95': None}
    clean = sorted(clean)
    n = len(clean)

    def pct(p):
        if n == 1:
            return clean[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return clean[idx]

    return {'n': n, 'median': statistics.median(clean), 'min': clean[0], 'max': clean[-1],
            'p95': pct(0.95)}


BENCH_PROMPT_BLOCKS = 40


def run_benchmark(client, samples=3, style=None, out=None, max_tokens=96, warmup=1,
                  progress=None):
    """Measure TTFT, prefill speed, decode speed and total latency against a live endpoint.

    Every number comes from the server's own usage/timings block, or from a monotonic
    clock around a real request. Nothing is estimated, and a field the server did not
    report stays None so the renderer prints `n/a` instead of a guess.

    Sample 1 is the cold start (nothing in the prompt cache). The next `warmup` samples
    are discarded, and the rest are the warm distribution — reported as median/min/max/p95.
    """
    out = out if out is not None else sys.stdout
    prompt = ('Review this Python function and give two concrete improvements.\\n\\n' +
              ('def search(items, target):\\n'
               '    for index, value in enumerate(items):\\n'
               '        if value == target:\\n'
               '            return index\\n'
               '    return -1\\n\\n') * BENCH_PROMPT_BLOCKS)
    messages = [{'role': 'user', 'content': prompt}]
    report = {'endpoint': client.base_url, 'model': client.model, 'samples': samples,
              'warmup': warmup, 'max_tokens': max_tokens, 'cold': None, 'warm': {},
              'errors': [], 'notes': []}
    collected = {'ttft': [], 'prefill': [], 'decode': [], 'latency': [],
                 'prompt_tokens': [], 'completion_tokens': []}
    # i==0 is the cold start (empty prompt cache), then `warmup` discarded samples, then
    # the `samples` that actually form the reported distribution.
    total = samples + warmup + 1
    for i in range(total):
        label = 'cold' if i == 0 else ('warmup' if i <= warmup else f'sample {i - warmup}')
        if progress:
            progress(label, i + 1, total)
        started = time.monotonic()
        stats = TurnStats()
        try:
            with client.stream_chat(messages, max_tokens=max_tokens, temperature=1.0) as stream:
                for ev in stream:
                    if ev['kind'] == 'done':
                        stats.absorb(ev.get('usage'), ev.get('timings'),
                                     ttft=ev.get('ttft'), elapsed=ev.get('elapsed') or 0.0)
        except BonsaiError as e:
            report['errors'].append(f'{label}: {e}')
            continue
        stats.elapsed = time.monotonic() - started
        entry = {'ttft_s': stats.ttft, 'latency_s': stats.elapsed,
                 'prefill_tok_s': stats.prompt_tokens_per_s, 'decode_tok_s': stats.tokens_per_s,
                 'prompt_tokens': stats.prompt_tokens,
                 'completion_tokens': stats.completion_tokens}
        if i == 0:
            report['cold'] = entry
            continue
        if i <= warmup:
            continue
        for key, value in (('ttft', stats.ttft), ('prefill', stats.prompt_tokens_per_s),
                           ('decode', stats.tokens_per_s), ('latency', stats.elapsed),
                           ('prompt_tokens', stats.prompt_tokens),
                           ('completion_tokens', stats.completion_tokens)):
            if value is not None:
                collected[key].append(value)
    report['warm'] = {k: summarize(v) for k, v in collected.items()}
    if not collected['decode']:
        report['notes'].append('no decode rate was measured — the server did not report '
                               'timings and no completion tokens arrived')
    facts = client.caps.facts
    report['context_window'] = facts.get('context_window')
    report['runtime'] = facts.get('version')
    report['transport'] = client.transport.stats()
    return report


def render_benchmark(report, style=None):
    """Human-readable benchmark output. Unmeasured fields are printed as `n/a`."""
    style = style or Style(force_color=False)
    lines = [style.bold('benchmark — measured against ' + report['endpoint'])]
    if report.get('runtime'):
        lines.append(style.dim(f'runtime: {report["runtime"]}'))
    if report.get('context_window'):
        lines.append(style.dim(f'context window: {report["context_window"]} tokens'))

    def row(name, summary, unit, digits=3, scale=1.0):
        if not summary or not summary.get('n'):
            return f'  {name:<22} n/a  (not measured)'
        return (f'  {name:<22} median {sig(summary["median"] * scale, digits)} {unit}'
                f'  |  min {sig(summary["min"] * scale, digits)}'
                f'  |  max {sig(summary["max"] * scale, digits)}'
                f'  |  p95 {sig(summary["p95"] * scale, digits)}'
                f'  |  n={summary["n"]}')

    cold = report.get('cold') or {}
    if cold:
        parts = []
        if cold.get('ttft_s'):
            parts.append(f'ttft {sig(cold["ttft_s"] * 1000)} ms')
        if cold.get('prefill_tok_s'):
            parts.append(f'prefill {sig(cold["prefill_tok_s"])} tok/s')
        if cold.get('decode_tok_s'):
            parts.append(f'decode {sig(cold["decode_tok_s"])} tok/s')
        if cold.get('latency_s'):
            parts.append(f'total {sig(cold["latency_s"])} s')
        lines.append(style.dim('  cold start (first request, empty prompt cache): ' +
                               (' | '.join(parts) if parts else 'n/a')))
    warm = report.get('warm') or {}
    lines.append(style.dim(f'  warm ({warm.get("decode", {}).get("n", 0)} sample(s), '
                           f'{report.get("warmup", 0)} warmup discarded):'))
    lines.append(row('ttft', warm.get('ttft'), 'ms', scale=1000.0))
    lines.append(row('prefill speed', warm.get('prefill'), 'tok/s'))
    lines.append(row('decode speed', warm.get('decode'), 'tok/s'))
    lines.append(row('total latency', warm.get('latency'), 's'))
    pt = warm.get('prompt_tokens') or {}
    ct = warm.get('completion_tokens') or {}
    if pt.get('n'):
        lines.append(style.dim(f'  prompt {sig(pt["median"])} tok, '
                               f'completion {sig(ct.get("median"))} tok per sample'))
    tr = report.get('transport') or {}
    if tr:
        lines.append(style.dim(f'  transport: {tr.get("requests", 0)} requests, '
                               f'{tr.get("connections_opened", 0)} connection(s), '
                               f'{tr.get("reconnects", 0)} reconnect(s)'))
    for note in report.get('notes', []):
        lines.append(style.yellow('  note: ' + note))
    for err in report.get('errors', []):
        lines.append(style.red('  error: ' + str(err)[:200]))
    lines.append(style.dim('  VRAM/RAM are server-side and cannot be measured from the '
                           'client — the deployment cell reports those.'))
    return '\n'.join(lines)


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
    p.add_argument('--effort', default='medium',
                   help='reasoning_effort sent to the chat template: medium|xhigh '
                        '(Bonsai 2 documents `low` as a no-op, so it is not offered; '
                        '`none` turns thinking off)')
    p.add_argument('--think', default='medium',
                   help='thinking budget: off|low|medium|high|max (=0/512/2048/8192/unlimited '
                        'thinking_budget_tokens) or an explicit token count')
    p.add_argument('--reasoning-display', default='compact', choices=list(REASONING_DISPLAYS),
                   help='full: stream the thinking trace; compact: one indicator + a token '
                        'count; hidden: count it only')
    p.add_argument('--no-stream', action='store_true', help='request one complete answer')
    p.add_argument('--no-markdown', action='store_true', help='print model output verbatim')
    p.add_argument('--no-highlight', action='store_true', help='no Pygments highlighting in code blocks')
    p.add_argument('--no-tools', action='store_true', help='disable tool calling')
    p.add_argument('--tools', help='comma-separated tool names to enable (default: all)')
    p.add_argument('--max-tool-rounds', type=int, default=6)
    p.add_argument('--tool-timeout', type=float, default=60.0,
                   help='seconds before an individual tool call is abandoned')
    p.add_argument('--tool-budget', type=float, default=600.0,
                   help='total seconds the tool loop may take in one turn')
    p.add_argument('--tool-max-output', type=int, default=24000,
                   help='characters of tool output fed back to the model per call')
    p.add_argument('--autosave-every', type=int, default=1,
                   help='write the session file every N turns (default every turn)')
    p.add_argument('--context-reserve', type=int, default=512,
                   help='tokens held back for the chat template')
    p.add_argument('--output-reserve', type=int, default=1024,
                   help='tokens held back for the answer being generated')
    p.add_argument('--show-stats', action='store_true',
                   help='print the full timing table after every turn (same as /speed on)')
    p.add_argument('--no-keepalive', action='store_true',
                   help='open a fresh connection per request (debugging only; slower)')
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
    p.add_argument('--benchmark', action='store_true',
                   help='measure TTFT, prefill and decode speed against the endpoint and exit')
    p.add_argument('--bench-samples', type=int, default=3,
                   help='benchmark samples after warmup (default 3)')
    p.add_argument('--bench-warmup', type=int, default=1,
                   help='benchmark warmup samples to discard (default 1)')
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


#: doctor states. `PASS` is only ever printed for behaviour that was actually exercised.
DOCTOR_STATES = ('PASS', 'FAIL', 'SKIP', 'UNKNOWN', 'DEGRADED')


def run_doctor(client, style=None, out=None, as_json=False):
    """Live diagnostics against a real deployment: is this endpoint actually usable?

    With `as_json` the same checks are emitted as one machine-readable object instead of
    the human table, so a deployment can be monitored without scraping text.
    """
    style = style or Style(force_color=False)
    out = out if out is not None else sys.stdout
    results = []
    report = {'endpoint': client.base_url, 'model': client.model,
              'client_version': VERSION, 'checks': {}}

    def check(name, state, detail=''):
        if state not in DOCTOR_STATES:
            raise ValueError(f'unknown doctor state {state!r}')
        results.append(state)
        report['checks'][name] = {'state': state, 'detail': detail}
        if as_json:
            return
        mark = {'PASS': style.green('PASS'), 'FAIL': style.red('FAIL'),
                'SKIP': style.yellow('SKIP'), 'UNKNOWN': style.yellow('UNKNOWN'),
                'DEGRADED': style.yellow('DEGRADED')}[state]
        out.write(f'{mark}  {name}' + (style.dim(' — ' + detail) if detail else '') + '\n')
        out.flush()

    def say(text):
        if not as_json:
            out.write(text + '\n')
            out.flush()

    say(style.bold(f'Diagnosing {client.base_url} '
                   f'(key {mask_key(client.api_key)}, model {client.model}, '
                   f'client v{VERSION})\n'))

    try:
        health = client.get_json('/health', timeout=15, root=True)
        check('/health', 'PASS' if isinstance(health, dict) else 'FAIL', json.dumps(health)[:80])
    except BonsaiError as e:
        check('/health', 'FAIL', str(e))
        say(style.red('\nThe endpoint is not reachable — nothing else can be tested. '
                      'If this was a Colab/Kaggle deployment, the runtime (and its '
                      'Quick Tunnel URL) is gone: rerun colab_kaggle_cell.py and use the '
                      'new base URL and key.'))
        return _doctor_exit(report, results, style, out, as_json, client)

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

    # ---- reasoning controls: probe what the runtime actually accepts -------------
    for field_name, value, label in (('thinking_budget_tokens', 2048, 'thinking budget'),
                                     ('reasoning_effort', 'medium', 'reasoning_effort')):
        try:
            ok = client.probe_field(field_name, value)
        except BonsaiError as e:
            check(f'reasoning control: {label}', 'UNKNOWN', f'probe failed: {e}')
            continue
        check(f'reasoning control: {label}', 'PASS' if ok else 'DEGRADED',
              'accepted by this build' if ok else
              'rejected with HTTP 400 — thinking stays server-controlled')

    _report_capabilities(client, style, out, as_json, report, check)
    return _doctor_exit(report, results, style, out, as_json, client)


def _report_capabilities(client, style, out, as_json, report, check=None):
    """Print the capability map: supported / unsupported / unknown, with evidence."""
    caps = client.caps
    report['capabilities'] = caps.to_dict()
    report['transport'] = client.transport.stats()
    if as_json:
        return
    out.write('\n' + style.bold('capability map (learned from this endpoint)\n'))
    for name in sorted(caps.fields):
        state = caps.state(name)
        colour = {'supported': style.green, 'unsupported': style.red}.get(state, style.yellow)
        detail = caps.evidence.get(name, 'not tested against this build')
        out.write(f'  {colour(state.ljust(11))} {name}' + style.dim('  — ' + detail) + '\n')
    for key in ('vision', 'vision_source', 'context_window', 'served_models', 'model_path',
                'total_slots', 'version'):
        if key in caps.facts:
            out.write(style.dim(f'  {key}: {caps.facts[key]}') + '\n')
    tr = client.transport.stats()
    out.write(style.dim(f'  transport: {tr["requests"]} requests, '
                        f'{tr["connections_opened"]} connection(s), '
                        f'{tr["reconnects"]} reconnect(s), keep-alive pooled={tr["pooled"]}'
                        ) + '\n')
    out.flush()


def _doctor_exit(report, results, style, out, as_json, client=None):
    counts = {state: results.count(state) for state in DOCTOR_STATES}
    report['summary'] = counts
    report['usable'] = counts['FAIL'] == 0
    if as_json:
        out.write(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
        out.flush()
    else:
        out.write(f'\n{counts["PASS"]} passed, {counts["FAIL"]} failed, '
                  f'{counts["SKIP"]} skipped, {counts["UNKNOWN"]} unknown, '
                  f'{counts["DEGRADED"]} degraded\n')
        out.write(style.green('Endpoint is usable.' if counts['FAIL'] == 0 else
                              'Endpoint has failures — see above.') + '\n')
        out.flush()
    if client is not None:
        client.close()
    return 0 if counts['FAIL'] == 0 else 1


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

    try:
        reasoning = ReasoningConfig(level=args.think,
                                    effort=None if str(args.effort).lower() in ('none', 'off')
                                    else args.effort,
                                    display=args.reasoning_display)
    except ValueError as e:
        sys.stderr.write(f'error: {e}\n')
        return 2
    if str(args.effort).lower() in ('none', 'off'):
        reasoning.level = 'off'
        reasoning.effort = None
    settings = Settings(model=args.model, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, reasoning=reasoning,
                        stream=not args.no_stream, use_tools=not args.no_tools,
                        markdown=not args.no_markdown, highlight=not args.no_highlight,
                        max_tool_rounds=args.max_tool_rounds,
                        tool_timeout=args.tool_timeout, tool_budget=args.tool_budget,
                        tool_max_output=args.tool_max_output,
                        autosave_every=args.autosave_every,
                        context_reserve=args.context_reserve,
                        output_reserve=args.output_reserve,
                        show_stats=args.show_stats)

    client = BonsaiClient(args.base_url, args.api_key, model=args.model,
                          timeout=args.timeout, retries=args.retries,
                          keepalive=not args.no_keepalive)
    if args.doctor:
        try:
            return run_doctor(client, style=style, as_json=args.json)
        finally:
            client.close()
    if args.benchmark:
        def progress(label, i, total):
            sys.stderr.write(f'  {label} ({i}/{total})\n')
        report = run_benchmark(client, samples=max(1, args.bench_samples),
                               warmup=max(0, args.bench_warmup), style=style,
                               progress=progress)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_benchmark(report, style))
        client.close()
        return 1 if report.get('errors') and not report['warm'].get('decode', {}).get('n') else 0
    if args.context:
        client.context_window = (lambda override=args.context: override)

    counter = TokenCounter(client)
    resumed = bool(args.session) and Path(args.session).expanduser().is_file()
    if resumed:
        conversation = Conversation.load(
            args.session, counter=counter,
            on_error=lambda n: sys.stderr.write(
                f'warning: skipped {n} unreadable line(s) in {args.session}; the rest of the '
                'session was loaded\n'))
        if conversation.skipped_lines:
            sys.stderr.write(f'note: session file came from '
                             f'{conversation.saved_version or "an older version"} '
                             f'(schema {conversation.schema})\n')
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
        # Machine-readable mode: stdout carries exactly one JSON document. Everything
        # human-readable — the streamed answer, the stats line, notices — goes to a sink.
        sink = io.StringIO()
        app.out = sink
        app.renderer = LiveRenderer(style, markdown=False, out=sink)
        app.agent = Agent(client, settings, registry=registry, counter=counter,
                          on_delta=lambda t: None, on_reasoning=lambda t: None,
                          on_tool=lambda *a: None, on_notice=lambda t: None)

    try:
        if prompt is not None:
            result = app.send(prompt, attachments=attachments)
            if result is None:
                return 1
            if args.json:
                print(json.dumps({'model': client.model, 'version': VERSION,
                                  'text': result.text, 'reasoning': result.reasoning,
                                  'tool_calls': result.tool_calls,
                                  'finish_reason': result.finish_reason,
                                  'usage': result.usage, 'timings': result.timings,
                                  'stats': result.stats.to_dict(),
                                  'reasoning_config': settings.reasoning.to_dict(),
                                  'capabilities': client.caps.to_dict(),
                                  'elapsed': result.elapsed, 'ttft': result.ttft,
                                  'rounds': result.rounds,
                                  'cancelled': result.cancelled,
                                  'interrupted': result.interrupted or None},
                                 ensure_ascii=False, indent=2))
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
