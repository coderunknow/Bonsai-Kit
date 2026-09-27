"""Server-sent events: incremental decoding and tool-call reconstruction."""

from __future__ import annotations

import json
import re

from ._meta import MAX_SSE_EVENT_BYTES
from .errors import BonsaiError, StreamInterrupted

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
