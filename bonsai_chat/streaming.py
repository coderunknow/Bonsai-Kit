"""ChatStream: response + decoder + accumulator, deterministically released."""

from __future__ import annotations

import copy
import json
import time

from .errors import StreamInterrupted
from .sse import SSEDecoder, accumulate_tool_calls, arguments_complete

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
        self._announced_tools = set()
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
        ready = self._absorb_tool_calls(delta.get('tool_calls'))
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
            if ready:
                return ready + [{'kind': 'delta', 'text': ct}]
            return {'kind': 'delta', 'text': ct}
        return ready or None

    def _absorb_tool_calls(self, tool_calls):
        """Fold streamed tool-call deltas in, and announce each call the moment its
        arguments are complete JSON.

        That announcement is what lets the agent start a tool while the model is still
        writing the rest of its turn, instead of after the stream ends. Each index is
        announced at most once; anything still unfinished at the end is announced from
        `tool_calls()` so no call is ever silently skipped.
        """
        accumulate_tool_calls(tool_calls, self.tool_acc)
        ready = []
        for index in sorted(self.tool_acc):
            if index in self._announced_tools:
                continue
            call = self.tool_acc[index]
            raw = (call.get('function') or {}).get('arguments')
            if not arguments_complete(raw):
                continue
            self._announced_tools.add(index)
            ready.append({'kind': 'tool_call', 'index': index,
                          'call': copy.deepcopy(call)})
        return ready

    def pending_tool_calls(self):
        """Completed calls the caller has not been told about yet (used at end of stream)."""
        out = []
        for index in sorted(self.tool_acc):
            if index in self._announced_tools:
                continue
            self._announced_tools.add(index)
            out.append({'kind': 'tool_call', 'index': index,
                        'call': copy.deepcopy(self.tool_acc[index])})
        return out

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
                        for item in self.pending_tool_calls():
                            yield item
                        yield self.done_event()
                        return
                    for item in (ev if isinstance(ev, list) else [ev]):
                        if item is not None:
                            yield item
            tail = self.decoder.close()
            if tail is not None:
                ev = self._event(tail)
                if ev is not None and ev is not _DONE:
                    for item in (ev if isinstance(ev, list) else [ev]):
                        yield item
            self.done = True
            self.drained = True          # natural EOF: the body is fully consumed
            for item in self.pending_tool_calls():
                yield item
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
