#!/usr/bin/env python3
"""
mock_bonsai_server.py — an offline stub of the OpenAI-compatible endpoint that
colab_kaggle_cell.py deploys.

This is NOT a model. It runs no inference and needs no GPU: it speaks the same wire
protocol as the PrismML llama.cpp fork (chat completions, SSE streaming with
`reasoning_content` and split `tool_calls`, /props, /tokenize, bearer auth, a text-only
server that rejects `image_url`) with scripted replies, so bonsai_chat.py can be exercised
end to end and the test-suite can assert on real HTTP traffic.

On top of the happy path it can reproduce the failure modes a real deployment actually
produces, deterministically and without randomness — see SCENARIOS. Set one with
`server.scenario = 'disconnect'` (next chat request only) or pass `scenario=` to the
constructor to make it sticky.

Use it to try the client before you have a deployment:

    python3 bonsai_chat.py --selftest            # runs scripted checks against this stub
    python3 -c "import mock_bonsai_server as m; s=m.MockBonsaiServer().start(); print(s.base_url)"
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = 'ternary-bonsai-2-27b'
DEFAULT_KEY = 'mock-key-0123456789abcdef0123456789'

DEMO_ANSWER = """**Bonsai 2** is a ternary reasoning model served through `llama.cpp`.

## Why C++ is fast

1. **Zero-cost abstractions** — templates resolve at compile time.
2. **Manual memory layout** — you decide what stays cache-resident.
3. **No runtime** — nothing runs between `main()` and your loop.

```cpp
#include <vector>
int sum(const std::vector<int>& v) {
    int acc = 0;
    for (int x : v) acc += x;
    return acc;
}
```

| Layer    | Cost        |
| -------- | ----------- |
| compile  | one-off     |
| runtime  | ~nothing    |

See the [language reference](https://en.cppreference.com) for details.
"""

WEATHER_CALL = [{'index': 0, 'id': 'call_mock_1', 'type': 'function',
                 'function': {'name': 'get_weather', 'arguments': '{"city": '}}]
WEATHER_CALL_TAIL = [{'index': 0, 'function': {'arguments': '"Lisbon"}'}}]

# Two calls interleaved by index, with the function *name* split across chunks and a
# multi-byte character split across argument chunks — the shape a real stream produces.
PARALLEL_CALLS = [
    [{'index': 0, 'id': 'call_a', 'type': 'function', 'function': {'name': 'get_', 'arguments': ''}},
     {'index': 1, 'id': 'call_b', 'type': 'function', 'function': {'name': 'current_', 'arguments': ''}}],
    [{'index': 0, 'function': {'name': 'weather', 'arguments': '{"city": "Lis'}},
     {'index': 1, 'function': {'name': 'time', 'arguments': '{"zone": "UT'}}],
    [{'index': 1, 'function': {'arguments': 'C"}'}}],
    [{'index': 0, 'function': {'arguments': 'bon\u00e9"}'}}],
]

# A tool call whose JSON never completes — the stream ends mid-arguments.
PARTIAL_CALL = [[{'index': 0, 'id': 'call_partial', 'type': 'function',
                  'function': {'name': 'get_weather', 'arguments': '{"city": "Lis'}}]]

#: Failure modes this stub can produce. Every one is deterministic.
SCENARIOS = (
    'ok',                  # normal scripted answer
    'slow-first-token',    # a delay before the first chunk (high TTFT)
    'slow-stream',         # a delay between every chunk
    'fragmented-sse',      # each SSE event split across several writes at odd boundaries
    'crlf-sse',            # CRLF line endings
    'keepalive-comments',  # SSE `:` comment keep-alives interleaved with data
    'multiline-data',      # a JSON payload spread over several `data:` lines
    'malformed-sse',       # a chunk that is not JSON at all, in the middle of the stream
    'partial-tool-call',   # tool-call JSON that never completes
    'parallel-tool-calls', # two interleaved calls with split names/arguments
    'no-usage',            # no usage chunk at all
    'no-finish-reason',    # the stream ends without a finish_reason
    'no-done',             # no `data: [DONE]` sentinel
    'reasoning-only',      # reasoning_content but no answer text
    'interleaved',         # reasoning and content alternate
    'disconnect',          # the server closes the socket mid-generation
    'timeout',             # the server stalls past the client's read timeout
    'invalid-json',        # HTTP 200 with a body that is not JSON
    'http-429', 'http-500', 'http-502', 'http-503',
    'context-overflow',    # 400 naming the context window
    'reject-effort',       # 400 naming reasoning_effort
    'reject-thinking-budget',   # 400 naming thinking_budget_tokens
    'reject-tools',        # 400 refusing tools (no --jinja)
)


def _sse(payload) -> bytes:
    return ('data: ' + json.dumps(payload) + '\n\n').encode()


class MockBonsaiServer:
    """Threaded stub server. `start()` returns self; `.base_url` is ready to use."""

    def __init__(self, api_key=DEFAULT_KEY, vision=False, context=8192,
                 reject_effort_once=False, reject_tools_once=False,
                 announce_modalities=True, scenario='ok', sticky_scenario=False,
                 reject_thinking_budget_once=False, reasoning='separate'):
        self.api_key = api_key
        self.vision = vision
        self.announce_modalities = announce_modalities
        self.context = context
        self.reject_effort_once = reject_effort_once
        self.reject_tools_once = reject_tools_once
        self.reject_thinking_budget_once = reject_thinking_budget_once
        self.reasoning = reasoning        # 'separate' | 'off' | 'none'
        #: applied to the next chat request, then reset to `default_scenario`
        self.scenario = scenario
        self.default_scenario = scenario if sticky_scenario else 'ok'
        self.sticky_scenario = sticky_scenario
        self.calls = []            # recorded request bodies, for assertions
        self.get_paths = []        # recorded GET paths, for assertions
        self.chat_requests = 0
        self.scenarios_served = []
        self._httpd = None
        self._thread = None

    # ------------------------------------------------------------------
    def set_scenario(self, name, sticky=False):
        """Arm a failure mode. Unknown names raise instead of silently passing."""
        if name not in SCENARIOS:
            raise ValueError(f'unknown scenario {name!r}; see mock_bonsai_server.SCENARIOS')
        self.scenario = name
        if sticky:
            self.sticky_scenario = True
            self.default_scenario = name
        return self

    def _take_scenario(self):
        name = self.scenario
        self.scenarios_served.append(name)
        self.scenario = self.default_scenario
        return name

    # ------------------------------------------------------------------
    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            server_version = 'MockBonsai/0.5'

            def log_message(self, *args):
                pass

            def _send(self, code, payload, ctype='application/json'):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self):
                auth = self.headers.get('Authorization', '')
                return auth == 'Bearer ' + outer.api_key

            def _body(self):
                length = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(length) if length else b''
                try:
                    return json.loads(raw.decode()) if raw else {}
                except ValueError:
                    return {'_unparsed': raw.decode('utf-8', 'replace')}

            # ----------------------------------------------------------
            def do_GET(self):
                outer.get_paths.append(self.path)
                if not self._authorized():
                    return self._send(401, {'error': {'message': 'invalid api key'}})
                if self.path == '/v1/models':
                    return self._send(200, {'object': 'list',
                                            'data': [{'id': MODEL, 'object': 'model'}]})
                if self.path == '/props':
                    props = {'default_generation_settings':
                             {'n_ctx': outer.context, 'model': 'mock-bonsai'},
                             'total_slots': 1}
                    if outer.announce_modalities:
                        props['modalities'] = {'vision': outer.vision, 'audio': False}
                    return self._send(200, props)
                if self.path == '/health':
                    return self._send(200, {'status': 'ok'})
                return self._send(404, {'error': {'message': 'unknown path ' + self.path}})

            def do_POST(self):
                outer.get_paths.append('POST ' + self.path)
                # Drain the body BEFORE deciding anything. An HTTP/1.1 keep-alive server
                # that replies without consuming the request leaves those bytes in the
                # socket, and the next request line it parses is the tail of this body —
                # which shows up as a spurious "Bad request syntax" 400. A real
                # llama-server drains or closes; the stub has to behave the same way for
                # connection-reuse tests to mean anything.
                body = self._body()
                if not self._authorized():
                    return self._send(401, {'error': {'message': 'invalid api key'}})
                if self.path == '/tokenize':
                    content = body.get('content') or ''
                    n = max(1, len(str(content)) // 4)
                    return self._send(200, {'tokens': list(range(n))})
                if self.path == '/v1/chat/completions':
                    outer.calls.append(body)
                    outer.chat_requests += 1
                    return self._chat(body)
                return self._send(404, {'error': {'message': 'unknown path ' + self.path}})

            # ----------------------------------------------------------
            def _chat(self, body):
                messages = body.get('messages') or []
                stream = bool(body.get('stream'))
                scenario = outer._take_scenario()
                self.scenario = scenario
                if outer.reject_effort_once and 'reasoning_effort' in body:
                    outer.reject_effort_once = False
                    scenario = self.scenario = 'reject-effort'
                if outer.reject_thinking_budget_once and 'thinking_budget_tokens' in body:
                    outer.reject_thinking_budget_once = False
                    scenario = self.scenario = 'reject-thinking-budget'
                if outer.reject_tools_once and body.get('tools'):
                    outer.reject_tools_once = False
                    scenario = self.scenario = 'reject-tools'
                # Scenarios that fail before any generation happens.
                if scenario.startswith('http-'):
                    code = int(scenario.split('-')[1])
                    return self._send(code, {'error': {'message': f'scripted {code}'}})
                if scenario == 'context-overflow':
                    return self._send(400, {'error': {'message':
                        'context size exceeded: the prompt is 9000 tokens and the context '
                        'window is 8192'}})
                if scenario == 'reject-effort':
                    return self._send(400, {'error': {
                        'message': "unknown field 'reasoning_effort'"}})
                if scenario == 'reject-thinking-budget':
                    return self._send(400, {'error': {
                        'message': "invalid value for 'thinking_budget_tokens'"}})
                if scenario == 'reject-tools':
                    return self._send(400, {'error': {
                        'message': 'this server was built without --jinja; tools are unsupported'}})
                if scenario == 'invalid-json' and not stream:
                    body_bytes = b'{this is not json'
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body_bytes)))
                    self.end_headers()
                    self.wfile.write(body_bytes)
                    return
                last_user = ''
                for m in reversed(messages):
                    if m.get('role') == 'user':
                        c = m.get('content')
                        if isinstance(c, list):
                            last_user = ' '.join(p.get('text', '') for p in c
                                                 if isinstance(p, dict))
                            if any(isinstance(p, dict) and p.get('type') == 'image_url'
                                   for p in c) and not outer.vision:
                                return self._send(400, {'error': {
                                    'message': 'this server does not support image input '
                                               '(no vision projector loaded)'}})
                        else:
                            last_user = str(c or '')
                        break
                tool_results = [m.get('content') for m in messages if m.get('role') == 'tool']
                tools = body.get('tools') or []
                if scenario == 'partial-tool-call':
                    return self._respond(body, stream, tool_call='partial')
                if scenario == 'parallel-tool-calls':
                    return self._respond(body, stream, tool_call='parallel')
                if not tool_results and tools and 'weather' in last_user.lower():
                    return self._respond(body, stream, tool_call=True)
                return self._respond(body, stream, tool_call=False,
                                     tool_output=tool_results[-1] if tool_results else None)

            # ----------------------------------------------------------
            def _respond(self, body, stream, tool_call, tool_output=None):
                scenario = getattr(self, 'scenario', 'ok')
                if tool_output:
                    answer = ('The tool reported: ' + str(tool_output) +
                              ' — so it is 21C and sunny in Lisbon.')
                else:
                    answer = DEMO_ANSWER
                if not stream:
                    if tool_call:
                        msg = {'role': 'assistant', 'content': None,
                               'tool_calls': [{'id': 'call_mock_1', 'type': 'function',
                                               'function': {'name': 'get_weather',
                                                            'arguments': '{"city": "Lisbon"}'}}]}
                        finish = 'tool_calls'
                    else:
                        msg = {'role': 'assistant', 'content': answer}
                        finish = 'stop'
                    payload = {'id': 'mock-1', 'object': 'chat.completion', 'model': MODEL,
                               'choices': [{'index': 0, 'message': msg,
                                            'finish_reason': finish}],
                               'usage': {'prompt_tokens': 42, 'completion_tokens': 64,
                                         'total_tokens': 106}}
                    if scenario == 'no-usage':
                        payload.pop('usage')
                    if scenario == 'no-finish-reason':
                        payload['choices'][0].pop('finish_reason')
                    return self._send(200, payload)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                try:
                    self._stream(body, tool_call, answer=answer)
                except (BrokenPipeError, ConnectionResetError):
                    return

            # ----------------------------------------------------------
            def _chunk(self, data: bytes):
                self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))
                self.wfile.flush()

            def _end_chunks(self):
                """Emit the real zero-length chunk that terminates a chunked body.

                Sending `0\r\n\r\n` *inside* a chunk (as an earlier version of this stub
                did) leaves the body unterminated: a client that reads by size blocks
                forever waiting for the terminator. A real llama-server terminates
                properly, so the stub must too.
                """
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()

            def _sse_write(self, payload):
                """Write one SSE event the way the scenario asks for it."""
                scenario = getattr(self, 'scenario', 'ok')
                raw = _sse(payload)
                if scenario == 'crlf-sse':
                    raw = raw.replace(b'\n', b'\r\n')
                if scenario == 'fragmented-sse':
                    # Split at boundaries that never line up with a line end, including
                    # inside the JSON, so the client's buffering is genuinely exercised.
                    for k in range(0, len(raw), 7):
                        self._chunk(raw[k:k + 7])
                    return
                if scenario == 'keepalive-comments':
                    self._chunk(b': keep-alive\n\n')
                self._chunk(raw)

            def _emit(self, delta, finish=None, usage=None, timings=None, choices=True):
                ev = {}
                if choices:
                    ch = {'index': 0, 'delta': delta}
                    if finish:
                        ch['finish_reason'] = finish
                    ev['choices'] = [ch]
                else:
                    ev['choices'] = []
                if usage:
                    ev['usage'] = usage
                if timings:
                    ev['timings'] = timings
                self._sse_write(ev)

            def _stream(self, body, tool_call, answer=DEMO_ANSWER):
                scenario = getattr(self, 'scenario', 'ok')
                reasoning_on = outer.reasoning != 'off'

                if scenario == 'slow-first-token':
                    time.sleep(0.35)
                if scenario == 'multiline-data':
                    # one JSON object spread across three `data:` lines
                    self._chunk(b'data: {"choices": [{"index": 0, "delta":\n')
                    self._chunk(b'data: {"content": "multi-line SSE payload "}}]}\n\n')
                if reasoning_on and scenario not in ('reasoning-only',):
                    for piece in ('Reasoning', ' about', ' the', ' question.'):
                        self._emit({'reasoning_content': piece + ' '})
                        if scenario == 'slow-stream':
                            time.sleep(0.05)
                if scenario == 'interleaved':
                    for i in range(3):
                        self._emit({'reasoning_content': f'(thought {i}) '})
                        self._emit({'content': f'answer {i} '})
                if scenario == 'malformed-sse':
                    self._chunk(b'data: {not json at all\n\n')
                if scenario == 'timeout':
                    time.sleep(5)
                if scenario == 'disconnect':
                    # Close mid-generation, without a terminating chunk or [DONE].
                    for piece in answer.split(' ')[:6]:
                        self._emit({'content': piece + ' '})
                    try:
                        self.wfile.flush()
                        self.close_connection = True
                        self.connection.close()
                    except OSError:
                        pass
                    return
                if tool_call == 'parallel':
                    for group in PARALLEL_CALLS:
                        for frag in group:
                            self._emit({'tool_calls': [frag]})
                    self._emit({}, finish='tool_calls')
                elif tool_call == 'partial':
                    for group in PARTIAL_CALL:
                        for frag in group:
                            self._emit({'tool_calls': [frag]})
                    # no finish_reason, no [DONE]: the call simply never completes
                    self._end_chunks()
                    return
                elif tool_call:
                    for frag in WEATHER_CALL:
                        self._emit({'tool_calls': [frag]})
                    for frag in WEATHER_CALL_TAIL:
                        self._emit({'tool_calls': [frag]})
                    self._emit({}, finish='tool_calls')
                elif scenario == 'reasoning-only':
                    for piece in ('Thinking', ' hard', ' about', ' it.'):
                        self._emit({'reasoning_content': piece + ' '})
                    self._emit({}, finish='stop')
                else:
                    for piece in answer.split(' '):
                        self._emit({'content': piece + ' '})
                        if scenario == 'slow-stream':
                            time.sleep(0.01)
                    if scenario != 'no-finish-reason':
                        self._emit({}, finish='stop')
                usage = {'prompt_tokens': 42, 'completion_tokens': 64, 'total_tokens': 106}
                timings = {'prompt_n': 42, 'predicted_n': 64, 'predicted_ms': 3200,
                           'prompt_ms': 800, 'tokens_per_second': 20.0,
                           'prompt_per_second': 52.5, 'predicted_per_second': 20.0}
                if scenario != 'no-usage':
                    self._sse_write({'choices': [], 'usage': usage, 'timings': timings})
                if scenario != 'no-done':
                    self._chunk(b'data: [DONE]\n\n')
                self._end_chunks()

        self._httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self.port = self._httpd.server_address[1]
        self.base_url = f'http://127.0.0.1:{self.port}/v1'
        return self

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


if __name__ == '__main__':  # pragma: no cover - manual smoke run
    srv = MockBonsaiServer().start()
    print('mock endpoint:', srv.base_url)
    print('api key     :', srv.api_key)
    print('Press Ctrl-C to stop.')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
