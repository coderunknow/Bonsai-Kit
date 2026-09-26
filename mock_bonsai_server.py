#!/usr/bin/env python3
"""
mock_bonsai_server.py — an offline stub of the OpenAI-compatible endpoint that
colab_kaggle_cell.py deploys.

This is NOT a model. It runs no inference and needs no GPU: it speaks the same wire
protocol as the PrismML llama.cpp fork (chat completions, SSE streaming with
`reasoning_content` and split `tool_calls`, /props, /tokenize, bearer auth, a text-only
server that rejects `image_url`) with scripted replies, so bonsai_chat.py can be exercised
end to end and the test-suite can assert on real HTTP traffic.

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


def _sse(payload) -> bytes:
    return ('data: ' + json.dumps(payload) + '\n\n').encode()


class MockBonsaiServer:
    """Threaded stub server. `start()` returns self; `.base_url` is ready to use."""

    def __init__(self, api_key=DEFAULT_KEY, vision=False, context=8192,
                 reject_effort_once=False, reject_tools_once=False,
                 announce_modalities=True):
        self.api_key = api_key
        self.vision = vision
        self.announce_modalities = announce_modalities
        self.context = context
        self.reject_effort_once = reject_effort_once
        self.reject_tools_once = reject_tools_once
        self.calls = []            # recorded request bodies, for assertions
        self.get_paths = []        # recorded GET paths, for assertions
        self.chat_requests = 0
        self._httpd = None
        self._thread = None

    # ------------------------------------------------------------------
    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            server_version = 'MockBonsai/0.3'

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
                if not self._authorized():
                    return self._send(401, {'error': {'message': 'invalid api key'}})
                body = self._body()
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
                if outer.reject_effort_once and 'reasoning_effort' in body:
                    outer.reject_effort_once = False
                    return self._send(400, {'error': {
                        'message': "unknown field 'reasoning_effort'"}})
                if outer.reject_tools_once and body.get('tools'):
                    outer.reject_tools_once = False
                    return self._send(400, {'error': {
                        'message': 'this server was built without --jinja; tools are unsupported'}})
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
                if not tool_results and tools and 'weather' in last_user.lower():
                    return self._respond(body, stream, tool_call=True)
                return self._respond(body, stream, tool_call=False,
                                     tool_output=tool_results[-1] if tool_results else None)

            def _respond(self, body, stream, tool_call, tool_output=None):
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
                    return self._send(200, {'id': 'mock-1', 'object': 'chat.completion',
                                            'model': MODEL, 'choices':
                                            [{'index': 0, 'message': msg, 'finish_reason': finish}],
                                            'usage': {'prompt_tokens': 42, 'completion_tokens': 64,
                                                      'total_tokens': 106}})
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                try:
                    self._stream(body, tool_call, answer=answer)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _chunk(self, data: bytes):
                self.wfile.write(b'%x\r\n%s\r\n' % (len(data), data))
                self.wfile.flush()

            def _stream(self, body, tool_call, answer=DEMO_ANSWER):
                for piece in ('Reasoning', ' about', ' the', ' question.'):
                    self._chunk(_sse({'choices': [{'index': 0, 'delta':
                                                   {'reasoning_content': piece + ' '}}]}))
                    time.sleep(0)
                if tool_call:
                    for frag in WEATHER_CALL:
                        self._chunk(_sse({'choices': [{'index': 0, 'delta':
                                                       {'tool_calls': [frag]}}]}))
                    for frag in WEATHER_CALL_TAIL:
                        self._chunk(_sse({'choices': [{'index': 0, 'delta':
                                                       {'tool_calls': [frag]}}]}))
                    self._chunk(_sse({'choices': [{'index': 0, 'delta': {},
                                                   'finish_reason': 'tool_calls'}]}))
                else:
                    for piece in answer.split(' '):
                        self._chunk(_sse({'choices': [{'index': 0, 'delta':
                                                       {'content': piece + ' '}}]}))
                    self._chunk(_sse({'choices': [{'index': 0, 'delta': {},
                                                   'finish_reason': 'stop'}]}))
                self._chunk(_sse({'choices': [], 'usage': {'prompt_tokens': 42,
                                                           'completion_tokens': 64,
                                                           'total_tokens': 106},
                                  'timings': {'prompt_n': 42, 'predicted_n': 64,
                                              'predicted_ms': 3200,
                                              'tokens_per_second': 20.0}}))
                self._chunk(b'data: [DONE]\n\n')
                self._chunk(b'0\r\n\r\n')

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
