#!/usr/bin/env python3
"""Offline tests for ``bonsai_chat --serve``: the local web UI and the /api proxy.

Everything runs against `mock_bonsai_server` as the upstream and stdlib `urllib` /
`http.client` as the client — no network, no GPU, no browser, no third-party HTTP
library. Security properties (the key never reaches a GET response, a header, or the
process output) each get their own test.

    python3 -m unittest -v test_web_server
"""
import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock as unittest_mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bonsai_chat as bc
from bonsai_chat import serve as serve_mod
from mock_bonsai_server import DEFAULT_KEY, MODEL, MockBonsaiServer

FULL_ANSWER_TAIL = 'en.cppreference.com'      # the tail of DEMO_ANSWER


def dead_endpoint():
    """A URL whose port nobody listens on: connection refused, deterministically."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return f'http://127.0.0.1:{port}/v1'


class ServeHarness(unittest.TestCase):
    """Mock upstream + ``--serve`` on port 0, in a thread, per test."""

    server_options = {}
    endpoint = None                 # None -> the mock; otherwise a fixed URL
    stall = None
    max_streams = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / 'config.json'
        self.mock = MockBonsaiServer(**self.server_options).start()
        self.addCleanup(self.mock.stop)
        endpoint = self.endpoint if self.endpoint is not None else self.mock.base_url
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        with contextlib.redirect_stdout(self.stdout), \
                contextlib.redirect_stderr(self.stderr):
            self.httpd = serve_mod.serve(
                '127.0.0.1', 0, endpoint, DEFAULT_KEY, MODEL,
                config_path=self.config_path, stall=self.stall,
                max_streams=self.max_streams)
        self.addCleanup(self._stop_httpd)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.httpd.server_port}'

    def _stop_httpd(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    # -- helpers ----------------------------------------------------------
    def get(self, path, timeout=15):
        try:
            with urllib.request.urlopen(self.base + path, timeout=timeout) as resp:
                return resp.status, dict(resp.headers), resp.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read().decode('utf-8', 'replace')

    def get_json(self, path):
        status, _, body = self.get(path)
        return status, json.loads(body)

    def post(self, path, payload, timeout=15):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode('utf-8', 'replace'))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode('utf-8', 'replace')
            try:
                body = json.loads(raw)
            except ValueError:
                body = {'error': raw}
            return exc.code, body

    def put(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='PUT')
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode('utf-8', 'replace'))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode('utf-8', 'replace')
            try:
                body = json.loads(raw)
            except ValueError:
                body = {'error': raw}
            return exc.code, body

    def chat_stream(self, payload, timeout=15):
        """POST /api/chat; returns the raw streaming response (caller closes)."""
        req = urllib.request.Request(
            self.base + '/api/chat', data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        return urllib.request.urlopen(req, timeout=timeout)

    def frames(self, resp, limit=None):
        """Yield (event, data) tuples from an SSE response, line by line."""
        out = []
        event = None
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode('utf-8', 'replace').rstrip('\n')
            if line.startswith('event: '):
                event = line[7:].strip()
            elif line.startswith('data: '):
                out.append((event, json.loads(line[6:])))
                event = None
                if limit and len(out) >= limit:
                    break
            elif line == '':
                continue
        return out

    def read_all_frames(self, resp, timeout=20):
        collected, event = [], None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                raw = resp.readline()
            except (TimeoutError, OSError):
                break
            if not raw:
                break
            line = raw.decode('utf-8', 'replace').rstrip('\n')
            if line.startswith('event: '):
                event = line[7:].strip()
            elif line.startswith('data: '):
                collected.append((event, json.loads(line[6:])))
                event = None
                if collected[-1][0] in ('done', 'error'):
                    break
        return collected

    def wait_for(self, condition, timeout=5, interval=0.05, message='condition'):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(interval)
        self.fail(f'timed out after {timeout}s waiting for {message}')


# ======================================================================
# Static files
# ======================================================================
class StaticTests(ServeHarness):

    def test_root_serves_the_index_html(self):
        status, headers, body = self.get('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers.get('Content-Type', ''))
        self.assertIn('<title>', body.lower())

    def test_ui_path_serves_files_from_dist(self):
        status, headers, body = self.get('/ui/index.html')
        self.assertEqual(status, 200)
        self.assertIn('<title>', body.lower())
        status2, _, body2 = self.get('/')
        self.assertEqual(body, body2, '/ and /ui/index.html must be the same page')

    def test_missing_asset_is_404(self):
        status, _, body = self.get('/ui/no-such-asset.js')
        self.assertEqual(status, 404)
        self.assertNotIn('Traceback', body)

    def test_path_traversal_is_blocked(self):
        for path in ('/ui/../../bonsai_chat/serve.py',
                     '/../bonsai_chat/serve.py',
                     '/ui/%2e%2e/%2e%2e/bonsai_chat/serve.py'):
            conn = HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=10)
            try:
                conn.request('GET', path)
                resp = conn.getresponse()
                body = resp.read().decode('utf-8', 'replace')
                self.assertEqual(resp.status, 404, path)
                self.assertNotIn('ThreadingHTTPServer', body, path)
                self.assertNotIn('def serve', body, path)
            finally:
                conn.close()

    def test_no_directory_listing(self):
        status, _, body = self.get('/ui/does-not-exist/')
        self.assertEqual(status, 404)
        self.assertNotIn('Index of', body)
        self.assertNotIn('directory', body.lower().replace('directory service', ''))

    def test_unknown_api_route_is_json_404(self):
        status, headers, body = self.get('/api/nope')
        self.assertEqual(status, 404)
        self.assertIn('json', headers.get('Content-Type', ''))
        self.assertIn('no such route', body)


# ======================================================================
# Health / models / capabilities / version
# ======================================================================
class InfoTests(ServeHarness):

    def test_health_returns_measured_latency(self):
        status, payload = self.get_json('/api/health')
        self.assertEqual(status, 200)
        self.assertTrue(payload['ok'])
        self.assertIsInstance(payload['latency_ms'], (int, float))
        self.assertGreaterEqual(payload['latency_ms'], 0)
        self.assertEqual(payload['model'], MODEL)
        self.assertIn(str(self.mock.port), payload['endpoint'])

    def test_health_dead_endpoint_is_null_latency_with_kind(self):
        httpd = serve_mod.serve('127.0.0.1', 0, dead_endpoint(), DEFAULT_KEY, MODEL,
                                config_path=self.config_path, stall=1.0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(
                    f'http://127.0.0.1:{httpd.server_port}/api/health',
                    timeout=15) as resp:
                payload = json.loads(resp.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
        self.assertFalse(payload['ok'])
        self.assertIsNone(payload['latency_ms'], 'a failed probe must not report a guess')
        self.assertEqual(payload['kind'], 'connection')
        self.assertIn('127.0.0.1', payload.get('error', ''))

    def test_health_without_endpoint(self):
        with unittest_mock.patch.dict(os.environ, {'BONSAI_BASE_URL': '',
                                                   'BONSAI_API_KEY': ''}):
            httpd = serve_mod.serve('127.0.0.1', 0, '', '',
                                    config_path=self.config_path)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                        f'http://127.0.0.1:{httpd.server_port}/api/health',
                        timeout=10) as resp:
                    payload = json.loads(resp.read())
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=5)
        self.assertFalse(payload['ok'])
        self.assertEqual(payload['kind'], 'connection')
        self.assertIn('no endpoint', payload['error'])
        self.assertIn('endpoint', payload)

    def test_models_returns_the_published_alias(self):
        status, payload = self.get_json('/api/models')
        self.assertEqual(status, 200)
        ids = [m.get('id') for m in payload.get('data', [])]
        self.assertIn(MODEL, ids)

    def test_capabilities_expose_facts_and_tristate_states(self):
        status, payload = self.get_json('/api/capabilities')
        self.assertEqual(status, 200)
        self.assertEqual(set(payload), {'fields', 'evidence', 'facts'})
        self.assertEqual(payload['facts'].get('context_window'), 8192)
        self.assertIs(payload['facts'].get('vision'), False,
                      'text-only mock must be reported as vision=False, not unknown')
        self.assertEqual(payload['facts'].get('vision_source'), '/props modalities')
        for name, state in payload['fields'].items():
            self.assertIn(state, ('supported', 'unsupported', 'unknown'),
                          f'{name} has a state outside the trichotomy')

    def test_version_matches_the_client(self):
        status, payload = self.get_json('/api/version')
        self.assertEqual(status, 200)
        self.assertEqual(payload['version'], bc.VERSION)


# ======================================================================
# Config round-trip
# ======================================================================
class ConfigTests(ServeHarness):

    def test_put_settings_round_trip_through_the_cli_config_file(self):
        status, payload = self.put('/api/config',
                                   {'settings': {'temperature': 0.4,
                                                 'max_tokens': 123}})
        self.assertEqual(status, 200)
        self.assertEqual(payload['settings']['temperature'], 0.4)
        self.assertEqual(payload['settings']['max_tokens'], 123)
        self.assertEqual(payload['provenance']['temperature'], 'config file')
        # GET sees exactly what PUT wrote
        status2, again = self.get_json('/api/config')
        self.assertEqual(status2, 200)
        self.assertEqual(again['settings']['temperature'], 0.4)
        self.assertEqual(again['settings']['max_tokens'], 123)
        # the CLI reads the same file
        data = json.loads(self.config_path.read_text())
        self.assertEqual(data['temperature'], 0.4)

    def test_config_file_written_through_the_api_is_0600(self):
        self.put('/api/config', {'settings': {'temperature': 0.5}})
        self.assertTrue(self.config_path.is_file())
        mode = self.config_path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600, 'the config file can hold keys — must be 0600')

    def test_config_never_returns_the_key(self):
        status, payload = self.get_json('/api/config')
        self.assertEqual(status, 200)
        self.assertTrue(payload['has_api_key'])
        self.assertNotIn('api_key', payload)
        self.assertNotIn(DEFAULT_KEY, json.dumps(payload))

    def test_config_rejects_bad_setting_types(self):
        status, payload = self.put('/api/config',
                                   {'settings': {'temperature': 'hot'}})
        self.assertEqual(status, 400)
        self.assertEqual(payload['kind'], 'bad_request')
        self.assertIn('temperature', payload['error'])
        self.assertFalse(self.config_path.exists(),
                         'an invalid PUT must not create a config file')

    def test_connect_updates_endpoint_in_memory(self):
        other = MockBonsaiServer().start()
        self.addCleanup(other.stop)
        status, payload = self.put('/api/config', {'endpoint': other.base_url,
                                                   'api_key': other.api_key})
        self.assertEqual(status, 200)
        self.assertEqual(payload['endpoint'], other.base_url)
        # the key itself is accepted but never echoed
        self.assertNotIn(other.api_key, json.dumps(payload))
        _, models = self.get_json('/api/models')
        self.assertEqual(models['data'][0]['id'], MODEL)
        status2, data = self.get_json('/api/config')
        self.assertNotIn(other.api_key, json.dumps(data))


# ======================================================================
# Security: the key never escapes the process
# ======================================================================
class SecurityTests(ServeHarness):

    GETS = ('/', '/ui/index.html', '/api/health', '/api/models',
            '/api/capabilities', '/api/version', '/api/config')

    def test_key_appears_in_no_get_response_body_or_header(self):
        for path in self.GETS:
            status, headers, body = self.get(path)
            self.assertNotIn(DEFAULT_KEY, body, f'key leaked in GET {path} body')
            joined = json.dumps(headers)
            self.assertNotIn(DEFAULT_KEY, joined, f'key leaked in GET {path} header')
            self.assertNotIn('Authorization', headers,
                             f'GET {path} must not carry an Authorization header')

    def test_key_absent_from_stdout_and_stderr_during_a_full_run(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.put('/api/config', {'endpoint': self.mock.base_url,
                                     'api_key': DEFAULT_KEY})
            self.get('/api/health')
            resp = self.chat_stream({'messages': [{'role': 'user', 'content': 'hi'}],
                                     'stream_id': 'leak-probe'})
            self.read_all_frames(resp)
            resp.close()
            self.get('/api/config')
        captured = out.getvalue() + err.getvalue()
        self.assertNotIn(DEFAULT_KEY, captured,
                         'the bearer key was written to the process output')
        self.assertNotIn('Authorization', captured)

    def test_bundle_contains_no_key(self):
        dist = serve_mod.DIST
        self.assertTrue(dist.is_dir(), 'dist must exist (npm run build)')
        offenders = []
        for path in sorted(dist.rglob('*')):
            if path.is_file():
                text = path.read_text(encoding='utf-8', errors='replace')
                if DEFAULT_KEY in text or 'mock-key' in text:
                    offenders.append(str(path.relative_to(dist)))
        self.assertEqual(offenders, [], f'committed bundle files contain a key: {offenders}')


# ======================================================================
# Streaming chat
# ======================================================================
class ChatTests(ServeHarness):

    def _chat_payload(self, stream_id='s1', **extra):
        payload = {'messages': [{'role': 'user', 'content': 'hello'}],
                   'stream_id': stream_id}
        payload.update(extra)
        return payload

    def test_delta_then_usage_then_done_with_real_counts(self):
        resp = self.chat_stream(self._chat_payload())
        frames = self.read_all_frames(resp)
        resp.close()
        events = [ev for ev, _ in frames]
        self.assertIn('delta', events)
        self.assertIn('usage', events)
        self.assertIn('done', events)
        self.assertLess(events.index('delta'), events.index('usage'))
        self.assertLess(events.index('usage'), events.index('done'))
        usage = next(d for ev, d in frames if ev == 'usage')
        self.assertEqual(usage['usage'],
                         {'prompt_tokens': 42, 'completion_tokens': 64,
                          'total_tokens': 106})
        done = next(d for ev, d in frames if ev == 'done')
        self.assertEqual(done['usage'], usage['usage'])
        self.assertEqual(done['finish_reason'], 'stop')
        self.assertFalse(done['stopped'])
        self.assertEqual(done['stream_id'], 's1')
        deltas = ''.join(d['text'] for ev, d in frames if ev == 'delta')
        self.assertTrue(deltas, 'no content deltas arrived')

    def test_reasoning_frames_are_separate_from_content(self):
        resp = self.chat_stream(self._chat_payload(stream_id='s2'))
        frames = self.read_all_frames(resp)
        resp.close()
        reasoning = ''.join(d['text'] for ev, d in frames if ev == 'reasoning')
        content = ''.join(d['text'] for ev, d in frames if ev == 'delta')
        self.assertTrue(reasoning, 'no reasoning frames arrived')
        self.assertIn('Reasoning', reasoning)
        self.assertNotIn('Reasoning', content,
                         'reasoning must never be merged into the answer')
        done = next(d for ev, d in frames if ev == 'done')
        self.assertIn('Reasoning', done['message'].get('reasoning_content', ''))

    def test_sampling_and_thinking_settings_reach_the_upstream(self):
        resp = self.chat_stream(self._chat_payload(
            stream_id='s3',
            sampling={'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
                      'min_p': 0.0, 'presence_penalty': 1.5, 'max_tokens': 64},
            thinking={'effort': 'xhigh', 'budget_tokens': 2048}))
        frames = self.read_all_frames(resp)
        resp.close()
        self.assertTrue(any(ev == 'done' for ev, _ in frames))
        sent = self.mock.calls[-1]
        self.assertEqual(sent['temperature'], 0.7)
        self.assertEqual(sent['top_p'], 0.8)
        self.assertEqual(sent['top_k'], 20)
        self.assertEqual(sent['min_p'], 0.0)
        self.assertEqual(sent['presence_penalty'], 1.5)
        self.assertEqual(sent['reasoning_effort'], 'xhigh')
        self.assertEqual(sent['thinking_budget_tokens'], 2048)

    def test_malformed_body_is_bad_request(self):
        conn = HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=10)
        try:
            conn.request('POST', '/api/chat', body='{not json',
                         headers={'Content-Type': 'application/json'})
            resp = conn.getresponse()
            body = json.loads(resp.read())
            self.assertEqual(resp.status, 400)
            self.assertEqual(body['kind'], 'bad_request')
        finally:
            conn.close()

    def test_missing_messages_is_bad_request(self):
        status, payload = self.post('/api/chat', {'stream_id': 'x'})
        self.assertEqual(status, 400)
        self.assertEqual(payload['kind'], 'bad_request')

    def test_bad_sampling_type_is_bad_request(self):
        status, payload = self.post('/api/chat', {
            'messages': [{'role': 'user', 'content': 'x'}],
            'sampling': {'temperature': 'hot'}})
        self.assertEqual(status, 400)
        self.assertIn('sampling.temperature', payload['error'])


# ======================================================================
# Stop, disconnect, concurrency
# ======================================================================
class StopTests(ServeHarness):

    def test_stop_cancels_midflight_and_retains_the_partial_answer(self):
        self.mock.scenario = 'slow-stream'
        resp = self.chat_stream({'messages': [{'role': 'user', 'content': 'story'}],
                                 'stream_id': 'stop-1'})
        seen = []
        event = None
        # read until the first content delta, then stop
        while True:
            raw = resp.readline()
            self.assertTrue(raw, 'stream ended before the first delta')
            line = raw.decode().rstrip('\n')
            if line.startswith('event: '):
                event = line[7:].strip()
            elif line.startswith('data: '):
                data = json.loads(line[6:])
                if event == 'delta':
                    seen.append(data['text'])
                    break
        status, stop = self.post('/api/stop', {'stream_id': 'stop-1'})
        self.assertEqual(status, 200)
        self.assertTrue(stop['ok'])
        self.assertTrue(stop['found'])
        frames = self.read_all_frames(resp)
        resp.close()
        done = next(d for ev, d in frames if ev == 'done')
        self.assertTrue(done['stopped'], 'a stopped turn must be marked stopped')
        partial = done['message'].get('content') or ''
        self.assertTrue(partial, 'the partial answer must be retained, not discarded')
        self.assertTrue(partial.startswith(seen[0]),
                        'the retained partial must include what had already streamed')
        self.assertNotIn(FULL_ANSWER_TAIL, partial,
                         'the stream must have been cut before the end')
        self.assertIsInstance(done.get('tokens'), int,
                              'a stopped turn needs a measured token count')
        self.assertGreater(done['tokens'], 0)
        # the registry drained: no orphan stream ids
        self.assertNotIn('stop-1', self.httpd.state.streams)
        # and the upstream connection was released (same seam the CLI tests use)
        self.wait_for(lambda: not self.httpd.state.client.transport._stream_outstanding,
                      message='transport release after stop')

    def test_stop_is_idempotent(self):
        status, payload = self.post('/api/stop', {'stream_id': 'never-existed'})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {'ok': True, 'found': False})
        status2, payload2 = self.post('/api/stop', {'stream_id': 'never-existed'})
        self.assertEqual(status2, 200)
        self.assertEqual(payload2, {'ok': True, 'found': False})

    def test_stop_requires_a_stream_id(self):
        status, payload = self.post('/api/stop', {})
        self.assertEqual(status, 400)
        self.assertEqual(payload['kind'], 'bad_request')

    def test_client_disconnect_stops_the_upstream_stream(self):
        self.mock.scenario = 'slow-stream'
        conn = HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=10)
        body = json.dumps({'messages': [{'role': 'user', 'content': 'story'}],
                           'stream_id': 'gone-1'})
        conn.request('POST', '/api/chat', body=body,
                     headers={'Content-Type': 'application/json',
                              'Content-Length': str(len(body))})
        resp = conn.getresponse()
        # read two frames so the stream is definitely in flight upstream
        read = 0
        while read < 2:
            line = resp.readline()
            if not line:
                break
            if line.startswith(b'event: '):
                read += 1
        self.assertEqual(read, 2)
        self.assertIn('gone-1', self.httpd.state.streams)
        conn.close()                         # the browser tab goes away
        self.wait_for(lambda: 'gone-1' not in self.httpd.state.streams,
                      message='registry drain after disconnect')
        self.wait_for(
            lambda: not self.httpd.state.client.transport._stream_outstanding,
            message='upstream stop() after client disconnect')

    def test_duplicate_stream_id_is_refused_while_active(self):
        self.mock.scenario = 'slow-stream'
        resp = self.chat_stream({'messages': [{'role': 'user', 'content': 'x'}],
                                 'stream_id': 'dup-1'})
        self.wait_for(lambda: 'dup-1' in self.httpd.state.streams,
                      message='stream registration')
        status, payload = self.post('/api/chat', {
            'messages': [{'role': 'user', 'content': 'y'}], 'stream_id': 'dup-1'})
        self.assertEqual(status, 409)
        self.assertEqual(payload['kind'], 'busy')
        resp.close()                         # disconnect path cleans the first up
        self.wait_for(lambda: 'dup-1' not in self.httpd.state.streams,
                      message='registry drain')


class ConcurrencyCapTests(ServeHarness):
    max_streams = 1

    def test_second_concurrent_stream_is_refused_not_queued(self):
        self.mock.scenario = 'slow-stream'
        resp = self.chat_stream({'messages': [{'role': 'user', 'content': 'a'}],
                                 'stream_id': 'cap-1'})
        self.wait_for(lambda: 'cap-1' in self.httpd.state.streams,
                      message='first stream registration')
        status, payload = self.post('/api/chat', {
            'messages': [{'role': 'user', 'content': 'b'}], 'stream_id': 'cap-2'})
        self.assertEqual(status, 429)
        self.assertEqual(payload['kind'], 'busy')
        self.assertIn('never queued', payload['error'])
        self.assertEqual(payload['limit'], 1)
        resp.close()
        self.wait_for(lambda: 'cap-1' not in self.httpd.state.streams,
                      message='registry drain')


# ======================================================================
# Error kinds
# ======================================================================
class ConnectionErrorTests(ServeHarness):
    endpoint = None                        # replaced in setUp with a dead URL

    def setUp(self):
        self.dead = dead_endpoint()
        self.endpoint = self.dead
        super().setUp()

    def test_connection_refused_maps_to_kind_connection(self):
        status, payload = self.post('/api/chat', {
            'messages': [{'role': 'user', 'content': 'x'}], 'stream_id': 'e1'})
        self.assertEqual(status, 502)
        self.assertEqual(payload['kind'], 'connection')


class UnauthorizedErrorTests(ServeHarness):

    def test_bad_key_maps_to_kind_unauthorized(self):
        self.put('/api/config', {'api_key': 'wrong-key-000'})
        status, payload = self.post('/api/chat', {
            'messages': [{'role': 'user', 'content': 'x'}], 'stream_id': 'e2'})
        self.assertEqual(status, 401)
        self.assertEqual(payload['kind'], 'unauthorized')
        self.assertNotIn(DEFAULT_KEY, json.dumps(payload))


class RemoteErrorTests(ServeHarness):
    server_options = {'scenario': 'http-500', 'sticky_scenario': True}

    def test_upstream_500_maps_to_kind_remote_with_status(self):
        status, payload = self.post('/api/chat', {
            'messages': [{'role': 'user', 'content': 'x'}], 'stream_id': 'e3'})
        self.assertEqual(status, 502)
        self.assertEqual(payload['kind'], 'remote')
        self.assertEqual(payload.get('status'), 500)


class MidStreamErrorTests(ServeHarness):

    def test_disconnect_midstream_maps_to_kind_mid_stream(self):
        self.mock.scenario = 'disconnect'
        resp = self.chat_stream({'messages': [{'role': 'user', 'content': 'x'}],
                                 'stream_id': 'e4'})
        frames = self.read_all_frames(resp)
        resp.close()
        kinds = [ev for ev, _ in frames]
        self.assertIn('error', kinds)
        self.assertNotIn('done', kinds, 'a broken stream must not look complete')
        error = next(d for ev, d in frames if ev == 'error')
        self.assertEqual(error['kind'], 'mid_stream')
        self.assertGreaterEqual(error['tokens_arrived'], 1,
                                'the UI must be told how much arrived before failure')
        deltas_before = sum(1 for ev, _ in frames
                            if ev == 'delta')
        self.assertGreaterEqual(error['tokens_arrived'], deltas_before - 1)


class StallErrorTests(ServeHarness):
    stall = 0.4

    def test_no_bytes_for_the_stall_window_maps_to_kind_stall(self):
        self.mock.scenario = 'timeout'      # mock goes silent for 5 seconds
        started = time.monotonic()
        resp = self.chat_stream({'messages': [{'role': 'user', 'content': 'x'}],
                                 'stream_id': 'e5'})
        frames = self.read_all_frames(resp, timeout=10)
        resp.close()
        elapsed = time.monotonic() - started
        error = next((d for ev, d in frames if ev == 'error'), None)
        self.assertIsNotNone(error, f'expected a stall error, got {frames}')
        self.assertEqual(error['kind'], 'stall')
        self.assertGreaterEqual(error['waited_seconds'], 0.3)
        self.assertLess(elapsed, 4.0,
                        'the watchdog must fire before the mock wakes up')
        self.assertIn('second', error['error'],
                      'the stall message must say how long it waited')


# ======================================================================
# Missing build output, doctor integration, output hygiene
# ======================================================================
class BuildAndDoctorTests(ServeHarness):

    def test_missing_dist_gives_a_build_instruction_not_a_traceback(self):
        absent = Path(self.tmp.name) / 'no-dist'
        with unittest_mock.patch.object(serve_mod, 'DIST', absent):
            with self.assertRaises(RuntimeError) as caught:
                serve_mod.serve('127.0.0.1', 0, self.mock.base_url, DEFAULT_KEY,
                                MODEL, config_path=self.config_path)
        message = str(caught.exception)
        self.assertIn('npm install && npm run build', message)
        self.assertIn('bonsai_chat/webui', message)
        self.assertNotIn('Traceback', message)

    def test_doctor_keeps_the_web_ui_out_of_its_default_checks(self):
        import bonsai_chat.doctor as doctor_mod
        client = bc.BonsaiClient(self.mock.base_url, DEFAULT_KEY, retries=1)
        out = io.StringIO()
        doctor_mod.run_doctor(client, out=out, as_json=True)
        printed = json.loads(out.getvalue())
        self.assertNotIn('local web UI', printed['checks'],
                         'the web server must stay out of default checks')
        self.assertNotIn('serve', json.dumps(list(printed['checks'])))

    def test_doctor_checks_the_web_ui_only_when_pointed_at_it(self):
        import bonsai_chat.doctor as doctor_mod
        client = bc.BonsaiClient(self.mock.base_url, DEFAULT_KEY, retries=1)
        self.addCleanup(client.close)
        out = io.StringIO()
        doctor_mod.run_doctor(client, out=out, as_json=True, serve_url=self.base)
        printed = json.loads(out.getvalue())
        self.assertIn('local web UI', printed['checks'])
        self.assertEqual(printed['checks']['local web UI']['state'], 'PASS')

        # and FAILs honestly when the URL stops answering
        out2 = io.StringIO()
        doctor_mod.run_doctor(client, out=out2, as_json=True,
                              serve_url=dead_endpoint())
        printed2 = json.loads(out2.getvalue())
        self.assertEqual(printed2['checks']['local web UI']['state'], 'FAIL')


if __name__ == '__main__':                   # pragma: no cover
    unittest.main(verbosity=2)
