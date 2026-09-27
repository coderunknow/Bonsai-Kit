#!/usr/bin/env python3
"""Evidence runner: a real `bonsai_chat.py --serve` against the mock, exercised end to end.

Starts `mock_bonsai_server` in-process (plus a tiny admin port so the UI test can arm
failure scenarios mid-session), launches the actual CLI:

    python3 bonsai_chat.py --serve --base-url ... --api-key ... --no-browser --serve-port 0

then performs a curl-level transcript (connect -> stream -> stop -> error kinds) and,
if npm/vitest are available, the gated live React session in
`webui/tests/live-session.test.tsx`. Everything is loopback; no GPU, no tunnel.

Usage:
    python3 evidence/live_session.py            # full evidence run
    python3 evidence/live_session.py --keep     # leave servers up and print the URL
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mock_bonsai_server import DEFAULT_KEY, MODEL, MockBonsaiServer  # noqa: E402


def log(title, body=''):
    print(f"\n===== {title} =====")
    if body:
        print(body.rstrip())


def http_json(url, payload=None, method=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ('POST' if data else 'GET'),
        headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', 'replace')
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode('utf-8', 'replace')
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, raw


def sse_transcript(base, payload, read_frames=None, stop_after=None):
    """POST /api/chat and return a readable transcript of the SSE frames."""
    req = urllib.request.Request(
        base + '/api/chat', data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    resp = urllib.request.urlopen(req, timeout=30)
    lines, event, content, stopped_at = [], None, '', None
    while True:
        raw = resp.readline()
        if not raw:
            lines.append('[connection closed]')
            break
        text = raw.decode('utf-8', 'replace').rstrip('\n')
        if text.startswith('event: '):
            event = text[7:]
        elif text.startswith('data: '):
            data = json.loads(text[6:])
            if event == 'delta':
                content += data.get('text', '')
            lines.append(f'{event}: {json.dumps(data)[:160]}')
            if stop_after and content and len(content) >= stop_after and stopped_at is None:
                stop_payload = json.dumps({'stream_id': payload.get('stream_id')}).encode()
                stop_req = urllib.request.Request(
                    base + '/api/stop', data=stop_payload,
                    headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(stop_req, timeout=10) as s:
                    lines.append(f'[browser pressed Stop -> {s.read().decode().strip()}]')
                stopped_at = len(content)
            if event in ('done', 'error'):
                break
    resp.close()
    return '\n'.join(lines), content


def main():
    keep = '--keep' in sys.argv
    mock = MockBonsaiServer().start()

    # Tiny admin server so the browser-side test can arm mock scenarios mid-session.
    class Admin(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(n) or b'{}')
            name = body.get('scenario')
            if name == 'ok':
                mock.scenario = mock.default_scenario = 'ok'
            elif name == 'disconnect-once':
                mock.scenario = 'disconnect'   # next chat only; then back to default
            elif name == 'slow-sticky':
                mock.set_scenario('slow-stream', sticky=True)
            else:
                self.send_error(400, f'unknown scenario {name}')
                return
            raw = json.dumps({'ok': True, 'scenario': mock.scenario}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    admin = ThreadingHTTPServer(('127.0.0.1', 0), Admin)
    threading.Thread(target=admin.serve_forever, daemon=True).start()

    env = dict(os.environ, PYTHONUNBUFFERED='1')
    proc = subprocess.Popen(
        [sys.executable, 'bonsai_chat.py', '--serve',
         '--base-url', mock.base_url, '--api-key', DEFAULT_KEY,
         '--serve-port', '0', '--no-browser'],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = None
    deadline = time.time() + 15
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        print(f'[serve] {line.rstrip()}')
        m = re.search(r'http://\S+', line)
        if m:
            base = m.group(0).rstrip('/')
            break
    if not base:
        proc.kill()
        raise SystemExit('--serve did not print a URL')
    admin_port = admin.server_address[1]
    print(f'[admin] scenario control on http://127.0.0.1:{admin_port}')

    try:
        # ---------- curl-level transcript ----------
        status, health = http_json(base + '/api/health')
        log('GET /api/health', f'{status} {json.dumps(health)}')
        status, models = http_json(base + '/api/models')
        log('GET /api/models', f'{status} {json.dumps(models)}')

        status, cfg = http_json(base + '/api/config')
        leaked = DEFAULT_KEY in json.dumps(cfg)
        log('GET /api/config',
            f'{status} key-in-body={leaked} has_api_key={cfg.get("has_api_key")} '
            f'provenance.temperature={cfg.get("provenance", {}).get("temperature")}')

        # a full stream, stopped mid-answer by /api/stop
        http_json(f'http://127.0.0.1:{admin_port}/', payload={'scenario': 'slow-sticky'})
        transcript, content = sse_transcript(
            base, {'messages': [{'role': 'user', 'content': 'Write a long story.'}],
                   'stream_id': 'curl-stop'},
            stop_after=40)
        log('POST /api/chat (slow) + POST /api/stop mid-flight', transcript)
        log('stopped partial (chars)', str(len(content)))

        # retry-worthy failure: next chat dies mid-stream, the one after succeeds
        http_json(f'http://127.0.0.1:{admin_port}/', payload={'scenario': 'disconnect-once'})
        transcript, _ = sse_transcript(
            base, {'messages': [{'role': 'user', 'content': 'This one breaks.'}],
                   'stream_id': 'curl-break'})
        log('POST /api/chat with scenario=disconnect (mid-stream failure)', transcript)
        http_json(f'http://127.0.0.1:{admin_port}/', payload={'scenario': 'ok'})
        transcript, _ = sse_transcript(
            base, {'messages': [{'role': 'user', 'content': 'This one works.'}],
                   'stream_id': 'curl-retry'})
        log('POST /api/chat after the failure (the retry)', transcript[:600])

        # ---------- live React session (jsdom, real fetch) ----------
        webui = ROOT / 'bonsai_chat' / 'webui'
        env2 = dict(env, BONSAI_LIVE_BASE=base, BONSAI_LIVE_ADMIN=f'http://127.0.0.1:{admin_port}',
                    BONSAI_LIVE_KEY=DEFAULT_KEY)
        run = subprocess.run(['npx', 'vitest', 'run', 'tests/live-session.test.tsx'],
                             cwd=webui, env=env2, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        log('vitest tests/live-session.test.tsx (live React session)', run.stdout)

        if keep:
            print('\n--keep: servers stay up. Press Enter to shut down.')
            try:
                input()
            except EOFError:
                time.sleep(3600)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        admin.shutdown()
        admin.server_close()
        mock.stop()


if __name__ == '__main__':
    main()
