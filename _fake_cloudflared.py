#!/usr/bin/env python3
"""Fake cloudflared Quick Tunnel.

Publishes a ``https://<something>.trycloudflare.com`` URL into its log — which is the
only way the real one announces it — and backs that URL with a real loopback TCP proxy
to the origin, so "public" requests in the offline harness are real HTTP requests that
really can fail when the tunnel dies. Can be told to publish late, never publish, or die
on cue (FAKE_TUNNEL_CONFIG).
"""

import os
import random
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path


class _Pipe(socketserver.BaseRequestHandler):
    """Raw byte pipe; `self.request` is the client socket for this handler."""

    def handle(self):
        try:
            upstream = socket.create_connection(self.server.target, timeout=30)
        except OSError:
            return
        upstream.settimeout(60)
        self.request.settimeout(60)
        done = threading.Event()

        def move(src, dst):
            try:
                while True:
                    data = src.recv(65536)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            done.set()

        a = threading.Thread(target=move, args=(self.request, upstream), daemon=True)
        b = threading.Thread(target=move, args=(upstream, self.request), daemon=True)
        a.start()
        b.start()
        a.join()
        b.join()
        for s in (upstream, self.request):
            try:
                s.close()
            except OSError:
                pass


class _Proxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    disable_nagle_algorithm = True

    def __init__(self, addr, target):
        self.target = target
        super().__init__(addr, _Pipe)


def main():
    import json
    argv = sys.argv[1:]
    cfg = {}
    try:
        cfg = json.loads(Path(os.environ['FAKE_TUNNEL_CONFIG']).read_text())
    except Exception:
        pass
    target = None
    for i, a in enumerate(argv):
        if a == '--url' and i + 1 < len(argv):
            target = argv[i + 1]
    if not target:
        sys.stderr.write('ERR --url is required\n')
        return 2
    host, _, port = target.rpartition(':')
    target_addr = (host.replace('http://', '').replace('https://', '') or '127.0.0.1',
                   int(port))

    run_dir = Path(os.environ.get('FAKE_RUN_DIR')
                 or Path(os.environ.get('FAKE_GPU_DIR', '/tmp/fake-gpu')).parent)
    if cfg.get('fail_immediately'):
        sys.stderr.write('ERR Unable to reach the Cloudflare edge network\n')
        return 1
    if cfg.get('no_url'):
        sys.stdout.write('INF Starting Hello World server\n')
        sys.stdout.flush()
        time.sleep(float(cfg.get('lifetime', 1.0)))
        return 1

    proxy = _Proxy(('127.0.0.1', 0), target_addr)
    proxy_port = proxy.server_address[1]
    url = 'https://bonsai-%08x.trycloudflare.com' % random.getrandbits(32)

    def serve():
        proxy.serve_forever(poll_interval=0.05)

    threading.Thread(target=serve, daemon=True).start()

    if cfg.get('publish_delay'):
        time.sleep(float(cfg['publish_delay']))
    sys.stdout.write('INF Thank you for trying Cloudflare Tunnel.\n')
    sys.stdout.write('INF |  %s  |\n' % url)
    sys.stdout.write('INF Your quick Tunnel has been created! Visit it at:\n')
    sys.stdout.write('INF %s\n' % url)
    sys.stdout.flush()
    try:
        map_path = run_dir / 'tunnel-map.json'
        mapping = {}
        if map_path.is_file():
            try:
                mapping = json.loads(map_path.read_text())
            except ValueError:
                mapping = {}
        mapping[url] = proxy_port
        tmp = map_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(mapping))
        os.replace(tmp, map_path)
    except OSError as e:
        sys.stderr.write('ERR could not write tunnel map: %s\n' % e)

    lifetime = float(cfg.get('lifetime', 0) or 0)
    if lifetime:
        deadline = time.monotonic() + lifetime
        while time.monotonic() < deadline:
            time.sleep(0.02)
        sys.stdout.write('ERR tunnel disconnected\n')
        sys.stdout.flush()
        try:
            proxy.shutdown()
        except Exception:
            pass
        return 1
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
