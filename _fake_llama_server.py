#!/usr/bin/env python3
"""Fake PrismML llama-server (behaviour half).

The executable the cell starts is a compiled launcher (see cell_harness.LAUNCHER_C) so
that /proc/<pid>/exe and argv[0] really are ``bin/cuda/llama-server``; this file is what
that launcher runs. It honours the flags the cell passes, writes a startup log in
llama.cpp's own format (including the ``KV buffer size = … MiB`` lines the cell measures
the per-token KV cost from), serves /health, /props, /slots, /v1/models and
/v1/chat/completions (streaming and not), requires the bearer key on every route, and
can be told — through FAKE_SERVER_CONFIG — to boot slowly, OOM above a given context,
crash after N requests, hang mid-generation, reject a flag, or advertise extra flags.

No network, no GPU, no model: replies are canned and deterministic.
"""

import json
import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Flags whose value may legitimately start with '-' (a urlsafe API key can). Parsing
# them with the usual "the next token does not start with a dash" rule would silently
# leave the server running with NO api key, which is exactly the failure an auth test
# must be able to see.
ALWAYS_VALUE_FLAGS = {'--api-key', '--alias', '--chat-template-kwargs',
                      '--chat-template', '--mmproj'}

VALUE_FLAGS = {
    '-m', '--model', '--alias', '--host', '--port', '--api-key', '-c', '--ctx-size',
    '-ngl', '--n-gpu-layers', '--split-mode', '--main-gpu', '-t', '--threads',
    '--parallel', '-np', '--chat-template-kwargs', '--temp', '--top-p', '--top-k',
    '--min-p', '-ub', '--ubatch-size', '-b', '--batch-size', '--cache-type-k',
    '--cache-ram', '--reasoning-budget', '--reasoning-format', '--mmproj',
    '--image-max-tokens', '--slot-save-path', '-fa', '--flash-attn',
}

BASE_HELP = """\
usage: llama-server [options]

options:
  -h, --help                 show this help message and exit
  --version                  show version and build info
  -m, --model FNAME          model path
  --alias STRING             set alias for model name (to be used by REST API)
  --host IP                  ip address to listen (default: 127.0.0.1)
  --port PORT                port to listen (default: 8080)
  --api-key KEY              API key to enable authentication
  -c, --ctx-size N           size of the prompt context
  -ngl, --n-gpu-layers N     number of layers to store in VRAM
  --split-mode MODE          how to split the model across GPUs: none|layer
  --main-gpu N               the GPU used for scratch and small tensors
  -fa, --flash-attn          enable Flash Attention
  -t, --threads N            number of threads to use during generation
  --parallel N               number of parallel slots
  --jinja                    use the Jinja chat template (native tool calling)
  --chat-template-kwargs JSON  extra kwargs for the chat template
  --temp N                   temperature
  --top-p N                  top-p sampling
  --top-k N                  top-k sampling
  --min-p N                  min-p sampling
  --mmproj FNAME             path to a multimodal projector
  --image-max-tokens N       downscale images to this many tokens
  --slot-save-path PATH      where to save prompt-cache state
  --cache-type-k TYPE        KV cache data type for the K tensor
"""


def load_config():
    path = os.environ.get('FAKE_SERVER_CONFIG')
    try:
        return json.loads(Path(path).read_text()) if path else {}
    except Exception:
        return {}


def parse_args(argv):
    opts, flags = {}, []
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith('-'):
            takes_value = tok in ALWAYS_VALUE_FLAGS or (
                tok in VALUE_FLAGS and i + 1 < len(argv)
                and not argv[i + 1].startswith('-'))
            if takes_value and i + 1 < len(argv):
                opts[tok] = argv[i + 1]
                i += 2
                continue
            flags.append(tok)
        i += 1
    return opts, flags


def first(opts, *names, default=None):
    for n in names:
        if n in opts:
            return opts[n]
    return default


class FakeState:
    """Cross-process bookkeeping: which servers hold VRAM, and how many requests
    this server has served (for the "crash after N" scenario)."""

    def __init__(self):
        gpu_dir = Path(os.environ.get('FAKE_GPU_DIR', '/tmp/fake-gpu'))
        self.run_dir = gpu_dir.parent
        self.gpu_dir = gpu_dir
        self.invocations = self.run_dir / 'invocations.jsonl'
        self.counter = self.run_dir / 'reqcount'
        self.pid = os.getppid()          # the launcher owns the PID the cell sees

    def record_start(self, argv, opts):
        try:
            with self.invocations.open('a') as f:
                f.write(json.dumps({'pid': self.pid, 'argv': argv,
                                    'ctx': int(first(opts, '-c', '--ctx-size', default='0')
                                               or 0)}) + '\n')
        except OSError:
            pass

    def claim_vram(self, per_gpu):
        entry = {'pid': self.pid, 'per_gpu': {str(k): int(v) for k, v in per_gpu.items()}}
        target = self.gpu_dir / 'apps' / ('%d.json' % self.pid)
        try:
            target.write_text(json.dumps(entry))
        except OSError:
            pass
        self._app_file = target
        return target

    def release_vram(self):
        try:
            getattr(self, '_app_file', None) and self._app_file.unlink()
        except OSError:
            pass

    def bump_requests(self):
        n = 0
        try:
            n = int(self.counter.read_text() or 0)
        except (OSError, ValueError):
            pass
        n += 1
        try:
            self.counter.write_text(str(n))
        except OSError:
            pass
        return n


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'FakePrismML/1.0'

    # -- plumbing ----------------------------------------------------
    def log_message(self, fmt, *args):
        sys.stderr.write('[fake-llama] ' + (fmt % args) + '\n')

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self):
        key = self.server.api_key
        if not key:
            return True
        return self.headers.get('Authorization', '') == 'Bearer ' + key

    def _require_auth(self):
        if not self._authorized():
            self._json(401, {'error': {'message': 'invalid api key', 'code': 401}})
            return False
        return True

    def _read_body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except ValueError:
            return {}

    # -- routes ------------------------------------------------------
    def do_GET(self):
        if not self._require_auth():
            return
        path = self.path.split('?')[0]
        if path in ('/health', '/healthz'):
            self._json(200, {'status': 'ok', 'slots_idle': self.server.slots,
                             'slots_processing': 0})
        elif path == '/slots':
            self._json(200, [{'id': i, 'model': self.server.alias, 'state': 0,
                              'n_ctx': self.server.ctx,
                              'cache_tokens': 0} for i in range(self.server.slots)])
        elif path == '/props':
            self._json(200, {
                'model_path': self.server.model_path,
                'model_alias': self.server.alias,
                'n_ctx': self.server.ctx,
                'total_slots': self.server.slots,
                'modalities': {'text': True, 'vision': self.server.vision},
                'default_generation_settings': {
                    'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.05},
                'build': {'stamp': self.server.stamp, 'fork': 'PrismML'},
            })
        elif path == '/v1/models':
            self._json(200, {'object': 'list', 'data': [
                {'id': self.server.alias, 'object': 'model', 'owned_by': 'prism-ml'}]})
        elif path == '/metrics':
            self._json(200, {'prompt_tokens_total': 0, 'tokens_predicted_total': 0})
        else:
            self._json(404, {'error': {'message': 'not found: ' + path, 'code': 404}})

    def do_POST(self):
        if not self._require_auth():
            return
        path = self.path.split('?')[0]
        body = self._read_body()
        n = self.server.state.bump_requests()
        if self.server.crash_after and n >= self.server.crash_after:
            def die():
                time.sleep(0.05)
                self.server.shutdown()
                os._exit(3)
            threading.Thread(target=die, daemon=True).start()
            self._json(500, {'error': {'message': 'simulated crash', 'code': 500}})
            return
        if path == '/v1/chat/completions':
            return self._chat(body)
        if path in ('/tokenize', '/v1/tokenize'):
            text = body.get('content') or body.get('input') or ''
            toks = [hash(t) % 30000 for t in text.split()]
            self._json(200, {'tokens': toks})
            return
        self._json(404, {'error': {'message': 'not found: ' + path, 'code': 404}})

    # -- chat --------------------------------------------------------
    def _chat(self, body):
        if self.server.reject_fields:
            for field in self.server.reject_fields:
                if field in body:
                    self._json(400, {'error': {
                        'message': "unsupported request field '%s'" % field,
                        'type': 'invalid_request_error', 'code': 400}})
                    return
        if self.server.hang:
            time.sleep(3600)
            return
        messages = body.get('messages') or []
        prompt_text = ' '.join(
            m.get('content') if isinstance(m.get('content'), str) else json.dumps(m.get('content'))
            for m in messages)
        prompt_tokens = max(120, min(len(prompt_text) // 4, max(1, self.server.ctx // 2)))
        want_tools = bool(body.get('tools')) and 'weather' in prompt_text.lower()
        max_tokens = int(body.get('max_tokens') or 256)
        completion_tokens = max(2, min(max_tokens, 96))
        reasoning = '' if body.get('thinking_budget_tokens') == 0 else (
            'Thinking about the request. The answer is short and deterministic. ')
        answer = ('The function performs a linear search over `items` and returns the '
                  'index of `target`, or -1 when it is absent. Two improvements: use '
                  '`enumerate` with a guard for empty input, and document the O(n) '
                  'complexity.'.strip())
        if 'Reply with the single word' in prompt_text:
            answer = 'ok'
        timings = {'prompt_per_second': self.server.prefill_rate,
                   'predicted_per_second': self.server.decode_rate,
                   'predicted_ms': completion_tokens * 40.0}
        usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
                 'total_tokens': prompt_tokens + completion_tokens, 'timings': timings}

        if want_tools:
            self._tool_reply(body, usage)
            return
        if body.get('stream'):
            self._stream(answer, reasoning, usage, body)
            return
        message = {'role': 'assistant', 'content': answer}
        if reasoning:
            message['reasoning_content'] = reasoning
        self._json(200, {'id': 'chatcmpl-fake', 'object': 'chat.completion',
                         'created': int(time.time()), 'model': self.server.alias,
                         'choices': [{'index': 0, 'message': message,
                                      'finish_reason': 'stop'}],
                         'usage': usage})

    def _tool_reply(self, body, usage):
        call = {'id': 'call_fake_1', 'type': 'function',
                'function': {'name': 'get_weather',
                             'arguments': json.dumps({'city': 'Lisbon'})}}
        message = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
        self._json(200, {'id': 'chatcmpl-fake', 'object': 'chat.completion',
                         'created': int(time.time()), 'model': self.server.alias,
                         'choices': [{'index': 0, 'message': message,
                                      'finish_reason': 'tool_calls'}],
                         'usage': usage})

    def _stream(self, answer, reasoning, usage, body):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

        created = int(time.time())
        words = reasoning.split() + answer.split()

        def chunk(payload):
            raw = ('data: ' + json.dumps(payload) + '\n\n').encode()
            self.wfile.write(b'%X\r\n' % len(raw) + raw + b'\r\n')
            self.wfile.flush()

        def delta(**kw):
            chunk({'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk',
                   'created': created, 'model': self.server.alias,
                   'choices': [{'index': 0, 'delta': kw, 'finish_reason': None}]})

        if body.get('thinking_budget_tokens') != 0:
            for word in reasoning.split():
                delta(reasoning_content=word + ' ')
                time.sleep(self.server.token_delay)
        for word in answer.split():
            delta(content=word + ' ')
            time.sleep(self.server.token_delay)
        chunk({'id': 'chatcmpl-fake', 'object': 'chat.completion.chunk', 'created': created,
               'model': self.server.alias,
               'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
               'usage': usage})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()
        self.wfile.write(b'0\r\n\r\n')
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    disable_nagle_algorithm = True      # or every benchmark measures 40 ms of harness

    def __init__(self, addr, cfg, opts, argv, state):
        super().__init__(addr, Handler)
        self.cfg = cfg
        self.api_key = first(opts, '--api-key', default='')
        self.alias = first(opts, '--alias', default='ternary-bonsai-2-27b')
        self.model_path = first(opts, '-m', '--model', default='')
        self.ctx = int(first(opts, '-c', '--ctx-size', default='8192') or 8192)
        self.stamp = cfg.get('stamp') or os.environ.get('FAKE_LLAMA_STAMP', 'prism-b10658')
        self.slots = int(first(opts, '--parallel', '-np',
                               default=cfg.get('slots', 1)) or 1)
        self.vision = bool(first(opts, '--mmproj', default=None))
        self.hang = bool(cfg.get('hang'))
        self.crash_after = cfg.get('crash_after')
        self.reject_fields = list(cfg.get('reject_fields') or ())
        self.decode_rate = float(cfg.get('decode_rate', 24.5))
        self.prefill_rate = float(cfg.get('prefill_rate', 1180.0))
        self.token_delay = float(cfg.get('token_delay', 0.0008))
        self.state = state
        if '--split-mode' in opts and opts['--split-mode'] == 'layer':
            self.decode_rate *= float(cfg.get('dual_factor', 1.0))


def main():
    argv = list(sys.argv)
    cfg = load_config()
    opts, flags = parse_args(argv)

    if '-h' in flags or '--help' in flags:
        extra = cfg.get('extra_flags') or []
        drop = set(cfg.get('help_drop') or ())
        text = BASE_HELP
        for line in extra:
            text += '  %s\n' % line
        if drop:
            kept = []
            for line in text.splitlines():
                head = re.split(r'\s{2,}', line.strip(), maxsplit=1)[0]
                if set(re.findall(r'-{1,2}[A-Za-z][\w-]*', head)) & drop:
                    continue
                kept.append(line)
            text = '\n'.join(kept) + '\n'
        sys.stdout.write(text)
        return 0
    if '--version' in flags:
        sys.stdout.write('version: 1 (prism fork)\n')
        sys.stdout.write('built with cc for x86_64-linux-gnu (prism fork)\n')
        return 0

    for flag in (cfg.get('reject_flags') or ()):
        if flag in opts or flag in flags:
            sys.stderr.write('error: unknown argument: %s\n' % flag)
            return 1

    state = FakeState()
    state.record_start(argv, opts)

    ctx = int(first(opts, '-c', '--ctx-size', default='8192') or 8192)
    oom_above = cfg.get('oom_above_ctx')
    if oom_above is not None and ctx > int(oom_above):
        sys.stdout.write('llama_model_loader: loaded meta data with 26 key-value pairs\n')
        sys.stdout.flush()
        time.sleep(float(cfg.get('oom_delay', 0.0)))
        sys.stdout.write(
            'ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8192.00 MiB\n'
            'CUDA error 2 at ggml-cuda.cu:412: out of memory\n'
            'ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 8589934592\n'
            'llama_init_from_model: failed to initialize the context: '
            'failed to allocate compute buffers\n')
        sys.stdout.flush()
        return 1

    if cfg.get('boot_delay'):
        time.sleep(float(cfg['boot_delay']))

    if cfg.get('oom_after') is not None:
        # Die of a CUDA OOM later on, the way a real server does when a long session
        # finally asks for more KV than the GPU has: the evidence is in the log, which
        # is exactly where the supervisor looks for it.
        def oom_later():
            time.sleep(float(cfg['oom_after']))
            sys.stdout.write(
                'ggml_backend_cuda_buffer_type_alloc_buffer: allocating 4096.00 MiB\n'
                'CUDA error 2 at ggml-cuda.cu:412: out of memory\n'
                'ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 4294967296\n'
                'llama_init_from_model: failed to initialize the context\n')
            sys.stdout.flush()
            state.release_vram()
            os._exit(1)
        threading.Thread(target=oom_later, daemon=True).start()

    port = int(first(opts, '--port', default='8080'))
    model_mib = 6970.0
    kv_mib = ctx * float(cfg.get('kv_mib_per_token', 0.30))
    per_gpu = {0: int(cfg.get('vram_mib', 7000))}
    if first(opts, '--split-mode') == 'layer':
        dual = cfg.get('dual_vram') or [4000, 4000]
        per_gpu = {i: int(v) for i, v in enumerate(dual)}
    state.claim_vram(per_gpu)

    sys.stdout.write('build: 1 (prism) with cc (GCC) for x86_64-linux-gnu\n')
    sys.stdout.write('system info: n_threads = 2, total_threads = 2\n')
    sys.stdout.write('llama_model_loader: loaded meta data with 26 key-value pairs\n')
    sys.stdout.write('llama_model_loader: - kv   0: general.architecture str  = qwen35\n')
    sys.stdout.write('llama_model_loader: - type f32:   65 tensors\n')
    sys.stdout.write('llama_model_load: model size = %.2f MiB\n' % model_mib)
    sys.stdout.write('llama_init_from_model: n_ctx_per_seq (%d) < n_ctx_train (262144)\n' % ctx)
    sys.stdout.write('llama_kv_cache_unified:      CUDA0 KV buffer size = %8.2f MiB\n' % kv_mib)
    if first(opts, '--split-mode') == 'layer':
        sys.stdout.write('llama_kv_cache_unified:      CUDA1 KV buffer size = %8.2f MiB\n'
                         % (kv_mib * 0.5))
    sys.stdout.write('llama_context:               CUDA0 compute buffer size = %7.2f MiB\n'
                     % 295.50)
    sys.stdout.write('llama_context:        CPU_Mapped compute buffer size = %7.2f MiB\n'
                     % 116.02)
    if cfg.get('mmproj_log') or opts.get('--mmproj'):
        sys.stdout.write('clip_model_loader: model size = 645.46 MiB '
                         '(-15.03 MiB multimodal projector)\n')
    sys.stdout.write('main: model loaded\n')
    sys.stdout.write('main: server is listening on 127.0.0.1:%d\n' % port)
    sys.stdout.flush()

    try:
        httpd = Server(('127.0.0.1', port), cfg, opts, argv, state)
    except OSError as exc:
        # Never leave a VRAM claim behind for a server that never came up.
        state.release_vram()
        sys.stderr.write('cannot listen on %d: %s\n' % (port, exc))
        return 1

    def config_watcher(httpd, state):
        """Re-read FAKE_SERVER_CONFIG so a test can change the server's fate mid-run.

        Real failures do not arrive conveniently before the process starts: a server
        that is told to OOM or hang *later* is the only way to test recovery honestly.
        """
        scheduled = {'oom': False, 'hang': httpd.hang}
        while True:
            time.sleep(0.2)
            try:
                cfg = json.loads(Path(os.environ['FAKE_SERVER_CONFIG']).read_text())
            except Exception:
                continue
            if cfg.get('crash_after') != httpd.crash_after:
                httpd.crash_after = cfg.get('crash_after')
            if bool(cfg.get('hang')) != scheduled['hang']:
                scheduled['hang'] = bool(cfg.get('hang'))
                httpd.hang = scheduled['hang']
            if cfg.get('oom_after') is not None and not scheduled['oom']:
                scheduled['oom'] = True
                delay = float(cfg['oom_after'])

                def oom_later():
                    time.sleep(delay)
                    sys.stdout.write(
                        'ggml_backend_cuda_buffer_type_alloc_buffer: '
                        'allocating 4096.00 MiB\n'
                        'CUDA error 2 at ggml-cuda.cu:412: out of memory\n'
                        'ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of '
                        'size 4294967296\n'
                        'llama_init_from_model: failed to initialize the context\n')
                    sys.stdout.flush()
                    state.release_vram()
                    os._exit(1)
                threading.Thread(target=oom_later, daemon=True).start()

    def cleanup(signum=None, frame=None):
        # Deliberately does NOT call httpd.shutdown(): a signal handler runs on the
        # same thread as serve_forever(), and shutdown() blocks until that loop
        # returns — a deadlock that would leave a "stopped" server holding its port.
        # Releasing the VRAM claim and exiting is what the real thing does.
        state.release_vram()
        os._exit(0)

    threading.Thread(target=config_watcher, args=(httpd, state), daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, cleanup)
    try:
        httpd.serve_forever(poll_interval=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        state.release_vram()
    return 0


if __name__ == '__main__':
    sys.exit(main())
