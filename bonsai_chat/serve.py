"""Local stdlib web server: serves the built UI and proxies ``/api/*`` to the model.

Same origin by construction. The browser only ever talks to this process; the bearer
key lives in this process's memory and never crosses into a response body, a header, a
log line or a URL. Every upstream call goes through :class:`BonsaiClient` — this module
opens no HTTP connection of its own to the model.

Routes
------
``GET /``, ``GET /ui/<path>`` (and any other non-``/api`` path) static files from
``bonsai_chat/webui/dist`` — traversal-safe, 404 on a miss, never a directory listing.

``GET  /api/health``        ``{ok, endpoint, model, latency_ms}`` — latency measured,
                            or ``null``; on failure also ``kind`` + ``error``.
``GET  /api/models``        ``BonsaiClient.models()`` verbatim.
``GET  /api/capabilities``  the capability map: ``fields`` / ``evidence`` / ``facts``.
``GET  /api/version``       ``{version}`` for client/UI skew warnings.
``GET  /api/config``        the CLI's config file + provenance; never the key.
``PUT  /api/config``        connect (endpoint, key in memory) + settings -> config file
                            (written 0600 via ``config.save_config``).
``POST /api/chat``          SSE: ``delta`` / ``reasoning`` / ``tool_call`` / ``usage``
                            / ``error`` / ``done``, flushed per frame.
``POST /api/stop``          stop an in-flight stream by id; idempotent.

Failure kinds (distinct JSON ``kind`` values): ``connection``, ``unauthorized``,
``remote``, ``mid_stream``, ``stall``, ``busy``, ``bad_request``.
"""
from __future__ import annotations

import json
import mimetypes
import os
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ._meta import DEFAULT_MODEL, VERSION
from .capabilities import CapabilityMap
from .client import BonsaiClient
from .config import (BOOL_KEYS, SETTING_KEYS, Config, find_config, load_config,
                     save_config)
from .conversation import Settings
from .errors import (BonsaiAPIError, BonsaiError, CapabilityError,
                     StreamInterrupted, TransportError)

#: Where ``vite build`` writes the committed front end.
DIST = Path(__file__).resolve().parent / 'webui' / 'dist'

#: Seconds without upstream bytes before a stream is declared stalled.
#: Override with ``BONSAI_SERVE_STALL`` (tests use a small value).
STALL_SECONDS = float(os.environ.get('BONSAI_SERVE_STALL') or 30)

#: Concurrent streams. Past this, /api/chat refuses with kind=busy — never queues.
MAX_STREAMS = int(os.environ.get('BONSAI_SERVE_MAX_STREAMS') or 4)

#: Sampling parameters forwarded to BonsaiClient.stream_chat, with their JSON types.
_SAMPLING = {'temperature': float, 'top_p': float, 'top_k': int, 'min_p': float,
             'presence_penalty': float, 'max_tokens': int}

#: Config-file settings whose JSON types are validated on PUT /api/config.
_INT_SETTINGS = {'max_tokens', 'max_tool_rounds', 'context_reserve', 'output_reserve',
                 'tool_max_output', 'autosave_every'}
_FLOAT_SETTINGS = {'temperature', 'top_p', 'context_budget'}
_MAX_BODY = 8 * 1024 * 1024
_DONE = object()


def _kind(exc):
    """Map an exception to the distinct error kinds the UI renders one per class."""
    if isinstance(exc, BonsaiAPIError):
        return 'unauthorized' if exc.status in (401, 403) else 'remote'
    if isinstance(exc, StreamInterrupted):
        reason = exc.reason or ''
        return 'stall' if 'timeout' in reason.lower() else 'mid_stream'
    if isinstance(exc, CapabilityError):
        return 'remote'
    if isinstance(exc, TimeoutError):        # includes socket.timeout
        return 'stall'
    if isinstance(exc, (TransportError, BonsaiError, OSError)):
        return 'connection'
    return 'remote'


def _status_for(kind):
    return {'unauthorized': 401, 'bad_request': 400, 'busy': 429}.get(kind, 502)


class _Bad(Exception):
    """A malformed request body — reported as 400 bad_request, never a traceback."""

    def __init__(self, message):
        self.message = str(message)
        super().__init__(self.message)


class _StreamEntry:
    """One in-flight stream: a private ChatStream + private client + the stop flag."""

    __slots__ = ('stream', 'client', 'stopped', 'started', 'closed')

    def __init__(self, stream, client):
        self.stream = stream
        self.client = client           # dedicated BonsaiClient: its own transport
        self.stopped = False
        self.started = time.monotonic()
        self.closed = False

    def close(self):
        """Stop upstream. Idempotent, and never blocks on the reader's in-flight read.

        ``HTTPResponse.close()`` waits for the buffer lock, which the reader thread
        holds while blocked inside ``recv()`` — on a silent upstream that wait would
        last until the read timeout. Shutting the raw socket down first makes the
        reader's ``recv()`` return immediately, so teardown completes in milliseconds
        and the remote serving slot is released deterministically.
        """
        if self.closed:
            return
        self.closed = True
        transport = self.client.transport
        poisoned = getattr(transport, '_conn', None)   # the streaming connection
        stream = self.stream
        if getattr(stream, 'closed', False):
            return                        # finished naturally; pooled conn is healthy
        sock = _raw_socket(stream)
        if sock is not None:
            try:
                sock.shutdown(2)          # socket.SHUT_RDWR: unblock the reader now
            except OSError:
                pass
        try:
            stream.close()
        except Exception:                 # pragma: no cover - defensive
            pass
        if poisoned is not None and getattr(transport, '_conn', None) is poisoned:
            # The socket we just shut down must not be re-pooled for the next
            # request. If another request already swapped in a fresh connection
            # (the stop path tokenizes the partial), leave that one alone.
            try:
                transport.close()
            except Exception:             # pragma: no cover - defensive
                pass


def _raw_socket(stream):
    """The socket behind a ChatStream's HTTPResponse, or None if it is gone."""
    try:
        resp = stream.resp
        return resp.fp.raw._sock
    except (AttributeError, ValueError, TypeError):
        return None


class _State:
    def __init__(self, endpoint='', key='', model=None, config_path=None,
                 cli_layer=None, stall=None, max_streams=None):
        self.endpoint = (endpoint or os.environ.get('BONSAI_BASE_URL') or '').strip()
        self.key = key or os.environ.get('BONSAI_API_KEY') or ''
        self.model = model or DEFAULT_MODEL
        self.cli_layer = dict(cli_layer or {})
        self.config_path = (Path(config_path).expanduser() if config_path
                            else (find_config() or Path('bonsai.json')))
        self.stall = STALL_SECONDS if stall is None else float(stall)
        self.max_streams = MAX_STREAMS if max_streams is None else int(max_streams)
        self.streams = {}
        self.lock = threading.RLock()
        self._client = None

    @property
    def client(self):
        """The single transport, built once the endpoint is known (else None)."""
        if self._client is None and self.endpoint:
            self._client = BonsaiClient(self.endpoint, self.key, model=self.model,
                                        retries=1)
        return self._client

    def connect(self, endpoint, key):
        """Adopt a new endpoint/key in process memory. The key is never persisted."""
        old = self._client
        self.endpoint = endpoint
        self.key = key
        self._client = None
        if old is not None and not self.streams:
            old.close()


# ======================================================================
# Request handling
# ======================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = 'BonsaiServe/' + VERSION
    protocol_version = 'HTTP/1.0'          # self-terminating responses, no chunking

    # -- plumbing --------------------------------------------------------
    def log_message(self, fmt, *args):      # noqa: D401 - silence access logs
        """Never write an access log: request lines must not carry secrets."""

    @property
    def state(self):
        return self.server.state

    def _json(self, obj, status=200):
        raw = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def _fail_json(self, message, kind, http_status=None, **extra):
        body = {'error': str(message), 'kind': kind}
        body.update(extra)
        self._json(body, http_status if http_status is not None else _status_for(kind))

    def _body(self):
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            raise _Bad('invalid Content-Length')
        if length < 0 or length > _MAX_BODY:
            raise _Bad('request body too large')
        raw = self.rfile.read(length) if length else b''
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError) as exc:
            raise _Bad(f'invalid JSON body: {exc}')
        if not isinstance(data, dict):
            raise _Bad('JSON body must be an object')
        return data

    def _sse(self, event, data):
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        self.wfile.write(f'event: {event}\ndata: {payload}\n\n'.encode('utf-8'))
        self.wfile.flush()

    def _path(self):
        return self.path.split('?', 1)[0].split('#', 1)[0]

    # -- GET -------------------------------------------------------------
    def do_GET(self):
        path = self._path()
        if path.startswith('/api/'):
            try:
                self._api_get(path)
            except _Bad as exc:
                self._fail_json(exc.message, 'bad_request', 400)
            except BrokenPipeError:
                pass
            except ConnectionResetError:
                pass
            return
        self._static(path)

    def _api_get(self, path):
        if path == '/api/health':
            self._json(self._health())
        elif path == '/api/models':
            client = self.state.client
            if client is None:
                self._fail_json('no endpoint configured', 'connection')
                return
            try:
                self._json(client.models())
            except BonsaiError as exc:
                self._fail_json(exc, _kind(exc))
        elif path == '/api/capabilities':
            self._json(self._capabilities())
        elif path == '/api/version':
            self._json({'version': VERSION})
        elif path == '/api/config':
            self._json(self._config_json())
        else:
            self._fail_json(f'no such route: {path}', 'bad_request', 404)

    def _health(self):
        state = self.state
        client = state.client
        if client is None:
            return {'ok': False, 'endpoint': '', 'model': state.model,
                    'latency_ms': None, 'kind': 'connection',
                    'error': 'no endpoint configured — enter the deployment URL and key'}
        started = time.monotonic()
        body = client.health()
        if body is not None:
            return {'ok': True, 'endpoint': client.base_url, 'model': client.model,
                    'latency_ms': round((time.monotonic() - started) * 1000, 2)}
        # health() swallowed the reason; ask once more to classify it honestly so the
        # UI can point at the key (401) instead of guessing "down" (refused).
        try:
            second = time.monotonic()
            client.get_json('/health', timeout=15, root=True)
            return {'ok': True, 'endpoint': client.base_url, 'model': client.model,
                    'latency_ms': round((time.monotonic() - second) * 1000, 2)}
        except BonsaiError as exc:
            return {'ok': False, 'endpoint': client.base_url, 'model': client.model,
                    'latency_ms': None, 'kind': _kind(exc), 'error': str(exc)}

    def _capabilities(self):
        client = self.state.client
        if client is None:
            # Nothing has been asked of an endpoint: everything is honestly unknown.
            return CapabilityMap().to_dict()
        try:
            client.props()                    # populates facts.context_window path
            client.context_window()
            client.probe_vision()             # reads /props modalities; probes only
            client.model_ids()                #   when the build does not announce
        except Exception:                     # facts stay as learned — never guessed
            pass
        # Probe the two reasoning controls the settings panel gates on (cached after
        # the first time; the same thing the doctor does). Without this the states
        # would stay `unknown` forever and the panel could not disable anything.
        for field_name, value in (('reasoning_effort', 'medium'),
                                  ('thinking_budget_tokens', 2048)):
            try:
                client.probe_field(field_name, value)
            except Exception:
                pass                          # probe failed -> stays unknown
        return client.caps.to_dict()

    def _config_json(self):
        state = self.state
        data, path = load_config(state.config_path)
        exists = path.is_file()
        cfg = Config(data, path=path if exists else None)
        defaults = {name: getattr(Settings(), name) for name in SETTING_KEYS}
        resolved = cfg.resolve(defaults, cli=state.cli_layer, environ=os.environ)
        client = state.client
        return {'path': str(path),
                'exists': exists,
                'settings': resolved,
                'provenance': cfg.provenance(),
                'endpoint': client.base_url if client else state.endpoint,
                'model': state.model,
                'has_api_key': bool(state.key)}

    # -- static ----------------------------------------------------------
    def _static(self, path):
        rel = path[1:] if path.startswith('/') else path
        if rel.startswith('ui/'):
            rel = rel[3:]
        elif rel == 'ui':
            rel = ''
        if rel.endswith('/'):
            rel += 'index.html'
        target = (DIST / rel).resolve() if rel else (DIST / 'index.html').resolve()
        root = DIST.resolve()
        try:
            inside = target.is_relative_to(root)
        except AttributeError:               # Python < 3.9
            inside = str(target).startswith(str(root) + os.sep) or target == root
        if not inside or not target.is_file():
            self.send_error(404, 'Not Found')
            return
        try:
            raw = target.read_bytes()
        except OSError:
            self.send_error(404, 'Not Found')
            return
        self.send_response(200)
        self.send_header('Content-Type',
                         mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    # -- PUT -------------------------------------------------------------
    def do_PUT(self):
        if self._path() != '/api/config':
            self._fail_json('no such route', 'bad_request', 404)
            return
        try:
            data = self._body()
            self._put_config(data)
        except _Bad as exc:
            self._fail_json(exc.message, 'bad_request', 400)
        except BrokenPipeError:
            pass

    def _put_config(self, data):
        state = self.state
        endpoint = data.get('endpoint')
        key = data.get('api_key')
        if endpoint is not None and not isinstance(endpoint, str):
            raise _Bad('endpoint must be a string')
        if key is not None and not isinstance(key, str):
            raise _Bad('api_key must be a string')
        if endpoint is not None:
            endpoint = endpoint.strip()
            if endpoint:
                try:
                    BonsaiClient(endpoint, key or state.key, model=state.model, retries=1)
                except BonsaiError as exc:
                    raise _Bad(f'endpoint: {exc}')
        settings = data.get('settings')
        if settings is not None and not isinstance(settings, dict):
            raise _Bad('settings must be an object')
        # 1. Persist settings through the CLI's own file (0600; never the key).
        if settings:
            incoming = {k: v for k, v in settings.items() if k in SETTING_KEYS}
            for name, value in incoming.items():
                if name in _INT_SETTINGS:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise _Bad(f'settings.{name} must be an integer')
                elif name in _FLOAT_SETTINGS:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise _Bad(f'settings.{name} must be a number')
                elif name in BOOL_KEYS:
                    if not isinstance(value, bool):
                        raise _Bad(f'settings.{name} must be true or false')
                elif value is not None and not isinstance(value, str):
                    raise _Bad(f'settings.{name} must be a string')
            existing, path = load_config(state.config_path)
            existing.update(incoming)
            save_config(existing, path)      # atomic, mode 0600
        # 2. Adopt the connection in process memory (key: memory only, this request).
        if endpoint is not None or key is not None:
            new_endpoint = (endpoint if endpoint is not None else state.endpoint).strip()
            new_key = key if key is not None else state.key
            state.connect(new_endpoint, new_key)
        self._json(self._config_json())

    # -- POST ------------------------------------------------------------
    def do_POST(self):
        path = self._path()
        try:
            if path == '/api/stop':
                self._stop()
            elif path == '/api/chat':
                self._chat()
            else:
                self._fail_json(f'no such route: {path}', 'bad_request', 404)
        except _Bad as exc:
            self._fail_json(exc.message, 'bad_request', 400)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

    def _stop(self):
        data = self._body()
        stream_id = data.get('stream_id')
        if not isinstance(stream_id, str) or not stream_id:
            raise _Bad('stream_id must be a non-empty string')
        state = self.state
        with state.lock:
            entry = state.streams.get(stream_id)
            if entry is not None:
                entry.stopped = True        # mark before close: the writer checks it
        if entry is not None:
            entry.close()
        # Idempotent: stopping an unknown or already-finished id succeeds too.
        self._json({'ok': True, 'found': entry is not None})

    def _chat(self):
        data = self._body()
        state = self.state
        messages = data.get('messages')
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(m, dict) and isinstance(m.get('role'), str)
                           for m in messages)):
            raise _Bad('messages must be a non-empty list of {role, content} objects')
        stream_id = data.get('stream_id') or f'stream-{time.time_ns()}'
        if not isinstance(stream_id, str) or not stream_id:
            raise _Bad('stream_id must be a non-empty string')
        params = self._sampling_params(data.get('sampling'))
        params.update(self._thinking_params(data.get('thinking')))
        client = state.client
        if client is None:
            self._fail_json('no endpoint configured — connect first', 'connection')
            return
        caps = client.caps
        if ('reasoning_effort' in params
                and caps.supported('reasoning_effort') is False):
            params.pop('reasoning_effort')
        if ('thinking_budget_tokens' in params
                and caps.supported('thinking_budget_tokens') is False):
            params.pop('thinking_budget_tokens')
        with state.lock:
            if stream_id in state.streams:
                self._fail_json('a stream with this stream_id is already running',
                                'busy', 409)
                return
            if len(state.streams) >= state.max_streams:
                self._fail_json(
                    f'concurrent stream limit reached ({state.max_streams}); '
                    'wait for a turn to finish — streams are refused, never queued',
                    'busy', 429, active=len(state.streams), limit=state.max_streams)
                return
        # A dedicated client per stream: HttpTransport owns exactly one connection
        # and one outstanding stream, so sharing it across concurrent streams (or
        # with the GET routes) would let one teardown kill another's socket. The
        # capability map is shared, so anything a stream learns (note_400/note_ok)
        # is visible to the settings panel and to the next turn.
        stream_client = BonsaiClient(state.endpoint, state.key, model=state.model,
                                     retries=1)
        stream_client.caps = client.caps
        try:
            stream = stream_client.stream_chat(messages, **params)
        except BonsaiError as exc:
            stream_client.close()
            extra = {}
            if isinstance(exc, BonsaiAPIError):
                extra['status'] = exc.status
                if exc.hint():
                    extra['hint'] = exc.hint()
            self._fail_json(exc, _kind(exc), **extra)
            return
        except (TypeError, ValueError) as exc:
            stream_client.close()
            raise _Bad(str(exc))
        entry = _StreamEntry(stream, stream_client)
        with state.lock:
            state.streams[stream_id] = entry
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self._pump(entry, stream_id)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The tab went away mid-answer: stop upstream so the remote serving
            # slot is not left busy.
            entry.close()
        finally:
            with state.lock:
                state.streams.pop(stream_id, None)
            entry.close()

    def _sampling_params(self, sampling):
        if sampling is None:
            return {}
        if not isinstance(sampling, dict):
            raise _Bad('sampling must be an object')
        params = {}
        for name, value in sampling.items():
            want = _SAMPLING.get(name)
            if want is None:                 # unknown keys are not forwarded
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _Bad(f'sampling.{name} must be a number')
            params[name] = want(value)
        return params

    def _thinking_params(self, thinking):
        if thinking is None:
            return {}
        if not isinstance(thinking, dict):
            raise _Bad('thinking must be an object')
        params = {}
        effort = thinking.get('effort')
        if effort is not None:
            if effort not in ('medium', 'xhigh', 'none'):
                raise _Bad("thinking.effort must be 'medium' or 'xhigh'")
            params['reasoning_effort'] = effort
        budget = thinking.get('budget_tokens')
        if budget is not None:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < -1:
                raise _Bad('thinking.budget_tokens must be an integer >= -1 '
                           '(-1 = unlimited, 0 = thinking off)')
            params['thinking_budget_tokens'] = budget
        return params

    # -- the streaming proxy --------------------------------------------
    def _pump(self, entry, stream_id):
        """Read ChatStream events on a worker thread; emit SSE frames from this one.

        The reader thread exists so a stalled upstream (no bytes for T seconds), a
        /api/stop from another request, and a dead browser can all be observed
        promptly — the stream's own iterator blocks inside a socket read.
        """
        state = self.state
        events = queue.Queue()

        def read():
            try:
                for ev in entry.stream:
                    events.put(ev)
            except BaseException as exc:      # noqa: BLE001 - forwarded, not swallowed
                events.put(exc)
            else:
                events.put(_DONE)

        threading.Thread(target=read, daemon=True,
                         name=f'bonsai-serve-{stream_id}').start()
        deltas = 0
        last = time.monotonic()
        while True:
            try:
                item = events.get(timeout=state.stall)
            except queue.Empty:
                waited = round(time.monotonic() - last, 3)
                if entry.stopped:
                    self._emit_stopped(entry, stream_id, deltas)
                    return
                try:
                    self._sse('error', {'kind': 'stall',
                                        'error': f'no data from the endpoint for '
                                                 f'{waited:g} seconds',
                                        'waited_seconds': waited,
                                        'tokens_arrived': deltas})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                entry.close()
                return
            if isinstance(item, BaseException):
                self._emit_failure(entry, item, stream_id, deltas)
                return
            if item is _DONE:
                # ChatStream always yields a done event before ending; this is a
                # defensive terminal frame, not a normal path.
                self._sse('done',
                          {'kind': 'done', 'stopped': bool(entry.stopped),
                           'stream_id': stream_id, 'usage': None,
                           'message': entry.stream.message()})
                return
            last = time.monotonic()
            kind = item.get('kind')
            if kind == 'done':
                self._emit_done(entry, item, stream_id, deltas)
                return
            if kind == 'delta':
                deltas += 1
            if kind in ('delta', 'reasoning', 'tool_call'):
                try:
                    self._sse(kind, item)
                except (BrokenPipeError, ConnectionResetError):
                    entry.close()            # closed tab: never leave the slot busy
                    return

    def _emit_done(self, entry, done, stream_id, deltas):
        usage = done.get('usage')
        stopped = bool(entry.stopped)
        payload = dict(done)
        payload['stream_id'] = stream_id
        payload['stopped'] = stopped
        if stopped:
            payload['tokens'] = self._stopped_tokens(entry, usage)
        if usage:
            self._sse('usage', {'kind': 'usage', 'usage': usage})
        self._sse('done', payload)

    def _emit_stopped(self, entry, stream_id, deltas):
        """/api/stop won the race: keep the partial answer, say it was stopped."""
        stream = entry.stream
        usage = getattr(stream, 'usage', None)
        payload = {'kind': 'done', 'stopped': True, 'stream_id': stream_id,
                   'message': stream.message(), 'tool_calls': stream.tool_calls(),
                   'finish_reason': stream.finish, 'usage': usage,
                   'chunks': stream.chunks,
                   'tokens': self._stopped_tokens(entry, usage),
                   'tokens_arrived': deltas}
        if usage:
            self._sse('usage', {'kind': 'usage', 'usage': usage})
        self._sse('done', payload)

    def _stopped_tokens(self, entry, usage):
        """A real, measured token count for a stopped turn: usage when the server
        reported it, otherwise the endpoint's own tokenizer on the partial text —
        never a guessed number."""
        if isinstance(usage, dict) and usage.get('completion_tokens') is not None:
            return usage['completion_tokens']
        text = (entry.stream.message() or {}).get('content') or ''
        if not text:
            return 0
        try:
            return entry.client.tokenize(text)
        except Exception:                    # pragma: no cover - tokenize already swallows
            return None

    def _emit_failure(self, entry, exc, stream_id, deltas):
        if entry.stopped:
            # Our own close aborted the read: the user pressed stop.
            self._emit_stopped(entry, stream_id, deltas)
            return
        kind = _kind(exc)
        payload = {'kind': kind, 'error': str(exc), 'tokens_arrived': deltas,
                   'stream_id': stream_id}
        if isinstance(exc, BonsaiAPIError):
            payload['status'] = exc.status
            hint = exc.hint()
            if hint:
                payload['hint'] = hint
        if kind == 'mid_stream' and getattr(exc, 'partial_text', ''):
            payload['partial_text'] = exc.partial_text[:20000]
        if kind == 'stall':
            payload['waited_seconds'] = round(time.monotonic() - entry.started, 3)
        self._sse('error', payload)


def serve(host='127.0.0.1', port=0, endpoint=None, key=None, model=None, *,
          config_path=None, cli_layer=None, open_browser=False, stall=None,
          max_streams=None):
    """Start the local web server; returns a ThreadingHTTPServer (caller serves).

    Binds ``127.0.0.1`` by default — this process holds the bearer key. Raises
    ``RuntimeError`` with build instructions when the committed UI is missing.
    """
    if not (DIST / 'index.html').is_file():
        raise RuntimeError(
            'web UI is not built — run `npm install && npm run build` in '
            'bonsai_chat/webui, then retry')
    state = _State(endpoint=endpoint, key=key, model=model,
                   config_path=config_path, cli_layer=cli_layer,
                   stall=stall, max_streams=max_streams)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.state = state                      # test seam: registry + transport stats
    url = f'http://{host}:{httpd.server_port}'
    print(f'Bonsai web UI: {url}', flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass                             # headless: the URL is printed either way
    return httpd
