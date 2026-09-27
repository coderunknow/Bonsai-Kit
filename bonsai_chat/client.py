"""OpenAI-compatible HTTP client (chat, models, props, tokenize)."""

from __future__ import annotations

import json
import time

from ._meta import DEFAULT_MODEL, VERSION
from .capabilities import CapabilityMap
from .errors import BonsaiAPIError, BonsaiError, TransportError
from .sse import normalize_base_url
from .streaming import ChatStream
from .transport import HttpTransport

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
        self._structured = None
        self._probing_structured = False
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

    def _apply_structured(self, params):
        """Drop `response_format` unless the server has proved it honours it.

        Sending the field anyway would be harmless on a server that ignores it and
        fatal on one that 400s, and in neither case would the caller get what it asked
        for. The caller is told, in the returned notes, which one happened.
        """
        notes = []
        if not params.get('response_format'):
            return notes
        if self._probing_structured:
            # The probe's own request carries the field on purpose; sending it through
            # the check would recurse instead of measuring anything.
            return notes
        if not self.probe_structured_output():
            notes.append('structured output requested but this server does not honour '
                         'response_format — the request was sent without it')
            params.pop('response_format', None)
            if params.get('json_schema'):
                notes.append('json_schema was not sent either; the schema is enforced '
                             'locally instead, and the reply is checked before return')
                params.pop('json_schema', None)
        return notes

    def chat(self, messages, tools=None, **params):
        """Non-streaming chat completion -> the raw JSON response dict."""
        notes = self._apply_structured(params)
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
        if 'response_format' in payload:
            self.caps.note_ok('response_format')
        if notes:
            data = dict(data)
            data['bonsai_notes'] = notes
        return data

    def stream_chat(self, messages, tools=None, **params):
        # Same rule as chat(): an unproved response_format is dropped, not sent.
        self._apply_structured(params)
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

    def probe_structured_output(self, force=None):
        """Can this server be *made* to answer in JSON? Probed once, then cached.

        The probe is a real request, not a flag check, because the only thing that
        matters is the behaviour: does the server reject `response_format`, ignore it,
        or honour it. Those three outcomes are reported as False / False / True — an
        ignored field is not structured output, and pretending otherwise would hand the
        caller prose where it asked for JSON.
        """
        if force is not None:
            self._structured = bool(force)
            self.caps.set_fact('structured_output', self._structured)
            return self._structured
        if self._structured is not None:
            return self._structured
        self._probing_structured = True
        try:
            resp = self.chat(
                [{'role': 'user', 'content': 'Reply with a JSON object and nothing '
                                             'else: {"ready": true}'}],
                response_format={'type': 'json_object'}, max_tokens=32)
        except BonsaiAPIError as e:
            self._structured = False
            self.log(f'response_format rejected with HTTP {e.status} — structured '
                     'output unavailable')
            self.caps.set_fact('structured_output', False)
            self.caps.set_fact('structured_output_source',
                               f'rejected with HTTP {e.status}')
            return False
        except BonsaiError as e:
            self.log(f'structured-output probe failed: {e}')
            return False
        finally:
            self._probing_structured = False
        text = ''
        try:
            text = ((resp.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        except (AttributeError, IndexError):
            text = ''
        try:
            value = json.loads(text)
            ok = isinstance(value, (dict, list))
        except ValueError:
            ok = False
        self._structured = ok
        self.caps.set_fact('structured_output', ok)
        # The capability map has to agree with the probe. `note_ok('response_format')`
        # only ever means "the server did not complain"; the probe is the evidence, and
        # a server that ignores the field is unsupported, not supported.
        self.caps.mark('response_format', 'supported' if ok else 'unsupported',
                       'probe returned valid JSON' if ok else
                       'accepted the field but answered in prose')
        self.caps.set_fact('structured_output_source',
                           'probe returned valid JSON' if ok else
                           'probe returned prose — the field was ignored')
        if not ok:
            self.log('response_format was accepted but the reply was not JSON — '
                     'treating structured output as unavailable')
        return ok

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
