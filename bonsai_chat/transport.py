"""One persistent HTTP/1.1 connection with a retry-safety policy."""

from __future__ import annotations

import http.client
import socket
import ssl
import time
import urllib.error, urllib.parse, urllib.request

from .errors import TransportError

class HttpTransport:
    """Persistent HTTP/1.1 connection to one host, with keep-alive and safe retries.

    Every request the client makes used to open a fresh TCP connection; against a
    Cloudflare tunnel that means a TLS handshake per call, which dominated the latency
    of short requests and of the token-counting round trips. This keeps one connection
    open, reuses it, and reconnects transparently when the far end has closed it.

    Retry safety is explicit. A failed call reports how far it got:
      * connect  — the socket never opened; the server saw nothing.  Safe to retry.
      * sent     — bytes left this process. The server may be generating. NOT retried
                   for POST; the caller gets a clear message instead of a duplicate.
      * response — the request completed but the reply was lost. NOT retried for POST.
    """

    def __init__(self, url, timeout=30, connect_timeout=15, opener=None, log=None,
                 keepalive=True, idle_ttl=3.0):
        parsed = urllib.parse.urlsplit(url)
        self.scheme = parsed.scheme or 'http'
        self.host = parsed.hostname or ''
        self.port = parsed.port or (443 if self.scheme == 'https' else 80)
        self.prefix = urllib.parse.urlunsplit((self.scheme, parsed.netloc, '', '', ''))
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.opener = opener                 # injectable for tests / non-http clients
        self.log = log or (lambda *a, **k: None)
        self.keepalive = keepalive
        self.idle_ttl = idle_ttl     # shorter than the server's keep-alive window
        self._conn = None
        self._stream_outstanding = False
        self.last_used = 0.0
        self.connections_opened = 0
        self.requests = 0
        self.reconnects = 0
        self._ssl_context = ssl.create_default_context() if self.scheme == 'https' else None

    # ------------------------------------------------------------------
    def _new_conn(self):
        self.close()
        if self.scheme == 'https':
            conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.connect_timeout,
                                               context=self._ssl_context)
        else:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.connect_timeout)
        self._conn = conn
        self.connections_opened += 1
        return conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._stream_outstanding = False

    # ------------------------------------------------------------------
    def request(self, method, url, body=None, headers=None, timeout=None, retry=None,
                stream=False):
        """One HTTP call -> http.client.HTTPResponse (caller must close it).

        `retry` overrides the retry policy; by default GET/HEAD may be retried freely and
        POST is retried only when we can prove the server never saw it.
        """
        if self.opener is not None:
            req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
            return self.opener.open(req, timeout=timeout or self.timeout)

        parts = urllib.parse.urlsplit(url)
        target = urllib.parse.urlunsplit(('', '', parts.path or '/', parts.query, ''))
        idempotent = method.upper() in ('GET', 'HEAD')
        allow = idempotent if retry is None else bool(retry)
        last = None
        for attempt in (1, 2):
            sent = False
            pooled = False
            try:
                conn = self._reusable_conn()
                pooled = conn is not None
                if conn is None:
                    conn = self._new_conn()
                conn.timeout = timeout or self.timeout
                conn.request(method.upper(), target, body=body,
                             headers=dict(headers or {}, Connection='keep-alive'))
                sent = True
                if conn.sock is not None:
                    # conn.timeout only applies to connect; the read deadline for a long
                    # generation is set on the socket itself.
                    try:
                        conn.sock.settimeout(timeout or self.timeout)
                    except OSError:
                        pass
                resp = conn.getresponse()
                self.requests += 1
                self.last_used = time.monotonic()
                if stream:
                    # Nothing may be multiplexed onto this connection until the caller
                    # hands the body back via release() — even if the caller forgets.
                    self._stream_outstanding = True
                if not self.keepalive or resp.will_close:
                    # Never pool a connection the peer is closing, or one whose body we
                    # may abandon mid-stream — the next request would read stale bytes.
                    self._conn = None
                return resp
            except http.client.HTTPException as e:
                self.close()
                last = e
                # A pooled connection that died before we wrote anything is the common
                # keep-alive-expiry case: provably nothing reached the server.
                if allow and not sent and (pooled or attempt == 1):
                    self.reconnects += 1
                    self.log(f'transport: {type(e).__name__} before send — reconnecting')
                    continue
                raise TransportError(f'{url}: {type(e).__name__}: {e}', url,
                                     request_sent=sent,
                                     phase='sent' if sent else 'connect') from None
            except (socket.timeout, TimeoutError) as e:
                self.close()
                last = e
                if allow and not sent and attempt == 1:
                    self.reconnects += 1
                    continue
                raise TransportError(
                    f'{url}: {"response" if sent else "connect"} timed out after '
                    f'{timeout or self.connect_timeout}s', url, request_sent=sent,
                    phase='response' if sent else 'connect') from None
            except OSError as e:
                self.close()
                last = e
                if allow and not sent and attempt == 1:
                    self.reconnects += 1
                    self.log(f'transport: {type(e).__name__} before send — reconnecting')
                    continue
                raise TransportError(f'cannot reach {url}: {e}', url, request_sent=sent,
                                     phase='sent' if sent else 'connect') from None
        raise TransportError(f'request to {url} failed: {last}', url, request_sent=True)

    def release(self, resp, drained):
        """Return a streamed response to the pool, or drop the connection.

        A body we stopped reading early leaves unread bytes in the socket; the next
        request on that connection would parse them as its response. `drained=False`
        therefore always costs us the connection — cheap compared to a corrupted reply.
        """
        self._stream_outstanding = False
        if not drained:
            self.close()

    def _reusable_conn(self):
        """Return the pooled connection unless it has been idle too long.

        llama-server (and Cloudflare's edge) close idle keep-alive connections after a
        few seconds. Reusing one that the peer already dropped turns into a failure at
        send time, which for a chat POST is not provably safe to retry. Dropping an
        idle connection here avoids that ambiguity almost entirely.
        """
        if not self.keepalive or self._conn is None:
            return None
        if self._stream_outstanding:
            # A stream is still open (or its reader was dropped without closing): the
            # socket holds unread bytes, so it cannot serve another request.
            self.close()
            self._stream_outstanding = False
            return None
        if time.monotonic() - self.last_used > self.idle_ttl:
            self.close()
            return None
        return self._conn

    def stats(self):
        return {'connections_opened': self.connections_opened, 'requests': self.requests,
                'reconnects': self.reconnects, 'pooled': self._conn is not None}
