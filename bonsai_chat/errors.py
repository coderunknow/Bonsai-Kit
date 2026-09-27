"""Errors shared by the client, the agent and the CLI."""

from __future__ import annotations

class BonsaiError(RuntimeError):
    """Base error for anything the client itself reports."""

class BonsaiAPIError(BonsaiError):
    """The endpoint answered with an HTTP error status."""

    def __init__(self, status, message, url=''):
        self.status = status
        self.message = message
        self.url = url
        super().__init__(f'HTTP {status}: {message}')

    def hint(self):
        s = (self.message or '').lower()
        if self.status in (401, 403):
            return ('authentication rejected — check BONSAI_API_KEY against the key the '
                    'deployment cell printed (it is printed exactly once).')
        if self.status == 404:
            return 'unknown path or model id — check BONSAI_BASE_URL ends in /v1 and the model name.'
        if self.status == 400 and 'context' in s:
            return 'the request exceeds the server context window — lower /max-tokens or /reset.'
        if self.status == 400:
            return 'the server rejected the payload; try /effort none or disable tools.'
        if self.status == 429:
            return 'rate limited or the single serving slot is busy — wait and retry.'
        if self.status >= 500:
            return 'server-side error; the notebook may be out of VRAM or restarting.'
        return ''

class CancelledByUser(Exception):
    """Raised when Ctrl-C interrupts the current turn (the session survives)."""

class TransportError(BonsaiError):
    """The HTTP layer could not complete a call. `request_sent` drives retry safety.

    A chat POST that the server may already be generating for must not be replayed
    blindly: doing so doubles the work and can hand the user a duplicated answer.
    So the transport records how far the request got, and only "never left this
    process" failures are retried automatically for non-idempotent calls.
    """

    #: request phases, in order; retry is safe for a POST only up to `sent=False`
    def __init__(self, message, url='', request_sent=False, phase='connect'):
        self.url = url
        self.request_sent = request_sent
        self.phase = phase
        super().__init__(message)

    @property
    def safe_to_retry_post(self):
        return not self.request_sent

class StreamInterrupted(BonsaiError):
    """The server stopped mid-generation. Carries whatever did arrive."""

    def __init__(self, message, partial_text='', partial_reasoning='', reason=''):
        self.partial_text = partial_text
        self.partial_reasoning = partial_reasoning
        self.reason = reason
        super().__init__(message)

class CapabilityError(BonsaiError):
    """A requested feature is not supported by this server build."""
