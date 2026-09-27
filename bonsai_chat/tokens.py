"""Token estimation and caching."""

from __future__ import annotations

import re

def estimate_tokens(text) -> int:
    """Rough token estimate used when /tokenize is unavailable."""
    if isinstance(text, list):
        return sum(estimate_tokens(part.get('text', '')) for part in text if isinstance(part, dict))
    if not text:
        return 0
    s = str(text)
    cjk = len(re.findall(r'[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]', s))
    return max(1, int(len(s) / 4) + cjk)

class TokenCounter:
    """Uses the server's own tokenizer when the endpoint exists, else a heuristic.

    Counts are memoized: budget trimming re-measures the same history on every turn and
    each measurement is a round trip to /tokenize.
    """

    CACHE_LIMIT = 1024

    def __init__(self, client=None):
        self.client = client
        self._server = None
        self._cache = {}

    def count(self, text) -> int:
        if isinstance(text, str):
            cached = self._cache.get(text)
            if cached is not None:
                return cached
        n = self._count_uncached(text)
        if isinstance(text, str):
            if len(self._cache) >= self.CACHE_LIMIT:
                self._cache.clear()
            self._cache[text] = n
        return n

    def using_server(self):
        """True once a real /tokenize round trip has succeeded."""
        return self._server is True

    def _count_uncached(self, text) -> int:
        if self._server is False:
            return estimate_tokens(text)
        if self.client is not None and isinstance(text, str) and text:
            n = self.client.tokenize(text)
            if n is None:
                self._server = False
            else:
                self._server = True
                return n
        return estimate_tokens(text)

    def messages(self, messages) -> int:
        total = 0
        for m in messages:
            total += self.count(m.get('content')) + 4
            for tc in m.get('tool_calls') or []:
                total += self.count((tc.get('function') or {}).get('arguments') or '') + 8
        return total
