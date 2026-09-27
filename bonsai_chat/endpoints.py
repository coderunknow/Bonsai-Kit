"""Several endpoints, one client, and a capability cache that is per endpoint.

A capability map (`/props`, `--help`-derived flags, whether the projector is loaded,
the context window) belongs to *one* server. Caching it globally and reusing it after
`--endpoint` switches would be a confident lie about hardware we have not asked, so
each endpoint carries its own map and its own token cache.

Two endpoints are never silently merged: switching endpoints builds a fresh client and
a fresh capability map, and the session says which endpoint produced each turn.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ._meta import DEFAULT_MODEL
from .errors import BonsaiError


class Endpoint:
    """One named server: where it is, how to authenticate, what to assume."""

    def __init__(self, name='default', base_url='', api_key='', model=None,
                 timeout=None, headers=None, note=''):
        self.name = str(name or 'default')
        self.base_url = str(base_url or '').rstrip('/')
        self.api_key = str(api_key or '')
        self.model = model or None
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.note = str(note or '')
        self.caps = None          # filled in on first use
        self.checked = False      # True once a request has actually succeeded here

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, name, spec):
        if isinstance(spec, str):
            spec = {'base_url': spec}
        spec = dict(spec or {})
        # `api_key_env` keeps secrets out of the config file: the file names the
        # variable, the environment holds the value.
        env_var = spec.pop('api_key_env', None)
        key = spec.pop('api_key', '')
        if env_var and not key:
            key = os.environ.get(str(env_var), '')
        return cls(name=spec.pop('name', name) or name,
                   base_url=spec.pop('base_url', '') or spec.pop('url', ''),
                   api_key=key, model=spec.pop('model', None),
                   timeout=spec.pop('timeout', None),
                   headers=spec.pop('headers', None) or {},
                   note=spec.pop('note', '') or '')

    def to_dict(self, redact=True):
        out = {'name': self.name, 'base_url': self.base_url, 'model': self.model}
        if self.timeout:
            out['timeout'] = self.timeout
        if self.headers:
            out['headers'] = self.headers
        if self.note:
            out['note'] = self.note
        if not redact and self.api_key:
            out['api_key'] = self.api_key
        return {k: v for k, v in out.items() if v not in (None, '')}

    # ------------------------------------------------------------------
    def masked(self):
        return f'{self.name}  {self.base_url or "(no url)"}'

    def describe(self):
        state = 'verified' if self.checked else 'not contacted yet'
        bits = [f'  {self.name:<14} {self.base_url or "(no url)"}   [{state}]']
        if self.model:
            bits.append(f'                 model override: {self.model}')
        if self.note:
            bits.append(f'                 {self.note}')
        if self.caps:
            known = ', '.join(f'{k}={v}' for k, v in sorted(self.caps.items()))
            bits.append(f'                 capabilities: {known}')
        return bits

    def client(self, **overrides):
        """Build the client for this endpoint. Capabilities start unknown, every time."""
        from .client import BonsaiClient
        kwargs = dict(base_url=self.base_url, api_key=self.api_key,
                      model=self.model or DEFAULT_MODEL)
        if self.timeout:
            kwargs['timeout'] = self.timeout
        kwargs.update(overrides)
        return BonsaiClient(**kwargs)


class EndpointRegistry:
    """Named endpoints, with one current one."""

    def __init__(self, endpoints=None, current=None):
        self.endpoints = {}
        for item in (endpoints or []):
            self.add(item)
        self.current_name = current or (next(iter(self.endpoints), None))

    # ------------------------------------------------------------------
    def add(self, endpoint):
        self.endpoints[endpoint.name] = endpoint
        return endpoint

    @classmethod
    def from_config(cls, spec, default_url=None, default_key=None, default_model=None):
        """Build from a config-file `endpoints` block (list or mapping)."""
        endpoints = []
        if isinstance(spec, dict):
            items = [(name, entry) for name, entry in spec.items()]
        elif isinstance(spec, list):
            items = [(e.get('name') or f'endpoint{i}', e)
                     for i, e in enumerate(spec) if isinstance(e, dict)]
        else:
            items = []
        for name, entry in items:
            endpoints.append(Endpoint.from_dict(name, entry))
        reg = cls(endpoints)
        # The environment (or --base-url/--api-key) is always available as `default`,
        # so a config file with no endpoints block changes nothing.
        default = Endpoint(name='default', base_url=default_url or '',
                           api_key=default_key or '', model=default_model)
        if 'default' in reg.endpoints:
            existing = reg.endpoints['default']
            existing.base_url = existing.base_url or default.base_url
            existing.api_key = existing.api_key or default.api_key
        else:
            reg.endpoints['default'] = default
        # `default` stays current unless the user asks for another one: a config file
        # that merely *lists* endpoints must not silently reroute the session.
        reg.current_name = 'default'
        return reg

    # ------------------------------------------------------------------
    def names(self):
        return list(self.endpoints)

    def get(self, name=None):
        name = name or self.current_name
        if name not in self.endpoints:
            raise BonsaiError(f'no endpoint named {name!r} '
                              f'(known: {", ".join(self.names()) or "none"})')
        return self.endpoints[name]

    @property
    def current(self):
        return self.get(self.current_name)

    def select(self, name):
        endpoint = self.get(name)              # raises on an unknown name
        self.current_name = name
        return endpoint

    def describe(self):
        lines = []
        for name in self.names():
            marker = '*' if name == self.current_name else ' '
            block = self.endpoints[name].describe()
            block[0] = marker + block[0][1:]
            lines.extend(block)
        return lines
