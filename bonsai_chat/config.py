"""Layered configuration, persona presets and precedence you can inspect.

Four sources feed one answer, lowest priority first:

    1. built-in defaults      (the values Settings already carries)
    2. the config file        (--config FILE, else the first file that exists)
    3. a persona preset       (--persona NAME, or `persona:` in the config file)
    4. environment variables  (BONSAI_TEMPERATURE, BONSAI_MAX_TOKENS, ...)
    5. command-line flags     (only the ones actually typed)

Command-line flags win, and the rule is *not* "the CLI namespace overwrites the
file": argparse fills in a value for every option whether or not it was typed, so a
default-filled flag would silently beat the file. `explicit_args()` re-parses the same
parser with `argparse.SUPPRESS` defaults, which yields exactly the options the user
typed and nothing else.

Nothing here guesses. `provenance()` reports where every resolved value came from, so
`/config` can show it and a user can see why 0.7 and not 1.0 was used.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

CONFIG_FILENAME = 'config.json'

#: Where a config file is looked for when --config is not given, first match wins.
SEARCH_PATHS = (
    'bonsai.json',
    '.bonsai.json',
    '~/.config/bonsai/config.json',
    '~/.bonsai.json',
)

#: Config-file keys that map onto a Settings attribute. Anything else in the file is
#: data, not a setting (personas, endpoints, mcpServers, tools).
SETTING_KEYS = (
    'model', 'temperature', 'top_p', 'max_tokens', 'stream', 'use_tools',
    'markdown', 'highlight', 'max_tool_rounds', 'context_budget',
    'context_reserve', 'output_reserve', 'tool_timeout', 'tool_budget',
    'tool_max_output', 'autosave_every', 'show_stats', 'system',
    'preexecute_tools',
)

#: Booleans, so the strings a JSON file can hold are read the way a human means them.
BOOL_KEYS = frozenset({'stream', 'use_tools', 'markdown', 'highlight', 'show_stats',
                       'preexecute_tools'})

#: `BONSAI_TEMPERATURE=0.3` and friends. The CLI still wins over these.
ENV_KEYS = {
    'BONSAI_MODEL': 'model',
    'BONSAI_TEMPERATURE': 'temperature',
    'BONSAI_TOP_P': 'top_p',
    'BONSAI_MAX_TOKENS': 'max_tokens',
    'BONSAI_SYSTEM': 'system',
    'BONSAI_PERSONA': 'persona',
    'BONSAI_ENDPOINT': 'endpoint',
}


def _coerce(key, value):
    if key in BOOL_KEYS:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ('1', 'true', 'yes', 'on')
    if key in ('temperature', 'top_p', 'context_budget'):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if key in ('max_tokens', 'max_tool_rounds', 'context_reserve', 'output_reserve',
               'tool_max_output', 'autosave_every'):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if key in ('tool_timeout', 'tool_budget'):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def deep_merge(base, override):
    """Merge `override` onto `base`, one level deep for dicts. Neither is mutated."""
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def find_config(explicit=None):
    """The config file to use, or None. An explicit path that is missing is an error."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    for candidate in SEARCH_PATHS:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    return None


def load_config(path, on_error=None):
    """Read a JSON config file. A broken file is reported and treated as empty."""
    path = Path(path).expanduser()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        if on_error:
            on_error(f'{path}: {exc}')
        return {}, path
    if not isinstance(data, dict):
        if on_error:
            on_error(f'{path}: expected a JSON object at the top level')
        return {}, path
    return data, path


def save_config(data, path):
    """Write the config file atomically, 0600: it can hold API keys."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def write_default_config(path, settings):
    """Create a starter config file from the current settings, without secrets."""
    data = {
        '_comment': 'Bonsai-Kit configuration. Command-line flags always win over '
                    'this file; `bonsai_chat.py --config --save` rewrites it.',
        'model': settings.model,
        'temperature': settings.temperature,
        'top_p': settings.top_p,
        'max_tokens': settings.max_tokens,
        'stream': settings.stream,
        'use_tools': settings.use_tools,
        'personas': {
            'concise': {'system': 'Answer in at most three sentences. No preamble.',
                        'temperature': 0.4},
            'code': {'system': 'You are a senior engineer. Prefer complete, runnable '
                               'code over description. Say when something is untested.',
                     'temperature': 0.3},
            'teacher': {'system': 'Explain step by step, then give a one-line summary.',
                        'temperature': 0.7},
        },
        'endpoints': {},
        'mcpServers': {},
    }
    return save_config(data, path)


def explicit_args(parser, argv):
    """The subset of `argv` the user actually typed, as a dict.

    Re-parses with `argparse.SUPPRESS` defaults so an option that merely *has* a
    default does not look like it was given. Unknown arguments are ignored: this runs
    as a probe, and the real parse is the one that reports errors.
    """
    try:
        probe = copy.deepcopy(parser)
    except Exception:                                   # pragma: no cover - defensive
        return {}
    # argparse.SUPPRESS (rather than a sentinel string) because argparse skips *type
    # conversion* for suppressed defaults — a sentinel would be fed to `type=int`
    # options and blow up. Required flags are relaxed: this is a probe, and the real
    # parse is the one that reports missing arguments.
    for action in probe._actions:
        action.default = argparse.SUPPRESS
        action.required = False
    try:
        namespace, _ = probe.parse_known_args(list(argv or ()))
    except (SystemExit, argparse.ArgumentError):        # pragma: no cover - defensive
        return {}
    return dict(vars(namespace))


class Config:
    """The resolved configuration, and where every value came from."""

    def __init__(self, data=None, path=None):
        self.data = dict(data or {})
        self.path = Path(path).expanduser() if path else None
        self.persona_name = None
        self.persona = {}
        self._provenance = {}
        self.notes = []

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, explicit_path=None, on_error=None):
        path = find_config(explicit_path)
        if path is None:
            if explicit_path and on_error:
                on_error(f'config file not found: {explicit_path}')
            return cls({}, None)
        data, path = load_config(path, on_error=on_error)
        return cls(data, path)

    # ------------------------------------------------------------------
    def personas(self):
        raw = self.data.get('personas') or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    def select_persona(self, name):
        """Apply a persona preset. Returns the preset, or None if there is no such one."""
        name = str(name or '').strip()
        if not name:
            return None
        personas = self.personas()
        if name not in personas:
            self.notes.append(f'no persona named {name!r} '
                              f'(known: {", ".join(sorted(personas)) or "none"})')
            return None
        self.persona_name = name
        self.persona = dict(personas[name])
        return self.persona

    def endpoints(self):
        raw = self.data.get('endpoints')
        if isinstance(raw, dict):
            out = []
            for name, entry in raw.items():
                if isinstance(entry, dict):
                    item = dict(entry)
                    item.setdefault('name', name)
                    out.append(item)
            return out
        if isinstance(raw, list):
            return [e for e in raw if isinstance(e, dict)]
        return []

    def mcp_servers(self):
        raw = self.data.get('mcpServers') or {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}

    # ------------------------------------------------------------------
    def resolve(self, defaults, cli=None, environ=None):
        """Fold the layers together. Returns {setting: value}.

        `defaults` is the baseline (normally the built-in Settings values), `cli` the
        options the user typed, `environ` the process environment.
        """
        environ = os.environ if environ is None else environ
        cli = dict(cli or {})
        resolved = {}
        self._provenance = {}

        def take(values, source):
            for key, value in values.items():
                if key not in SETTING_KEYS and key != 'persona':
                    continue
                resolved[key] = _coerce(key, value)
                self._provenance[key] = source

        take({k: v for k, v in defaults.items() if k in SETTING_KEYS}, 'default')
        file_values = {k: v for k, v in self.data.items() if k in SETTING_KEYS}
        take(file_values, 'config file' if self.path else 'config')
        take({k: v for k, v in self.persona.items() if k in SETTING_KEYS},
             f'persona {self.persona_name}' if self.persona_name else 'persona')

        env_values = {}
        for var, key in ENV_KEYS.items():
            if environ.get(var):
                env_values[key] = environ[var]
        take(env_values, 'environment')
        take({k: v for k, v in cli.items()}, 'command line')
        return resolved

    def provenance(self):
        return dict(self._provenance)

    def describe(self):
        """Lines for `/config`: value, and where it came from."""
        prov = self.provenance()
        width = max((len(k) for k in prov), default=8)
        lines = []
        for key in sorted(prov):
            lines.append(f'  {key:<{width}}  {prov[key]}')
        return lines

    def source_of(self, key):
        return self._provenance.get(key)
