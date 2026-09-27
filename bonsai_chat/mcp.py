"""Minimal MCP (Model Context Protocol) client, stdlib only.

Speaks JSON-RPC 2.0 over a server's stdio — the transport every MCP server supports —
and registers what it finds into the existing `ToolRegistry`, so an MCP tool goes
through exactly the same approval, timeout, output-cap and sandbox rules as a built-in
one. There is deliberately no second tool-calling mechanism in this codebase.

What this is not: it does not trust the remote server. Tool *output* is data, never
instructions, and it is fed back to the model as a tool result like any other. Names
are namespaced `server__tool` unless they are already unique, so two servers exporting
`search` cannot shadow each other.

Servers are configured the same way Claude Desktop configures them, so an existing
`mcpServers` block can be pasted in unchanged:

    "mcpServers": {"files": {"command": "python3", "args": ["-m", "mcp_server_files"]}}
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

from .errors import BonsaiError

PROTOCOL_VERSION = '2024-11-05'
CLIENT_INFO = {'name': 'bonsai-kit', 'version': '0.6.0'}


class McpError(BonsaiError):
    """An MCP server said no, or said something that is not JSON-RPC."""


class McpServer:
    """One MCP server process, its tools, and a serialised request/response channel.

    A single stdio pair cannot interleave two requests, so every exchange holds a lock:
    concurrent tool calls (the agent pre-executes them) queue instead of corrupting
    each other's framing.
    """

    def __init__(self, name, command, args=None, env=None, cwd=None, timeout=30.0):
        self.name = str(name)
        self.command = str(command)
        self.args = [str(a) for a in (args or [])]
        self.env_extra = {str(k): str(v) for k, v in (env or {}).items()}
        self.cwd = str(cwd) if cwd else None
        self.timeout = float(timeout)
        self.proc = None
        self._lock = threading.Lock()
        self._id = 0
        self._tools = []
        self.server_info = {}
        self.closed = False

    # ------------------------------------------------------------------
    def start(self):
        env = dict(os.environ)
        env.update(self.env_extra)
        try:
            self.proc = subprocess.Popen(
                [self.command] + self.args, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                cwd=self.cwd, text=True, bufsize=1)
        except OSError as exc:
            raise McpError(f'{self.name}: cannot start {self.command}: {exc}') from None
        try:
            result = self.request('initialize', {
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {},
                'clientInfo': CLIENT_INFO,
            })
        except McpError as exc:
            self.close()
            raise McpError(f'{self.name}: initialize failed — {exc}') from None
        self.server_info = result.get('serverInfo') or {}
        try:
            self.notify('notifications/initialized')
        except McpError:
            pass                      # a server that ignores notifications is still usable
        self._tools = self.list_tools()
        return self.server_info

    def close(self):
        self.closed = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self.proc.kill()
                except OSError:
                    pass
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    def _send(self, payload):
        if not self.proc or self.proc.stdin is None:
            raise McpError(f'{self.name}: server is not running')
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpError(f'{self.name}: write failed ({exc})') from None

    def _read(self):
        """Read one JSON-RPC message, skipping notifications and non-JSON noise."""
        deadline = time.monotonic() + self.timeout
        while True:
            if not self.proc or self.proc.stdout is None:
                raise McpError(f'{self.name}: server is not running')
            if time.monotonic() > deadline:
                raise McpError(f'{self.name}: no response within {self.timeout:.0f}s')
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise McpError(f'{self.name}: server closed its output'
                               + (f' (exit {code})' if code is not None else ''))
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue                                  # stderr-style chatter on stdout
            if isinstance(msg, dict) and 'id' in msg:
                return msg

    def request(self, method, params=None):
        with self._lock:
            self._id += 1
            rid = self._id
            self._send({'jsonrpc': '2.0', 'id': rid, 'method': method,
                        'params': params if params is not None else {}})
            while True:
                msg = self._read()
                if msg.get('id') != rid:
                    continue                              # a stray late reply
                if 'error' in msg:
                    err = msg['error'] or {}
                    raise McpError(f'{self.name}: {err.get("message") or err} '
                                   f'(code {err.get("code")})')
                return msg.get('result') or {}

    def notify(self, method, params=None):
        with self._lock:
            self._send({'jsonrpc': '2.0', 'method': method,
                        'params': params if params is not None else {}})

    # ------------------------------------------------------------------
    def list_tools(self):
        result = self.request('tools/list')
        tools = result.get('tools') or []
        return [t for t in tools if isinstance(t, dict) and t.get('name')]

    @property
    def tools(self):
        return list(self._tools)

    def call_tool(self, name, arguments=None, max_chars=None):
        """Call one tool and return its text. Non-text parts are described, not dropped."""
        result = self.request('tools/call', {'name': name,
                                             'arguments': arguments or {}})
        parts = result.get('content') or []
        chunks = []
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                continue
            kind = part.get('type')
            if kind == 'text':
                chunks.append(str(part.get('text') or ''))
            elif kind == 'image':
                chunks.append(f'[{part.get("mimeType") or "image"}, '
                              f'{len(str(part.get("data") or ""))} base64 characters]')
            elif kind == 'resource':
                res = part.get('resource') or {}
                chunks.append(str(res.get('text') or res.get('uri') or ''))
            else:
                chunks.append(f'[{kind or "content"}]')
        if result.get('isError'):
            chunks.insert(0, 'ERROR: ')
        out = '\n'.join(c for c in chunks if c) or '(no content returned)'
        if max_chars and len(out) > max_chars:
            out = out[:max_chars] + f'\n... [truncated {len(out) - max_chars} chars]'
        return out


# ---------------------------------------------------------------------------
def namespace(server_name, tool_name, taken):
    """`server__tool`, unless `tool_name` is free or already namespaced."""
    if tool_name not in taken:
        return tool_name
    candidate = f'{server_name}__{tool_name}'
    if candidate in taken:
        n = 2
        while f'{candidate}_{n}' in taken:
            n += 1
        candidate = f'{candidate}_{n}'
    return candidate


def register_mcp(registry, servers, on_note=None, max_chars=None):
    """Start each configured MCP server and register its tools.

    Returns a list of `(server, registered_names)`. A server that will not start is
    reported and skipped — one broken MCP server must not take the session down, and it
    must never be quietly pretended to work either.
    """
    registered = []
    for name, spec in sorted((servers or {}).items()):
        if not isinstance(spec, dict) or not spec.get('command'):
            if on_note:
                on_note(f'MCP {name}: no "command" in its config — skipped')
            continue
        server = McpServer(name, spec['command'], args=spec.get('args'),
                           env=spec.get('env'), cwd=spec.get('cwd'),
                           timeout=float(spec.get('timeout') or 30))
        try:
            info = server.start()
        except McpError as exc:
            if on_note:
                on_note(str(exc))
            continue
        names = []
        for tool in server.tools:
            schema = tool.get('inputSchema') or {'type': 'object', 'properties': {}}
            if not isinstance(schema, dict):
                schema = {'type': 'object', 'properties': {}}
            tool_name = str(tool['name'])
            exposed = namespace(name, tool_name, set(registry.tools))
            description = str(tool.get('description') or f'MCP tool {tool_name} '
                              f'from {name}')
            registry.add(_mcp_tool(exposed, description, schema, server, tool_name,
                                   max_chars))
            names.append(exposed)
        version = info.get('version') or ''
        if on_note:
            on_note(f'MCP {name}: {len(names)} tool(s)'
                    + (f' (server v{version})' if version else '')
                    + (f' — {", ".join(names)}' if names else ''))
        registered.append((server, names))
    return registered


def _mcp_tool(exposed, description, schema, server, remote_name, max_chars):
    from .tools import Tool

    def call(args):
        if server.closed or not server.proc:
            raise McpError(f'{server.name}: not connected')
        return server.call_tool(remote_name, args or {}, max_chars=max_chars)

    # MCP servers run outside the sandbox; calling one is the user's explicit choice,
    # so it is declared `write` and goes through the normal approval path.
    return Tool(exposed, description, schema, call, risk='write')


def close_all(registered):
    for server, _names in registered:
        server.close()
