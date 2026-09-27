"""Sandboxed tool registry with approval, timeout and output caps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import subprocess
import threading
import time
import urllib.error, urllib.parse, urllib.request

from ._meta import VERSION
from .errors import BonsaiError
from .images import ImageAttachment

class ToolError(BonsaiError):
    """A tool refused or failed; the message goes back to the model so it can recover."""

class ToolTimeout(ToolError):
    """A tool call exceeded its wall-clock budget and was abandoned.

    The worker thread is not killed (Python cannot do that safely); it is left to finish
    on its own and the model is told the call timed out, which is the honest outcome.
    """

    def __init__(self, timeout):
        self.timeout = timeout
        super().__init__(f'tool exceeded its {timeout:.0f}s budget')

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: object
    risk: str = 'safe'          # safe | write | dangerous
    enabled: bool = True

    def schema(self):
        return {'type': 'function',
                'function': {'name': self.name, 'description': self.description,
                             'parameters': self.parameters}}

@dataclass
class ToolContext:
    sandbox: Path
    approve: object = None            # fn(name, args, risk) -> bool
    notify: object = None             # fn(text) -> None
    http_timeout: int = 30
    auto_approve: bool = False
    max_output: int = 8000

    def say(self, text):
        if self.notify:
            self.notify(text)

    def ask(self, name, args, risk):
        if self.auto_approve:
            return True
        if self.approve is None:
            return risk == 'safe'
        return bool(self.approve(name, args, risk))

    def inside(self, path):
        """Resolve a path and keep it inside the sandbox."""
        p = Path(str(path)).expanduser()
        p = p if p.is_absolute() else (self.sandbox / p)
        try:
            resolved = p.resolve()
            root = self.sandbox.resolve()
        except OSError as e:
            raise ToolError(f'cannot resolve path {path}: {e}') from None
        if resolved != root and root not in resolved.parents:
            raise ToolError(f'{path} is outside the sandbox ({root}); refusing.')
        return resolved

    def truncate_to(self, text, limit):
        """Cap tool output at `limit` characters, saying so, so the model is not misled."""
        if limit is None or limit <= 0 or len(text) <= limit:
            return text
        head = limit * 3 // 4
        tail = limit // 4
        return (text[:head] +
                f'\n… [{len(text) - head - tail} characters omitted to stay inside the '
                f'{limit}-character tool output limit] …\n' + text[-tail:])

    def truncate(self, text):
        text = str(text)
        if len(text) <= self.max_output:
            return text
        return (text[:self.max_output] +
                f'\n... [truncated {len(text) - self.max_output} chars — narrow the request]')

_MATH_FUNCS = {
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
    'sqrt': __import__('math').sqrt, 'floor': __import__('math').floor,
    'ceil': __import__('math').ceil, 'log': __import__('math').log,
    'log2': __import__('math').log2, 'log10': __import__('math').log10,
    'exp': __import__('math').exp, 'sin': __import__('math').sin, 'cos': __import__('math').cos,
    'tan': __import__('math').tan, 'atan2': __import__('math').atan2, 'pi': __import__('math').pi,
    'e': __import__('math').e, 'radians': __import__('math').radians,
    'degrees': __import__('math').degrees, 'hypot': __import__('math').hypot,
}

def _safe_eval(node):
    import ast as _ast
    import operator as _op
    binops = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.Div: _op.truediv,
              _ast.FloorDiv: _op.floordiv, _ast.Mod: _op.mod, _ast.Pow: _op.pow}
    unops = {_ast.UAdd: _op.pos, _ast.USub: _op.neg}
    if isinstance(node, _ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, _ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError('only numbers are allowed in expressions')
    if isinstance(node, _ast.BinOp) and type(node.op) in binops:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, _ast.Pow) and abs(right) > 64:
            raise ToolError('exponent too large')
        return binops[type(node.op)](left, right)
    if isinstance(node, _ast.UnaryOp) and type(node.op) in unops:
        return unops[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
        fn = _MATH_FUNCS.get(node.func.id)
        if fn is None or not callable(fn):
            raise ToolError(f'function {node.func.id!r} is not available')
        return fn(*[_safe_eval(a) for a in node.args])
    if isinstance(node, _ast.Name):
        val = _MATH_FUNCS.get(node.id)
        if val is None:
            raise ToolError(f'unknown name {node.id!r}')
        return val
    raise ToolError('unsupported expression syntax')

class ToolRegistry:
    """Built-in tools + a decorator for adding your own.

    Everything the model can touch is confined to `sandbox` and anything that writes or
    executes goes through the approval callback first.
    """

    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.tools = {}
        self.calls_run = 0
        self.last_duration = 0.0
        self._register_builtins()

    # ------------------------------------------------------------------
    def tool(self, name, description, parameters, risk='safe'):
        def deco(fn):
            self.tools[name] = Tool(name, description, parameters, fn, risk=risk)
            return fn
        return deco

    def add(self, tool: Tool):
        self.tools[tool.name] = tool

    def _register_builtins(self):
        ctx = self.ctx

        @self.tool('read_file',
                   'Read a text file inside the sandbox and return its contents.',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string', 'description': 'File path, relative to the sandbox'},
                                   'max_lines': {'type': 'integer', 'description': 'Maximum lines to return (default 200)'}},
                    'required': ['path']})
        def read_file(args):
            path = ctx.inside(args['path'])
            if not path.is_file():
                raise ToolError(f'no such file: {args["path"]}')
            max_lines = int(args.get('max_lines') or 200)
            try:
                text = path.read_text(errors='replace')
            except OSError as e:
                raise ToolError(f'cannot read {args["path"]}: {e}') from None
            lines = text.splitlines()
            head = lines[:max_lines]
            out = '\n'.join(head)
            if len(lines) > max_lines:
                out += f'\n... [{len(lines) - max_lines} more lines; raise max_lines to see them]'
            return ctx.truncate(out or '(empty file)')

        @self.tool('list_dir',
                   'List the entries of a directory inside the sandbox with sizes.',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string', 'description': 'Directory (default: sandbox root)'}},
                    'required': []})
        def list_dir(args):
            path = ctx.inside(args.get('path') or '.')
            if not path.is_dir():
                raise ToolError(f'not a directory: {args.get("path")}')
            rows = []
            for entry in sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))[:200]:
                if entry.is_dir():
                    rows.append(f'{entry.name}/')
                else:
                    rows.append(f'{entry.name}  ({entry.stat().st_size} bytes)')
            return ctx.truncate('\n'.join(rows) or '(empty directory)')

        @self.tool('search_text',
                   'Search file contents in the sandbox with a regular expression and return matching lines.',
                   {'type': 'object',
                    'properties': {'pattern': {'type': 'string', 'description': 'Regular expression'},
                                   'path': {'type': 'string', 'description': 'Directory or file to search (default: sandbox root)'},
                                   'glob': {'type': 'string', 'description': 'Filename glob, e.g. *.py (default *)'},
                                   'max_results': {'type': 'integer', 'description': 'Maximum matches (default 25)'}},
                    'required': ['pattern']})
        def search_text(args):
            root = ctx.inside(args.get('path') or '.')
            try:
                rx = re.compile(args['pattern'], re.IGNORECASE)
            except re.error as e:
                raise ToolError(f'invalid regular expression: {e}') from None
            limit = int(args.get('max_results') or 25)
            glob = args.get('glob') or '*'
            files = [root] if root.is_file() else sorted(root.rglob(glob))
            hits, scanned = [], 0
            for f in files:
                if not f.is_file() or f.stat().st_size > 2_000_000:
                    continue
                scanned += 1
                if scanned > 2000:
                    hits.append('... [stopped after scanning 2000 files]')
                    break
                try:
                    for n, line in enumerate(f.read_text(errors='replace').splitlines(), 1):
                        if rx.search(line):
                            try:
                                rel = f.relative_to(ctx.sandbox.resolve())
                            except ValueError:
                                rel = f
                            hits.append(f'{rel}:{n}: {line.strip()[:200]}')
                            if len(hits) >= limit:
                                return ctx.truncate('\n'.join(hits))
                except OSError:
                    continue
            return ctx.truncate('\n'.join(hits) or 'no matches')

        @self.tool('write_file',
                   'Write text to a file inside the sandbox (creates or overwrites).',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                    'required': ['path', 'content']},
                   risk='write')
        def write_file(args):
            path = ctx.inside(args['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get('content', '')))
            return f'wrote {path.stat().st_size} bytes to {args["path"]}'

        @self.tool('run_shell',
                   'Run a shell command in the sandbox and return combined stdout/stderr. Needs approval.',
                   {'type': 'object',
                    'properties': {'command': {'type': 'string'},
                                   'timeout': {'type': 'integer', 'description': 'Seconds (default 60)'}},
                    'required': ['command']},
                   risk='dangerous')
        def run_shell(args):
            cmd = str(args['command'])
            timeout = min(int(args.get('timeout') or 60), 300)
            try:
                proc = subprocess.run(cmd, shell=True, cwd=str(ctx.sandbox), capture_output=True,
                                      text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise ToolError(f'command timed out after {timeout}s') from None
            out = (proc.stdout or '') + (('\n[stderr]\n' + proc.stderr) if proc.stderr else '')
            return ctx.truncate(out.strip() + f'\n[exit code {proc.returncode}]')

        @self.tool('http_get',
                   'Fetch a URL and return the beginning of the response body. Needs approval.',
                   {'type': 'object',
                    'properties': {'url': {'type': 'string'},
                                   'max_bytes': {'type': 'integer', 'description': 'Default 8000'}},
                    'required': ['url']},
                   risk='dangerous')
        def http_get(args):
            url = str(args['url'])
            if not re.match(r'^https?://', url):
                raise ToolError('only http(s) URLs are allowed')
            limit = int(args.get('max_bytes') or 8000)
            req = urllib.request.Request(url, headers={'User-Agent': f'bonsai-chat/{VERSION}'})
            try:
                with urllib.request.urlopen(req, timeout=ctx.http_timeout) as r:
                    body = r.read(limit + 1)
                    ctype = r.headers.get('Content-Type', '')
            except Exception as e:
                raise ToolError(f'request failed: {e}') from None
            text = body[:limit].decode('utf-8', 'replace')
            more = f'\n... [truncated to {limit} bytes]' if len(body) > limit else ''
            return ctx.truncate(f'[content-type: {ctype}]\n{text}{more}')

        @self.tool('calculator',
                   'Evaluate an arithmetic expression. Supports + - * / // % ** and '
                   'sqrt, log, log2, log10, exp, sin, cos, tan, atan2, hypot, floor, ceil, '
                   'abs, round, min, max, sum, pow, pi, e.',
                   {'type': 'object',
                    'properties': {'expression': {'type': 'string', 'description': 'e.g. 2*(3+4)/sqrt(2)'}},
                    'required': ['expression']})
        def calculator(args):
            import ast as _ast
            expr = str(args['expression'])
            if len(expr) > 500:
                raise ToolError('expression too long')
            try:
                tree = _ast.parse(expr, mode='eval')
            except SyntaxError as e:
                raise ToolError(f'cannot parse expression: {e.msg}') from None
            value = _safe_eval(tree)
            return f'{expr} = {value}'

        @self.tool('current_time',
                   'Current date and time (UTC by default).',
                   {'type': 'object',
                    'properties': {'utc_offset_hours': {'type': 'number', 'description': 'Optional offset from UTC'}},
                    'required': []})
        def current_time(args):
            import datetime as _dt
            offset = args.get('utc_offset_hours')
            if offset is None:
                now = _dt.datetime.now(_dt.timezone.utc)
                label = 'UTC'
            else:
                tz = _dt.timezone(_dt.timedelta(hours=float(offset)))
                now = _dt.datetime.now(tz)
                label = f'UTC{float(offset):+g}'
            return now.strftime(f'%Y-%m-%d %H:%M:%S {label} (%A)')

        @self.tool('image_inspect',
                   'Measure an image file: format, pixel dimensions, Pillow colour statistics '
                   'and (when tesseract is installed) OCR text. Use this when the user mentions '
                   'an image and the server cannot see it.',
                   {'type': 'object',
                    'properties': {'path': {'type': 'string'},
                                   'ocr': {'type': 'boolean', 'description': 'Attempt OCR if available'}},
                    'required': ['path']})
        def image_inspect(args):
            path = ctx.inside(args['path'])
            att = ImageAttachment.load(path, encode=False, use_ocr=bool(args.get('ocr')))
            return ctx.truncate(att.text_card(vision=False))

    # ------------------------------------------------------------------
    def schemas(self):
        return [t.schema() for t in self.tools.values() if t.enabled]

    def names(self):
        return sorted(self.tools)

    def set_enabled(self, name, enabled):
        if name not in self.tools:
            raise ToolError(f'unknown tool: {name}')
        self.tools[name].enabled = bool(enabled)

    def parse_arguments(self, raw):
        """Model-supplied arguments -> dict, with a repairable error message on bad JSON."""
        if raw in (None, ''):
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            raise ToolError(f'arguments were not valid JSON ({e}); send them as a JSON object') from None
        if not isinstance(parsed, dict):
            raise ToolError('arguments must be a JSON object')
        return parsed

    def execute(self, name, raw_args, timeout=None, max_output=None):
        """Run one tool call; always returns a string for the model (errors included).

        `timeout` bounds a single call and `max_output` bounds what is fed back, so a tool
        that hangs or dumps a 50 MB log cannot stall or overflow the turn. Approval is
        asked *before* the timed section — a prompt waiting on the user is not a hang.
        """
        tool = self.tools.get(name)
        if tool is None:
            return f'ERROR: unknown tool {name!r}. Available: {", ".join(self.names())}'
        if not tool.enabled:
            return f'ERROR: tool {name!r} is disabled in this session.'
        try:
            args = self.parse_arguments(raw_args)
        except ToolError as e:
            return f'ERROR: {e}'
        if tool.risk != 'safe' and not self.ctx.ask(name, args, tool.risk):
            return (f'DENIED: the user declined to run {name} with {json.dumps(args)[:300]}. '
                    'Do not retry it; answer with what you already know.')
        self.ctx.say(f'⚙ {name} {json.dumps(args, ensure_ascii=False)[:200]}')
        started = time.monotonic()
        try:
            result = self._call(tool.func, args, timeout)
        except ToolTimeout:
            # Must precede the generic ToolError handler: ToolTimeout subclasses it.
            return (f'ERROR: {name} was still running after {timeout:.0f}s and was abandoned. '
                    'Report that it timed out; do not retry it.')
        except ToolError as e:
            return f'ERROR: {e}'
        except Exception as e:
            return f'ERROR: {type(e).__name__}: {e}'
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        self.last_duration = time.monotonic() - started
        self.calls_run += 1
        return self.ctx.truncate_to(result, max_output) if max_output else result

    @staticmethod
    def _call(func, args, timeout):
        """Run `func` with a wall-clock bound.

        The function runs in a daemon thread so a hung tool cannot wedge the chat loop.
        The thread is not killed (Python cannot do that safely); it is abandoned and the
        model is told the call timed out, which is the honest outcome.
        """
        if not timeout or timeout <= 0:
            return func(args)
        box = {}

        def runner():
            try:
                box['value'] = func(args)
            except BaseException as e:      # noqa: BLE001 - surfaced to the model
                box['error'] = e

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise ToolTimeout(timeout)
        if 'error' in box:
            raise box['error']
        return box.get('value')
