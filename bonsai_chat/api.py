"""The importable API — one object that does the whole job.

This is the surface to write against if you want Bonsai inside your own program. It is
deliberately small, it is the same code path the CLI uses (there is no second
implementation behind it), and it never raises where a returned object would do:

    from bonsai_chat.api import Bonsai

    bot = Bonsai(base_url='http://127.0.0.1:8080/v1', api_key='...')
    print(bot.ask('What is 6*7?').text)
    for chunk in bot.stream('Write a haiku about tensors'):
        print(chunk, end='', flush=True)

Everything measurable comes back on the result: `.usage`, `.timings`, `.ttft`,
`.elapsed`, `.rounds`, `.tool_calls`, and `.stats` for the full record. Values the
server did not report stay `None`; nothing is filled in with a plausible guess.

Constructing a `Bonsai` contacts nothing. The first request is when the endpoint is
probed, and `--doctor` (or `bot.doctor()`) is the way to find out what it supports
before you rely on it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._meta import DEFAULT_MODEL, VERSION
from .agent import Agent
from .conversation import Conversation, Settings
from .errors import BonsaiError
from .tokens import TokenCounter

__all__ = ['Bonsai', 'Result', 'run_batch']


class Result:
    """One turn, in a shape that is convenient from Python.

    Wraps the agent's `TurnResult` rather than replacing it, so `res.raw` always has the
    original with its full stats.
    """

    def __init__(self, raw):
        self.raw = raw
        self.text = raw.text
        self.reasoning = raw.reasoning
        self.usage = raw.usage
        self.timings = raw.timings
        self.ttft = raw.ttft
        self.elapsed = raw.elapsed
        self.rounds = raw.rounds
        self.tool_calls = list(raw.tool_calls)
        self.cancelled = raw.cancelled
        self.interrupted = raw.interrupted
        self.stats = raw.stats

    def __str__(self):
        return self.text or ''

    def to_dict(self):
        out = {'text': self.text, 'usage': self.usage, 'timings': self.timings,
               'ttft_ms': round(self.ttft * 1000, 1) if self.ttft else None,
               'elapsed_s': round(self.elapsed, 3), 'rounds': self.rounds,
               'tool_calls': len(self.tool_calls), 'cancelled': self.cancelled,
               'interrupted': self.interrupted}
        if self.reasoning:
            out['reasoning'] = self.reasoning
        return out


class Bonsai:
    """A conversation with one Bonsai endpoint, with tools, sessions and budgets."""

    def __init__(self, base_url=None, api_key=None, model=None, settings=None,
                 tools=True, sandbox=None, auto_approve=False, config_path=None,
                 endpoint=None, mcp_servers=None, budget=None, log=None):
        from .client import BonsaiClient
        from .config import Config
        from .endpoints import EndpointRegistry
        from .tools import ToolContext, ToolRegistry

        self.config = (Config.load(config_path, on_error=lambda m: (log or print)(m))
                       if config_path else Config())
        self.endpoints = EndpointRegistry.from_config(
            self.config.endpoints(), default_url=base_url, default_key=api_key,
            default_model=model)
        if endpoint:
            self.endpoints.select(endpoint)
        current = self.endpoints.current
        self.client = current.client()
        self.client.log = log or self.client.log
        self.settings = settings or Settings(
            model=model or current.model or DEFAULT_MODEL)
        self.counter = TokenCounter(self.client)
        self.conversation = Conversation(counter=self.counter)
        ctx = ToolContext(sandbox=Path(sandbox or '.').resolve(),
                          auto_approve=auto_approve,
                          notify=lambda t: (log or (lambda _m: None))(t))
        self.registry = ToolRegistry(ctx) if tools else None
        self.mcp = []
        servers = mcp_servers if mcp_servers is not None else self.config.mcp_servers()
        if servers and self.registry is not None:
            from .mcp import register_mcp
            self.mcp = register_mcp(self.registry, servers,
                                    on_note=lambda m: (log or (lambda _m: None))(m))
        self.budget = budget
        self.log = log or (lambda *a, **k: None)
        self._executor = None

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        from .mcp import close_all
        close_all(self.mcp)
        self.mcp = []
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self.client.close()

    # ------------------------------------------------------------------
    @property
    def _pool(self):
        from concurrent.futures import ThreadPoolExecutor
        if self._executor is None and self.settings.preexecute_tools:
            self._executor = ThreadPoolExecutor(max_workers=2,
                                                thread_name_prefix='bonsai-tool')
        return self._executor

    def agent(self, **hooks):
        return Agent(self.client, self.settings, registry=self.registry,
                     counter=self.counter, executor=self._pool, **hooks)

    # ------------------------------------------------------------------
    def ask(self, text='', attachments=None, regenerate=False, **hooks):
        """Send one message and wait for the whole answer."""
        reason = self.budget.over() if self.budget else None
        if reason:
            raise BonsaiError(reason)
        result = self.agent(**hooks).run(self.conversation, text,
                                         attachments=attachments,
                                         regenerate=regenerate)
        if self.budget:
            self.budget.note(result.usage)
        return Result(result)

    def stream(self, text='', attachments=None, **hooks):
        """Yield answer text as it arrives. The final `Result` arrives via `.result`."""
        chunks = []
        result_box = {}

        def on_delta(chunk):
            chunks.append(chunk)

        def on_reasoning(chunk):
            pass

        hooks.setdefault('on_delta', on_delta)
        hooks.setdefault('on_reasoning', on_reasoning)
        result = self.agent(**hooks).run(self.conversation, text,
                                         attachments=attachments)
        result_box['result'] = Result(result)
        if self.budget:
            self.budget.note(result.usage)
        yield from chunks
        self.last_result = result_box['result']

    # ------------------------------------------------------------------
    def tool(self, name, description, parameters, risk='safe'):
        """Register a Python function as a tool the model can call.

            @bot.tool('weather', 'Current weather for a city',
                      {'type': 'object',
                       'properties': {'city': {'type': 'string'}},
                       'required': ['city']})
            def weather(args):
                return 'raining'
        """
        if self.registry is None:
            raise BonsaiError('tools were disabled for this Bonsai instance')
        return self.registry.tool(name, description, parameters, risk=risk)

    # ------------------------------------------------------------------
    def save(self, path):
        return self.conversation.save(path)

    def load(self, path):
        self.conversation = Conversation.load(path, counter=self.counter)
        return self.conversation

    def reset(self, system=None):
        self.conversation = Conversation(counter=self.counter)
        if system is not None:
            self.conversation.set_system(system)
        return self.conversation

    def doctor(self):
        from .doctor import diagnose
        return diagnose(self.client, model=self.settings.model)

    def facts(self):
        return self.client.server_facts()


# ======================================================================
# Batch / non-interactive mode
#
# The same client, the same retry policy, the same tool loop — one item at a time,
# one fresh conversation per item, and a per-item error recorded rather than raised so
# a single bad line cannot lose the other 999.
# ======================================================================
def run_batch(items, bot=None, client=None, settings=None, on_item=None,
              max_items=None):
    """Run a list of requests non-interactively.

    Each item is a dict with either `messages` (a full list) or `prompt` (a string),
    and optionally `id`, `system`, `max_tokens`, `temperature`, `response_format`.

    Returns a list of result dicts, in input order, each with `ok`, and with `error`
    set when the item failed.
    """
    own = bot is None
    bot = bot or Bonsai()
    client = client or bot.client
    base_settings = settings or bot.settings
    results = []
    limit = len(items) if max_items is None else min(len(items), max_items)
    for index, item in enumerate(items[:limit]):
        if not isinstance(item, dict):
            results.append({'id': f'item{index}', 'ok': False,
                            'error': 'not a JSON object'})
            continue
        item_id = str(item.get('id') or f'item{index}')
        try:
            settings = _batch_settings(base_settings, item)
            conversation = Conversation(counter=bot.counter)
            if item.get('system') or item.get('prompt') is None:
                conversation.set_system(item.get('system') or
                                        (conversation.system_text or ''))
            messages = item.get('messages')
            if isinstance(messages, list) and messages:
                for message in messages:
                    # `tool` is allowed so the wire check below can reject an orphan
                    # result here rather than have the server reject the whole request.
                    if isinstance(message, dict) and message.get('role') in (
                            'system', 'user', 'assistant', 'tool'):
                        conversation.add(message)
            else:
                conversation.add_user(str(item.get('prompt') or ''))
            problems = conversation.validate_wire()
            if problems:
                raise BonsaiError('; '.join(problems))
            agent = Agent(client, settings, registry=bot.registry,
                          counter=bot.counter)
            raw = agent.run(conversation)
            if raw.cancelled or raw.interrupted:
                raise BonsaiError(raw.interrupted or 'cancelled')
            result = Result(raw).to_dict()
            result.update({'id': item_id, 'ok': True})
        except Exception as exc:                    # noqa: BLE001 - one item, one error
            result = {'id': item_id, 'ok': False,
                      'error': f'{type(exc).__name__}: {exc}'}
        results.append(result)
        if bot.budget:
            bot.budget.note(result.get('usage'))
        if on_item:
            on_item(result)
        if bot.budget and bot.budget.over():
            reason = bot.budget.over()
            for remaining in items[index + 1:limit]:
                results.append({'id': str(remaining.get('id')
                                          if isinstance(remaining, dict) else '?'),
                                'ok': False, 'error': f'not run — {reason}'})
            break
    if own:
        bot.close()
    return results


def _batch_settings(base, item):
    """A copy of the settings with this item's per-request overrides applied."""
    import copy
    settings = copy.copy(base)
    for key, attr in (('max_tokens', 'max_tokens'), ('temperature', 'temperature'),
                      ('top_p', 'top_p')):
        if item.get(key) is not None:
            setattr(settings, attr, item[key])
    if item.get('response_format'):
        settings.json_schema = item['response_format']
    return settings


def read_batch(path):
    """Read a .jsonl (or a JSON array) batch file. Bad lines are reported, not fatal."""
    text = Path(path).expanduser().read_text(encoding='utf-8')
    stripped = text.lstrip()
    if stripped.startswith('['):
        try:
            data = json.loads(stripped)
        except ValueError as exc:
            raise BonsaiError(f'{path}: {exc}') from None
        return [d for d in data if isinstance(d, dict)]
    items = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except ValueError as exc:
            items.append({'id': f'line{lineno}', 'ok': False,
                          '_parse_error': f'{exc}'})
    return items


def write_batch(results, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
    return path
