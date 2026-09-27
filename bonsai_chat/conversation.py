"""Conversation history, turn-group trimming, compaction and session files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import time

from ._meta import DEFAULT_MODEL, VERSION
from .reasoning import ReasoningConfig, parse_effort, parse_think_level
from .tokens import TokenCounter

DEFAULT_SYSTEM = (
    'You are Bonsai, a precise coding and reasoning assistant served from a single-slot '
    'llama.cpp endpoint. Answer in the language the user writes in. Use Markdown: fenced '
    'code blocks with a language tag, tables for comparisons, and short paragraphs. When a '
    'tool would give a better answer than guessing, call it; report tool output faithfully '
    'and say plainly when you do not know.')

WIRE_ROLES = ('system', 'user', 'assistant', 'tool')

SESSION_SCHEMA = 2

class Settings:
    """Everything the user can change about a turn, in one place.

    `reasoning` (a ReasoningConfig) is the single source of truth for thinking; `effort`
    and `think` are accessors over it so older callers, the `/effort` command and the
    `effort=`/`think=` constructor keywords all keep working without a second copy of the
    state to fall out of sync.
    """

    def __init__(self, model=DEFAULT_MODEL, temperature=1.0, top_p=0.95, max_tokens=2048,
                 reasoning=None, effort=None, think=None, stream=True, use_tools=True,
                 markdown=True, highlight=True, max_tool_rounds=6, context_budget=0.75,
                 context_reserve=512, output_reserve=1024, tool_timeout=60.0,
                 tool_budget=600.0, tool_max_output=24000, autosave_every=1,
                 show_stats=False, preexecute_tools=True, system=None,
                 json_schema=None, budget=None):
        #: run a tool as soon as the model finishes its arguments, while the rest of
        #: the turn is still being generated
        self.preexecute_tools = bool(preexecute_tools)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.reasoning = reasoning or ReasoningConfig('medium', 'medium', 'compact')
        if effort is not None:
            self.effort = effort
        if think is not None:
            self.think = think
        self.stream = stream
        self.use_tools = use_tools
        self.markdown = markdown
        self.highlight = highlight
        self.max_tool_rounds = max_tool_rounds
        self.context_budget = context_budget
        self.context_reserve = context_reserve      # tokens for the chat template
        self.output_reserve = output_reserve        # tokens for the answer being written
        self.tool_timeout = tool_timeout            # per tool call
        self.tool_budget = tool_budget              # total wall clock for one turn's tools
        self.tool_max_output = tool_max_output      # characters fed back per call
        self.autosave_every = autosave_every        # turns between session writes
        self.show_stats = show_stats                # /speed: always print the timing table
        self.json_schema = json_schema              # enforced locally when sent is unsafe
        self.budget = budget                        # a Budget, or None
        if system is not None:
            self.system = system
        else:
            self.system = None

    def __repr__(self):
        return (f'Settings(model={self.model!r}, temperature={self.temperature}, '
                f'top_p={self.top_p}, max_tokens={self.max_tokens}, '
                f'reasoning={self.reasoning.describe()!r}, stream={self.stream}, '
                f'use_tools={self.use_tools})')

    # -- reasoning accessors ---------------------------------------------------
    @property
    def effort(self):
        return self.reasoning.effort or 'none'

    @effort.setter
    def effort(self, value):
        v = str(value or '').lower()
        if v in ('none', 'off', ''):
            self.reasoning.level = 'off'
            self.reasoning.effort = None
        else:
            self.reasoning.effort = parse_effort(v)
            if self.reasoning.level == 'off':
                self.reasoning.level = 'medium'

    @property
    def think(self):
        return self.reasoning.level

    @think.setter
    def think(self, value):
        self.reasoning.level = parse_think_level(value)

    # --------------------------------------------------------------------------
    def api_params(self, caps=None):
        """Sampling + reasoning fields for one request, filtered by real capability."""
        params = {'temperature': self.temperature, 'top_p': self.top_p,
                  'max_tokens': self.max_tokens}
        fields, _ = self.reasoning.wire(caps)
        params.update(fields)
        # `response_format` is *asked for* here and *decided on* by the client, which
        # drops it unless this server proved it honours the field (see
        # BonsaiClient._apply_structured). Asking is free; assuming is not.
        if self.json_schema is not None:
            params['response_format'] = {'type': 'json_object'}
            params['json_schema'] = self.json_schema
        return {k: v for k, v in params.items() if v is not None}

    def reasoning_notes(self, caps=None):
        return self.reasoning.wire(caps)[1]

class TurnStats:
    """Measured numbers for one turn. Every field is either measured or absent.

    Nothing here is estimated: if the server did not report a value, the property returns
    None and the UI prints `n/a` rather than a plausible-looking number.
    """

    def __init__(self):
        self.ttft = None
        self.elapsed = 0.0
        self.prompt_tokens = None
        self.completion_tokens = None
        self.reasoning_tokens = None
        self.prompt_tokens_per_s = None
        self.tokens_per_s = None
        self.rounds = 0
        self.tool_calls = 0
        self.cancelled = False
        self.interrupted = None
        self.context_used = None
        self.context_window = None

    # ------------------------------------------------------------------
    def absorb(self, usage=None, timings=None, ttft=None, elapsed=0.0):
        usage = usage or {}
        timings = dict(timings or {})
        # Some builds nest the timing block inside usage instead of at the top level.
        if isinstance(usage.get('timings'), dict):
            for k, v in usage['timings'].items():
                timings.setdefault(k, v)
        if usage.get('prompt_tokens'):
            self.prompt_tokens = usage['prompt_tokens']
        if usage.get('completion_tokens'):
            self.completion_tokens = usage['completion_tokens']
        rc = usage.get('reasoning_tokens')
        if rc is None and isinstance(usage.get('completion_tokens_details'), dict):
            rc = usage['completion_tokens_details'].get('reasoning_tokens')
        if rc:
            self.reasoning_tokens = rc
        self.ttft = self.ttft if self.ttft is not None else ttft
        self.elapsed += elapsed or 0.0
        for src, dst in (('prompt_per_second', 'prompt_tokens_per_s'),
                         ('tokens_per_second', 'tokens_per_s'),
                         ('predicted_per_second', 'tokens_per_s'),
                         ('prompt_n', '_pn'), ('predicted_n', '_gn')):
            v = timings.get(src)
            if v and dst != '_pn' and dst != '_gn':
                setattr(self, dst, float(v))
        # Fall back to the server's own token/ms timings when it does not send rates.
        if self.tokens_per_s is None and timings.get('predicted_ms') and (
                timings.get('predicted_n') or self.completion_tokens):
            n = timings.get('predicted_n') or self.completion_tokens
            self.tokens_per_s = float(n) / (float(timings['predicted_ms']) / 1000.0)
        if self.prompt_tokens_per_s is None and timings.get('prompt_ms') and (
                timings.get('prompt_n') or self.prompt_tokens):
            n = timings.get('prompt_n') or self.prompt_tokens
            self.prompt_tokens_per_s = float(n) / (float(timings['prompt_ms']) / 1000.0)

    def rate(self):
        if self.tokens_per_s:
            return float(self.tokens_per_s)
        if self.elapsed and self.completion_tokens:
            # Wall-clock fallback: includes network and tool time, so it is labelled as
            # such by the caller and never printed as a decode rate.
            return self.completion_tokens / self.elapsed
        return 0.0

    @property
    def total_tokens(self):
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    def context_utilization(self):
        if not self.context_window or self.context_used is None:
            return None
        return 100.0 * self.context_used / self.context_window

    def to_dict(self):
        return {'ttft_ms': round(self.ttft * 1000, 1) if self.ttft else None,
                'elapsed_s': round(self.elapsed, 3),
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'reasoning_tokens': self.reasoning_tokens,
                'total_tokens': self.total_tokens,
                'prompt_tokens_per_s': round(self.prompt_tokens_per_s, 1)
                if self.prompt_tokens_per_s else None,
                'tokens_per_s': round(self.tokens_per_s, 2) if self.tokens_per_s else None,
                'rounds': self.rounds, 'tool_calls': self.tool_calls,
                'cancelled': self.cancelled, 'interrupted': self.interrupted,
                'context_used': self.context_used, 'context_window': self.context_window}

class Conversation:
    """Message history with token-budget trimming that never orphans a tool result."""

    #: bumped when the on-disk session format changes in an incompatible way
    SCHEMA = SESSION_SCHEMA

    def __init__(self, system=DEFAULT_SYSTEM, counter=None):
        self.counter = counter or TokenCounter()
        self.messages = []
        self.system_text = None
        self.schema = SESSION_SCHEMA
        self.saved_version = None
        self.saved_branches = None
        self.skipped_lines = 0
        if system:
            self.set_system(system)

    # ------------------------------------------------------------------
    def set_system(self, text):
        self.system_text = text
        if self.messages and self.messages[0]['role'] == 'system':
            if text:
                self.messages[0] = {'role': 'system', 'content': text}
            else:
                self.messages.pop(0)
        elif text:
            self.messages.insert(0, {'role': 'system', 'content': text})

    def add(self, message):
        self.messages.append(message)
        return message

    def add_user(self, content):
        return self.add({'role': 'user', 'content': content})

    def add_assistant(self, message):
        wire = {'role': 'assistant', 'content': message.get('content')}
        if message.get('tool_calls'):
            wire['tool_calls'] = message['tool_calls']
            wire['content'] = message.get('content')
        if wire['content'] is None and not wire.get('tool_calls'):
            wire['content'] = ''
        return self.add(wire)

    def add_tool_result(self, tool_call_id, content, name=None):
        msg = {'role': 'tool', 'tool_call_id': tool_call_id, 'content': content}
        if name:
            msg['name'] = name
        return self.add(msg)

    def last_user_index(self):
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i]['role'] == 'user':
                return i
        return None

    def wire(self):
        """Messages in exactly the shape the API expects."""
        out = []
        for m in self.messages:
            if m['role'] not in WIRE_ROLES:
                continue
            if m['role'] == 'tool':
                out.append({'role': 'tool', 'tool_call_id': m.get('tool_call_id'),
                            'content': m.get('content', '')})
                continue
            item = {'role': m['role'], 'content': m.get('content', '')}
            if m.get('tool_calls'):
                item['tool_calls'] = m['tool_calls']
            out.append(item)
        return out

    def tokens(self):
        return self.counter.messages(self.wire())

    def truncate(self, index):
        """Cut the history back to `index` messages."""
        if index < len(self.messages):
            removed = len(self.messages) - index
            del self.messages[index:]
            return removed
        return 0

    def drop_last_turn(self):
        """Remove the newest user turn and everything after it (for /undo and /retry)."""
        i = self.last_user_index()
        if i is None:
            return 0
        removed = len(self.messages) - i
        del self.messages[i:]
        return removed

    def turn_groups(self):
        """Split the history into (start, end) spans, one per user turn.

        A group is a user message plus every assistant/tool message that answers it. All
        removal happens at group granularity: dropping anything finer can orphan a `tool`
        message from the `tool_calls` it belongs to, and llama.cpp rejects the whole
        request when that happens.
        """
        start = 0
        while start < len(self.messages) and self.messages[start]['role'] == 'system':
            start += 1                     # system + any compaction digest are never dropped
        groups = []
        begin = start
        for i in range(start + 1, len(self.messages)):
            if self.messages[i]['role'] == 'user':
                if i > begin:
                    groups.append((begin, i))
                begin = i
        if begin < len(self.messages):
            groups.append((begin, len(self.messages)))
        return groups

    def trim(self, budget):
        """Drop the oldest complete turns until the history fits `budget` tokens.

        Removal happens per turn group so a `tool` message can never survive without the
        `tool_calls` it belongs to. Token counts come from the counter's cache, so the
        repeated re-measurement inside this loop costs one server round trip per new
        string, not one per iteration.
        """
        dropped = 0
        while self.tokens() > budget and len(self.messages) > 2:
            groups = self.turn_groups()
            if not groups:
                break
            start, end = groups[0]
            del self.messages[start:end]
            dropped += end - start
        return dropped

    def compact(self, budget, keep_recent=2, summarizer=None):
        """Compact old turns into a single system-visible digest instead of dropping them.

        `trim` throws history away; `compact` preserves what it can. The oldest turns are
        summarised (by `summarizer`, else by a cheap extractive digest) and folded into a
        `[earlier in this conversation]` block placed after the system message. The most
        recent `keep_recent` turns, the system prompt and every tool pair in the kept
        region are preserved verbatim.

        Returns a dict describing what happened so the UI can tell the user.
        """
        groups = self.turn_groups()
        if len(groups) <= keep_recent:
            return {'compacted': 0, 'kept': len(groups), 'tokens_before': self.tokens(),
                    'tokens_after': self.tokens()}
        before = self.tokens()
        keep_from = groups[-keep_recent][0] if keep_recent else len(self.messages)
        old = self.messages[1 if (self.messages and self.messages[0]['role'] == 'system')
                            else 0:keep_from]
        digest = (summarizer(old) if summarizer else self._extractive_digest(old))
        head = 0
        while head < len(self.messages) and self.messages[head]['role'] == 'system':
            head += 1
        self.messages = (self.messages[:head] +
                         [{'role': 'system',
                           'content': ('[earlier in this conversation — compacted]\n' + digest)}] +
                         self.messages[keep_from:])
        after = self.tokens()
        # Still over budget after compaction? Fall back to whole-turn trimming.
        dropped = self.trim(budget) if after > budget else 0
        return {'compacted': len(old), 'kept': keep_recent, 'tokens_before': before,
                'tokens_after': self.tokens(), 'trimmed': dropped}

    @staticmethod
    def _extractive_digest(messages):
        """A cheap, honest digest: who said what, first line only. No model call."""
        lines = []
        for m in messages:
            body = m.get('content')
            if isinstance(body, list):
                body = ' '.join(p.get('text', '') for p in body if isinstance(p, dict))
            body = ' '.join((body or '').split())
            role = m['role']
            if role == 'user':
                lines.append('- user: ' + body[:200])
            elif role == 'assistant':
                if m.get('tool_calls'):
                    names = ', '.join((tc.get('function') or {}).get('name', '?')
                                      for tc in m['tool_calls'])
                    lines.append(f'- assistant called tool(s): {names}')
                elif body:
                    lines.append('- assistant: ' + body[:200])
            elif role == 'tool':
                lines.append(f'- tool {m.get("name") or m.get("tool_call_id")}: ' + body[:120])
        return '\n'.join(lines) or '(nothing recorded)'

    def validate_wire(self):
        """Check the outgoing message list for the things llama.cpp rejects.

        Returns a list of human-readable problems. Cheap enough to run before every
        request, and it turns a confusing server-side 400 into a precise local message.
        """
        problems = []
        wire = self.wire()
        pending = set()
        for i, m in enumerate(wire):
            role = m.get('role')
            if role == 'assistant' and m.get('tool_calls'):
                for tc in m['tool_calls']:
                    pending.add(tc.get('id'))
            if role == 'tool':
                cid = m.get('tool_call_id')
                if cid not in pending:
                    problems.append(f'message {i}: orphan tool result (no matching tool_call '
                                    f'for id {cid!r})')
                else:
                    pending.discard(cid)
            if role == 'assistant' and not m.get('tool_calls') and not m.get('content'):
                problems.append(f'message {i}: empty assistant message')
        if pending:
            problems.append(f'unanswered tool_call(s): {", ".join(sorted(str(p) for p in pending))}')
        return problems

    # ------------------------------------------------------------------
    def save(self, path, branches=None):
        """Atomic write: temp file in the same directory, fsync, then rename.

        A notebook runtime can be killed at any moment; a torn session file must not cost
        the user their conversation.

        `branches` is written as a second meta line. Session schema 2 is unchanged and
        readers that do not know about branches skip the line and read the messages.
        """
        path = Path(path).expanduser()
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as fh:
            fh.write(json.dumps({'_meta': True, 'schema': SESSION_SCHEMA,
                                 'version': VERSION,
                                 'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                                ensure_ascii=False) + '\n')
            if branches:
                fh.write(json.dumps({'_meta': True, 'branches': branches},
                                    ensure_ascii=False) + '\n')
            for m in self.messages:
                fh.write(json.dumps(m, ensure_ascii=False) + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path

    @classmethod
    def load(cls, path, counter=None, on_error=None):
        """Load a session file, tolerating corruption and older formats.

        A truncated last line (killed mid-write), a stray blank, or a file from an older
        schema must not crash the client. Bad lines are skipped and reported; the rest of
        the conversation is kept.
        """
        path = Path(path).expanduser()
        conv = cls(system=None, counter=counter)
        skipped = 0
        for lineno, line in enumerate(path.read_text(encoding='utf-8', errors='replace')
                                   .splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if not isinstance(msg, dict):
                skipped += 1
                continue
            if msg.get('_meta'):
                # Only a line that *carries* a schema may set it: the extra branches
                # meta line has none, and defaulting it to 1 would downgrade a v2 file.
                if msg.get('schema') is not None:
                    conv.schema = msg['schema']
                conv.saved_version = msg.get('version')
                if msg.get('branches'):
                    conv.saved_branches = msg['branches']
                continue
            if msg.get('role') not in WIRE_ROLES:
                skipped += 1
                continue
            if msg.get('role') == 'system' and not conv.messages:
                conv.set_system(msg.get('content') or '')
            else:
                conv.messages.append(msg)
        conv.skipped_lines = skipped
        if skipped and on_error:
            on_error(skipped)
        # Drop anything that would be rejected by the server, e.g. a tool result whose
        # assistant tool_call line was the one that got truncated away.
        conv._repair_orphans()
        return conv

    def _repair_orphans(self):
        """Remove tool results with no matching tool_call, and vice versa."""
        known = set()
        for m in self.messages:
            for tc in m.get('tool_calls') or []:
                known.add(tc.get('id'))
        self.messages = [m for m in self.messages
                         if m.get('role') != 'tool' or m.get('tool_call_id') in known]
        answered = {m.get('tool_call_id') for m in self.messages if m.get('role') == 'tool'}
        for m in self.messages:
            if m.get('tool_calls'):
                m['tool_calls'] = [tc for tc in m['tool_calls'] if tc.get('id') in answered]
                if not m['tool_calls']:
                    m.pop('tool_calls', None)
                    if not m.get('content'):
                        m['content'] = ''

    def export_markdown(self, path, title='Bonsai session'):
        lines = [f'# {title}', '']
        for m in self.messages:
            role = m['role']
            body = m.get('content')
            if isinstance(body, list):
                body = '\n'.join(p.get('text', f"[{p.get('type')}]") for p in body if isinstance(p, dict))
            body = body or ''
            if role == 'system':
                lines += ['## System', '', '```', body, '```', '']
            elif role == 'user':
                lines += ['## You', '', body, '']
            elif role == 'assistant':
                lines += ['## Bonsai', '']
                if m.get('tool_calls'):
                    lines += ['```json',
                              json.dumps(m['tool_calls'], ensure_ascii=False, indent=2),
                              '```', '']
                if body:
                    lines += [body, '']
            elif role == 'tool':
                lines += [f'### tool result — `{m.get("name") or m.get("tool_call_id")}`', '',
                          '```', body[:4000], '```', '']
        path = Path(path).expanduser()
        path.write_text('\n'.join(lines), encoding='utf-8')
        return path

def mask_key(key):
    if not key:
        return '(no key)'
    if len(key) <= 12:
        return key[:2] + '…'
    return key[:6] + '…' + key[-4:]

@dataclass
class TurnResult:
    """What one user turn produced. `stats` holds the measured numbers."""
    text: str = ''
    reasoning: str = ''
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ''
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    elapsed: float = 0.0
    ttft: float = 0.0
    rounds: int = 0
    cancelled: bool = False
    interrupted: str = ''
    stats: TurnStats = field(default_factory=TurnStats)

    @property
    def prompt_tokens(self):
        return self.stats.prompt_tokens or 0

    @property
    def completion_tokens(self):
        return self.stats.completion_tokens or 0

    def rate(self):
        return self.stats.rate()
