"""Bounded tool-calling agent loop."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .conversation import TurnResult
from .errors import CancelledByUser, StreamInterrupted
from .images import build_user_message
from .tokens import TokenCounter

class Agent:
    """Drives one turn: stream the answer, run any tool calls, feed results back, repeat.

    Three guarantees this class is responsible for:

    1. Reasoning never becomes answer text. `reasoning_content` is collected separately
       and is never written into the assistant history as `content`.
    2. A cancelled or interrupted generation never becomes a completed assistant message.
       Whatever the model produced this turn is rolled back, the user's message stays, and
       the user is told exactly what happened.
    3. The tool loop is bounded — by rounds, by per-call time, by total wall clock, and by
       the size of what gets fed back — so an agentic loop cannot run away.
    """

    def __init__(self, client, settings, registry=None, counter=None,
                 on_delta=None, on_reasoning=None, on_tool=None, on_notice=None,
                 on_tool_start=None, executor=None):
        self.client = client
        self.settings = settings
        self.registry = registry
        self.counter = counter or TokenCounter(client)
        self.on_delta = on_delta or (lambda t: None)
        self.on_reasoning = on_reasoning or (lambda t: None)
        self.on_tool = on_tool or (lambda name, args, result: None)
        self.on_notice = on_notice or (lambda t: None)
        self.on_tool_start = on_tool_start or (lambda name: None)
        self.executor = executor
        self.turn_started = None

    # ------------------------------------------------------------------
    def run(self, conversation, text='', attachments=None, regenerate=False):
        settings = self.settings
        vision = False
        if attachments:
            vision = self.client.probe_vision()
            if vision:
                self.on_notice(f'attaching {len(attachments)} image(s) as pixels '
                               '(server has a vision projector)')
            else:
                self.on_notice('server is text-only — sending measured image facts instead of pixels')
            message = build_user_message(text, attachments, vision)
        else:
            message = {'role': 'user', 'content': text or ''}
        # An empty turn is never worth sending: it would reach the server as a user
        # message with no content, which is both useless and, for a model, a confusing
        # instruction to continue from nothing.
        if not regenerate and (text or attachments or message.get('content')):
            conversation.add(message)
        # Everything from here on is produced by the model; a cancel rolls back to this
        # mark so no half-written tool group can survive into the history.
        rollback_mark = len(conversation.messages)

        tools = self.registry.schemas() if (self.registry and settings.use_tools) else None
        for note in settings.reasoning_notes(self.client.caps):
            self.on_notice(note)
        result = TurnResult()
        cancelled = False
        interrupted = ''
        self._preexecuted = {}
        turn_started = self.turn_started = time.monotonic()
        try:
            for round_no in range(1, settings.max_tool_rounds + 1):
                result.rounds = round_no
                result.stats.rounds = round_no
                assistant, usage, timings, finish = self._one_request(
                    conversation.wire(), tools, result.stats)
                conversation.add_assistant(assistant)
                if finish:
                    result.finish_reason = finish
                if assistant.get('reasoning_content'):
                    result.reasoning = ((result.reasoning + '\n') if result.reasoning else '') + \
                        assistant['reasoning_content']
                calls = assistant.get('tool_calls') or []
                if not calls:
                    result.text = assistant.get('content') or ''
                    break
                result.tool_calls.extend(calls)
                if round_no >= settings.max_tool_rounds:
                    self.on_notice(f'stopped after {round_no} tool rounds (raise --max-tool-rounds)')
                    result.text = assistant.get('content') or ''
                    break
                over_budget = False
                # Calls whose arguments completed during the stream were started then;
                # collect them in the order the model asked for them.
                pending = self._preexecuted or {}
                self._preexecuted = {}
                for call in calls:
                    fn = call.get('function') or {}
                    name = fn.get('name') or '?'
                    call_id = call.get('id') or f'call_{round_no}'
                    future = pending.pop(call_id, None)
                    if time.monotonic() - turn_started > settings.tool_budget and future is None:
                        answer = (f'ERROR: the tool budget for this turn '
                                  f'({settings.tool_budget:.0f}s) is exhausted; {name} was not run.')
                        over_budget = True
                    elif future is not None:
                        answer = self._collect(future, name)
                    else:
                        answer = self.registry.execute(
                            name, fn.get('arguments'), timeout=settings.tool_timeout,
                            max_output=settings.tool_max_output) if self.registry else \
                            'ERROR: tools are disabled'
                    conversation.add_tool_result(call_id, answer, name)
                    self.on_tool(name, fn.get('arguments') or '', answer)
                for _call_id, future in pending.items():
                    future.cancel()
                if over_budget:
                    self.on_notice('tool budget exhausted — asking the model to answer with '
                                   'what it has')
        except CancelledByUser:
            cancelled = True
            self._rollback(conversation, rollback_mark)
            self.on_notice('cancelled — the partial answer was discarded and the history is '
                           'unchanged (your message is still there; /retry to run it again)')
        except StreamInterrupted as e:
            interrupted = e.reason or 'stream interrupted'
            self._rollback(conversation, rollback_mark)
            self.on_notice(f'the stream ended early ({e.reason or "disconnected"}) after '
                           f'{len(e.partial_text)} characters; nothing was added to the '
                           'history, so no half-finished answer is remembered. Use /retry.')
        result.cancelled = cancelled
        result.interrupted = interrupted
        result.stats.cancelled = cancelled
        result.stats.interrupted = interrupted or None
        result.stats.elapsed = time.monotonic() - turn_started
        result.elapsed = result.stats.elapsed
        result.ttft = result.stats.ttft or 0.0
        result.usage = {k: v for k, v in (('prompt_tokens', result.stats.prompt_tokens),
                                          ('completion_tokens', result.stats.completion_tokens),
                                          ('total_tokens', result.stats.total_tokens))
                        if v is not None}
        result.timings = {}
        result.stats.tool_calls = len(result.tool_calls)
        return result

    # ------------------------------------------------------------------
    def _start_tool(self, call):
        """Run a tool the moment the model has finished writing its arguments.

        The call runs on a worker so it overlaps the tail of the generation instead of
        waiting for it. Bounded by the same per-turn tool budget and the same registry
        as everything else, and it is skipped (falling back to the post-stream path)
        when pre-execution is off or no registry exists.
        """
        settings = self.settings
        if not (self.registry and settings.use_tools and settings.preexecute_tools):
            return
        if self.executor is None:
            return
        fn = call.get('function') or {}
        name = fn.get('name') or '?'
        if time.monotonic() - (self.turn_started or time.monotonic()) > settings.tool_budget:
            return
        self.on_tool_start(name)
        self._preexecuted[call.get('id') or name] = self.executor.submit(
            self.registry.execute, name, fn.get('arguments'),
            settings.tool_timeout, settings.tool_max_output)

    def _collect(self, future, name):
        """Take a pre-executed call's result, turning any failure into tool output.

        A tool that raises still has to produce a message: the model needs to see that
        the call failed, and why, instead of seeing nothing at all.
        """
        try:
            return future.result(timeout=max(1.0, self.settings.tool_timeout))
        except Exception as exc:                       # noqa: BLE001 - reported, not hidden
            return f'ERROR: {name} failed: {type(exc).__name__}: {exc}'

    @staticmethod
    def _rollback(conversation, mark):
        """Drop everything the model produced this turn, keeping the user's message."""
        if len(conversation.messages) > mark:
            del conversation.messages[mark:]

    # ------------------------------------------------------------------
    def _one_request(self, messages, tools, stats):
        settings = self.settings
        params = settings.api_params(self.client.caps)
        if not settings.stream:
            resp = self.client.chat(messages, tools=tools, **params)
            choice = (resp.get('choices') or [{}])[0]
            msg = choice.get('message') or {}
            finish = choice.get('finish_reason') or ''
            stats.absorb(resp.get('usage'), resp.get('timings'), ttft=None, elapsed=0.0)
            text = msg.get('content') or ''
            if text:
                self.on_delta(text)
            rc = msg.get('reasoning_content')
            if rc:
                self.on_reasoning(rc)
            return msg, resp.get('usage'), resp.get('timings'), finish
        # `with` is not cosmetic: it closes the HTTP response the moment the turn ends,
        # including on the `return` below. Leaving that to the garbage collector let a
        # finished stream's unread bytes leak into the next request on the same socket.
        with self.client.stream_chat(messages, tools=tools, **params) as stream:
            try:
                for ev in stream:
                    if ev['kind'] == 'delta':
                        self.on_delta(ev['text'])
                    elif ev['kind'] == 'reasoning':
                        self.on_reasoning(ev['text'])
                    elif ev['kind'] == 'tool_call':
                        self._start_tool(ev['call'])
                    elif ev['kind'] == 'done':
                        self.client.note_accepted(stream.sent_fields)
                        stats.absorb(ev.get('usage'), ev.get('timings'),
                                     ttft=ev.get('ttft'), elapsed=ev.get('elapsed') or 0.0)
                        return (ev['message'], ev.get('usage'), ev.get('timings'),
                                ev.get('finish_reason'))
            except KeyboardInterrupt:
                stats.cancelled = True
                raise CancelledByUser() from None
            msg = {'role': 'assistant', 'content': stream.message().get('content', '')}
            stats.absorb(None, None, ttft=None, elapsed=0.0)
            return msg, None, None, None
