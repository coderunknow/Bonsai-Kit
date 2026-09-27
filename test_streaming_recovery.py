#!/usr/bin/env python3
"""v0.5.0 regression tests: streaming robustness, cancellation, recovery, context,
reasoning control, transport safety and diagnostics.

Everything runs offline against mock_bonsai_server, which reproduces each failure mode
deterministically. The code under test is the shipped client, over real HTTP.

    python3 -m unittest -v test_streaming_recovery
"""
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bonsai_chat as bc
from mock_bonsai_server import DEFAULT_KEY, SCENARIOS, MockBonsaiServer


def plain_style(width=100):
    return bc.Style(force_color=False, width=width)


class MockTestCase(unittest.TestCase):
    """Starts the stub with a scenario and hands out a client + app."""

    server_options = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = MockBonsaiServer(**self.server_options).start()
        self.addCleanup(self.server.stop)
        self.out = io.StringIO()
        self.client = bc.BonsaiClient(self.server.base_url, DEFAULT_KEY, retries=1)
        self.addCleanup(self.client.close)
        self.settings = bc.Settings(max_tokens=256)

    def app(self, input_lines=(), registry=None, **kw):
        registry = registry or self.registry()
        conversation = bc.Conversation(system=bc.DEFAULT_SYSTEM,
                                       counter=bc.TokenCounter(self.client))
        inputs = list(input_lines)

        def fake_input(prompt=''):
            if not inputs:
                raise EOFError
            return inputs.pop(0)

        return bc.make_app(self.client, self.settings, plain_style(), registry=registry,
                           conversation=conversation, sandbox=self.root, out=self.out,
                           input_fn=fake_input, auto_approve=kw.pop('auto_approve', True))

    def registry(self):
        ctx = bc.ToolContext(sandbox=self.root, auto_approve=True)
        registry = bc.ToolRegistry(ctx)
        registry.add(bc.Tool('get_weather', 'Current weather for a city',
                             {'type': 'object',
                              'properties': {'city': {'type': 'string'}}, 'required': ['city']},
                             func=lambda a: f'21C and sunny in {a.get("city")}'))
        return registry

    def transcript(self):
        return self.out.getvalue()


# ======================================================================
# SSE decoding
# ======================================================================
class SSEDecoderTests(unittest.TestCase):

    def feed(self, chunks):
        d = bc.SSEDecoder()
        events = []
        for c in chunks:
            events.extend(d.feed(c))
        tail = d.close()
        if tail is not None:
            events.append(tail)
        return events, d

    def test_fragmented_across_arbitrary_byte_boundaries(self):
        raw = b'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
        one_byte = [raw[i:i + 1] for i in range(len(raw))]
        events, _ = self.feed(one_byte)
        self.assertEqual(events, ['{"a": 1}', '{"b": 2}'])

    def test_crlf_and_lf_are_both_accepted(self):
        events, _ = self.feed([b'data: a\r\n\r\ndata: b\n\n'])
        self.assertEqual(events, ['a', 'b'])

    def test_comments_and_keepalives_are_ignored_but_counted(self):
        events, d = self.feed([b': keep-alive\n\n', b':ping\n\n', b'data: x\n\n'])
        self.assertEqual(events, ['x'])
        self.assertEqual(d.comments, 2)

    def test_multiline_data_is_joined_with_newlines(self):
        events, _ = self.feed([b'data: {"a":\n', b'data: 1}\n\n'])
        self.assertEqual(events, ['{"a":\n1}'])

    def test_exactly_one_leading_space_is_stripped(self):
        events, _ = self.feed([b'data:  padded\n\n', b'data:tight\n\n'])
        self.assertEqual(events, [' padded', 'tight'])

    def test_unknown_fields_are_ignored(self):
        events, _ = self.feed([b'event: message\nid: 7\nretry: 100\ndata: x\n\n'])
        self.assertEqual(events, ['x'])

    def test_trailing_event_without_a_blank_line_is_flushed(self):
        events, _ = self.feed([b'data: last'])
        self.assertEqual(events, ['last'])

    def test_done_sentinel_survives_fragmentation(self):
        events, _ = self.feed([b'data: [DO', b'NE]\n\n'])
        self.assertEqual(events, ['[DONE]'])

    def test_oversized_event_is_refused_rather_than_buffered_forever(self):
        d = bc.SSEDecoder(max_event_bytes=64)
        with self.assertRaises(bc.StreamInterrupted):
            for _ in range(20):
                list(d.feed(b'data: ' + b'x' * 32 + b'\n'))

    def test_str_input_is_accepted(self):
        events, _ = self.feed(['data: text\n\n'])
        self.assertEqual(events, ['text'])


# ======================================================================
# Tool-call reconstruction
# ======================================================================
class ToolCallReconstructionTests(unittest.TestCase):

    def test_name_and_arguments_split_across_many_chunks(self):
        acc = {}
        for frag in ({'index': 0, 'id': 'c', 'function': {'name': 'get_'}},
                     {'index': 0, 'function': {'name': 'wea'}},
                     {'index': 0, 'function': {'name': 'ther', 'arguments': '{"ci'}},
                     {'index': 0, 'function': {'arguments': 'ty": "Lis'}},
                     {'index': 0, 'function': {'arguments': 'bon"}'}}):
            bc.accumulate_tool_calls([frag], acc)
        call = acc[0]
        self.assertEqual(call['function']['name'], 'get_weather')
        self.assertEqual(json.loads(call['function']['arguments']), {'city': 'Lisbon'})

    def test_multiple_interleaved_calls_stay_separate(self):
        acc = {}
        groups = [
            ({'index': 0, 'id': 'a', 'function': {'name': 'one', 'arguments': '{'}},
             {'index': 1, 'id': 'b', 'function': {'name': 'two', 'arguments': '{'}}),
            ({'index': 1, 'function': {'arguments': '"y": 2}'}},
             {'index': 0, 'function': {'arguments': '"x": 1}'}}),
        ]
        for group in groups:
            for frag in group:
                bc.accumulate_tool_calls([frag], acc)
        calls = [acc[k] for k in sorted(acc)]
        self.assertEqual([c['function']['name'] for c in calls], ['one', 'two'])
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'x': 1})
        self.assertEqual(json.loads(calls[1]['function']['arguments']), {'y': 2})

    def test_unicode_arguments_survive_fragmentation(self):
        acc = {}
        text = '{"city": "Lisboa — ☃"}'
        for ch in text:
            bc.accumulate_tool_calls([{'index': 0, 'function': {'arguments': ch}}], acc)
        self.assertEqual(json.loads(acc[0]['function']['arguments'])['city'], 'Lisboa — ☃')

    def test_incomplete_json_is_detected_and_never_executed(self):
        self.assertFalse(bc.arguments_complete('{"city": "Lis'))
        self.assertTrue(bc.arguments_complete('{"city": "Lisbon"}'))
        self.assertTrue(bc.arguments_complete(''))
        self.assertTrue(bc.arguments_complete(None))

    def test_missing_index_defaults_to_zero(self):
        acc = {}
        bc.accumulate_tool_calls([{'function': {'name': 'f', 'arguments': '{}'}}], acc)
        self.assertIn(0, acc)

    def test_non_string_arguments_are_serialised(self):
        acc = {}
        bc.accumulate_tool_calls([{'index': 0, 'function': {'arguments': {'a': 1}}}], acc)
        self.assertEqual(json.loads(acc[0]['function']['arguments']), {'a': 1})


class StreamedToolCallTests(MockTestCase):

    def test_parallel_calls_are_reassembled_from_the_wire(self):
        self.server.set_scenario('parallel-tool-calls')
        calls = None
        with self.client.stream_chat([{'role': 'user', 'content': 'go'}],
                                     tools=[{'type': 'function', 'function': {
                                         'name': 'x', 'parameters': {}}}]) as stream:
            for ev in stream:
                if ev['kind'] == 'done':
                    calls = ev['tool_calls']
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['function']['name'], 'get_weather')
        self.assertEqual(json.loads(calls[0]['function']['arguments'])['city'], 'Lisboné')
        self.assertEqual(calls[1]['function']['name'], 'current_time')

    def test_partial_tool_call_is_reported_not_executed(self):
        """A call whose JSON never completes must not be handed to the tool."""
        self.server.set_scenario('partial-tool-call')
        registry = self.registry()
        executed = []
        original = registry.tools['get_weather'].func
        registry.tools['get_weather'].func = lambda a: (executed.append(a), 'ran')[1]
        app = self.app(registry=registry)
        app.send('call the tool')
        self.assertEqual(executed, [], 'an incomplete tool call was executed')

    def test_malformed_sse_chunk_is_skipped_and_the_stream_continues(self):
        self.server.set_scenario('malformed-sse')
        with self.client.stream_chat([{'role': 'user', 'content': 'hi'}]) as stream:
            events = list(stream)
        done = events[-1]
        self.assertEqual(done['kind'], 'done')
        self.assertTrue(done['message']['content'].strip())
        self.assertGreater(stream.malformed, 0)


# ======================================================================
# Stream robustness
# ======================================================================
class StreamRobustnessTests(MockTestCase):

    def stream_of(self, scenario):
        self.server.set_scenario(scenario)
        with self.client.stream_chat([{'role': 'user', 'content': 'hi'}]) as stream:
            events = list(stream)
        return events, stream

    def test_missing_usage_and_finish_reason_do_not_break_the_turn(self):
        events, stream = self.stream_of('no-usage')
        self.assertIsNone(events[-1]['usage'])
        events, stream = self.stream_of('no-finish-reason')
        self.assertIsNone(events[-1]['finish_reason'])
        self.assertTrue(events[-1]['message']['content'])

    def test_missing_done_sentinel_still_completes(self):
        events, stream = self.stream_of('no-done')
        self.assertEqual(events[-1]['kind'], 'done')
        self.assertTrue(stream.drained)

    def test_reasoning_only_stream_has_no_answer_text(self):
        events, stream = self.stream_of('reasoning-only')
        msg = events[-1]['message']
        self.assertEqual(msg['content'], '')
        self.assertTrue(msg['reasoning_content'])

    def test_interleaved_reasoning_and_content_stay_separate(self):
        events, stream = self.stream_of('interleaved')
        msg = events[-1]['message']
        self.assertIn('answer 0', msg['content'])
        self.assertIn('thought 0', msg['reasoning_content'])
        self.assertNotIn('thought', msg['content'])

    def test_fragmented_sse_is_reassembled(self):
        events, stream = self.stream_of('fragmented-sse')
        self.assertTrue(events[-1]['message']['content'].strip())

    def test_crlf_sse_is_reassembled(self):
        events, _ = self.stream_of('crlf-sse')
        self.assertTrue(events[-1]['message']['content'].strip())

    def test_keepalive_comments_do_not_end_the_stream(self):
        events, stream = self.stream_of('keepalive-comments')
        self.assertTrue(events[-1]['message']['content'].strip())
        self.assertGreater(stream.decoder.comments, 0)

    def test_server_disconnect_is_reported_with_the_partial_text(self):
        self.server.set_scenario('disconnect')
        with self.assertRaises(bc.StreamInterrupted) as ctx:
            with self.client.stream_chat([{'role': 'user', 'content': 'hi'}]) as stream:
                for _ in stream:
                    pass
        self.assertTrue(ctx.exception.partial_text, 'partial text was not preserved')

    def test_disconnect_does_not_become_a_completed_assistant_message(self):
        self.server.set_scenario('disconnect')
        app = self.app()
        before = [m['role'] for m in app.conversation.messages]
        result = app.send('hello there')
        roles = [m['role'] for m in app.conversation.messages]
        self.assertEqual(roles, before + ['user'],
                         'a broken stream wrote an assistant message into history')
        self.assertTrue(result.interrupted)
        self.assertIn('stream ended early', self.transcript())

    def test_http_error_status_is_surfaced_as_an_api_error(self):
        for scenario, code in (('http-429', 429), ('http-500', 500), ('http-502', 502),
                               ('http-503', 503)):
            self.server.set_scenario(scenario)
            with self.assertRaises(bc.BonsaiAPIError) as ctx:
                self.client.chat([{'role': 'user', 'content': 'hi'}])
            self.assertEqual(ctx.exception.status, code)

    def test_non_json_200_body_is_an_error_not_a_crash(self):
        self.server.set_scenario('invalid-json')
        with self.assertRaises(Exception):
            self.client.chat([{'role': 'user', 'content': 'hi'}])

    def test_context_overflow_produces_an_actionable_hint(self):
        self.server.set_scenario('context-overflow')
        with self.assertRaises(bc.BonsaiAPIError) as ctx:
            self.client.chat([{'role': 'user', 'content': 'hi'}])
        self.assertIn('context', ctx.exception.hint())

    def test_stream_close_is_idempotent_and_releases_the_connection(self):
        with self.client.stream_chat([{'role': 'user', 'content': 'hi'}]) as stream:
            next(iter(stream))
        stream.close()
        stream.close()
        self.assertTrue(stream.closed)
        self.assertFalse(self.client.transport._stream_outstanding)

    def test_abandoned_stream_cannot_poison_the_next_request(self):
        """Abandon a stream mid-body, then make a normal request on the same client."""
        stream = self.client.stream_chat([{'role': 'user', 'content': 'hi'}])
        it = iter(stream)
        next(it)                       # read one event, then walk away
        del it
        reply = self.client.chat([{'role': 'user', 'content': 'hi'}], max_tokens=16)
        self.assertTrue(reply['choices'][0]['message']['content'])


# ======================================================================
# Cancellation
# ======================================================================
class CancellationTests(MockTestCase):

    def test_ctrl_c_stops_the_turn_and_leaves_history_unchanged(self):
        app = self.app()
        before = [m['role'] for m in app.conversation.messages]
        calls = []
        original_delta = app.renderer.delta

        def delta_then_interrupt(text):
            calls.append(text)
            original_delta(text)
            if len(calls) == 3:
                raise KeyboardInterrupt

        app.agent.on_delta = delta_then_interrupt
        result = app.send('tell me a long story')
        self.assertTrue(result.cancelled)
        roles = [m['role'] for m in app.conversation.messages]
        self.assertEqual(roles, before + ['user'],
                         'a cancelled turn wrote an assistant message into history')
        self.assertNotIn('[cancelled]',
                         json.dumps(app.conversation.messages, default=str))
        self.assertIn('cancelled', self.transcript())

    def test_cancelled_turn_does_not_break_the_next_turn(self):
        app = self.app()
        n = [0]
        original = app.renderer.delta

        def interrupt_once(text):
            n[0] += 1
            original(text)
            if n[0] == 2:
                raise KeyboardInterrupt

        app.agent.on_delta = interrupt_once
        app.send('first')
        app.agent.on_delta = original
        result = app.send('second')
        self.assertFalse(result.cancelled)
        self.assertTrue(result.text)
        roles = [m['role'] for m in app.conversation.messages]
        self.assertEqual(roles, ['system', 'user', 'user', 'assistant'])

    def test_the_http_response_is_closed_on_cancellation(self):
        app = self.app()
        n = [0]
        original = app.renderer.delta

        def interrupt(text):
            n[0] += 1
            original(text)
            if n[0] == 2:
                raise KeyboardInterrupt

        app.agent.on_delta = interrupt
        app.send('hello')
        self.assertFalse(self.client.transport._stream_outstanding,
                         'a cancelled stream left the connection marked busy')


# ======================================================================
# Retry safety
# ======================================================================
class RetrySafetyTests(unittest.TestCase):

    def test_a_post_that_reached_the_server_is_never_replayed(self):
        attempts = []

        class FakeTransport(bc.HttpTransport):
            def __init__(self):
                super().__init__('http://127.0.0.1:9/v1', timeout=1, connect_timeout=1)

            def request(self, method, url, body=None, headers=None, timeout=None,
                        retry=None, stream=False):
                attempts.append((method, url))
                raise bc.TransportError('lost after send', url, request_sent=True,
                                        phase='response')

        client = bc.BonsaiClient('http://127.0.0.1:9/v1', 'k', retries=3)
        client.transport = FakeTransport()
        with self.assertRaises(bc.BonsaiError) as ctx:
            client.request('/chat/completions', {'a': 1})
        self.assertEqual(len(attempts), 1, 'the chat POST was replayed')
        self.assertIn('may already be generating', str(ctx.exception))

    def test_a_get_is_retried_when_it_never_arrived(self):
        attempts = []

        class FakeTransport(bc.HttpTransport):
            def __init__(self):
                super().__init__('http://127.0.0.1:9/v1', timeout=1, connect_timeout=1)

            def request(self, method, url, body=None, headers=None, timeout=None,
                        retry=None, stream=False):
                attempts.append(method)
                raise bc.TransportError('connect refused', url, request_sent=False,
                                        phase='connect')

        client = bc.BonsaiClient('http://127.0.0.1:9/v1', 'k', retries=3)
        client.transport = FakeTransport()
        client.log = lambda *a, **k: None
        with self.assertRaises(bc.BonsaiError):
            client.request('/props', root=True)
        self.assertEqual(len(attempts), 3)

    def test_transport_error_knows_when_retry_is_safe(self):
        self.assertTrue(bc.TransportError('x', request_sent=False).safe_to_retry_post)
        self.assertFalse(bc.TransportError('x', request_sent=True).safe_to_retry_post)


# ======================================================================
# Transport
# ======================================================================
class TransportTests(MockTestCase):

    def test_requests_reuse_one_connection(self):
        for _ in range(6):
            self.client.health()
        stats = self.client.transport.stats()
        self.assertEqual(stats['connections_opened'], 1,
                         f'opened {stats["connections_opened"]} connections for 6 requests')
        self.assertTrue(stats['pooled'])

    def test_keepalive_can_be_disabled(self):
        client = bc.BonsaiClient(self.server.base_url, DEFAULT_KEY, retries=1,
                                 keepalive=False)
        self.addCleanup(client.close)
        for _ in range(3):
            client.health()
        self.assertEqual(client.transport.stats()['connections_opened'], 3)

    def test_stale_pooled_connection_is_dropped_after_the_idle_ttl(self):
        self.client.transport.idle_ttl = 0.0
        self.client.health()
        self.assertIsNone(self.client.transport._reusable_conn())

    def test_release_marks_the_connection_unusable_when_the_body_was_not_drained(self):
        tr = self.client.transport
        tr._stream_outstanding = True
        tr.release(object(), drained=False)
        self.assertFalse(tr._stream_outstanding)
        self.assertIsNone(tr._conn)


# ======================================================================
# Reasoning control
# ======================================================================
class ReasoningConfigTests(unittest.TestCase):

    def test_levels_map_to_the_documented_budgets(self):
        self.assertEqual(bc.THINK_BUDGETS,
                         {'off': 0, 'low': 512, 'medium': 2048, 'high': 8192, 'max': -1})
        for level, budget in bc.THINK_BUDGETS.items():
            self.assertEqual(bc.ReasoningConfig(level=level).budget(), budget)

    def test_explicit_token_budget_is_accepted(self):
        self.assertEqual(bc.ReasoningConfig(level='4096').budget(), 4096)
        self.assertEqual(bc.parse_think_level('4096'), 4096)

    def test_aliases_resolve(self):
        self.assertEqual(bc.parse_think_level('none'), 'off')
        self.assertEqual(bc.parse_think_level('unlimited'), 'max')
        self.assertEqual(bc.parse_effort('default'), 'xhigh')
        self.assertEqual(bc.parse_effort('high'), 'xhigh')

    def test_unknown_values_raise_instead_of_being_sent(self):
        with self.assertRaises(ValueError):
            bc.parse_think_level('turbo')
        with self.assertRaises(ValueError):
            bc.parse_effort('extreme')

    def test_wire_fields_are_suppressed_only_after_a_real_400(self):
        caps = bc.CapabilityMap()
        fields, notes = bc.ReasoningConfig('high', 'xhigh').wire(caps)
        self.assertEqual(fields, {'thinking_budget_tokens': 8192, 'reasoning_effort': 'xhigh'})
        self.assertEqual(notes, [])
        caps.note_400('thinking_budget_tokens', "invalid value for 'thinking_budget_tokens'")
        fields, notes = bc.ReasoningConfig('high', 'xhigh').wire(caps)
        self.assertNotIn('thinking_budget_tokens', fields)
        self.assertEqual(len(notes), 1)

    def test_capability_map_is_tri_state(self):
        caps = bc.CapabilityMap()
        self.assertIsNone(caps.supported('tools'))
        caps.mark('tools', 'supported')
        self.assertTrue(caps.supported('tools'))
        caps.note_400('tools', 'no jinja')
        self.assertFalse(caps.supported('tools'))
        self.assertEqual(caps.state('tools'), 'unsupported')


class ReasoningBehaviourTests(MockTestCase):

    def test_thinking_budget_reaches_the_server(self):
        app = self.app()
        app.settings.think = 'high'
        app.send('hello')
        self.assertEqual(self.server.calls[-1].get('thinking_budget_tokens'), 8192)

    def test_thinking_off_sends_budget_zero(self):
        app = self.app()
        app.command('/think off')
        app.send('hello')
        self.assertEqual(self.server.calls[-1].get('thinking_budget_tokens'), 0)

    def test_changing_thinking_does_not_touch_history(self):
        app = self.app()
        app.send('hello')
        count = len(app.conversation.messages)
        app.command('/think max')
        app.command('/effort xhigh')
        self.assertEqual(len(app.conversation.messages), count)
        self.assertEqual(app.settings.reasoning.level, 'max')
        self.assertEqual(app.settings.reasoning.effort, 'xhigh')

    def test_a_build_that_rejects_the_budget_is_reported_not_faked(self):
        self.server.reject_thinking_budget_once = True
        app = self.app()
        app.send('hello')
        self.assertEqual(self.client.caps.state('thinking_budget_tokens'), 'unsupported')
        app.send('hello again')
        self.assertNotIn('thinking_budget_tokens', self.server.calls[-1])
        self.assertIn('server-controlled', self.transcript())

    def test_reasoning_content_never_enters_history_as_answer_text(self):
        app = self.app()
        result = app.send('hello')
        self.assertTrue(result.reasoning)
        assistant = [m for m in app.conversation.messages if m['role'] == 'assistant'][-1]
        self.assertNotIn('Reasoning about', assistant.get('content') or '')
        wire = app.conversation.wire()
        for m in wire:
            self.assertNotIn('reasoning_content', m)

    def test_reasoning_display_modes(self):
        for mode in bc.REASONING_DISPLAYS:
            renderer = bc.LiveRenderer(plain_style(), markdown=False, out=io.StringIO(),
                                       reasoning_display=mode)
            renderer.reasoning('thinking hard ')
            renderer.delta('answer')
            renderer.finish()
            text = renderer.out.getvalue()
            if mode == 'hidden':
                self.assertNotIn('thinking hard', text)
            elif mode == 'full':
                self.assertIn('thinking hard', text)
            else:
                self.assertNotIn('thinking hard', text)
                self.assertEqual(renderer.reasoning_chars, len('thinking hard '))

    def test_think_command_rejects_nonsense_without_changing_state(self):
        app = self.app()
        app.settings.think = 'high'
        app.command('/think turbo')
        self.assertEqual(app.settings.reasoning.level, 'high')
        self.assertIn('unknown thinking level', self.transcript())


# ======================================================================
# Context management
# ======================================================================
class ContextTests(MockTestCase):

    def make_conv(self, turns=6, size=400):
        conv = bc.Conversation(counter=bc.TokenCounter())
        for i in range(turns):
            conv.add_user(f'question {i} ' + 'x' * size)
            conv.add_assistant({'role': 'assistant', 'content': f'answer {i} ' + 'y' * size})
        return conv

    def test_budget_reserves_room_for_the_template_and_the_answer(self):
        app = self.app()
        window = self.client.context_window()
        app.settings.max_tokens = 256
        budget = app.context_budget_tokens()
        self.assertLessEqual(budget, window - app.settings.context_reserve - 256)
        self.assertGreater(budget, 0)

    def test_trim_drops_whole_turns_and_reports_what_it_dropped(self):
        conv = self.make_conv(turns=20)
        before = len(conv.messages)
        dropped = conv.trim(600)
        self.assertGreater(dropped, 0)
        self.assertLess(len(conv.messages), before)
        self.assertLessEqual(conv.tokens(), 600)
        self.assertEqual(conv.messages[0]['role'], 'system',
                         'the system message must survive trimming')

    def test_trim_never_orphans_a_tool_result(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        for i in range(12):
            conv.add_user('q' * 300)
            conv.add_assistant({'role': 'assistant', 'content': None,
                                'tool_calls': [{'id': f'c{i}', 'type': 'function',
                                                'function': {'name': 'f',
                                                             'arguments': '{}'}}]})
            conv.add_tool_result(f'c{i}', 'r' * 300, 'f')
            conv.add_assistant({'role': 'assistant', 'content': 'a' * 300})
        conv.trim(900)
        self.assertEqual(conv.validate_wire(), [])

    def test_validate_wire_detects_orphans(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('hi')
        conv.messages.append({'role': 'tool', 'tool_call_id': 'ghost', 'content': 'x'})
        problems = conv.validate_wire()
        self.assertTrue(any('orphan' in p for p in problems))
        conv._repair_orphans()
        self.assertEqual(conv.validate_wire(), [])

    def test_compact_keeps_recent_turns_and_summarises_the_rest(self):
        conv = self.make_conv(turns=10)
        before_tokens = conv.tokens()
        report = conv.compact(500, keep_recent=2)
        self.assertGreater(report['compacted'], 0)
        self.assertLess(report['tokens_after'], before_tokens)
        self.assertEqual(conv.messages[0]['role'], 'system')
        self.assertIn('compacted', conv.messages[1]['content'])
        self.assertEqual(conv.validate_wire(), [])

    def test_compact_command_tells_the_user_what_happened(self):
        app = self.app()
        for i in range(6):
            app.conversation.add_user('q' * 200)
            app.conversation.add_assistant({'role': 'assistant', 'content': 'a' * 200})
        app.command('/compact')
        self.assertIn('compacted', self.transcript())
        self.assertEqual(app.totals['compactions'], 1)

    def test_compact_preserves_tool_pairs_in_the_kept_region(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        for i in range(6):
            conv.add_user('q' * 200)
            conv.add_assistant({'role': 'assistant', 'content': None,
                                'tool_calls': [{'id': f'c{i}', 'type': 'function',
                                                'function': {'name': 'f',
                                                             'arguments': '{}'}}]})
            conv.add_tool_result(f'c{i}', 'r' * 100, 'f')
            conv.add_assistant({'role': 'assistant', 'content': 'a' * 100})
        conv.compact(400, keep_recent=2)
        self.assertEqual(conv.validate_wire(), [])

    def test_turn_groups_split_on_user_messages(self):
        conv = self.make_conv(turns=3)
        groups = conv.turn_groups()
        self.assertEqual(len(groups), 3)
        for start, end in groups:
            self.assertEqual(conv.messages[start]['role'], 'user')


# ======================================================================
# Session persistence
# ======================================================================
class SessionTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_roundtrip_keeps_messages_and_records_the_schema(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('hello')
        conv.add_assistant({'role': 'assistant', 'content': 'hi'})
        path = conv.save(self.root / 's.jsonl')
        loaded = bc.Conversation.load(path)
        self.assertEqual([m['role'] for m in loaded.messages], ['system', 'user', 'assistant'])
        self.assertEqual(loaded.schema, bc.SESSION_SCHEMA)
        self.assertEqual(loaded.skipped_lines, 0)

    def test_truncated_last_line_is_skipped_not_fatal(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('hello')
        conv.add_assistant({'role': 'assistant', 'content': 'hi'})
        path = conv.save(self.root / 's.jsonl')
        with path.open('a', encoding='utf-8') as fh:
            fh.write('{"role": "user", "content": "cut off mid-wr')
        loaded = bc.Conversation.load(path)
        self.assertEqual(loaded.skipped_lines, 1)
        self.assertEqual([m['role'] for m in loaded.messages], ['system', 'user', 'assistant'])

    def test_garbage_file_does_not_crash_the_loader(self):
        path = self.root / 'bad.jsonl'
        path.write_text('not json\n\x00\x01binary\n[1,2,3]\n', encoding='utf-8')
        loaded = bc.Conversation.load(path)
        self.assertEqual(loaded.messages, [])
        self.assertGreaterEqual(loaded.skipped_lines, 3)

    def test_files_from_an_older_version_still_load(self):
        """v0.3/v0.4 wrote bare JSONL with no _meta line."""
        path = self.root / 'old.jsonl'
        path.write_text(json.dumps({'role': 'system', 'content': 'old system'}) + '\n' +
                        json.dumps({'role': 'user', 'content': 'old question'}) + '\n',
                        encoding='utf-8')
        loaded = bc.Conversation.load(path)
        self.assertEqual([m['role'] for m in loaded.messages], ['system', 'user'])
        self.assertEqual(loaded.system_text, 'old system')

    def test_orphaned_tool_result_in_a_file_is_repaired_on_load(self):
        path = self.root / 'orphan.jsonl'
        path.write_text(
            json.dumps({'role': 'user', 'content': 'hi'}) + '\n' +
            json.dumps({'role': 'tool', 'tool_call_id': 'missing', 'content': 'x'}) + '\n',
            encoding='utf-8')
        loaded = bc.Conversation.load(path)
        self.assertEqual([m['role'] for m in loaded.messages], ['user'])

    def test_saved_file_is_owner_only(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('secret')
        path = conv.save(self.root / 's.jsonl')
        mode = oct(path.stat().st_mode & 0o777)
        self.assertEqual(mode, '0o600')

    def test_no_temp_file_is_left_behind(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('hi')
        conv.save(self.root / 's.jsonl')
        self.assertEqual([p.name for p in self.root.iterdir()], ['s.jsonl'])

    def test_autosave_cadence_is_honoured(self):
        server = MockBonsaiServer().start()
        self.addCleanup(server.stop)
        client = bc.BonsaiClient(server.base_url, DEFAULT_KEY, retries=1)
        self.addCleanup(client.close)
        settings = bc.Settings(max_tokens=64, autosave_every=3)
        out = io.StringIO()
        conv = bc.Conversation(counter=bc.TokenCounter(client))
        path = self.root / 'auto.jsonl'
        app = bc.make_app(client, settings, plain_style(), conversation=conv,
                          sandbox=self.root, session_path=path, out=out,
                          input_fn=lambda *a: '', auto_approve=True)
        app.send('one')
        app.send('two')
        self.assertFalse(path.exists(), 'autosaved before the configured cadence')
        app.send('three')
        self.assertTrue(path.exists())


# ======================================================================
# Tool budgets
# ======================================================================
class ToolBudgetTests(MockTestCase):

    def test_slow_tool_is_abandoned_at_its_timeout(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root, auto_approve=True))
        registry.add(bc.Tool('sleeper', 'sleeps', {'type': 'object', 'properties': {}},
                             func=lambda a: time.sleep(5)))
        started = time.monotonic()
        out = registry.execute('sleeper', '{}', timeout=0.3)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIn('timed out', out)

    def test_large_tool_output_is_capped_with_a_notice(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root, auto_approve=True))
        registry.add(bc.Tool('shout', 'shouts', {'type': 'object', 'properties': {}},
                             func=lambda a: 'x' * 10000))
        out = registry.execute('shout', '{}', max_output=500)
        self.assertLess(len(out), 700)
        self.assertIn('omitted', out)

    def test_tool_budget_stops_the_loop(self):
        server = MockBonsaiServer(sticky_scenario=False).start()
        self.addCleanup(server.stop)
        client = bc.BonsaiClient(server.base_url, DEFAULT_KEY, retries=1)
        self.addCleanup(client.close)
        settings = bc.Settings(max_tokens=64, tool_budget=0.0, max_tool_rounds=4)
        registry = self.registry()
        conv = bc.Conversation(counter=bc.TokenCounter(client))
        app = bc.make_app(client, settings, plain_style(), registry=registry,
                          conversation=conv, sandbox=self.root, out=self.out,
                          input_fn=lambda *a: '', auto_approve=True)
        result = app.send('What is the weather in Lisbon? Call the tool.')
        self.assertTrue(any('budget' in (m.get('content') or '')
                            for m in conv.messages if m['role'] == 'tool'))
        self.assertIn('tool budget exhausted', self.transcript())

    def test_denial_still_reaches_the_model(self):
        registry = self.registry()
        registry.ctx.ask = lambda name, args, risk: False
        registry.tools['get_weather'].risk = 'dangerous'
        out = registry.execute('get_weather', '{"city": "Lisbon"}')
        self.assertIn('DENIED', out)


# ======================================================================
# Diagnostics and benchmarking
# ======================================================================
class DiagnosticsTests(MockTestCase):

    def test_doctor_json_is_machine_readable(self):
        buf = io.StringIO()
        code = bc.run_doctor(self.client, style=plain_style(), out=buf, as_json=True)
        payload = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload['client_version'], bc.VERSION)
        self.assertIn('capabilities', payload)
        self.assertIn('/health', payload['checks'])
        self.assertEqual(payload['checks']['/health']['state'], 'PASS')

    def test_doctor_reports_a_rejected_reasoning_control_as_degraded(self):
        self.server.reject_thinking_budget_once = True
        buf = io.StringIO()
        bc.run_doctor(self.client, style=plain_style(), out=buf)
        self.assertIn('DEGRADED', buf.getvalue())
        self.assertIn('thinking budget', buf.getvalue())

    def test_doctor_lists_unknown_capabilities_honestly(self):
        buf = io.StringIO()
        bc.run_doctor(self.client, style=plain_style(), out=buf)
        text = buf.getvalue()
        self.assertIn('capability map', text)
        self.assertIn('not tested against this build', text)

    def test_caps_command_shows_learned_state(self):
        app = self.app()
        app.send('hello')
        app.command('/caps')
        self.assertIn('thinking_budget_tokens', self.transcript())
        self.assertIn('supported', self.transcript())


class BenchmarkTests(MockTestCase):

    def test_benchmark_measures_real_numbers(self):
        report = bc.run_benchmark(self.client, samples=2, warmup=0)
        self.assertEqual(report['warm']['decode']['n'], 2)
        self.assertGreater(report['warm']['decode']['median'], 0)
        self.assertIsNotNone(report['cold'])
        self.assertIsNotNone(report['cold']['ttft_s'])

    def test_benchmark_reports_failures_instead_of_silent_zeros(self):
        self.server.set_scenario('http-500', sticky=True)
        report = bc.run_benchmark(self.client, samples=1, warmup=0)
        self.assertEqual(report['warm']['decode']['n'], 0)
        self.assertTrue(report['errors'])
        text = bc.render_benchmark(report, plain_style())
        self.assertIn('n/a', text)
        self.assertIn('error', text)

    def test_precision_is_not_overstated(self):
        self.assertEqual(bc.sig(147.123456), 147.0)
        self.assertEqual(bc.sig(0.00456789), 0.00457)
        self.assertEqual(bc.sig(None), None)
        summary = bc.summarize([10.0, 20.0, 30.0])
        self.assertEqual(summary['median'], 20.0)
        self.assertEqual(summary['min'], 10.0)
        self.assertEqual(summary['max'], 30.0)
        self.assertEqual(bc.summarize([])['n'], 0)

    def test_rendered_report_has_no_invented_vram(self):
        report = bc.run_benchmark(self.client, samples=1, warmup=0)
        text = bc.render_benchmark(report, plain_style())
        self.assertIn('cannot be measured from the client', text)


# ======================================================================
# Output coalescing
# ======================================================================
class RendererTests(unittest.TestCase):

    def test_small_writes_are_coalesced_into_fewer_flushes(self):
        class CountingIO(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        out = CountingIO()
        renderer = bc.LiveRenderer(plain_style(), markdown=False, out=out, coalesce=True)
        for i in range(50):
            renderer.delta('t')
        renderer.flush()
        self.assertLess(out.flushes, 20, f'{out.flushes} flushes for 50 one-char writes')
        self.assertEqual(out.getvalue(), 't' * 50)

    def test_disabling_coalescing_flushes_every_write(self):
        class CountingIO(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        out = CountingIO()
        renderer = bc.LiveRenderer(plain_style(), markdown=False, out=out, coalesce=False)
        for _ in range(10):
            renderer.delta('t')
        self.assertGreaterEqual(out.flushes, 10)

    def test_a_broken_pipe_does_not_crash_the_renderer(self):
        class Broken(io.StringIO):
            def write(self, s):
                raise BrokenPipeError

        renderer = bc.LiveRenderer(plain_style(), markdown=False, out=Broken(),
                                   coalesce=False)
        renderer.delta('text')
        renderer.finish()

    def test_stats_line_omits_unmeasured_values(self):
        stats = bc.TurnStats()
        stats.completion_tokens = 12
        stats.elapsed = 1.5
        line = bc.format_stats(stats, plain_style())
        self.assertIn('ttft n/a', line)
        self.assertIn('12 tok out', line)
        self.assertNotIn('decode', line)


class StatsTests(unittest.TestCase):

    def test_usage_block_is_absorbed(self):
        stats = bc.TurnStats()
        stats.absorb({'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150},
                     {'prompt_ms': 1000, 'prompt_n': 100, 'predicted_ms': 2500,
                      'predicted_n': 50}, ttft=0.4, elapsed=3.0)
        self.assertEqual(stats.prompt_tokens, 100)
        self.assertEqual(stats.prompt_tokens_per_s, 100.0)
        self.assertEqual(stats.tokens_per_s, 20.0)
        self.assertEqual(stats.ttft, 0.4)

    def test_nested_timings_inside_usage_are_found(self):
        stats = bc.TurnStats()
        stats.absorb({'prompt_tokens': 10, 'completion_tokens': 5,
                      'timings': {'tokens_per_second': 25.0}})
        self.assertEqual(stats.tokens_per_s, 25.0)

    def test_context_utilization(self):
        stats = bc.TurnStats()
        stats.context_used = 4096
        stats.context_window = 8192
        self.assertEqual(stats.context_utilization(), 50.0)

    def test_nothing_is_invented_when_the_server_reports_nothing(self):
        stats = bc.TurnStats()
        self.assertIsNone(stats.prompt_tokens)
        self.assertIsNone(stats.tokens_per_s)
        self.assertEqual(stats.rate(), 0.0)
        self.assertIsNone(stats.to_dict()['ttft_ms'])


class MockScenarioCoverageTests(unittest.TestCase):

    def test_every_declared_scenario_is_known(self):
        for name in SCENARIOS:
            server = MockBonsaiServer()
            server.set_scenario(name)      # raises on an unknown name

    def test_unknown_scenario_is_refused(self):
        with self.assertRaises(ValueError):
            MockBonsaiServer().set_scenario('not-a-scenario')


if __name__ == '__main__':
    unittest.main(verbosity=2)
