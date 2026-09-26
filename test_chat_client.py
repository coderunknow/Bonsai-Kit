#!/usr/bin/env python3
"""Offline tests for bonsai_chat.py — no GPU, no model, no network.

Everything here runs against mock_bonsai_server.MockBonsaiServer, a stub that speaks the
same wire protocol as the llama.cpp server the deployment cell starts (SSE streaming with
`reasoning_content`, split `tool_calls`, /props, /tokenize, bearer auth, text-only image
rejection), so the client code paths that ship are the ones being executed.

    python3 -m unittest -v test_chat_client
"""
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bonsai_chat as bc
from mock_bonsai_server import DEFAULT_KEY, MockBonsaiServer


def make_png(path, width=6, height=4, rgb=(10, 130, 200)):
    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress((b'\x00' + bytes(rgb) * width) * height)
    Path(path).write_bytes(bc.PNG_MAGIC + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) +
                           chunk(b'IEND', b''))
    return Path(path)


def plain_style(width=100):
    return bc.Style(force_color=False, width=width)


class ServerTestCase(unittest.TestCase):
    """Base class that starts/stops the stub and hands out a client + app."""

    server_options = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = MockBonsaiServer(**self.server_options).start()
        self.addCleanup(self.server.stop)
        self.out = io.StringIO()
        self.client = bc.BonsaiClient(self.server.base_url, DEFAULT_KEY, retries=1)
        self.settings = bc.Settings(max_tokens=256)

    def app(self, registry=None, system=bc.DEFAULT_SYSTEM, input_lines=(), **kw):
        registry = registry or self.registry()
        conversation = bc.Conversation(system=system, counter=bc.TokenCounter(self.client))
        inputs = list(input_lines)

        def fake_input(prompt=''):
            if not inputs:
                raise EOFError
            return inputs.pop(0)

        app = bc.make_app(self.client, self.settings, plain_style(), registry=registry,
                          conversation=conversation, sandbox=self.root, out=self.out,
                          input_fn=fake_input, auto_approve=kw.pop('auto_approve', True))
        return app

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


class TransportTests(ServerTestCase):

    def test_base_url_normalization(self):
        for raw, want in (('https://h.trycloudflare.com', 'https://h.trycloudflare.com/v1'),
                          ('https://h.trycloudflare.com/v1', 'https://h.trycloudflare.com/v1'),
                          ('https://h.trycloudflare.com/v1/', 'https://h.trycloudflare.com/v1'),
                          ('127.0.0.1:8080', 'http://127.0.0.1:8080/v1')):
            self.assertEqual(bc.normalize_base_url(raw), want)
        with self.assertRaises(bc.BonsaiError):
            bc.normalize_base_url('')

    def test_models_and_context_and_tokenize(self):
        self.assertEqual(self.client.model_ids(), [bc.DEFAULT_MODEL])
        self.assertEqual(self.client.context_window(), 8192)
        counter = bc.TokenCounter(self.client)
        self.assertEqual(counter.count('x' * 400), 100)   # server-side /tokenize
        self.assertGreater(bc.estimate_tokens('hello world'), 0)

    def test_token_counter_falls_back_when_endpoint_missing(self):
        counter = bc.TokenCounter(bc.BonsaiClient('http://127.0.0.1:1/v1', 'x', retries=1))
        self.assertGreater(counter.count('hello world'), 0)

    def test_sse_parser_handles_multiline_data_and_comments(self):
        class FakeResponse:
            def __init__(self, lines):
                self.lines = lines

            def __iter__(self):
                return iter(self.lines)

        events = list(bc.iter_sse_events(FakeResponse([
            b': keep-alive\r\n', b'data: {"a":\r\n', b'data: 1}\r\n', b'\r\n',
            b'data: [DONE]\r\n', b'\r\n'])))
        self.assertEqual(events, ['{"a":\n1}', '[DONE]'])

    def test_tool_call_fragments_are_reassembled(self):
        acc = {}
        bc.accumulate_tool_calls([{'index': 0, 'id': 'c1',
                                   'function': {'name': 'get_', 'arguments': '{"ci'}}], acc)
        bc.accumulate_tool_calls([{'index': 0,
                                   'function': {'name': 'weather', 'arguments': 'ty": "Lis"}'}}], acc)
        calls = [acc[k] for k in sorted(acc)]
        self.assertEqual(calls[0]['id'], 'c1')
        self.assertEqual(calls[0]['function']['name'], 'get_weather')
        self.assertEqual(json.loads(calls[0]['function']['arguments']), {'city': 'Lis'})

    def test_unauthorized_request_reports_status_and_hint(self):
        bad = bc.BonsaiClient(self.server.base_url, 'not-the-key', retries=1)
        with self.assertRaises(bc.BonsaiAPIError) as ctx:
            bad.models()
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn('authentication rejected', ctx.exception.hint())

    def test_unreachable_endpoint_explains_the_dead_tunnel(self):
        dead = bc.BonsaiClient('http://127.0.0.1:1/v1', 'x', retries=1, timeout=2)
        with self.assertRaises(bc.BonsaiError) as ctx:
            dead.chat([{'role': 'user', 'content': 'hi'}])
        self.assertIn('cannot reach', str(ctx.exception))
        self.assertIn('Quick Tunnel', str(ctx.exception))

    def test_vision_probe_detects_text_only_server(self):
        self.assertFalse(self.client.probe_vision())
        self.assertFalse(self.client.probe_vision())   # cached

    def test_reasoning_effort_is_dropped_after_a_400(self):
        self.server.reject_effort_once = True
        self.settings.effort = 'medium'
        app = self.app()
        result = app.send('Say something.')
        self.assertTrue(result.text)
        self.assertFalse(self.client.supports_reasoning_effort)
        self.assertNotIn('reasoning_effort', self.server.calls[-1])

    def test_tools_are_dropped_after_a_400(self):
        self.server.reject_tools_once = True
        app = self.app()
        result = app.send('What is the weather in Lisbon? Call the tool.')
        self.assertTrue(result.text)
        self.assertFalse(self.client.supports_tools)


class StreamingTests(ServerTestCase):

    def test_streaming_answer_with_usage_and_timings(self):
        app = self.app()
        result = app.send('Explain why C++ can be fast in 3 sentences.')
        self.assertIn('Zero-cost abstractions', result.text)
        self.assertEqual(result.completion_tokens, 64)
        self.assertEqual(result.prompt_tokens, 42)
        self.assertEqual(result.finish_reason, 'stop')
        self.assertAlmostEqual(result.rate(), 20.0, places=1)
        self.assertIn('Reasoning', result.reasoning)

    def test_reasoning_is_not_echoed_back_into_history(self):
        app = self.app()
        app.send('hi')
        assistant = [m for m in app.conversation.messages if m['role'] == 'assistant'][-1]
        self.assertNotIn('reasoning_content', assistant)

    def test_non_streaming_mode(self):
        self.settings.stream = False
        app = self.app()
        result = app.send('Explain why C++ can be fast.')
        self.assertIn('Zero-cost abstractions', result.text)
        self.assertEqual(result.completion_tokens, 64)

    def test_wire_messages_match_the_openai_shape(self):
        app = self.app()
        app.send('hi')
        wire = app.conversation.wire()
        self.assertEqual([m['role'] for m in wire], ['system', 'user', 'assistant'])
        for m in wire:
            self.assertIsInstance(m['content'], str)
        sent = self.server.calls[-1]
        self.assertEqual(sent['model'], bc.DEFAULT_MODEL)
        self.assertTrue(sent['stream'])
        self.assertEqual(sent['stream_options'], {'include_usage': True})


class ToolCallingTests(ServerTestCase):

    def test_tool_is_called_result_fed_back_and_answer_produced(self):
        app = self.app()
        result = app.send('What is the weather in Lisbon right now? Call the tool.')
        self.assertEqual([tc['function']['name'] for tc in result.tool_calls], ['get_weather'])
        self.assertEqual(json.loads(result.tool_calls[0]['function']['arguments']),
                         {'city': 'Lisbon'})
        roles = [m['role'] for m in app.conversation.messages]
        self.assertEqual(roles, ['system', 'user', 'assistant', 'tool', 'assistant'])
        tool_msg = app.conversation.messages[3]
        self.assertEqual(tool_msg['tool_call_id'], 'call_mock_1')
        self.assertIn('21C and sunny in Lisbon', tool_msg['content'])
        self.assertGreaterEqual(result.rounds, 2)
        self.assertIn('21C and sunny in Lisbon', result.text)

    def test_denied_tool_is_reported_to_the_model(self):
        registry = self.registry()
        app = self.app(registry=registry, auto_approve=False)
        registry.tools['get_weather'].risk = 'dangerous'
        registry.ctx.auto_approve = False
        registry.ctx.approve = lambda name, args, risk: False
        app.send('What is the weather in Lisbon? Call the tool.')
        tool_msg = [m for m in app.conversation.messages if m['role'] == 'tool'][0]
        self.assertIn('DENIED', tool_msg['content'])

    def test_approval_callback_is_invoked_with_args(self):
        seen = []
        registry = self.registry()
        registry.tools['get_weather'].risk = 'write'
        registry.ctx.auto_approve = False
        app = self.app(registry=registry, auto_approve=False)
        registry.ctx.approve = lambda name, args, risk: seen.append((name, args, risk)) or True
        app.send('What is the weather in Lisbon? Call the tool.')
        self.assertEqual(seen, [('get_weather', {'city': 'Lisbon'}, 'write')])

    def test_tool_round_cap_is_enforced(self):
        self.settings.max_tool_rounds = 1
        app = self.app()
        result = app.send('What is the weather in Lisbon? Call the tool.')
        self.assertEqual(result.rounds, 1)
        self.assertIn('tool rounds', self.transcript())

    def test_malformed_arguments_produce_a_recoverable_error(self):
        registry = self.registry()
        out = registry.execute('calculator', '{not json')
        self.assertIn('ERROR', out)
        self.assertIn('not valid JSON', out)

    def test_unknown_tool_is_reported(self):
        registry = self.registry()
        self.assertIn('unknown tool', registry.execute('nope', '{}'))

    def test_disabled_tool_is_reported(self):
        registry = self.registry()
        registry.set_enabled('calculator', False)
        self.assertIn('disabled', registry.execute('calculator', '{"expression": "1+1"}'))

    def test_calculator_evaluates_and_refuses_builtins(self):
        registry = self.registry()
        self.assertEqual(registry.execute('calculator', '{"expression": "2*(3+4)/2"}'),
                         '2*(3+4)/2 = 7.0')
        self.assertIn('sqrt', registry.execute('calculator', '{"expression": "sqrt(16)"}'))
        self.assertIn('ERROR', registry.execute('calculator', '{"expression": "__import__(\'os\')"}'))
        self.assertIn('ERROR', registry.execute('calculator', '{"expression": "open(\'x\')"}'))

    def test_file_tools_stay_inside_the_sandbox(self):
        registry = self.registry()
        (self.root / 'notes.txt').write_text('line one\nline two\n')
        (self.root / 'sub').mkdir()
        (self.root / 'sub' / 'code.py').write_text('def answer():\n    return 42\n')
        self.assertIn('line one', registry.execute('read_file', '{"path": "notes.txt"}'))
        listing = registry.execute('list_dir', '{"path": "."}')
        self.assertIn('notes.txt', listing)
        self.assertIn('sub/', listing)
        hits = registry.execute('search_text', '{"pattern": "def answer"}')
        self.assertIn('code.py:1', hits)
        escape = registry.execute('read_file', '{"path": "../../../etc/passwd"}')
        self.assertIn('outside the sandbox', escape)
        written = registry.execute('write_file', '{"path": "out.txt", "content": "hello"}')
        self.assertIn('wrote', written)
        self.assertEqual((self.root / 'out.txt').read_text(), 'hello')

    def test_write_file_cannot_leave_the_sandbox(self):
        registry = self.registry()
        self.assertIn('outside the sandbox',
                      registry.execute('write_file', '{"path": "../escape.txt", "content": "x"}'))
        self.assertFalse((self.root.parent / 'escape.txt').exists())

    def test_image_inspect_tool_reports_measured_facts(self):
        registry = self.registry()
        make_png(self.root / 'shot.png', width=8, height=5)
        out = registry.execute('image_inspect', '{"path": "shot.png"}')
        self.assertIn('8x5 px', out)
        self.assertIn('TEXT-ONLY', out)

    def test_current_time_tool(self):
        registry = self.registry()
        self.assertIn('UTC', registry.execute('current_time', '{}'))


class ImageTests(ServerTestCase):

    def test_png_header_parsed_without_pillow(self):
        path = make_png(self.root / 'a.png', width=17, height=9)
        att = bc.ImageAttachment.load(path, encode=False)
        self.assertEqual((att.fmt, att.width, att.height), ('png', 17, 9))
        self.assertEqual(att.header_extra['bit_depth'], 8)
        self.assertEqual(att.header_extra['color_type'], 'rgb')

    def test_other_formats_are_sniffed_from_magic_bytes(self):
        jpeg = self.root / 'b.jpg'
        jpeg.write_bytes(b'\xff\xd8\xff\xe0' + struct.pack('>H', 14) + b'\x00' * 12 +
                         b'\xff\xc0' + struct.pack('>H', 17) + b'\x08' +
                         struct.pack('>HH', 33, 55) + b'\x03')
        fmt, dims, _ = bc.header_info(jpeg.read_bytes())
        self.assertEqual((fmt, dims), ('jpeg', (55, 33)))
        gif = self.root / 'c.gif'
        gif.write_bytes(b'GIF89a' + struct.pack('<HH', 12, 7) + b'\x00' * 10)
        self.assertEqual(bc.header_info(gif.read_bytes())[1], (12, 7))
        bmp = self.root / 'd.bmp'
        bmp.write_bytes(b'BM' + b'\x00' * 16 + struct.pack('<ii', 21, 11) + b'\x00' * 10)
        self.assertEqual(bc.header_info(bmp.read_bytes())[1], (21, 11))
        self.assertEqual(bc.header_info(b'not an image at all')[0], None)

    def test_webp_variants(self):
        vp8x = b'RIFF' + struct.pack('<I', 30) + b'WEBPVP8X' + struct.pack('<I', 10) + \
            b'\x00' * 4 + (100 - 1).to_bytes(3, 'little') + (50 - 1).to_bytes(3, 'little')
        self.assertEqual(bc.webp_dimensions(vp8x)[0:2], (100, 50))

    def test_non_image_is_rejected(self):
        text = self.root / 'notes.txt'
        text.write_text('hello')
        with self.assertRaises(bc.BonsaiError):
            bc.ImageAttachment.load(text)
        with self.assertRaises(bc.BonsaiError):
            bc.ImageAttachment.load(self.root / 'missing.png')

    def test_text_only_card_says_the_model_cannot_see_pixels(self):
        att = bc.ImageAttachment.load(make_png(self.root / 'e.png'), encode=False)
        card = att.text_card(vision=False)
        self.assertIn('TEXT-ONLY', card)
        self.assertIn('6x4 px', card)
        self.assertIn('PNG', card)
        self.assertIn('aspect 3:2', card)

    def test_vision_card_is_a_caption(self):
        att = bc.ImageAttachment.load(make_png(self.root / 'f.png'), encode=False)
        card = att.text_card(vision=True)
        self.assertIn('vision projector', card)
        self.assertNotIn('TEXT-ONLY', card)

    def test_build_user_message_switches_on_vision_support(self):
        att = bc.ImageAttachment.load(make_png(self.root / 'g.png'), encode=True)
        self.assertTrue(att.data_url.startswith('data:image/png;base64,'))
        blind = bc.build_user_message('what is this?', [att], vision=False)
        self.assertIsInstance(blind['content'], str)
        self.assertIn('Image attached: g.png', blind['content'])
        seeing = bc.build_user_message('what is this?', [att], vision=True)
        self.assertIsInstance(seeing['content'], list)
        self.assertEqual(seeing['content'][-1]['type'], 'image_url')

    def test_text_only_image_facts_reach_the_server(self):
        att = bc.ImageAttachment.load(make_png(self.root / 'h.png', width=9, height=2),
                                      encode=True)
        app = self.app()
        app.send('what is in this image?', attachments=[att])
        sent = self.server.calls[-1]['messages'][-1]['content']
        self.assertIn('9x2 px', sent)
        self.assertIn('TEXT-ONLY', sent)

    def test_pillow_statistics_are_included_when_available(self):
        if bc._PILImage is None:
            self.skipTest('Pillow not installed')
        att = bc.ImageAttachment.load(make_png(self.root / 'i.png'), encode=False)
        self.assertIn('mean colour', att.text_card(vision=False))


class MarkdownTests(unittest.TestCase):

    def render(self, text, width=90):
        return bc.render_markdown(text, plain_style(width))

    def test_headings_and_rule(self):
        out = self.render('# Title\n\ntext\n\n---\n')
        self.assertIn('Title', out)
        self.assertIn('─' * 10, out)

    def test_lists_nest_and_wrap(self):
        out = self.render('- alpha\n- beta\n  - gamma\n- ' + 'long ' * 60)
        self.assertIn('alpha', out)
        self.assertIn('  • gamma', out) if '•' in out else self.assertIn('gamma', out)
        self.assertTrue(all(len(line) <= 90 for line in out.split('\n')))

    def test_ordered_and_task_lists(self):
        out = self.render('1. first\n2. second\n\n- [x] done\n- [ ] todo\n')
        self.assertIn('first', out)
        self.assertIn('[x] done', out)
        self.assertIn('[ ] todo', out)

    def test_code_fence_is_boxed_and_language_labelled(self):
        out = self.render('```python\nprint("hi")\nx = 1\n```\n')
        self.assertIn('python', out)
        self.assertIn('print("hi")', out)
        self.assertIn('┌─', out)
        self.assertIn('└', out)

    def test_table_is_drawn(self):
        out = self.render('| name | value |\n| --- | --- |\n| a | 1 |\n| bb | 22 |\n')
        self.assertIn('┼', out)
        self.assertIn('│', out)
        self.assertIn('name', out)
        self.assertIn('22', out)

    def test_blockquote(self):
        out = self.render('> quoted words\n> still quoted\n')
        self.assertIn('│', out)
        self.assertIn('quoted words', out)

    def test_inline_styles_are_applied_and_plain_without_colour(self):
        styled = bc.render_markdown('**bold** and `code` and *em* and [link](http://x.y)\n',
                                    bc.Style(force_color=True, width=90))
        self.assertIn('\033[', styled)
        plain = self.render('**bold** and `code` and [link](http://x.y)\n')
        self.assertNotIn('\033[', plain)
        self.assertIn('bold', plain)
        self.assertIn('code', plain)
        self.assertIn('http://x.y', plain)

    def test_html_entities_are_unescaped(self):
        self.assertIn('a < b && c > d', self.render('a &lt; b &amp;&amp; c &gt; d\n'))

    def test_streamed_rendering_matches_one_shot(self):
        text = '## H\n\npara **one**\n\n- x\n- y\n\n```js\nlet a = 1;\n```\n'
        one_shot = self.render(text)
        writer = bc.MarkdownWriter(plain_style(90))
        streamed = ''.join(writer.feed(c) for c in text) + writer.finish()
        self.assertEqual(streamed, one_shot)

    def test_wrap_ansi_ignores_escape_width(self):
        text = bc.Style(force_color=True).bold('word ') * 40
        wrapped = bc.wrap_ansi(text, 40)
        self.assertTrue(all(bc.visible_len(line) <= 40 for line in wrapped.split('\n')))


class ConversationTests(ServerTestCase):

    def test_trim_drops_whole_turns_and_never_orphans_tool_messages(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('first question ' + 'x' * 300)
        conv.add_assistant({'role': 'assistant', 'content': None,
                            'tool_calls': [{'id': 'c1', 'type': 'function',
                                            'function': {'name': 'read_file',
                                                         'arguments': '{"path": "a"}'}}]})
        conv.add_tool_result('c1', 'file body', 'read_file')
        conv.add_assistant({'role': 'assistant', 'content': 'answer one ' + 'y' * 300})
        for i in range(30):
            conv.add_user('q' + str(i) + ' ' + 'x' * 200)
            conv.add_assistant({'role': 'assistant', 'content': 'a' + str(i) + ' ' + 'y' * 200})
        dropped = conv.trim(600)
        self.assertGreater(dropped, 0)
        wire = conv.wire()
        for i, m in enumerate(wire):
            if m['role'] == 'tool':
                self.assertTrue(any(tc['id'] == m['tool_call_id']
                                    for tc in wire[i - 1].get('tool_calls', [])))
        self.assertLessEqual(conv.tokens(), 600)

    def test_drop_last_turn_and_undo(self):
        conv = bc.Conversation()
        conv.add_user('q1')
        conv.add_assistant({'role': 'assistant', 'content': 'a1'})
        conv.add_user('q2')
        self.assertEqual(conv.drop_last_turn(), 1)
        self.assertEqual([m['role'] for m in conv.messages], ['system', 'user', 'assistant'])

    def test_save_load_roundtrip_and_markdown_export(self):
        conv = bc.Conversation(counter=bc.TokenCounter())
        conv.add_user('hello')
        conv.add_assistant({'role': 'assistant', 'content': 'hi there'})
        path = conv.save(self.root / 's.jsonl')
        self.assertEqual(path.read_text().count('\n'), 3)
        loaded = bc.Conversation.load(path)
        self.assertEqual([m['role'] for m in loaded.messages], ['system', 'user', 'assistant'])
        md = conv.export_markdown(self.root / 's.md')
        self.assertIn('## You', md.read_text())
        self.assertIn('## Bonsai', md.read_text())

    def test_wire_shape_keeps_tool_calls(self):
        conv = bc.Conversation()
        conv.add_assistant({'role': 'assistant', 'content': None,
                            'tool_calls': [{'id': 'c', 'type': 'function',
                                            'function': {'name': 'f', 'arguments': '{}'}}]})
        wire = conv.wire()[-1]
        self.assertEqual(wire['tool_calls'][0]['function']['name'], 'f')
        self.assertIsNone(wire['content'])


class ReplTests(ServerTestCase):

    def test_slash_help_tools_context_and_quit(self):
        app = self.app(input_lines=['/help', '/tools', '/context', '/usage', '/quit'])
        self.assertEqual(app.run(), 0)
        out = self.transcript()
        self.assertIn('Commands:', out)
        self.assertIn('read_file', out)
        self.assertIn('context window', out)
        self.assertIn('0 turn(s)', out)

    def test_plain_line_is_sent_to_the_model(self):
        app = self.app(input_lines=['Explain why C++ can be fast.', '/quit'])
        app.run()
        self.assertIn('Zero-cost abstractions', self.transcript())
        self.assertIn('Ternary Bonsai 2 27B', self.transcript())

    def test_multiline_input_is_joined(self):
        app = self.app(input_lines=['first line \\', 'second line', '/quit'])
        app.run()
        sent = self.server.calls[0]['messages'][1]['content']
        self.assertIn('first line', sent)
        self.assertIn('second line', sent)

    def test_undo_and_retry_commands(self):
        app = self.app(input_lines=['hello', '/undo', '/retry', '/quit'])
        app.run()
        self.assertEqual([m['role'] for m in app.conversation.messages], ['system'])
        self.assertIn('nothing to retry yet', self.transcript())

    def test_settings_commands(self):
        app = self.app(input_lines=['/temp 0.3', '/topp 0.8', '/max-tokens 128',
                                    '/effort high', '/stream off', '/markdown off',
                                    '/model other-model', '/quit'])
        app.run()
        self.assertEqual(self.settings.temperature, 0.3)
        self.assertEqual(self.settings.top_p, 0.8)
        self.assertEqual(self.settings.max_tokens, 128)
        self.assertEqual(self.settings.effort, 'high')
        self.assertFalse(self.settings.stream)
        self.assertFalse(self.settings.markdown)
        self.assertEqual(self.client.model, 'other-model')

    def test_invalid_setting_reports_an_error_instead_of_crashing(self):
        app = self.app(input_lines=['/temp notanumber', '/bogus', '/quit'])
        self.assertEqual(app.run(), 0)
        self.assertIn('failed', self.transcript())
        self.assertIn('unknown command /bogus', self.transcript())

    def test_image_command_queues_an_attachment(self):
        make_png(self.root / 'pic.png', width=5, height=5)
        app = self.app(input_lines=[f'/image {self.root / "pic.png"}', 'what is it?', '/quit'])
        app.run()
        self.assertIn('Image attached: pic.png', self.transcript())
        self.assertIn('5x5 px', self.server.calls[-1]['messages'][-1]['content'])
        self.assertEqual(app.pending_images, [])

    def test_tools_toggle(self):
        app = self.app(input_lines=['/tools calculator off', '/tools off', '/tools on', '/quit'])
        app.run()
        self.assertFalse(app.registry.tools['calculator'].enabled)
        self.assertTrue(self.settings.use_tools)

    def test_system_prompt_command(self):
        app = self.app(input_lines=['/system You are terse.', 'hi', '/quit'])
        app.run()
        self.assertEqual(self.server.calls[0]['messages'][0]['content'], 'You are terse.')

    def test_session_save_load_export_commands(self):
        app = self.app(input_lines=['hi', f'/save {self.root / "s.jsonl"}',
                                    f'/export {self.root / "s.md"}', '/quit'])
        app.run()
        self.assertTrue((self.root / 's.jsonl').is_file())
        self.assertIn('## You', (self.root / 's.md').read_text())
        loaded = self.app(input_lines=[f'/load {self.root / "s.jsonl"}', '/history', '/quit'])
        loaded.run()
        self.assertIn('[user] hi', self.transcript())

    def test_autosave_session_file_is_written(self):
        session = self.root / 'auto.jsonl'
        registry = self.registry()
        conv = bc.Conversation(counter=bc.TokenCounter(self.client))
        app = bc.make_app(self.client, self.settings, plain_style(), registry=registry,
                          conversation=conv, sandbox=self.root, out=self.out,
                          session_path=session, input_fn=lambda p='': '/quit',
                          auto_approve=True)
        app.send('remember me')
        self.assertTrue(session.is_file())
        self.assertIn('remember me', session.read_text())

    def test_api_error_is_reported_with_a_hint_not_a_traceback(self):
        self.client.api_key = 'wrong'
        app = self.app()
        self.assertIsNone(app.send('hello'))
        self.assertIn('HTTP 401', self.transcript())
        self.assertIn('hint:', self.transcript())


class CliTests(ServerTestCase):

    def run_cli(self, argv):
        buffer = io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = buffer, buffer
        try:
            code = bc.main(argv)
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        return code, buffer.getvalue()

    def test_one_shot_prompt(self):
        code, out = self.run_cli(['--base-url', self.server.base_url, '--api-key', DEFAULT_KEY,
                                  '--plain', '--no-tools', '-p', 'Explain why C++ is fast.'])
        self.assertEqual(code, 0)
        self.assertIn('Zero-cost abstractions', out)

    def test_one_shot_json_output(self):
        code, out = self.run_cli(['--base-url', self.server.base_url, '--api-key', DEFAULT_KEY,
                                  '--no-tools', '--json', '-p', 'hello'])
        self.assertEqual(code, 0)
        payload = json.loads(out[out.index('{'):])
        self.assertIn('Zero-cost abstractions', payload['text'])
        self.assertEqual(payload['usage']['completion_tokens'], 64)
        self.assertIn('Reasoning', payload['reasoning'])

    def test_piped_stdin_becomes_the_prompt(self):
        old_stdin = sys.stdin
        sys.stdin = io.StringIO('Explain why C++ is fast.\n')
        try:
            code, out = self.run_cli(['--base-url', self.server.base_url,
                                      '--api-key', DEFAULT_KEY, '--plain', '--no-tools'])
        finally:
            sys.stdin = old_stdin
        self.assertEqual(code, 0)
        self.assertIn('Zero-cost abstractions', out)

    def test_image_flag_attaches_a_file(self):
        png = make_png(self.root / 'cli.png', width=3, height=7)
        code, out = self.run_cli(['--base-url', self.server.base_url, '--api-key', DEFAULT_KEY,
                                  '--plain', '--no-tools', '--image', str(png),
                                  '-p', 'describe it'])
        self.assertEqual(code, 0)
        self.assertIn('3x7 px', out)

    def test_missing_image_exits_cleanly(self):
        code, out = self.run_cli(['--base-url', self.server.base_url, '--api-key', DEFAULT_KEY,
                                  '--image', str(self.root / 'nope.png'), '-p', 'x'])
        self.assertEqual(code, 2)
        self.assertIn('image not found', out)

    def test_missing_endpoint_exits_with_guidance(self):
        env = os.environ.pop('BONSAI_BASE_URL', None)
        try:
            code, out = self.run_cli(['-p', 'hi'])
        finally:
            if env is not None:
                os.environ['BONSAI_BASE_URL'] = env
        self.assertEqual(code, 2)
        self.assertIn('BONSAI_BASE_URL', out)

    def test_unknown_tool_name_warns(self):
        code, out = self.run_cli(['--base-url', self.server.base_url, '--api-key', DEFAULT_KEY,
                                  '--tools', 'calculator,nope', '-p', 'What is 2+2? Use a tool.'])
        self.assertEqual(code, 0)
        self.assertIn('unknown tool(s): nope', out)

    def test_selftest_command_passes(self):
        code, out = self.run_cli(['--selftest', '--plain'])
        self.assertEqual(code, 0)
        self.assertIn('self-test checks passed', out)
        self.assertNotIn('FAIL', out)


class VisionServerTests(ServerTestCase):
    """The same client against a build that does have a vision projector."""

    server_options = {'vision': True}

    def test_pixels_are_sent_when_the_server_can_see(self):
        att = bc.ImageAttachment.load(make_png(self.root / 'v.png'), encode=True)
        app = self.app()
        app.send('what is in this image?', attachments=[att])
        sent = self.server.calls[-1]['messages'][-1]['content']
        self.assertIsInstance(sent, list)
        self.assertEqual(sent[-1]['type'], 'image_url')
        self.assertTrue(sent[-1]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertIn('vision projector', self.transcript())

    def test_vision_is_read_from_props_modalities_without_a_probe_request(self):
        app = self.app()
        att = bc.ImageAttachment.load(make_png(self.root / 'm.png'), encode=True)
        app.send('look', attachments=[att])
        self.assertTrue(self.client.probe_vision())
        probes = [c for c in self.server.calls if c.get('max_tokens') == 8]
        self.assertEqual(probes, [], 'a 1x1 probe request should not be needed')

    def test_large_image_is_downscaled_when_pillow_is_available(self):
        if bc._PILImage is None:
            self.skipTest('Pillow not installed')
        big = self.root / 'big.png'
        make_png(big, width=3000, height=2000)
        att = bc.ImageAttachment.load(big, max_side=512)
        self.assertLessEqual(att.payload_bytes, big.stat().st_size)
        self.assertIn('data:image/jpeg;base64,', att.data_url)


class PropsVisionProbeTests(ServerTestCase):
    """Older builds have no `modalities` in /props, so the 1x1 probe is the fallback."""

    server_options = {'vision': True, 'announce_modalities': False}

    def test_probe_is_used_when_props_says_nothing_about_modalities(self):
        app = self.app()
        att = bc.ImageAttachment.load(make_png(self.root / 'p.png'), encode=True)
        app.send('look', attachments=[att])
        self.assertTrue(self.client.probe_vision())
        probes = [c for c in self.server.calls if c.get('max_tokens') == 8]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]['messages'][0]['content'][-1]['type'], 'image_url')

    def test_the_probe_runs_only_once_per_session(self):
        att = bc.ImageAttachment.load(make_png(self.root / 'w.png'), encode=True)
        app = self.app()
        app.send('first', attachments=[att])
        app.send('second', attachments=[att])
        probes = [c for c in self.server.calls if c.get('max_tokens') == 8]
        self.assertEqual(len(probes), 1)


class EfficiencyTests(ServerTestCase):

    def test_context_window_is_probed_once_per_session(self):
        app = self.app()
        app.send('first message')
        app.send('second message')
        app.send('third message')
        props = [p for p in self.server.get_paths if p == '/props']
        self.assertEqual(len(props), 1, f'/props was requested {len(props)} times')

    def test_repeated_token_counts_do_not_hit_the_server(self):
        counter = bc.TokenCounter(self.client)
        counter.count('the same sentence twice')
        after_first = len([p for p in self.server.get_paths if p == 'POST /tokenize'])
        counter.count('the same sentence twice')
        after_second = len([p for p in self.server.get_paths if p == 'POST /tokenize'])
        self.assertEqual(after_first, 1)
        self.assertEqual(after_second, 1)

    def test_llama_cpp_native_routes_are_at_the_server_root(self):
        """/props, /tokenize and /health are NOT under /v1 in llama.cpp."""
        self.client.context_window()
        bc.TokenCounter(self.client).count('some text to count')
        self.client.health()
        self.assertIn('/props', self.server.get_paths)
        self.assertIn('POST /tokenize', self.server.get_paths)
        self.assertIn('/health', self.server.get_paths)
        self.assertNotIn('/v1/props', self.server.get_paths)
        self.assertNotIn('POST /v1/tokenize', self.server.get_paths)


class DoctorTests(ServerTestCase):

    def test_doctor_passes_against_a_healthy_endpoint(self):
        code, out = self.run_doctor()
        self.assertEqual(code, 0)
        self.assertIn('Endpoint is usable.', out)
        self.assertNotIn('FAIL', out)
        for line in ('/health', '/v1/models', '/props context window', '/tokenize',
                     'bearer auth enforced', 'chat completion', 'streaming (SSE)',
                     'native tool calling', 'vision (image input)'):
            self.assertIn(line, out)

    def test_doctor_reports_a_dead_endpoint_and_exits_nonzero(self):
        dead = bc.BonsaiClient('http://127.0.0.1:1/v1', 'x', retries=1, timeout=2)
        code = bc.run_doctor(dead, style=plain_style(), out=self.out)
        self.assertEqual(code, 1)
        self.assertIn('not reachable', self.transcript())

    def test_doctor_flags_a_server_without_jinja_tool_support(self):
        self.server.reject_tools_once = True
        code, out = self.run_doctor()
        self.assertIn('native tool calling', out)
        self.assertIn('fell back to no-tool mode', out)

    def run_doctor(self):
        code = bc.run_doctor(self.client, style=plain_style(), out=self.out)
        return code, self.transcript()


class SessionFileTests(unittest.TestCase):

    def test_state_file_is_not_required(self):
        self.assertTrue(callable(bc.main))

    def test_optional_features_reported(self):
        feats = bc.optional_features()
        self.assertEqual(set(feats), {'pillow', 'pygments', 'pytesseract', 'tesseract'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
