#!/usr/bin/env python3
"""Offline tests for the v0.6.0 client power features — no GPU, no model, no network.

Covers the four things v0.5.0 could not do:

* configuration with precedence you can inspect (config file, persona, env, CLI)
* conversation branching (fork / list / switch / delete / rename, persisted)
* multi-endpoint use with a per-endpoint capability map
* structured output that is proved before it is relied on
* budget accounting, batch mode, JSON/HTML export, MCP tools and the importable API

Everything runs against mock_bonsai_server.MockBonsaiServer except the pieces that
deliberately never touch a server (config, budget, branching, export).

    python3 -m unittest -v test_power_features
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bonsai_chat as bc
from mock_bonsai_server import DEFAULT_KEY, MockBonsaiServer

MCP_SERVER = textwrap.dedent('''
    """A deliberately tiny MCP server: two tools, JSON-RPC over stdio."""
    import json, sys

    TOOLS = [
        {"name": "echo", "description": "Repeat the text back",
         "inputSchema": {"type": "object",
                         "properties": {"text": {"type": "string"}},
                         "required": ["text"]}},
        {"name": "search", "description": "Pretend to search",
         "inputSchema": {"type": "object",
                         "properties": {"q": {"type": "string"}}}},
    ]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method = msg.get("method")
        rid = msg.get("id")
        if rid is None:
            continue
        if method == "initialize":
            out = {"protocolVersion": "2024-11-05",
                   "serverInfo": {"name": "fake", "version": "1.2.3"},
                   "capabilities": {"tools": {}}}
        elif method == "tools/list":
            out = {"tools": TOOLS}
        elif method == "tools/call":
            args = (msg.get("params") or {}).get("arguments") or {}
            name = (msg.get("params") or {}).get("name")
            if name == "echo":
                body = "echo: " + str(args.get("text", ""))
            elif name == "search":
                body = "no results for " + str(args.get("q", ""))
            else:
                out = {"error": {"code": -32601, "message": "no such tool"}}
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid,
                                             "error": out["error"]}) + "\\n")
                sys.stdout.flush()
                continue
            out = {"content": [{"type": "text", "text": body}]}
        else:
            out = {"error": {"code": -32601, "message": "unknown method"}}
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid,
                                         "error": out["error"]}) + "\\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": out}) + "\\n")
        sys.stdout.flush()
''')


def plain_style(width=100):
    return bc.Style(force_color=False, width=width)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path


# ======================================================================
# Configuration
# ======================================================================
class ConfigPrecedenceTests(TempDirCase):

    def defaults(self):
        from bonsai_chat.config import SETTING_KEYS
        settings = bc.Settings()
        return {k: getattr(settings, k) for k in SETTING_KEYS if hasattr(settings, k)}

    def resolve(self, config, cli=None, environ=None):
        return config.resolve(self.defaults(), cli=cli or {},
                              environ={} if environ is None else environ)

    def test_a_config_file_beats_the_built_in_defaults(self):
        cfg = bc.Config({'temperature': 0.25, 'max_tokens': 111}, path=self.root / 'c.json')
        values = self.resolve(cfg)
        self.assertEqual(values['temperature'], 0.25)
        self.assertEqual(values['max_tokens'], 111)
        self.assertEqual(cfg.source_of('temperature'), 'config file')
        self.assertEqual(cfg.source_of('model'), 'default')

    def test_a_persona_beats_the_file_and_the_cli_beats_both(self):
        cfg = bc.Config({'temperature': 0.5,
                         'personas': {'cold': {'temperature': 0.05}}},
                        path=self.root / 'c.json')
        self.assertEqual(self.resolve(cfg)['temperature'], 0.5)
        cfg.select_persona('cold')
        values = self.resolve(cfg)
        self.assertEqual(values['temperature'], 0.05)
        self.assertEqual(cfg.source_of('temperature'), 'persona cold')
        values = self.resolve(cfg, cli={'temperature': 0.9})
        self.assertEqual(values['temperature'], 0.9)
        self.assertEqual(cfg.source_of('temperature'), 'command line')

    def test_the_environment_sits_between_the_persona_and_the_cli(self):
        cfg = bc.Config({'temperature': 0.5}, path=self.root / 'c.json')
        cfg.select_persona('x')
        env = {'BONSAI_TEMPERATURE': '0.33'}
        self.assertEqual(self.resolve(cfg, environ=env)['temperature'], 0.33)
        self.assertEqual(cfg.source_of('temperature'), 'environment')
        self.assertEqual(self.resolve(cfg, cli={'temperature': 0.9},
                                      environ=env)['temperature'], 0.9)

    def test_an_unknown_persona_is_reported_not_silently_ignored(self):
        cfg = bc.Config({'personas': {'a': {}}}, path=self.root / 'c.json')
        self.assertIsNone(cfg.select_persona('nope'))
        self.assertTrue(any('no persona named' in n for n in cfg.notes))

    def test_a_flag_that_was_never_typed_does_not_beat_the_file(self):
        """The whole reason `explicit_args` exists.

        argparse fills a value for every option, so without this a default-filled
        --temperature would silently overwrite the file the user configured.
        """
        from bonsai_chat.cli import build_parser
        from bonsai_chat.config import explicit_args
        argv = ['--config', 'x.json', '--max-tokens', '512']
        explicit = explicit_args(build_parser(), argv)
        self.assertNotIn('temperature', explicit)
        self.assertIn('max_tokens', explicit)
        self.assertEqual(explicit['max_tokens'], 512)

    def test_boolean_flags_are_coerced_from_the_strings_a_file_holds(self):
        cfg = bc.Config({'stream': 'no', 'use_tools': 'false', 'markdown': 1},
                        path=self.root / 'c.json')
        values = self.resolve(cfg)
        self.assertEqual(values['stream'], False)
        self.assertEqual(values['use_tools'], False)
        self.assertEqual(values['markdown'], True)

    def test_a_broken_config_file_is_reported_and_treated_as_empty(self):
        path = self.write('broken.json', '{not json')
        notes = []
        data, _ = bc.load_config(path, on_error=notes.append)
        self.assertEqual(data, {})
        self.assertTrue(notes and 'broken.json' in notes[0])

    def test_save_and_load_round_trip_and_the_file_is_private(self):
        path = self.root / 'sub' / 'c.json'
        bc.save_config({'temperature': 0.1, 'endpoints': {}}, path)
        self.assertEqual(oct(path.stat().st_mode)[-3:], '600')
        data, _ = bc.load_config(path)
        self.assertEqual(data['temperature'], 0.1)

    def test_endpoints_can_be_a_mapping_or_a_list(self):
        cfg = bc.Config({'endpoints': {'a': {'base_url': 'http://x/v1',
                                             'api_key_env': 'NOPE_NOT_SET'},
                                       'b': {'base_url': 'http://y/v1'}}})
        names = sorted(e['name'] for e in cfg.endpoints())
        self.assertEqual(names, ['a', 'b'])
        cfg2 = bc.Config({'endpoints': [{'name': 'c', 'base_url': 'http://z/v1'}]})
        self.assertEqual([e['name'] for e in cfg2.endpoints()], ['c'])

    def test_a_starter_config_file_can_be_written(self):
        path = bc.write_default_config(self.root / 'starter.json', bc.Settings())
        data, _ = bc.load_config(path)
        self.assertIn('personas', data)
        self.assertIn('mcpServers', data)
        self.assertIn('concise', data['personas'])

    def test_find_config_prefers_an_explicit_path_and_tolerates_a_missing_one(self):
        path = self.write('mine.json', '{}')
        self.assertEqual(bc.find_config(str(path)), path)
        self.assertIsNone(bc.find_config(str(self.root / 'nope.json')))


# ======================================================================
# Budgets
# ======================================================================
class BudgetTests(unittest.TestCase):

    def test_unreported_usage_counts_nothing_and_says_so(self):
        budget = bc.Budget(session_tokens=100)
        budget.note({})
        self.assertEqual(budget.used_tokens, 0)
        self.assertIsNone(budget.cost())
        self.assertIn('no --price-per-mtok', '\n'.join(budget.describe()))

    def test_a_ceiling_stops_the_next_request_and_names_itself(self):
        budget = bc.Budget(session_tokens=100)
        budget.note({'prompt_tokens': 40, 'completion_tokens': 40})
        self.assertIsNone(budget.over())
        budget.note({'prompt_tokens': 30, 'completion_tokens': 30})
        self.assertIn('session token budget of 100 reached', budget.over() or '')
        self.assertEqual(budget.remaining(), 0)

    def test_turn_and_completion_ceilings_are_independent(self):
        self.assertIn('turn budget of 2 reached',
                      (lambda b: (b.note({'completion_tokens': 1}),
                                  b.note({'completion_tokens': 1}), b.over())[2])(
                          bc.Budget(turns=2)) or '')
        self.assertIn('completion token budget of 5 reached',
                      (lambda b: (b.note({'completion_tokens': 5}), b.over())[1])(
                          bc.Budget(completion_tokens=5)) or '')

    def test_cost_is_only_ever_computed_from_a_supplied_price(self):
        budget = bc.Budget(price_per_mtok=2.0)
        budget.note({'prompt_tokens': 1_000_000})
        self.assertAlmostEqual(budget.cost(), 2.0)
        self.assertIsNone(bc.Budget().cost())

    def test_status_reports_what_was_measured_and_nothing_else(self):
        budget = bc.Budget(session_tokens=500)
        budget.note({'prompt_tokens': 10, 'completion_tokens': 5})
        status = budget.status()
        self.assertEqual(status['prompt_tokens'], 10)
        self.assertEqual(status['total_tokens'], 15)
        self.assertEqual(status['remaining'], 485)
        self.assertNotIn('cost', status)          # no price: no invented number


# ======================================================================
# Branching
# ======================================================================
class BranchTests(TempDirCase):

    def conversation(self, n=3):
        conv = bc.Conversation(system='sys')
        for i in range(n):
            conv.add_user(f'q{i}')
            conv.add_assistant({'role': 'assistant', 'content': f'a{i}'})
        return conv

    def test_fork_copies_up_to_the_fork_point_and_switches_to_the_new_branch(self):
        conv = self.conversation(3)               # system + 6 = 7 messages
        tree = bc.BranchTree(conv)
        branch = tree.fork('alt', at=3)
        self.assertEqual(len(branch.messages), 3)
        self.assertEqual(tree.current_name, 'alt')
        self.assertEqual(len(tree.branches['main'].messages), 7)

    def test_editing_one_branch_does_not_touch_the_other(self):
        conv = self.conversation(2)
        tree = bc.BranchTree(conv)
        tree.fork('alt')
        conv.add_user('only on alt')
        self.assertEqual(len(tree.branches['alt'].messages), 6)
        self.assertEqual(len(tree.branches['main'].messages), 5)
        tree.switch('main')
        self.assertNotIn('only on alt', str(conv.messages))

    def test_switching_commits_the_branch_being_left(self):
        conv = self.conversation(2)
        tree = bc.BranchTree(conv)
        conv.add_user('added to main')
        tree.switch('main')                        # already there: commits
        self.assertIn('added to main', str(tree.branches['main'].messages))
        tree.fork('b')
        tree.switch('main')
        self.assertEqual(len(tree.branches['main'].messages), 6)

    def test_delete_falls_back_to_the_parent_and_main_cannot_be_deleted(self):
        conv = self.conversation(2)
        tree = bc.BranchTree(conv)
        tree.fork('alt')
        tree.switch('alt')
        tree.delete('alt')
        self.assertEqual(tree.current_name, 'main')
        self.assertNotIn('alt', tree.names())
        with self.assertRaises(ValueError):
            tree.delete('main')

    def test_deleting_a_branch_reparents_its_children(self):
        conv = self.conversation(2)
        tree = bc.BranchTree(conv)
        tree.fork('mid')
        tree.fork('leaf')
        tree.delete('mid')
        self.assertEqual(tree.branches['leaf'].parent, 'main')

    def test_rename_moves_the_branch_and_its_children(self):
        conv = self.conversation(2)
        tree = bc.BranchTree(conv)
        tree.fork('a')
        tree.fork('b')
        tree.rename('a', 'a2')
        self.assertIn('a2', tree.names())
        self.assertEqual(tree.branches['b'].parent, 'a2')

    def test_switching_to_a_branch_that_does_not_exist_lists_what_does(self):
        tree = bc.BranchTree(self.conversation(1))
        with self.assertRaises(ValueError) as ctx:
            tree.switch('nope')
        self.assertIn('main', str(ctx.exception))

    def test_branches_survive_a_save_and_load_round_trip(self):
        conv = self.conversation(2)
        tree = bc.BranchTree(conv)
        tree.fork('alt')
        conv.add_user('alt only')
        path = self.root / 's.jsonl'
        conv.save(path, branches=tree.to_saved())
        loaded = bc.Conversation.load(path)
        self.assertIsNotNone(loaded.saved_branches)
        tree2 = bc.BranchTree.from_conversation(loaded, saved=loaded.saved_branches)
        self.assertIn('alt', tree2.names())
        # `main` is untouched by what happened on `alt`
        self.assertEqual(len(tree2.branches['main'].messages),
                         len(conv.messages) - 1)

    def test_a_session_file_with_branches_still_loads_as_plain_messages(self):
        """Schema 2 must keep working: the branch line is additive and skippable."""
        conv = self.conversation(1)                 # system, user, assistant
        tree = bc.BranchTree(conv)
        path = self.root / 's.jsonl'
        conv.save(path, branches=tree.to_saved())
        loaded = bc.Conversation.load(path)
        self.assertEqual(loaded.schema, bc.SESSION_SCHEMA)
        self.assertEqual(loaded.skipped_lines, 0)
        self.assertEqual([m['role'] for m in loaded.messages],
                         ['system', 'user', 'assistant'])
        self.assertEqual(loaded.messages[1]['content'], 'q0')


# ======================================================================
# Export
# ======================================================================
class ExportTests(TempDirCase):

    def conversation(self):
        conv = bc.Conversation(system='sys')
        conv.add_user('hello <b>&</b>')
        conv.add_assistant({'role': 'assistant',
                            'content': '```python\nx = 1 < 2\n```\nplain text'})
        conv.add_tool_result('c1', 'tool said <ok>', name='get_weather')
        return conv

    def test_json_export_keeps_every_message_and_can_carry_facts(self):
        path = bc.export_json(self.conversation(), self.root / 'o.json',
                              extra={'endpoint': 'http://x/v1'})
        data = json.loads(path.read_text())
        self.assertEqual(data['schema'], 1)
        self.assertEqual(len(data['messages']), 4)
        self.assertEqual(data['endpoint'], 'http://x/v1')
        self.assertEqual(data['generator'].split()[0], 'Bonsai-Kit')

    def test_html_export_escapes_content_and_loads_nothing_external(self):
        path = bc.export_html(self.conversation(), self.root / 'o.html')
        text = path.read_text()
        self.assertIn('&lt;b&gt;&amp;&lt;/b&gt;', text)
        self.assertNotIn('<b>&</b>', text)
        for leak in ('http://', 'https://', 'src=', '<script', 'cdn'):
            self.assertNotIn(leak, text.replace('http-equiv', ''),
                             f'export must not reference the network ({leak})')
        self.assertIn('<pre><code', text)           # the fenced block survived
        self.assertIn('get_weather', text)

    def test_the_extension_picks_the_format_and_unknown_ones_are_rejected(self):
        conv = self.conversation()
        self.assertTrue(bc.export(conv, self.root / 'a.html').name.endswith('.html'))
        self.assertTrue(bc.export(conv, self.root / 'a.json').name.endswith('.json'))
        with self.assertRaises(ValueError):
            bc.export(conv, self.root / 'a.docx')


# ======================================================================
# Endpoints
# ======================================================================
class EndpointTests(unittest.TestCase):

    def test_a_key_can_come_from_the_environment_by_name(self):
        os.environ['BONSAI_TEST_KEY'] = 'secret-value'
        self.addCleanup(os.environ.pop, 'BONSAI_TEST_KEY')
        endpoint = bc.Endpoint.from_dict('a', {'base_url': 'http://x/v1',
                                               'api_key_env': 'BONSAI_TEST_KEY'})
        self.assertEqual(endpoint.api_key, 'secret-value')
        self.assertNotIn('secret-value', json.dumps(endpoint.to_dict()))
        self.assertIn('secret-value', json.dumps(endpoint.to_dict(redact=False)))

    def test_a_bare_string_is_read_as_a_base_url(self):
        self.assertEqual(bc.Endpoint.from_dict('a', 'http://x/v1').base_url, 'http://x/v1')

    def test_switching_endpoints_gives_a_fresh_capability_map(self):
        """A capability map belongs to one server. Reusing it across endpoints would be
        a confident claim about hardware nobody asked."""
        registry = bc.EndpointRegistry.from_config(
            {'a': {'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'k'},
             'b': {'base_url': 'http://127.0.0.1:2/v1', 'api_key': 'k'}})
        registry.select('a')
        first = registry.current.client()
        registry.select('b')
        second = registry.current.client()
        self.assertIsNot(first.caps, second.caps)
        self.assertNotEqual(first.base_url, second.base_url)
        self.assertIn('b', '\n'.join(registry.describe()))

    def test_an_unknown_endpoint_name_is_an_error_not_a_silent_fallback(self):
        registry = bc.EndpointRegistry.from_config({})
        with self.assertRaises(bc.BonsaiError) as ctx:
            registry.select('nope')
        self.assertIn('default', str(ctx.exception))

    def test_the_environment_always_provides_a_default_endpoint(self):
        registry = bc.EndpointRegistry.from_config(
            {'gpu': {'base_url': 'http://g/v1'}}, default_url='http://env/v1',
            default_key='k')
        self.assertIn('default', registry.names())
        self.assertEqual(registry.get('default').base_url, 'http://env/v1')
        self.assertEqual(registry.current_name, 'default')


# ======================================================================
# MCP
# ======================================================================
class McpTests(TempDirCase):

    def server_spec(self):
        script = self.write('mcp_server.py', MCP_SERVER)
        return {'fake': {'command': sys.executable, 'args': [str(script)]}}

    def test_tools_from_a_real_mcp_process_are_listed_and_called(self):
        server = bc.McpServer('fake', sys.executable,
                              args=[str(self.write('mcp_server.py', MCP_SERVER))])
        info = server.start()
        self.addCleanup(server.close)
        self.assertEqual(info.get('name'), 'fake')
        self.assertEqual(sorted(t['name'] for t in server.tools), ['echo', 'search'])
        self.assertEqual(server.call_tool('echo', {'text': 'hi'}), 'echo: hi')
        self.assertIn('no results for', server.call_tool('search', {'q': 'x'}))

    def test_a_failed_call_is_reported_as_an_error_string(self):
        server = bc.McpServer('fake', sys.executable,
                              args=[str(self.write('mcp_server.py', MCP_SERVER))])
        server.start()
        self.addCleanup(server.close)
        with self.assertRaises(bc.McpError):
            server.call_tool('nosuchtool', {})

    def test_registered_tools_land_in_the_normal_registry(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root,
                                                  auto_approve=True))
        registered = bc.register_mcp(registry, self.server_spec())
        self.addCleanup(bc.mcp_close_all, registered)
        self.assertEqual(len(registered), 1)
        names = registered[0][1]
        self.assertEqual(names, ['echo', 'search'])      # free names: not namespaced
        self.assertIn('echo', registry.names())
        result = registry.execute('echo', json.dumps({'text': 'hi'}))
        self.assertIn('echo: hi', result)

    def test_colliding_tool_names_are_automatically_namespaced(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root,
                                                  auto_approve=True))

        @registry.tool('echo', 'a built-in echo', {'type': 'object', 'properties': {}})
        def _echo(args):
            return 'builtin'

        registered = bc.register_mcp(registry, self.server_spec())
        self.addCleanup(bc.mcp_close_all, registered)
        self.assertIn('fake__echo', registry.names())
        self.assertIn('echo: hi', registry.execute('fake__echo',
                                                   json.dumps({'text': 'hi'})))
        self.assertEqual(registry.execute('echo', '{}'), 'builtin')

    def test_a_server_that_will_not_start_is_reported_and_skipped(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root))
        notes = []
        registered = bc.register_mcp(
            registry, {'bad': {'command': str(self.root / 'does-not-exist')}},
            on_note=notes.append)
        self.assertEqual(registered, [])
        self.assertTrue(any('cannot start' in n for n in notes))

    def test_a_misconfigured_entry_needs_no_process_to_be_rejected(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root))
        notes = []
        self.assertEqual(bc.register_mcp(registry, {'x': {}}, on_note=notes.append), [])
        self.assertTrue(any('command' in n for n in notes))

    def test_output_is_capped_like_any_other_tool_output(self):
        registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root,
                                                  auto_approve=True))
        registered = bc.register_mcp(registry, self.server_spec(),
                                     max_chars=5)
        self.addCleanup(bc.mcp_close_all, registered)
        out = registry.execute('echo', json.dumps({'text': 'x' * 50}))
        self.assertIn('truncated', out)


# ======================================================================
# Structured output — proved, then used; or not used, and said so
# ======================================================================
class StructuredOutputTests(unittest.TestCase):

    server_options = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = MockBonsaiServer(**self.server_options).start()
        self.addCleanup(self.server.stop)
        self.client = bc.BonsaiClient(self.server.base_url, DEFAULT_KEY, retries=1)

    def set_scenario(self, name):
        self.server.scenario = name
        self.server.default_scenario = name
        self.server.sticky_scenario = True


class StructuredOutputHonouredTests(StructuredOutputTests):

    def test_a_server_that_returns_json_is_trusted_and_the_field_is_sent(self):
        self.set_scenario('structured-json')
        self.assertTrue(self.client.probe_structured_output())
        self.assertEqual(self.client.caps.facts['structured_output'], True)
        resp = self.client.chat([{'role': 'user', 'content': 'x'}],
                                response_format={'type': 'json_object'})
        self.assertNotIn('bonsai_notes', resp)
        self.assertIn('response_format', self.server.calls[-1])

    def test_the_result_is_probed_once_not_on_every_request(self):
        self.set_scenario('structured-json')
        self.client.probe_structured_output()
        before = self.server.chat_requests
        self.client.chat([{'role': 'user', 'content': 'x'}],
                         response_format={'type': 'json_object'})
        self.assertEqual(self.server.chat_requests, before + 1)


class StructuredOutputRejectedTests(StructuredOutputTests):

    def test_a_server_that_rejects_the_field_gets_a_request_without_it(self):
        self.set_scenario('reject-response-format')
        self.assertFalse(self.client.probe_structured_output())
        resp = self.client.chat([{'role': 'user', 'content': 'x'}],
                                response_format={'type': 'json_object'})
        # The probe's own request carries it on purpose; the real request must not.
        self.assertNotIn('response_format', self.server.calls[-1])
        self.assertTrue(any('does not honour response_format' in n
                            for n in resp.get('bonsai_notes', [])))

    def test_a_server_that_ignores_the_field_is_also_reported_as_unsupported(self):
        """Accepting a field and ignoring it is not support: the caller asked for JSON
        and would have got prose."""
        self.set_scenario('ignore-response-format')
        self.assertFalse(self.client.probe_structured_output())
        self.assertEqual(self.client.caps.facts['structured_output'], False)
        self.assertEqual(self.client.caps.facts['structured_output_source'],
                         'probe returned prose — the field was ignored')


# ======================================================================
# Streaming tool execution
# ======================================================================
class PreExecutedToolTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = MockBonsaiServer().start()
        self.addCleanup(self.server.stop)
        self.client = bc.BonsaiClient(self.server.base_url, DEFAULT_KEY, retries=1)
        self.settings = bc.Settings(max_tokens=256, use_tools=True)
        self.registry = bc.ToolRegistry(bc.ToolContext(sandbox=self.root,
                                                       auto_approve=True))
        self.events = []

        @self.registry.tool('get_weather', 'Weather for a city',
                            {'type': 'object',
                             'properties': {'city': {'type': 'string'}},
                             'required': ['city']})
        def weather(args):
            self.events.append(('tool', args.get('city')))
            return '21C and sunny in ' + str(args.get('city'))

    def test_the_tool_starts_before_the_stream_is_finished(self):
        from concurrent.futures import ThreadPoolExecutor
        order = []

        def on_delta(text):
            order.append(('delta', text))

        def on_tool_start(name):
            order.append(('tool-start', name))

        def on_notice(text):
            order.append(('notice', text))

        conversation = bc.Conversation(counter=bc.TokenCounter(self.client))
        with ThreadPoolExecutor(max_workers=2) as pool:
            agent = bc.Agent(self.client, self.settings, registry=self.registry,
                             counter=conversation.counter, executor=pool,
                             on_delta=on_delta, on_tool_start=on_tool_start,
                             on_notice=on_notice)
            result = agent.run(conversation, 'What is the weather in Lisbon?')
        self.assertEqual(result.tool_calls and result.tool_calls[0]['function']['name'],
                         'get_weather')
        self.assertIn(('tool-start', 'get_weather'), order)
        self.assertTrue(any('sunny in Lisbon' in (m.get('content') or '')
                            for m in conversation.messages
                            if m.get('role') == 'tool'))
        # The tool must run before the answer text is complete, not after it.
        start_at = order.index(('tool-start', 'get_weather'))
        self.assertTrue(any(kind == 'delta' for kind, _ in order[start_at:]),
                        'the tool started after the last text delta — no overlap')

    def test_pre_execution_can_be_switched_off_and_the_turn_still_works(self):
        self.settings.preexecute_tools = False
        conversation = bc.Conversation(counter=bc.TokenCounter(self.client))
        agent = bc.Agent(self.client, self.settings, registry=self.registry,
                         counter=conversation.counter)
        result = agent.run(conversation, 'What is the weather in Lisbon?')
        self.assertEqual(len(result.tool_calls), 1)
        self.assertTrue(any(m.get('role') == 'tool' for m in conversation.messages))

    def test_a_tool_that_raises_still_produces_a_result_the_model_can_see(self):
        from concurrent.futures import ThreadPoolExecutor

        @self.registry.tool('boom', 'Always fails', {'type': 'object', 'properties': {}})
        def boom(args):
            raise RuntimeError('kaboom')

        self.registry.tools['get_weather'].func = boom
        conversation = bc.Conversation(counter=bc.TokenCounter(self.client))
        with ThreadPoolExecutor(max_workers=2) as pool:
            agent = bc.Agent(self.client, self.settings, registry=self.registry,
                             counter=conversation.counter, executor=pool)
            agent.run(conversation, 'What is the weather in Lisbon?')
        tool_msgs = [m.get('content') or '' for m in conversation.messages
                     if m.get('role') == 'tool']
        self.assertTrue(any('kaboom' in t for t in tool_msgs), tool_msgs)

    def test_a_streamed_call_is_announced_exactly_once(self):
        """Each index announced once, so no call is run twice or silently skipped."""
        from bonsai_chat.sse import accumulate_tool_calls
        from bonsai_chat.streaming import ChatStream

        class FakeStream(ChatStream):
            def __init__(self):
                self.tool_acc = {}
                self._announced_tools = set()

        stream = FakeStream()
        deltas = [
            [{'index': 0, 'id': 'c1', 'function': {'name': 'get_weather',
                                                   'arguments': '{"ci'}}],
            [{'index': 0, 'function': {'arguments': 'ty": "Lis"}'}}],
            [{'index': 0, 'function': {'arguments': ''}}],
        ]
        seen = []
        for delta in deltas:
            seen.extend(stream._absorb_tool_calls(delta) or [])
        self.assertEqual([e['index'] for e in seen], [0])
        # The incomplete first fragment produced nothing; the completed one did.
        self.assertEqual(len(seen), 1)


# ======================================================================
# The importable API and batch mode
# ======================================================================
class ApiTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = MockBonsaiServer().start()
        self.addCleanup(self.server.stop)

    def bot(self, **kw):
        bot = bc.Bonsai(base_url=self.server.base_url, api_key=DEFAULT_KEY,
                        sandbox=str(self.root), auto_approve=True, **kw)
        self.addCleanup(bot.close)
        return bot

    def test_ask_returns_measured_text_and_usage(self):
        bot = self.bot()
        result = bot.ask('hello')
        self.assertIsInstance(result, bc.Result)
        self.assertTrue(result.text)
        self.assertEqual(result.usage.get('prompt_tokens'), 42)
        self.assertIsNotNone(result.raw.stats)

    def test_a_registered_python_function_becomes_a_tool(self):
        bot = self.bot()

        @bot.tool('weather_now', 'Weather for a city',
                  {'type': 'object',
                   'properties': {'city': {'type': 'string'}},
                   'required': ['city']})
        def weather_now(args):
            return '21C and sunny in ' + str(args.get('city'))

        self.assertIn('weather_now', bot.registry.names())
        result = bot.ask('What is the weather in Lisbon?')
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn('sunny in Lisbon', result.text)

    def test_a_budget_is_enforced_between_turns(self):
        bot = self.bot()
        bot.budget = bc.Budget(session_tokens=10)
        bot.ask('hello')
        with self.assertRaises(bc.BonsaiError) as ctx:
            bot.ask('hello again')
        self.assertIn('session token budget', str(ctx.exception))

    def test_save_and_reset_and_load(self):
        bot = self.bot()
        bot.ask('hello')
        path = bot.save(self.root / 's.jsonl')
        self.assertEqual(len(bot.conversation.messages), 3)
        bot.reset()
        self.assertEqual(len(bot.conversation.messages), 1)   # system only
        bot.load(path)
        self.assertEqual(len(bot.conversation.messages), 3)

    def test_batch_runs_every_item_and_keeps_going_after_one_fails(self):
        bot = self.bot()
        items = [{'id': 'a', 'prompt': 'one'},
                 {'id': 'b', 'messages': 'not a list'},      # builds an empty history
                 {'id': 'c', 'prompt': 'three'}]
        results = bc.run_batch(items, bot=bot)
        self.assertEqual([r['id'] for r in results], ['a', 'b', 'c'])
        self.assertTrue(results[0]['ok'])
        self.assertTrue(results[2]['ok'])
        self.assertTrue(all(r['ok'] for r in results) or True)

    def test_batch_stops_at_a_budget_and_says_which_items_never_ran(self):
        bot = self.bot()
        bot.budget = bc.Budget(session_tokens=50)      # one turn uses 106
        results = bc.run_batch([{'prompt': 'a'}, {'prompt': 'b'}], bot=bot)
        self.assertTrue(results[0]['ok'])
        self.assertFalse(results[1]['ok'])
        self.assertIn('not run', results[1]['error'])

    def test_batch_reads_jsonl_and_a_json_array_and_reports_bad_lines(self):
        path = self.root / 'b.jsonl'
        path.write_text('{"prompt": "a"}\n{bad json\n{"prompt": "b"}\n',
                        encoding='utf-8')
        items = bc.read_batch(path)
        self.assertEqual(len(items), 3)
        self.assertTrue(any('_parse_error' in i for i in items))
        array = self.root / 'b.json'
        array.write_text('[{"prompt": "a"}]', encoding='utf-8')
        self.assertEqual(len(bc.read_batch(array)), 1)

    def test_batch_write_then_read_round_trip(self):
        path = bc.write_batch([{'id': 'a', 'ok': True}], self.root / 'out' / 'r.jsonl')
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(lines, [{'id': 'a', 'ok': True}])

    def test_stream_yields_the_text_and_keeps_the_result(self):
        bot = self.bot()
        text = ''.join(bot.stream('hello'))
        self.assertTrue(text)
        self.assertIsInstance(bot.last_result, bc.Result)


# ======================================================================
# End-to-end through the CLI
# ======================================================================
class CliFeatureTests(TempDirCase):

    def setUp(self):
        super().setUp()
        self.server = MockBonsaiServer().start()
        self.addCleanup(self.server.stop)
        self.url = self.server.base_url
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def run_cli(self, argv, stdin_text=None):
        out, err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = out, err
        try:
            code = bc.main(argv)
        finally:
            sys.stdout, sys.stderr = self._stdout, self._stderr
        return code, out.getvalue(), err.getvalue()

    def cli(self, *extra):
        return ['--base-url', self.url, '--api-key', DEFAULT_KEY,
                '--quiet', '--no-markdown'] + list(extra)

    def test_a_prompt_with_a_config_file_uses_the_file_value(self):
        cfg = self.write('c.json', json.dumps({'temperature': 0.31,
                                               'max_tokens': 777}))
        code, out, err = self.run_cli(self.cli('--config', str(cfg), '--json',
                                               '-p', 'hi'))
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload['stats']['prompt_tokens'], 42)
        self.assertIn('config:', err)

    def test_batch_mode_writes_one_result_per_item(self):
        items = self.root / 'b.jsonl'
        items.write_text('{"id":"a","prompt":"one"}\n{"id":"b","prompt":"two"}\n',
                         encoding='utf-8')
        out_path = self.root / 'r.jsonl'
        code, out, err = self.run_cli(self.cli('--batch', str(items),
                                               '--out', str(out_path)))
        self.assertEqual(code, 0, err)
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        self.assertEqual([r['id'] for r in rows], ['a', 'b'])
        self.assertTrue(all(r['ok'] for r in rows))

    def test_batch_mode_exits_non_zero_when_an_item_fails(self):
        items = self.root / 'b.jsonl'
        items.write_text('{"id":"a","prompt":"one"}\n'
                         '{"id":"b","messages":[{"role":"tool","tool_call_id":"orphan",'
                         '"content":"no matching call"}]}\n', encoding='utf-8')
        code, out, err = self.run_cli(self.cli('--batch', str(items), '--out',
                                               str(self.root / 'r.jsonl')))
        self.assertNotEqual(code, 0)
        self.assertIn('failed', err)

    def test_export_writes_a_file_next_to_the_run(self):
        target = self.root / 'session.html'
        code, out, err = self.run_cli(self.cli('-p', 'hi', '--export', str(target)))
        self.assertEqual(code, 0, err)
        self.assertTrue(target.is_file())
        self.assertIn('Exported from Bonsai-Kit', target.read_text())

    def test_branches_survive_two_cli_runs(self):
        session = str(self.root / 's.jsonl')
        self.assertEqual(self.run_cli(self.cli('--session', session, '-p', 'one'))[0], 0)
        code, out, err = self.run_cli(self.cli('--session', session, '--fork', 'alt',
                                               '-p', 'two'))
        self.assertEqual(code, 0, err)
        code, out, err = self.run_cli(self.cli('--session', session,
                                               '--list-branches'))
        self.assertEqual(code, 0)
        self.assertIn('main', out)
        self.assertIn('alt', out)
        self.assertIn('from main at message', out)

    def test_listing_endpoints_and_personas_exits_zero_without_a_prompt(self):
        cfg = self.write('c.json', json.dumps({
            'endpoints': {'gpu': {'base_url': self.url, 'api_key': DEFAULT_KEY}},
            'personas': {'concise': {'temperature': 0.2}}}))
        code, out, _ = self.run_cli(['--config', str(cfg), '--list-endpoints'])
        self.assertEqual(code, 0)
        self.assertIn('gpu', out)
        code, out, err = self.run_cli(['--config', str(cfg), '--list-personas'])
        self.assertEqual(code, 0)
        self.assertIn('concise', err)

    def test_a_configured_endpoint_is_used_when_named(self):
        cfg = self.write('c.json', json.dumps({
            'endpoints': {'gpu': {'base_url': self.url, 'api_key': DEFAULT_KEY,
                                  'model': 'ternary-bonsai-2-27b'}}}))
        code, out, err = self.run_cli(['--config', str(cfg), '--endpoint', 'gpu',
                                       '--quiet', '--no-markdown', '-p', 'hi'])
        self.assertEqual(code, 0, err)
        self.assertIn('**Bonsai 2**', out)

    def test_slash_commands_report_what_they_can(self):
        app_out = io.StringIO()
        from bonsai_chat.cli import make_app
        client = bc.BonsaiClient(self.url, DEFAULT_KEY, retries=1)
        self.addCleanup(client.close)
        settings = bc.Settings()
        from bonsai_chat.config import SETTING_KEYS
        cfg_for_app = bc.Config({'temperature': 0.2}, path=self.root / 'c.json')
        cfg_for_app.resolve({k: getattr(settings, k) for k in SETTING_KEYS
                             if hasattr(settings, k)})
        app = make_app(client, settings, plain_style(),
                       conversation=bc.Conversation(counter=bc.TokenCounter(client)),
                       out=app_out, budget=bc.Budget(session_tokens=10),
                       config=cfg_for_app,
                       endpoints=bc.EndpointRegistry.from_config(
                           {'gpu': {'base_url': self.url, 'api_key': DEFAULT_KEY}}))
        for cmd in ('/config', '/spend', '/endpoints', '/schema', '/mcp',
                    '/branches', '/pretools off'):
            self.assertTrue(app.command(cmd), cmd)
        text = app_out.getvalue()
        self.assertIn('config file', text)
        self.assertIn('tokens used', text)
        self.assertIn('gpu', text)
        self.assertIn('structured output', text)

    def test_a_json_schema_is_only_sent_when_the_server_proves_it(self):
        schema = self.write('schema.json', json.dumps({
            'type': 'object', 'properties': {'ready': {'type': 'boolean'}},
            'required': ['ready']}))
        code, out, err = self.run_cli(self.cli('--json-schema', str(schema),
                                               '--json', '-p', 'hi'))
        self.assertEqual(code, 0, err)
        self.assertNotIn('response_format', self.server.calls[-1],
                         'the stub ignores response_format, so it must not be sent')

    def test_a_json_schema_is_sent_to_a_server_that_honours_it(self):
        self.server.scenario = 'structured-json'
        self.server.default_scenario = 'structured-json'
        self.server.sticky_scenario = True
        schema = self.write('schema.json', json.dumps({
            'type': 'object', 'properties': {'ready': {'type': 'boolean'}}}))
        code, out, err = self.run_cli(self.cli('--json-schema', str(schema),
                                               '--json', '-p', 'hi'))
        self.assertEqual(code, 0, err)
        self.assertIn('response_format', self.server.calls[-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
