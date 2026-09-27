"""Offline protocol self-test."""

from __future__ import annotations

from pathlib import Path
import io
import struct
import sys

from ._meta import DEFAULT_MODEL
from .agent import Agent
from .app import ChatApp
from .client import BonsaiClient
from .conversation import Conversation, DEFAULT_SYSTEM, SESSION_SCHEMA, Settings
from .doctor import run_doctor
from .errors import BonsaiAPIError, BonsaiError
from .images import ImageAttachment, PNG_MAGIC, build_user_message
from .markdown import render_markdown
from .render import LiveRenderer
from .style import Style
from .tokens import TokenCounter
from .tools import Tool, ToolContext, ToolRegistry

def make_png(path, width=4, height=3, rgb=(200, 100, 50)):
    """Write a tiny valid PNG (used by --selftest; no third-party dependency)."""
    import zlib

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    row = b'\x00' + bytes(rgb) * width
    idat = zlib.compress(row * height)
    Path(path).write_bytes(PNG_MAGIC + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) +
                           chunk(b'IEND', b''))
    return Path(path)

def run_selftest(style=None, out=None):
    """Exercise the real client code paths against the offline protocol stub."""
    import tempfile
    from mock_bonsai_server import DEFAULT_KEY, MockBonsaiServer

    style = style or Style(force_color=False)
    out = out if out is not None else sys.stdout
    results = []

    def check(name, ok, detail=''):
        results.append(bool(ok))
        out.write(f'{"PASS" if ok else "FAIL"}  {name}' + (f' — {detail}' if detail else '') + '\n')
        out.flush()

    server = MockBonsaiServer().start()
    tmp = tempfile.TemporaryDirectory()
    try:
        root = Path(tmp.name)
        client = BonsaiClient(server.base_url, DEFAULT_KEY, model=DEFAULT_MODEL, retries=1)
        settings = Settings(effort='medium', max_tokens=512)

        check('models endpoint lists the alias', DEFAULT_MODEL in client.model_ids(),
              ', '.join(client.model_ids()))
        check('/props reports the context window', client.context_window() == 8192,
              f'{client.context_window()} tokens')
        counter = TokenCounter(client)
        n = counter.count('hello world, this is a tokenisation probe')
        check('/tokenize counts server-side', n > 0, f'{n} tokens')

        ctx = ToolContext(sandbox=root, notify=lambda t: None, auto_approve=True)
        registry = ToolRegistry(ctx)
        registry.add(Tool('get_weather', 'Current weather for a city',
                          {'type': 'object', 'properties': {'city': {'type': 'string'}},
                           'required': ['city']},
                          func=lambda a: '21°C and sunny in ' + str(a.get('city'))))
        conv = Conversation(system=DEFAULT_SYSTEM, counter=counter)
        app = ChatApp(client, settings, style, registry, conv,
                      session_path=root / 'session.jsonl', quiet=True,
                      input_fn=lambda *_: '', out=out, auto_approve=True)
        app.renderer = LiveRenderer(style, markdown=False, out=out)
        app.agent = Agent(client, settings, registry=registry, counter=counter,
                          on_delta=app.renderer.delta, on_reasoning=app.renderer.reasoning,
                          on_tool=lambda *a: None, on_notice=app.renderer.notice)

        answer = app.send('Explain why C++ can be fast in 3 sentences.')
        check('streaming chat produced an answer', bool(answer and answer.text.strip()),
              f'{len(answer.text)} chars' if answer else 'no result')
        check('streaming reported usage', bool(answer and answer.completion_tokens),
              f'{answer.completion_tokens} completion tokens' if answer else '')
        check('reasoning deltas arrived', bool(answer and answer.reasoning),
              f'{len(answer.reasoning)} chars of thinking' if answer else '')

        conv2 = Conversation(system=DEFAULT_SYSTEM, counter=counter)
        app2 = ChatApp(client, settings, style, registry, conv2, out=out, auto_approve=True)
        app2.renderer = LiveRenderer(style, markdown=False, out=out)
        app2.agent = Agent(client, settings, registry=registry, counter=counter,
                           on_delta=app2.renderer.delta, on_reasoning=app2.renderer.reasoning,
                           on_tool=lambda *a: None, on_notice=app2.renderer.notice)
        tooled = app2.send('What is the weather in Lisbon right now? Call the tool.')
        roles = [m['role'] for m in conv2.messages]
        check('model requested a tool', bool(tooled and tooled.tool_calls),
              ', '.join(tc['function']['name'] for tc in (tooled.tool_calls if tooled else [])))
        check('tool result was fed back', 'tool' in roles, ' -> '.join(roles))
        check('tool loop finished with an answer',
              bool(tooled and tooled.text.strip() and tooled.rounds >= 2),
              f'{tooled.rounds if tooled else 0} round(s)')

        png = make_png(root / 'chart.png')
        attachment = ImageAttachment.load(png, use_ocr=False)
        check('image header parsed without Pillow', attachment.width == 4 and attachment.height == 3,
              f'{attachment.fmt} {attachment.width}x{attachment.height}')
        check('vision probe detects a text-only server', client.probe_vision() is False)
        card = attachment.text_card(vision=False)
        check('text-only image card states the limits', 'TEXT-ONLY' in card and '4x3 px' in card)
        msg = build_user_message('what is in this image?', [attachment], vision=False)
        check('image facts are sent to the model', 'Image attached' in msg['content'])

        rendered = render_markdown('## Title\n\n- one\n- two\n\n```python\nprint(1)\n```\n',
                                   Style(force_color=False))
        check('markdown renderer emits blocks', '┌─' in rendered and 'Title' in rendered and
              'print(1)' in rendered)
        table = render_markdown('| a | b |\n| --- | --- |\n| 1 | 2 |\n', Style(force_color=False))
        check('markdown renderer draws tables', '┼' in table and '│' in table)

        # ------------------------------------------------------------ v0.6.0 power features
        from .config import Config
        from .budget import Budget
        from .branches import BranchTree
        from .endpoints import EndpointRegistry
        from .export import export
        from .api import run_batch

        # Configuration: precedence is a claim the client must be able to show, not
        # just implement.
        cfg = Config({'temperature': 0.25, 'max_tokens': 999,
                      'personas': {'concise': {'temperature': 0.05}}},
                     path=root / 'c.json')
        defaults = {k: getattr(settings, k) for k in
                    ('temperature', 'max_tokens') if hasattr(settings, k)}
        cfg.resolve(defaults)
        check('a config file value beats the built-in default',
              cfg.source_of('temperature') == 'config file')
        cfg.select_persona('concise')
        cfg.resolve(defaults, cli={'temperature': 0.9})
        check('the typed flag beats both the file and the persona',
              cfg.source_of('temperature') == 'command line')

        # Budgets: measured, and silent about cost until a price is supplied.
        budget = Budget(session_tokens=100)
        budget.note({'prompt_tokens': 30, 'completion_tokens': 30})
        check('a token budget counts what the server reported',
              budget.used_tokens == 60 and budget.over() is None)
        budget.note({'prompt_tokens': 30, 'completion_tokens': 30})
        check('a token budget stops the next turn', budget.over() is not None,
              budget.over() or '')
        check('cost stays unknown until a price is given', Budget().cost() is None)

        # Branching: a fork is a copy, not an alias.
        branched = Conversation(system=DEFAULT_SYSTEM, counter=TokenCounter())
        branched.add_user('one')
        branched.add_assistant({'role': 'assistant', 'content': 'two'})
        tree = BranchTree(branched)
        tree.fork('alt')
        branched.add_user('only on alt')
        check('a fork is independent of the branch it came from',
              len(tree.branches['main'].messages) == 3 and
              len(tree.branches['alt'].messages) == 4)
        saved = branched.save(root / 'branched.jsonl', branches=tree.to_saved())
        reloaded = Conversation.load(saved)
        check('branches survive a save/load round trip',
              reloaded.saved_branches is not None and reloaded.schema == SESSION_SCHEMA)

        # Endpoints: a capability map belongs to the server it came from.
        registry_eps = EndpointRegistry.from_config(
            {'a': {'base_url': 'http://127.0.0.1:1/v1', 'api_key': 'k'},
             'b': {'base_url': 'http://127.0.0.1:2/v1', 'api_key': 'k'}})
        registry_eps.select('a')
        first_caps = registry_eps.current.client().caps
        registry_eps.select('b')
        check('switching endpoints starts a fresh capability map',
              registry_eps.current.client().caps is not first_caps)

        # Export: self-contained, escaped, no network references.
        html_path = export(branched, root / 'session.html')
        html_text = html_path.read_text()
        check('HTML export is self-contained',
              '<div class="msg user">' in html_text and 'http://' not in html_text)

        # Batch: the same client, the same retry policy, one conversation per item.
        from .api import Bonsai
        bot = Bonsai(base_url=server.base_url, api_key=DEFAULT_KEY,
                     sandbox=str(root), auto_approve=True)
        try:
            batch_results = run_batch([{'id': 'a', 'prompt': 'one'},
                                       {'id': 'b', 'prompt': 'two'}], bot=bot)
        finally:
            bot.close()
        check('batch mode runs every item',
              len(batch_results) == 2 and all(r['ok'] for r in batch_results),
              ', '.join(f"{r['id']}:{'ok' if r['ok'] else r['error']}"
                        for r in batch_results))

        # Structured output: proved before use, and reported either way.
        server.scenario = 'ignore-response-format'
        server.default_scenario = 'ignore-response-format'
        server.sticky_scenario = True
        prove = BonsaiClient(server.base_url, DEFAULT_KEY, retries=1)
        honoured = prove.probe_structured_output()
        check('a server that ignores response_format is not trusted', honoured is False,
              str(prove.caps.facts.get('structured_output_source')))
        server.scenario = server.default_scenario = 'ok'
        server.sticky_scenario = False

        trimmed = Conversation(system=DEFAULT_SYSTEM, counter=TokenCounter())
        for i in range(40):
            trimmed.add_user('question ' + str(i) + ' ' + ('x' * 400))
            trimmed.add_assistant({'role': 'assistant', 'content': 'answer ' + str(i) + ' ' + ('y' * 400)})
        before = len(trimmed.messages)
        trimmed.trim(400)
        wired = trimmed.wire()
        orphans = [i for i, m in enumerate(wired) if m['role'] == 'tool']
        check('history trimming keeps the budget', trimmed.tokens() <= 400,
              f'{before} -> {len(trimmed.messages)} messages, {trimmed.tokens()} tokens')
        check('trimming never orphans tool results', not orphans)

        bad = BonsaiClient(server.base_url, 'wrong-key', model=DEFAULT_MODEL, retries=1)
        try:
            bad.models()
            check('bad key is rejected', False, 'no error raised')
        except BonsaiAPIError as e:
            check('bad key is rejected', e.status == 401 and bool(e.hint()),
                  f'HTTP {e.status}')
        try:
            dead = BonsaiClient('http://127.0.0.1:1/v1', 'x', retries=1, timeout=2)
            dead.chat([{'role': 'user', 'content': 'hi'}])
            check('dead endpoint explains itself', False, 'no error raised')
        except BonsaiError as e:
            check('dead endpoint explains itself', 'cannot reach' in str(e))
        doctor_out = io.StringIO()
        doctor_code = run_doctor(client, style=style, out=doctor_out)
        check('--doctor passes against the stub endpoint', doctor_code == 0,
              doctor_out.getvalue().strip().splitlines()[-2] if doctor_out.getvalue() else '')
    finally:
        server.stop()
        tmp.cleanup()

    passed = sum(results)
    out.write(f'\n{passed}/{len(results)} self-test checks passed\n')
    out.flush()
    return 0 if passed == len(results) else 1
