"""Argument parsing and process entry point."""

from __future__ import annotations

from pathlib import Path
import argparse
import io
import json
import os
import signal
import sys

import base64
import mimetypes
import shutil

from ._meta import DEFAULT_MODEL, VERSION
from .agent import Agent
from .app import ChatApp
from .benchmark import render_benchmark, run_benchmark
from .client import BonsaiClient
from .conversation import Conversation, DEFAULT_SYSTEM, Settings
from .doctor import run_doctor
from .errors import BonsaiAPIError, BonsaiError
from .images import load_attachments
from .reasoning import REASONING_DISPLAYS, ReasoningConfig
from .render import LiveRenderer
from .selftest import run_selftest
from .style import Style
from .tokens import TokenCounter
from .tools import ToolContext, ToolRegistry

def build_parser():
    p = argparse.ArgumentParser(
        prog='bonsai_chat.py',
        description='Chat client for a Ternary Bonsai 2 27B OpenAI-compatible endpoint.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Environment: BONSAI_BASE_URL, BONSAI_API_KEY, BONSAI_MODEL, BONSAI_SESSION.\n'
               'The deployment cell (colab_kaggle_cell.py) prints the base URL and key.')
    p.add_argument('--serve', action='store_true', help='serve the local web UI and API proxy')
    p.add_argument('--serve-host', default='127.0.0.1')
    p.add_argument('--serve-port', type=int, default=0)
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--base-url', default=os.environ.get('BONSAI_BASE_URL'),
                   help='e.g. https://<tunnel-host>/v1')
    p.add_argument('--api-key', default=os.environ.get('BONSAI_API_KEY'),
                   help='bearer key printed by the deployment cell')
    p.add_argument('--model', default=os.environ.get('BONSAI_MODEL', DEFAULT_MODEL))
    p.add_argument('-p', '--prompt', help='one-shot prompt instead of the interactive loop')
    p.add_argument('--image', action='append', default=[], help='attach an image (repeatable)')
    p.add_argument('--ocr', action='store_true', help='OCR attached images with tesseract')
    p.add_argument('--image-max-side', type=int, default=1024,
                   help='downscale attached images to this many pixels (default 1024)')
    p.add_argument('--system', default=None, help='system prompt text')
    p.add_argument('--system-file', default=None, help='read the system prompt from a file')
    p.add_argument('--session', default=os.environ.get('BONSAI_SESSION'),
                   help='autosave/load the conversation to this .jsonl file')
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--effort', default='medium',
                   help='reasoning_effort sent to the chat template: medium|xhigh '
                        '(Bonsai 2 documents `low` as a no-op, so it is not offered; '
                        '`none` turns thinking off)')
    p.add_argument('--think', default='medium',
                   help='thinking budget: off|low|medium|high|max (=0/512/2048/8192/unlimited '
                        'thinking_budget_tokens) or an explicit token count')
    p.add_argument('--reasoning-display', default='compact', choices=list(REASONING_DISPLAYS),
                   help='full: stream the thinking trace; compact: one indicator + a token '
                        'count; hidden: count it only')
    p.add_argument('--no-stream', action='store_true', help='request one complete answer')
    p.add_argument('--no-markdown', action='store_true', help='print model output verbatim')
    p.add_argument('--no-highlight', action='store_true', help='no Pygments highlighting in code blocks')
    p.add_argument('--no-tools', action='store_true', help='disable tool calling')
    p.add_argument('--tools', help='comma-separated tool names to enable (default: all)')
    p.add_argument('--max-tool-rounds', type=int, default=6)
    p.add_argument('--tool-timeout', type=float, default=60.0,
                   help='seconds before an individual tool call is abandoned')
    p.add_argument('--tool-budget', type=float, default=600.0,
                   help='total seconds the tool loop may take in one turn')
    p.add_argument('--tool-max-output', type=int, default=24000,
                   help='characters of tool output fed back to the model per call')
    p.add_argument('--autosave-every', type=int, default=1,
                   help='write the session file every N turns (default every turn)')
    p.add_argument('--context-reserve', type=int, default=512,
                   help='tokens held back for the chat template')
    p.add_argument('--output-reserve', type=int, default=1024,
                   help='tokens held back for the answer being generated')
    p.add_argument('--show-stats', action='store_true',
                   help='print the full timing table after every turn (same as /speed on)')
    p.add_argument('--no-keepalive', action='store_true',
                   help='open a fresh connection per request (debugging only; slower)')
    p.add_argument('--sandbox', default=None, help='directory tools may touch (default: cwd)')
    p.add_argument('--auto-approve', action='store_true',
                   help='let the model write files and run commands without asking')
    p.add_argument('--timeout', type=int, default=600, help='streaming timeout in seconds')
    p.add_argument('--retries', type=int, default=3)
    p.add_argument('--context', type=int, default=None, help='override the context window in tokens')
    p.add_argument('--plain', action='store_true', help='disable colour')
    p.add_argument('--color', action='store_true',
                   help='force colour even when stdout is not a terminal')
    p.add_argument('--width', type=int, default=None, help='render width (default: terminal width)')
    p.add_argument('--json', action='store_true', help='one-shot mode: print the raw result as JSON')
    p.add_argument('--quiet', action='store_true', help='suppress the banner')
    p.add_argument('--selftest', action='store_true',
                   help='run the client against a local protocol stub and exit')
    p.add_argument('--doctor', action='store_true',
                   help='diagnose a real endpoint (health, models, context, auth, chat, '
                        'streaming, tool calling, vision) and exit')
    p.add_argument('--benchmark', action='store_true',
                   help='measure TTFT, prefill and decode speed against the endpoint and exit')
    p.add_argument('--bench-samples', type=int, default=3,
                   help='benchmark samples after warmup (default 3)')
    p.add_argument('--bench-warmup', type=int, default=1,
                   help='benchmark warmup samples to discard (default 1)')
    p.add_argument('--mock', action='store_true',
                   help='run against the bundled protocol stub (scripted replies, no model, '
                        'no GPU) to try the client without a deployment')
    # ---------------------------------------------------------------- configuration
    p.add_argument('--config', default=None, metavar='FILE',
                   help='config file (default: the first of bonsai.json, .bonsai.json, '
                        '~/.config/bonsai/config.json, ~/.bonsai.json that exists)')
    p.add_argument('--save-config', action='store_true',
                   help='write a starter config file (with --config FILE) and exit')
    p.add_argument('--persona', default=None,
                   help='apply a persona preset from the config file')
    p.add_argument('--list-personas', action='store_true',
                   help='list the personas in the config file and exit')
    # ---------------------------------------------------------------- endpoints
    p.add_argument('--endpoint', default=None,
                   help='use the named endpoint from the config file')
    p.add_argument('--list-endpoints', action='store_true',
                   help='list configured endpoints and exit')
    # ---------------------------------------------------------------- budgets
    p.add_argument('--budget-tokens', type=int, default=None,
                   help='stop after this many tokens have been used in this session')
    p.add_argument('--budget-turns', type=int, default=None,
                   help='stop after this many turns')
    p.add_argument('--budget-prompt-tokens', type=int, default=None,
                   help='stop after this many prompt tokens')
    p.add_argument('--price-per-mtok', type=float, default=None,
                   help='your price per million tokens, so /budget can show a cost '
                        '(the client has no price list of its own)')
    # ---------------------------------------------------------------- structured io
    p.add_argument('--json-schema', default=None, metavar='FILE',
                   help='ask for JSON matching this schema; only sent if the server '
                        'proves it honours response_format, otherwise enforced locally')
    p.add_argument('--batch', default=None, metavar='FILE',
                   help='run a .jsonl (or JSON array) of requests non-interactively')
    p.add_argument('--out', default=None, metavar='FILE',
                   help='with --batch: where to write the results (default: stdout)')
    p.add_argument('--export', default=None, metavar='FILE',
                   help='export the conversation (.json or .html) when the run ends')
    # ---------------------------------------------------------------- branching
    p.add_argument('--branch', default=None,
                   help='with --session: work on this branch')
    p.add_argument('--fork', default=None, metavar='NAME',
                   help='with --session: fork a new branch at the loaded point')
    p.add_argument('--list-branches', action='store_true',
                   help='with --session: list branches and exit')
    p.add_argument('--no-pretools', action='store_true',
                   help='do not start tools mid-stream (run them after the turn instead)')
    p.add_argument('--version', action='version', version='bonsai_chat ' + VERSION)
    return p

def make_app(client, settings, style, registry=None, conversation=None, sandbox=None,
             session_path=None, use_ocr=False, image_max_side=1024, auto_approve=False,
             quiet=False, input_fn=None, out=None, tree=None, budget=None,
             config=None, endpoints=None, mcp=()):
    """Assemble a ChatApp exactly the way the CLI does (tests and --selftest use this too)."""
    out = out if out is not None else sys.stdout
    sandbox = Path(sandbox).expanduser().resolve() if sandbox else Path.cwd().resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    registry = registry or ToolRegistry(ToolContext(sandbox=sandbox, auto_approve=auto_approve))
    conversation = conversation or Conversation(counter=TokenCounter(client))
    app = ChatApp(client, settings, style, registry, conversation,
                  session_path=Path(session_path).expanduser() if session_path else None,
                  use_ocr=use_ocr, image_max_side=image_max_side,
                  input_fn=input_fn, out=out, auto_approve=auto_approve, quiet=quiet,
                  tree=tree, budget=budget, config=config, endpoints=endpoints, mcp=mcp)
    registry.ctx.approve = app.approve
    registry.ctx.notify = app.renderer.notice
    return app

def _apply_config(args, explicit, style, note):
    """Fold config file, persona and environment into `args`. CLI flags still win.

    Returns the `Config` actually used. An explicitly-typed flag beats the file; a flag
    that merely has a default does not, which is why `explicit_args()` re-parses with
    argparse.SUPPRESS defaults.
    """
    from .config import Config, write_default_config
    # --save-config creates the file, so "it is not there yet" is not worth reporting.
    config = Config.load(args.config, on_error=None if args.save_config else note)
    if args.save_config:
        target = Path(args.config or 'bonsai.json').expanduser()
        write_default_config(target, Settings())
        note('wrote ' + str(target))
        return None
    if args.list_personas:
        personas = config.personas()
        note('personas: ' + (', '.join(sorted(personas)) or
                             '(none defined; see --save-config)'))
        if config.path:
            note('config file: ' + str(config.path))
        return None
    persona = explicit.get('persona') or os.environ.get('BONSAI_PERSONA') or \
        config.data.get('persona')
    if persona:
        config.select_persona(persona)
    # Map the CLI's flag names onto the config's setting names before resolving, so
    # `--max-tokens 512` means max_tokens=512 and not an unrelated `max_tokens` default.
    alias = {'max_tokens': 'max_tokens', 'no_stream': None, 'no_tools': None,
             'no_markdown': None, 'no_highlight': None}
    cli = {}
    for key, value in explicit.items():
        if key in alias and alias[key] is None:
            continue
        cli[key] = value
    if explicit.get('no_stream'):
        cli['stream'] = False
    if explicit.get('no_tools'):
        cli['use_tools'] = False
    if explicit.get('no_markdown'):
        cli['markdown'] = False
    if explicit.get('no_highlight'):
        cli['highlight'] = False
    if config.path:
        note(f'config: {config.path}'
             + (f' (persona {config.persona_name})' if config.persona_name else ''))
    defaults = {name: getattr(Settings(), name) for name in (
        'model', 'temperature', 'top_p', 'max_tokens', 'stream', 'use_tools',
        'markdown', 'highlight', 'max_tool_rounds', 'context_budget',
        'context_reserve', 'output_reserve', 'tool_timeout', 'tool_budget',
        'tool_max_output', 'autosave_every', 'show_stats', 'preexecute_tools')}
    resolved = config.resolve(defaults, cli=cli, environ=os.environ)
    for key, value in resolved.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return config


def _run_batch(args, client, settings, registry=None):
    """Non-interactive mode: one request per line, the same client and retry policy."""
    from .api import read_batch, run_batch, write_batch
    from .conversation import Conversation
    from .tokens import TokenCounter
    from .tools import ToolContext, ToolRegistry
    try:
        items = read_batch(args.batch)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f'error: cannot read --batch {args.batch}: {exc}\n')
        return 2
    from .budget import Budget
    budget = Budget(session_tokens=args.budget_tokens, turns=args.budget_turns,
                    prompt_tokens=args.budget_prompt_tokens,
                    price_per_mtok=args.price_per_mtok)
    registry = registry or (None if args.no_tools else ToolRegistry(ToolContext(
        sandbox=Path(args.sandbox).expanduser().resolve() if args.sandbox
        else Path.cwd().resolve(), auto_approve=args.auto_approve)))
    counter = TokenCounter(client)

    class _Shim:
        """run_batch wants a Bonsai-ish owner; give it exactly what it needs."""
        def __init__(self):
            self.client = client
            self.settings = settings
            self.registry = registry
            self.counter = counter
            self.budget = budget

        def close(self):
            pass

    results = run_batch(items, bot=_Shim(), client=client, settings=settings)
    payload = results if args.json else results
    if args.out:
        write_batch(results, args.out)
        failed = sum(1 for r in results if not r.get('ok'))
        sys.stderr.write(f'{len(results)} item(s) -> {args.out} '
                         f'({failed} failed)\n')
        return 1 if failed else 0
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.json else None,
                     default=str) if args.json else chr(10).join(
                         json.dumps(r, ensure_ascii=False, default=str) for r in results))
    failed = sum(1 for r in results if not r.get('ok'))
    return 1 if failed else 0


def _export(args, app):
    if not args.export:
        return None
    try:
        from .export import export
        facts = {'endpoint': app.client.base_url, 'model': app.client.model,
                 'capabilities': app.client.caps.to_dict(), 'usage': app.totals}
        if app.budget:
            facts['budget'] = app.budget.status()
        return export(app.conversation, args.export, extra=facts)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f'error: export failed: {exc}\n')
        return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    style = Style(force_color=False if args.plain else (True if args.color else None),
                  width=args.width)
    notes = []

    def note(text):
        notes.append(str(text))

    if args.selftest:
        return run_selftest(style=style)
    if args.serve:
        from .serve import serve
        from .config import SETTING_KEYS, explicit_args
        explicit = explicit_args(parser, argv or ())
        cli_layer = {k: v for k, v in explicit.items() if k in SETTING_KEYS}
        try:
            httpd = serve(args.serve_host, args.serve_port, args.base_url, args.api_key,
                          args.model, config_path=args.config, cli_layer=cli_layer,
                          open_browser=not args.no_browser)
        except RuntimeError as exc:
            sys.stderr.write(f'error: {exc}\n')
            return 1
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return 0

    from .config import explicit_args
    explicit = explicit_args(parser, argv or ())
    config = _apply_config(args, explicit, style, note)
    for text in notes:
        sys.stderr.write(text + '\n')
    if config is None:                       # --save-config / --list-personas handled it
        return 0

    from .budget import Budget
    from .endpoints import EndpointRegistry
    endpoints = EndpointRegistry.from_config(
        config.endpoints(), default_url=args.base_url, default_key=args.api_key,
        default_model=args.model)
    if args.list_endpoints:
        print('\n'.join(endpoints.describe()))
        return 0
    if args.endpoint and args.endpoint in endpoints.names():
        endpoints.select(args.endpoint)
    current = endpoints.current
    args.base_url = current.base_url or args.base_url
    if current.api_key:
        args.api_key = current.api_key
    if current.model and 'model' not in explicit:
        args.model = current.model

    mock_server = None
    if args.mock:
        from mock_bonsai_server import DEFAULT_KEY as MOCK_KEY, MockBonsaiServer
        mock_server = MockBonsaiServer().start()
        args.base_url = args.base_url or mock_server.base_url
        args.api_key = args.api_key or MOCK_KEY
        sys.stderr.write(f'mock endpoint {args.base_url} — scripted replies, '
                         'this is a protocol stub and not the model\n')

    if not args.base_url:
        sys.stderr.write(
            'error: no endpoint. Export BONSAI_BASE_URL (the deployment cell prints it as '
            '"Base URL: https://<host>/v1") or pass --base-url. '
            'Try --selftest to exercise the client offline.\n')
        return 2

    explicit_system = args.system is not None or bool(args.system_file)
    system = None
    if args.system is not None:
        system = args.system
    elif args.system_file:
        system = Path(args.system_file).expanduser().read_text(encoding='utf-8')

    try:
        reasoning = ReasoningConfig(level=args.think,
                                    effort=None if str(args.effort).lower() in ('none', 'off')
                                    else args.effort,
                                    display=args.reasoning_display)
    except ValueError as e:
        sys.stderr.write(f'error: {e}\n')
        return 2
    if str(args.effort).lower() in ('none', 'off'):
        reasoning.level = 'off'
        reasoning.effort = None
    settings = Settings(model=args.model, temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, reasoning=reasoning,
                        stream=not args.no_stream, use_tools=not args.no_tools,
                        markdown=not args.no_markdown, highlight=not args.no_highlight,
                        max_tool_rounds=args.max_tool_rounds,
                        tool_timeout=args.tool_timeout, tool_budget=args.tool_budget,
                        tool_max_output=args.tool_max_output,
                        autosave_every=args.autosave_every,
                        context_reserve=args.context_reserve,
                        output_reserve=args.output_reserve,
                        show_stats=args.show_stats,
                        preexecute_tools=not args.no_pretools)

    client = BonsaiClient(args.base_url, args.api_key, model=args.model,
                          timeout=args.timeout, retries=args.retries,
                          keepalive=not args.no_keepalive)
    if args.doctor:
        try:
            return run_doctor(client, style=style, as_json=args.json,
                              serve_url=os.environ.get('BONSAI_SERVE_URL'))
        finally:
            client.close()
    if args.benchmark:
        def progress(label, i, total):
            sys.stderr.write(f'  {label} ({i}/{total})\n')
        report = run_benchmark(client, samples=max(1, args.bench_samples),
                               warmup=max(0, args.bench_warmup), style=style,
                               progress=progress)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print(render_benchmark(report, style))
        client.close()
        return 1 if report.get('errors') and not report['warm'].get('decode', {}).get('n') else 0
    budget = Budget(session_tokens=args.budget_tokens, turns=args.budget_turns,
                    prompt_tokens=args.budget_prompt_tokens,
                    price_per_mtok=args.price_per_mtok)
    settings.budget = budget
    if args.json_schema:
        try:
            settings.json_schema = json.loads(
                Path(args.json_schema).expanduser().read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            sys.stderr.write(f'error: cannot read --json-schema {args.json_schema}: {e}\n')
            client.close()
            return 2

    if args.context:
        client.context_window = (lambda override=args.context: override)

    if args.batch:
        return _run_batch(args, client, settings, registry=None)

    counter = TokenCounter(client)
    resumed = bool(args.session) and Path(args.session).expanduser().is_file()
    if resumed:
        conversation = Conversation.load(
            args.session, counter=counter,
            on_error=lambda n: sys.stderr.write(
                f'warning: skipped {n} unreadable line(s) in {args.session}; the rest of the '
                'session was loaded\n'))
        if conversation.skipped_lines:
            sys.stderr.write(f'note: session file came from '
                             f'{conversation.saved_version or "an older version"} '
                             f'(schema {conversation.schema})\n')
        if explicit_system:
            conversation.set_system(system or '')
    else:
        conversation = Conversation(system=system if explicit_system else DEFAULT_SYSTEM,
                                    counter=counter)

    registry = ToolRegistry(ToolContext(
        sandbox=Path(args.sandbox).expanduser().resolve() if args.sandbox else Path.cwd().resolve(),
        auto_approve=args.auto_approve))
    if args.tools:
        wanted = {t.strip() for t in args.tools.split(',') if t.strip()}
        for name in registry.names():
            registry.set_enabled(name, name in wanted)
        unknown = wanted - set(registry.names())
        if unknown:
            sys.stderr.write(f'warning: unknown tool(s): {", ".join(sorted(unknown))}\n')

    from .branches import BranchTree
    tree = None
    if args.session:
        tree = BranchTree.from_conversation(conversation,
                                            saved=conversation.saved_branches)
        if args.list_branches:
            print(chr(10).join(tree.describe()))
            return 0
        if args.fork:
            tree.fork(args.fork)
            sys.stderr.write(f"forked branch '{args.fork}' from "
                             f"'{tree.branches[args.fork].parent}'\n")
            # Persist the fork immediately: a branch that only exists in memory is a
            # branch the next run cannot find.
            conversation.save(args.session, branches=tree.to_saved())
        elif args.branch:
            tree.switch(args.branch)
            sys.stderr.write(f"on branch '{args.branch}'\n")
        conversation = tree.conversation

    mcp_servers = []
    servers = config.mcp_servers()
    if servers and not args.no_tools:
        from .mcp import register_mcp
        mcp_servers = register_mcp(
            registry, servers,
            on_note=lambda m: sys.stderr.write(f'MCP: {m}\n'),
            max_chars=max(1000, args.tool_max_output))

    app = make_app(client, settings, style, registry=registry, conversation=conversation,
                   session_path=args.session, use_ocr=args.ocr,
                   image_max_side=args.image_max_side, auto_approve=args.auto_approve,
                   quiet=args.quiet, tree=tree, budget=budget, config=config,
                   endpoints=endpoints, mcp=mcp_servers)
    if args.quiet:
        app.banner = lambda: None

    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            prompt = piped

    try:
        attachments = load_attachments(args.image, max_side=args.image_max_side,
                                       use_ocr=args.ocr) if args.image else []
    except BonsaiError as e:
        sys.stderr.write(f'error: {e}\n')
        return 2
    if attachments:
        app.show_cards(attachments)

    if args.json:
        # Machine-readable mode: stdout carries exactly one JSON document. Everything
        # human-readable — the streamed answer, the stats line, notices — goes to a sink.
        sink = io.StringIO()
        app.out = sink
        app.renderer = LiveRenderer(style, markdown=False, out=sink)
        app.agent = Agent(client, settings, registry=registry, counter=counter,
                          on_delta=lambda t: None, on_reasoning=lambda t: None,
                          on_tool=lambda *a: None, on_notice=lambda t: None)

    try:
        if prompt is not None:
            reason = budget.over() if budget else None
            if reason:
                sys.stderr.write(f'error: {reason}\n')
                return 1
            result = app.send(prompt, attachments=attachments)
            if result is None:
                return 1
            if budget:
                budget.note(result.usage)
            if args.json:
                print(json.dumps({'model': client.model, 'version': VERSION,
                                  'text': result.text, 'reasoning': result.reasoning,
                                  'tool_calls': result.tool_calls,
                                  'finish_reason': result.finish_reason,
                                  'usage': result.usage, 'timings': result.timings,
                                  'stats': result.stats.to_dict(),
                                  'reasoning_config': settings.reasoning.to_dict(),
                                  'capabilities': client.caps.to_dict(),
                                  'elapsed': result.elapsed, 'ttft': result.ttft,
                                  'rounds': result.rounds,
                                  'cancelled': result.cancelled,
                                  'interrupted': result.interrupted or None},
                                 ensure_ascii=False, indent=2))
            _export(args, app)
            return 0
        code = app.run()
        _export(args, app)
        return code
    except KeyboardInterrupt:
        sys.stderr.write('\ninterrupted\n')
        return 130
    except BonsaiError as e:
        sys.stderr.write(f'error: {e}\n')
        hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
        if hint:
            sys.stderr.write(f'hint: {hint}\n')
        return 1
    finally:
        from .mcp import close_all
        close_all(mcp_servers)
        if 'app' in dir() and getattr(app, 'executor', None) is not None:
            app.executor.shutdown(wait=False)
        if mock_server is not None:
            mock_server.stop()

if __name__ == '__main__':
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
