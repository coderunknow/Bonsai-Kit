"""The interactive chat application and its slash commands."""

from __future__ import annotations

import json
import shutil
import statistics
import sys

from ._meta import VERSION
from ._optional import optional_features, pytesseract
from .agent import Agent
from .benchmark import render_benchmark, run_benchmark
from .conversation import Conversation, mask_key
from .tokens import TokenCounter
from .doctor import run_doctor
from .errors import BonsaiAPIError, BonsaiError
from .images import load_attachments
from .markdown import MarkdownWriter
from .reasoning import REASONING_DISPLAYS, THINK_BUDGETS
from .render import LiveRenderer, format_stats, format_stats_block

HELP_TEXT = """\
Commands:

Conversation
  /help                     this help
  /system <text>            replace the system prompt (empty text clears it)
  /reset                    start a fresh conversation
  /undo                     drop the last exchange
  /retry                    regenerate the previous answer
  /history                  show the conversation so far
  /cancel                   clear the queued turn (Ctrl-C interrupts a running generation)

Thinking
  /think [off|low|medium|high|max|<tokens>]
                            thinking budget: 0 / 512 / 2048 / 8192 / unlimited
  /effort [medium|xhigh]    reasoning_effort sent to the chat template
  /reasoning full|compact|hidden
                            how the thinking trace is displayed

Sampling and output
  /temp <0..2>  /topp <0..1>  /max-tokens <n>
  /stream on|off            toggle streaming
  /markdown on|off          toggle terminal Markdown rendering
  /speed [on|off]           print the full timing table after every turn
  /stats                    session totals + last-turn measurements

Context and tools
  /context                  window, history tokens, budget, reserves, tokenizer
  /compact [keep_turns]     summarise old turns instead of dropping them
  /tools                    list tools; /tools <name> on|off toggles one; /tools off all

Attachments and diagnostics
  /image <path> [...]       attach image(s) to the next message
  /ocr on|off               run tesseract OCR on attached images (needs pytesseract)
  /vision on|off|probe      override or re-probe whether the server can see images
  /caps                     what this server actually supports (supported/unsupported/unknown)
  /doctor [--json]          live endpoint diagnostics
  /bench [samples]          measure TTFT, prefill and decode against this endpoint
  /model [id]               show or change the model id
  /usage                    cumulative tokens and timings for this session
  /save [file]  /load <file>  /export [file.json|.html|.md]

Configuration, endpoints, branching, budgets
  /config                   which config file is in use and where each value came from
  /persona [name]           apply a persona preset from the config file
  /endpoints [name]         list endpoints, or switch to one (capabilities re-probed)
  /spend                    tokens, turns and cost against the budget you set
  /schema <file.json>|off   ask for structured output (only sent if the server proves it)
  /mcp                      MCP servers connected and the tools they advertise
  /branches                 list conversation branches
  /branch <name>            switch branches
  /fork <name>              fork a new branch at the current point
  /branch del <name>  /branch rename <old> <new>
  /pretools on|off          start a tool the moment its arguments are complete

  /quit                     leave (Ctrl-D also works; Ctrl-C cancels just this turn)

Anything else is sent to the model. End a line with a backslash to keep typing.

Precedence, lowest to highest: built-in defaults -> config file -> persona ->
environment (BONSAI_*) -> the flags you typed. /config shows you the winner.
"""

class ChatApp:
    """The interactive loop: prompt -> slash command or model turn -> rendered answer."""

    def __init__(self, client, settings, style, registry, conversation, renderer=None,
                 session_path=None, use_ocr=False, image_max_side=1024,
                 input_fn=None, out=None, auto_approve=False, quiet=False,
                 tree=None, budget=None, config=None, endpoints=None, mcp=()):
        self.client = client
        self.settings = settings
        self.style = style
        self.registry = registry
        self.conversation = conversation
        self.out = out if out is not None else sys.stdout
        self.renderer = renderer or LiveRenderer(style, markdown=settings.markdown,
                                                 highlight=settings.highlight, out=self.out)
        self.session_path = session_path
        self.use_ocr = use_ocr
        self.image_max_side = image_max_side
        self.input_fn = input_fn or input
        self.auto_approve = auto_approve
        self.quiet = quiet
        self.pending_images = []
        self.totals = {'turns': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                       'reasoning_tokens': 0, 'tool_calls': 0, 'seconds': 0.0,
                       'ttfts': [], 'rates': [], 'compactions': 0, 'cancels': 0}
        self.last_stats = None
        self._autosave_counter = 0
        self.renderer.reasoning_display = settings.reasoning.display
        self._cancelled = False
        self.tree = tree                 # BranchTree over `conversation`, or None
        self.budget = budget             # Budget, or None
        self.config = config             # Config, for /config provenance
        self.endpoints = endpoints       # EndpointRegistry, for /endpoints
        self.mcp = list(mcp or ())       # registered MCP servers, for /mcp and shutdown
        self.executor = None
        if settings.preexecute_tools and registry is not None:
            from concurrent.futures import ThreadPoolExecutor
            # One worker: tools pre-executed mid-stream must not out-run the turn, and
            # a second worker would only queue work behind the first.
            self.executor = ThreadPoolExecutor(max_workers=2,
                                               thread_name_prefix='bonsai-tool')
        self.agent = Agent(client, settings, registry=registry,
                           counter=conversation.counter, executor=self.executor,
                           on_delta=self.renderer.delta,
                           on_reasoning=self.renderer.reasoning,
                           on_tool=self.renderer.tool,
                           on_notice=self.renderer.notice,
                           on_tool_start=lambda name: self.renderer.notice(
                               f'running {name} (started while the model was still '
                               'writing)'))

    # ------------------------------------------------------------------
    def say(self, text=''):
        self.out.write(text + '\n')
        self.out.flush()

    def banner(self):
        s = self.style
        feats = optional_features()
        extras = ', '.join(k for k, v in feats.items() if v) or 'none (standard library only)'
        self.say(s.bold(s.cyan('Ternary Bonsai 2 27B — chat client v' + VERSION)))
        self.say(s.dim(f'endpoint {self.client.base_url}  key {mask_key(self.client.api_key)}'))
        models = self.client.model_ids()
        self.say(s.dim(f'model {self.client.model}  '
                       f'(served: {", ".join(models) if models else "unknown"})'))
        ctx = self.client.context_window()
        self.say(s.dim(f'context {ctx} tokens, budget {self.context_budget_tokens()} | '
                       f'tools {"on: " + ", ".join(self.registry.names()) if self.settings.use_tools else "off"}'))
        self.say(s.dim(f'thinking: {self.settings.reasoning.describe()} | '
                       f'streaming {"on" if self.settings.stream else "off"} | '
                       f'tokenizer {"server" if self.conversation.counter.using_server() else "estimate"}'))
        self.say(s.dim(f'optional extras: {extras} | /help for commands'))
        self.say()

    # ------------------------------------------------------------------
    def context_budget_tokens(self):
        """Tokens of history we are allowed to send.

        The server's context window has to hold the prompt *and* the answer, plus whatever
        the chat template adds. Sending history right up to the window guarantees a 400 on
        the turn that matters, so both reserves come off the top.
        """
        window = self.client.context_window()
        soft = int(window * self.settings.context_budget)
        reserve = self.settings.context_reserve + min(self.settings.max_tokens,
                                                      self.settings.output_reserve)
        return max(1024, min(soft, window - reserve))

    # ------------------------------------------------------------------
    def approve(self, name, args, risk):
        """Interactive gate for tools that write or execute."""
        if self.auto_approve:
            return True
        self.say()
        self.say(self.style.yellow(f'⚠ the model wants to run {name} [{risk}]'))
        self.say(self.style.dim('  args: ' + json.dumps(args, ensure_ascii=False)[:600]))
        while True:
            try:
                answer = self.input_fn(self.style.bold('  allow? [y]es / [n]o / [a]lways: ')).strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.say()
                return False
            if answer in ('y', 'yes'):
                return True
            if answer in ('n', 'no', ''):
                return False
            if answer in ('a', 'always'):
                self.auto_approve = True
                return True
            self.say(self.style.dim('  please answer y, n or a'))

    # ------------------------------------------------------------------
    def show_cards(self, attachments):
        """Print what the model will be told about each attached file."""
        for a in attachments:
            self.say(self.style.dim(a.text_card(self.client.probe_vision())))
        if attachments:
            self.say()

    def send(self, text, attachments=None, regenerate=False):
        attachments = attachments or []
        budget = self.context_budget_tokens()
        dropped = self.conversation.trim(budget)
        if dropped:
            self.renderer.notice(f'trimmed {dropped} oldest message(s) to fit the '
                                 f'{budget}-token context budget (/compact keeps a summary '
                                 'instead)')
        problems = self.conversation.validate_wire()
        if problems:
            self.renderer.notice('history repaired before sending: ' + '; '.join(problems[:3]))
            self.conversation._repair_orphans()
        self._cancelled = False
        try:
            result = self.agent.run(self.conversation, text=text, attachments=attachments,
                                    regenerate=regenerate)
        except BonsaiError as e:
            self.renderer.finish()
            self.renderer.error(str(e))
            hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
            if hint:
                self.say(self.style.dim('  hint: ' + hint))
            return None
        self.renderer.finish()
        result.stats.context_used = self.conversation.tokens()
        result.stats.context_window = self.client.context_window()
        self.last_stats = result.stats
        self.totals['turns'] += 1
        self.totals['prompt_tokens'] += result.prompt_tokens
        self.totals['completion_tokens'] += result.completion_tokens
        self.totals['reasoning_tokens'] += result.stats.reasoning_tokens or 0
        self.totals['tool_calls'] += len(result.tool_calls)
        self.totals['seconds'] += result.elapsed or 0.0
        if result.cancelled:
            self.totals['cancels'] += 1
        if result.stats.ttft:
            self.totals['ttfts'].append(result.stats.ttft)
        if result.stats.tokens_per_s:
            self.totals['rates'].append(result.stats.tokens_per_s)
        if self.settings.show_stats:
            self.say(format_stats_block(result.stats, self.style))
        else:
            self.say(self.style.dim(format_stats(result.stats, self.style)))
        self.say()
        self._autosave_counter += 1
        if self.session_path and self._autosave_counter >= max(1, self.settings.autosave_every):
            self._autosave_counter = 0
            try:
                self.conversation.save(self.session_path,
                                       branches=self.tree.to_saved() if self.tree else None)
            except OSError as e:
                self.say(self.style.dim(f'(could not autosave session: {e})'))
        return result

    # ------------------------------------------------------------------
    def command(self, line):
        """Handle one slash command. Returns False to exit the loop."""
        parts = line[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ''
        arg = parts[1].strip() if len(parts) > 1 else ''
        s = self.style
        try:
            if cmd in ('q', 'quit', 'exit'):
                return False
            if cmd == 'help':
                self.say(HELP_TEXT)
            elif cmd == 'system':
                self.conversation.set_system(arg)
                self.say(s.dim('system prompt updated' if arg else 'system prompt cleared'))
            elif cmd == 'reset':
                system = self.conversation.system_text
                self.conversation = Conversation(system=system, counter=self.conversation.counter)
                self.agent = Agent(self.client, self.settings, registry=self.registry,
                                   counter=self.conversation.counter,
                                   executor=self.executor,
                                   on_delta=self.renderer.delta, on_reasoning=self.renderer.reasoning,
                                   on_tool=self.renderer.tool, on_notice=self.renderer.notice,
                                   on_tool_start=self.agent.on_tool_start)
                if self.tree is not None:
                    self.tree.sync()
                self.say(s.dim('conversation cleared'))
            elif cmd == 'undo':
                n = self.conversation.drop_last_turn()
                self.say(s.dim(f'removed {n} message(s)' if n else 'nothing to undo'))
            elif cmd == 'retry':
                i = self.conversation.last_user_index()
                if i is None:
                    self.say(s.dim('nothing to retry yet'))
                else:
                    text = self.conversation.messages[i].get('content')
                    if isinstance(text, list):
                        text = ' '.join(p.get('text', '') for p in text if isinstance(p, dict))
                    self.conversation.drop_last_turn()
                    self.say(s.dim('regenerating …'))
                    self.send(text or '')
            elif cmd == 'image':
                if not arg:
                    self.say(s.dim('queued: ' + ', '.join(a.display_name for a in self.pending_images)
                                   if self.pending_images else 'usage: /image <path> [more paths]'))
                else:
                    loaded = load_attachments(arg.split(), max_side=self.image_max_side,
                                              use_ocr=self.use_ocr)
                    self.pending_images.extend(loaded)
                    for a in loaded:
                        self.say(s.dim(a.text_card(self.client.probe_vision())))
            elif cmd == 'ocr':
                self.use_ocr = arg.lower() in ('on', '1', 'true', 'yes')
                self.say(s.dim('OCR ' + ('on' if self.use_ocr else 'off') +
                               ('' if (pytesseract and shutil.which('tesseract')) or not self.use_ocr
                                else ' — but pytesseract/tesseract is not installed')))
            elif cmd == 'tools':
                self._tools_command(arg)
            elif cmd == 'model':
                if arg:
                    self.client.model = arg
                    self.settings.model = arg
                    self.say(s.dim('model set to ' + arg))
                else:
                    self.say(s.dim(f'model {self.client.model}; served: '
                                   f'{", ".join(self.client.model_ids()) or "unknown"}'))
            elif cmd == 'temp':
                self.settings.temperature = float(arg)
                self.say(s.dim(f'temperature {self.settings.temperature}'))
            elif cmd == 'topp':
                self.settings.top_p = float(arg)
                self.say(s.dim(f'top_p {self.settings.top_p}'))
            elif cmd in ('max-tokens', 'maxtokens'):
                self.settings.max_tokens = int(arg)
                self.say(s.dim(f'max_tokens {self.settings.max_tokens}'))
            elif cmd == 'effort':
                if not arg:
                    self.say(s.dim('reasoning_effort: ' + str(self.settings.reasoning.effort)
                                   + ' — Bonsai 2 accepts medium|xhigh (low is a documented '
                                     'no-op and is not offered)'))
                else:
                    self.settings.effort = arg
                    self.say(s.dim(f'reasoning_effort {self.settings.reasoning.effort} '
                                   f'({self.settings.reasoning.describe()})'))
            elif cmd in ('think', 'thinking', 'budget'):
                self._think_command(arg)
            elif cmd == 'reasoning':
                self._reasoning_display_command(arg)
            elif cmd == 'stats':
                self._stats_command()
            elif cmd == 'speed':
                self.settings.show_stats = not self.settings.show_stats if arg == '' else \
                    arg.lower() in ('on', '1', 'true', 'yes', 'full')
                self.say(s.dim('per-turn timing table ' +
                               ('on' if self.settings.show_stats else 'off') +
                               ' (one-line summary otherwise)'))
            elif cmd == 'compact':
                self._compact_command(arg)
            elif cmd == 'cancel':
                self.pending_images = []
                self.renderer.finish()
                self.say(s.dim('cleared the queued turn. Ctrl-C interrupts a generation that '
                               'is already running; the partial answer is discarded and the '
                               'history is left untouched.'))
            elif cmd == 'doctor':
                run_doctor(self.client, style=self.style, out=self.out,
                           as_json=arg.strip() == '--json')
            elif cmd in ('caps', 'capabilities'):
                self._caps_command()
            elif cmd == 'bench':
                self._bench_command(arg)
            elif cmd == 'stream':
                self.settings.stream = arg.lower() in ('on', '1', 'true', 'yes')
                self.say(s.dim('streaming ' + ('on' if self.settings.stream else 'off')))
            elif cmd == 'markdown':
                self.settings.markdown = arg.lower() in ('on', '1', 'true', 'yes')
                self.renderer.markdown = self.settings.markdown
                self.renderer.writer = (MarkdownWriter(self.style, highlight=self.settings.highlight)
                                        if self.settings.markdown else None)
                self.say(s.dim('markdown ' + ('on' if self.settings.markdown else 'off')))
            elif cmd == 'vision':
                if arg.lower() == 'probe':
                    self.client._vision = None
                    self.say(s.dim('vision: ' + str(self.client.probe_vision())))
                elif arg:
                    self.client.probe_vision(force=arg.lower() in ('on', '1', 'true', 'yes'))
                    self.say(s.dim('vision forced ' + str(self.client._vision)))
                else:
                    self.say(s.dim('vision: ' + str(self.client.probe_vision())))
            elif cmd == 'context':
                window = self.client.context_window()
                used = self.conversation.tokens()
                budget = self.context_budget_tokens()
                self.say(s.dim(f'context window {window} | history {used} tokens over '
                               f'{len(self.conversation.messages)} messages '
                               f'({100.0 * used / window:.0f}% of window) | '
                               f'budget {budget} (reserve '
                               f'{self.settings.context_reserve}+'
                               f'{min(self.settings.max_tokens, self.settings.output_reserve)} '
                               f'for template + answer) | tokenizer '
                               f'{"server /tokenize" if self.conversation.counter.using_server() else "estimate"}'))
            elif cmd == 'history':
                for m in self.conversation.messages:
                    body = m.get('content')
                    if isinstance(body, list):
                        body = ' '.join(p.get('text', '') for p in body if isinstance(p, dict))
                    body = (body or '').replace('\n', ' ')
                    self.say(s.dim(f'[{m["role"]}] ') + body[:300])
            elif cmd == 'usage':
                t = self.totals
                self.say(s.dim(f'{t["turns"]} turn(s) | {t["completion_tokens"]} tokens out, '
                               f'{t["prompt_tokens"]} in, {t["reasoning_tokens"]} reasoning | '
                               f'{t["tool_calls"]} tool call(s) | '
                               f'{t["seconds"]:.1f}s of model time | '
                               f'{t["cancels"]} cancelled, {t["compactions"]} compacted'))
            elif cmd == 'save':
                path = self.conversation.save(
                    arg or self.session_path or 'bonsai-session.jsonl',
                    branches=self.tree.to_saved() if self.tree else None)
                self.say(s.dim('saved ' + str(path)))
            elif cmd == 'load':
                if not arg:
                    self.say(s.dim('usage: /load <file.jsonl>'))
                else:
                    self.conversation = Conversation.load(arg, counter=self.conversation.counter)
                    if self.tree is not None:
                        from .branches import BranchTree
                        self.tree = BranchTree.from_conversation(
                            self.conversation, saved=self.conversation.saved_branches)
                    self.agent = Agent(self.client, self.settings, registry=self.registry,
                                       counter=self.conversation.counter,
                                       on_delta=self.renderer.delta, on_reasoning=self.renderer.reasoning,
                                       on_tool=self.renderer.tool, on_notice=self.renderer.notice)
                    self.say(s.dim(f'loaded {len(self.conversation.messages)} messages from {arg}'))
            elif cmd == 'export':
                path = arg or 'bonsai-session.json'
                if path.endswith('.md'):
                    written = self.conversation.export_markdown(path)
                else:
                    from .export import export
                    facts = {'endpoint': self.client.base_url,
                             'model': self.client.model,
                             'capabilities': self.client.caps.to_dict(),
                             'usage': self.totals}
                    if self.budget:
                        facts['budget'] = self.budget.status()
                    written = export(self.conversation, path, extra=facts)
                self.say(s.dim('exported ' + str(written)))
            elif cmd == 'config':
                self._config_command(arg)
            elif cmd == 'persona':
                self._persona_command(arg)
            elif cmd in ('endpoints', 'endpoint'):
                self._endpoint_command(arg)
            elif cmd in ('spend', 'budget-report'):
                self._budget_command()
            elif cmd == 'schema':
                self._schema_command(arg)
            elif cmd == 'mcp':
                self._mcp_command()
            elif cmd == 'pretools':
                self.settings.preexecute_tools = arg.lower() in ('on', '1', 'true', 'yes')
                self.say(s.dim('pre-executing tools mid-stream ' +
                               ('on' if self.settings.preexecute_tools else 'off')))
            elif cmd in ('branches', 'branch'):
                self._branch_command(cmd, arg)
            elif cmd == 'fork':
                self._fork_command(arg)
            else:
                self.say(s.red(f'unknown command /{cmd} — try /help'))
        except (ValueError, BonsaiError, OSError, json.JSONDecodeError) as e:
            self.say(s.red(f'{cmd} failed: {e}'))
        return True

    # ------------------------------------------------------------------
    def _config_command(self, arg):
        """Where every setting came from. Precedence you can read beats precedence you
        have to guess at."""
        s = self.style
        if not self.config:
            self.say(s.dim('no config file in use (looked for bonsai.json, '
                           '.bonsai.json, ~/.config/bonsai/config.json, ~/.bonsai.json)'))
            return
        self.say(s.dim('config file: ' + str(self.config.path or '(none)')))
        rows = self.config.describe()
        self.say(chr(10).join(rows) if rows else s.dim('  (nothing resolved from a file)'))
        if self.config.persona_name:
            self.say(s.dim(f'  persona        {self.config.persona_name}'))
        for note in self.config.notes:
            self.say(s.dim('  note: ' + note))

    def _persona_command(self, arg):
        s = self.style
        if not self.config or not self.config.personas():
            self.say(s.dim('no personas defined (add a "personas" object to the config '
                           'file: name -> {system, temperature, ...})'))
            return
        if not arg:
            self.say(s.dim('personas: ' + ', '.join(sorted(self.config.personas()))))
            return
        preset = self.config.select_persona(arg)
        if preset is None:
            self.say(s.red(f'no such persona: {arg}'))
            return
        from .config import SETTING_KEYS
        applied = []
        for key, value in preset.items():
            if key in SETTING_KEYS and hasattr(self.settings, key):
                setattr(self.settings, key, value)
                applied.append(key)
            elif key == 'system':
                self.conversation.set_system(value)
                applied.append('system')
        self.say(s.dim(f'persona {arg}: applied {", ".join(applied) or "nothing"}'))

    def _endpoint_command(self, arg):
        s = self.style
        if not self.endpoints:
            self.say(s.dim('one endpoint (no "endpoints" block in the config file)'))
            return
        if not arg:
            self.say(chr(10).join(self.endpoints.describe()))
            return
        try:
            endpoint = self.endpoints.select(arg)
        except BonsaiError as e:
            self.say(s.red(str(e)))
            return
        self.client.close()
        self.client = endpoint.client()
        self.client.model = self.settings.model
        self.conversation.counter = TokenCounter(self.client)
        self.agent = Agent(self.client, self.settings, registry=self.registry,
                           counter=self.conversation.counter, executor=self.executor,
                           on_delta=self.renderer.delta,
                           on_reasoning=self.renderer.reasoning,
                           on_tool=self.renderer.tool, on_notice=self.renderer.notice,
                           on_tool_start=self.agent.on_tool_start)
        self.say(s.dim(f'endpoint -> {endpoint.name} ({endpoint.base_url}) — '
                       'capabilities are probed again for this server'))

    def _budget_command(self):
        s = self.style
        if not self.budget:
            self.say(s.dim('no budget set (--budget-tokens N, --budget-turns N, '
                           '--price-per-mtok X)'))
            return
        self.say(chr(10).join(self.budget.describe()))

    def _schema_command(self, arg):
        """Ask for structured output. The client only sends the field if this server
        proved it honours it, and says which one happened."""
        s = self.style
        if not arg:
            state = 'on' if self.settings.json_schema is not None else 'off'
            self.say(s.dim(f'structured output: {state} — /schema <file.json> to set a '
                           'JSON schema, /schema off to clear'))
            return
        if arg.lower() in ('off', 'none', 'clear'):
            self.settings.json_schema = None
            self.say(s.dim('structured output off'))
            return
        try:
            schema = json.loads(Path(arg).expanduser().read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            self.say(s.red(f'cannot read schema {arg}: {e}'))
            return
        self.settings.json_schema = schema
        supported = self.client.probe_structured_output()
        self.say(s.dim('structured output on — ' + (
            'this server honours response_format, so the schema is sent'
            if supported else
            'this server does NOT honour response_format, so the request goes out '
            'without it and the reply is checked locally instead')))

    def _mcp_command(self):
        s = self.style
        if not self.mcp:
            self.say(s.dim('no MCP servers connected (add a "mcpServers" block to the '
                           'config file — the same shape Claude Desktop uses)'))
            return
        for server, names in self.mcp:
            self.say(s.dim(f'  {server.name}: {len(names)} tool(s) — '
                           f'{", ".join(names) or "(none advertised)"}'))

    def _branch_command(self, cmd, arg):
        s = self.style
        if self.tree is None:
            self.say(s.dim('branching needs a session file (--session FILE)'))
            return
        bits = arg.split()
        if not bits:
            self.say(chr(10).join(self.tree.describe()))
            return
        action, rest = bits[0].lower(), ' '.join(bits[1:])
        try:
            if action == 'del' or action == 'delete':
                self.tree.delete(rest)
                self.say(s.dim(f'deleted branch {rest} -> now on {self.tree.current_name}'))
            elif action == 'rename':
                old, new = (rest.split(None, 1) + [''])[:2]
                self.tree.rename(old, new)
                self.say(s.dim(f'renamed {old} -> {new}'))
            else:
                branch = self.tree.switch(arg)
                self.say(s.dim(f'on branch {branch.name} '
                               f'({len(branch.messages)} messages)'))
        except ValueError as e:
            self.say(s.red(str(e)))
        self._after_branch_change()

    def _fork_command(self, arg):
        s = self.style
        if self.tree is None:
            self.say(s.dim('branching needs a session file (--session FILE)'))
            return
        if not arg:
            self.say(s.dim('usage: /fork <name>  (forks from the current point)'))
            return
        try:
            branch = self.tree.fork(arg)
        except ValueError as e:
            self.say(s.red(str(e)))
            return
        self.say(s.dim(f'forked {branch.name} from {branch.parent} at message '
                       f'{branch.fork_at} — you are now on it'))
        self._after_branch_change()

    def _after_branch_change(self):
        """The conversation object changed identity underneath the agent and renderer."""
        self.conversation = self.tree.conversation
        self.agent = Agent(self.client, self.settings, registry=self.registry,
                           counter=self.conversation.counter, executor=self.executor,
                           on_delta=self.renderer.delta,
                           on_reasoning=self.renderer.reasoning,
                           on_tool=self.renderer.tool, on_notice=self.renderer.notice,
                           on_tool_start=self.agent.on_tool_start)

    # ------------------------------------------------------------------
    def _think_command(self, arg):
        """Change the thinking budget without touching history or restarting anything."""
        s = self.style
        if not arg:
            budgets = '  '.join(f'{k}={v if v != -1 else "unlimited"}'
                                for k, v in THINK_BUDGETS.items())
            self.say(s.dim(f'thinking: {self.settings.reasoning.label()} | levels: {budgets}'))
            self.say(s.dim('usage: /think off|low|medium|high|max  or  /think <tokens>'))
            return
        try:
            self.settings.think = arg
        except ValueError as e:
            self.say(s.red(str(e)))
            return
        supported = self.client.caps.supported('thinking_budget_tokens')
        note = ''
        if supported is False:
            note = ' — but this build rejects thinking_budget_tokens, so thinking stays server-controlled'
        elif supported is None:
            note = ' — will be confirmed on the next request'
        self.say(s.dim(f'thinking {self.settings.reasoning.label()}{note}'))

    def _reasoning_display_command(self, arg):
        s = self.style
        if arg.lower() not in REASONING_DISPLAYS:
            self.say(s.dim('usage: /reasoning full|compact|hidden  (current: '
                           + self.settings.reasoning.display + ')'))
            return
        self.settings.reasoning.display = arg.lower()
        self.renderer.reasoning_display = arg.lower()
        self.say(s.dim('reasoning display: ' + arg.lower() +
                       (' — the trace is streamed' if arg == 'full' else
                        ' — one indicator + a token count' if arg == 'compact' else
                        ' — never shown, only counted')))

    def _stats_command(self):
        s = self.style
        t = self.totals
        self.say(s.bold('session'))
        self.say(s.dim(f'  {t["turns"]} turn(s), {t["seconds"]:.1f}s of model time, '
                       f'{t["cancels"]} cancelled'))
        self.say(s.dim(f'  {t["completion_tokens"]} completion tokens, {t["prompt_tokens"]} '
                       f'prompt tokens, {t["reasoning_tokens"]} reasoning tokens, '
                       f'{t["tool_calls"]} tool call(s), {t["compactions"]} compaction(s)'))
        if t['ttfts']:
            self.say(s.dim(f'  ttft      median {statistics.median(t["ttfts"]) * 1000:.0f} ms | '
                           f'min {min(t["ttfts"]) * 1000:.0f} | max {max(t["ttfts"]) * 1000:.0f} '
                           f'({len(t["ttfts"])} sample(s))'))
        if t['rates']:
            self.say(s.dim(f'  decode    median {statistics.median(t["rates"]):.1f} tok/s | '
                           f'min {min(t["rates"]):.1f} | max {max(t["rates"]):.1f}'))
        tr = self.client.transport.stats()
        self.say(s.dim(f'  transport {tr["requests"]} requests over '
                       f'{tr["connections_opened"]} connection(s), '
                       f'{tr["reconnects"]} reconnect(s), pooled={tr["pooled"]}'))
        if self.last_stats:
            self.say(s.bold('last turn'))
            self.say(format_stats_block(self.last_stats, s))

    def _caps_command(self):
        s = self.style
        caps = self.client.caps
        self.say(s.bold('server capabilities (learned from this endpoint, not assumed)'))
        for name in sorted(caps.fields):
            state = caps.state(name)
            colour = {'supported': s.green, 'unsupported': s.red}.get(state, s.yellow)
            detail = caps.evidence.get(name, '')
            self.say(f'  {colour(state.ljust(11))} {name}' + (s.dim('  — ' + detail) if detail else ''))
        for k in ('vision', 'context_window', 'served_models', 'version'):
            if k in caps.facts:
                self.say(s.dim(f'  {k}: {caps.facts[k]}'))

    def _compact_command(self, arg):
        s = self.style
        budget = self.context_budget_tokens()
        keep = 2
        if arg.isdigit():
            keep = max(0, int(arg))
        before = len(self.conversation.messages)
        report = self.conversation.compact(budget, keep_recent=keep)
        self.totals['compactions'] += 1
        self.say(s.dim(f'compacted {report["compacted"]} message(s) into a summary, kept the '
                       f'last {report["kept"]} turn(s): {report["tokens_before"]} -> '
                       f'{report["tokens_after"]} tokens '
                       f'({len(self.conversation.messages)} messages, was {before})'))
        if report.get('trimmed'):
            self.say(s.dim(f'still over budget afterwards — also trimmed '
                           f'{report["trimmed"]} message(s)'))

    def _bench_command(self, arg):
        """Measure this endpoint for real: TTFT, prefill and decode over N samples."""
        samples = 3
        bits = arg.split()
        if bits and bits[0].isdigit():
            samples = max(1, min(10, int(bits[0])))
        self.say(self.style.dim(f'benchmarking {samples} sample(s) against '
                                f'{self.client.base_url} (the prompt is fixed on purpose, '
                                'so the second and later samples measure prompt-cache reuse)'))
        report = run_benchmark(self.client, samples=samples, style=self.style, out=self.out)
        self.say(render_benchmark(report, self.style))

    def _tools_command(self, arg):
        s = self.style
        if not arg:
            for name in self.registry.names():
                tool = self.registry.tools[name]
                state = s.green('on ') if tool.enabled else s.dim('off')
                self.say(f'  {state} {name:<14} {s.dim(tool.risk):<12} '
                         f'{tool.description.splitlines()[0][:70]}')
            return
        bits = arg.split()
        if bits[0].lower() == 'off' and len(bits) == 1:
            self.settings.use_tools = False
            self.say(s.dim('tool calling disabled for this session'))
            return
        if bits[0].lower() == 'on' and len(bits) == 1:
            self.settings.use_tools = True
            self.say(s.dim('tool calling enabled'))
            return
        if len(bits) == 2:
            self.registry.set_enabled(bits[0], bits[1].lower() in ('on', '1', 'true', 'yes'))
            self.say(s.dim(f'{bits[0]} -> {bits[1]}'))
        else:
            self.say(s.dim('usage: /tools | /tools <name> on|off | /tools on | /tools off'))

    # ------------------------------------------------------------------
    def run(self):
        s = self.style
        self.banner()
        if self.pending_images:
            pass
        while True:
            try:
                line = self._read_input()
            except EOFError:
                self.say()
                break
            except KeyboardInterrupt:
                self.say(s.dim('\n(interrupted — /quit to leave)'))
                continue
            if line is None:
                break
            if not line:
                continue
            if line.startswith('/'):
                if not self.command(line):
                    break
                continue
            attachments, self.pending_images = self.pending_images, []
            self.send(line, attachments=attachments)
        return 0

    def _read_input(self):
        prompt = self.style.bold(self.style.cyan('you › '))
        chunks = []
        while True:
            raw = self.input_fn(prompt if not chunks else self.style.dim('… '))
            if raw is None:
                return None
            if raw.endswith('\\'):
                chunks.append(raw[:-1])
                continue
            chunks.append(raw)
            line = '\n'.join(chunks).strip()
            return line
