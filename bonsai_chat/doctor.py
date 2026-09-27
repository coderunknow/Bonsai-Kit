"""Endpoint diagnostics, human- and machine-readable."""

from __future__ import annotations

import json
import sys

from ._meta import VERSION
from .client import BonsaiClient
from .conversation import mask_key
from .errors import BonsaiAPIError, BonsaiError
from .style import Style
from .tokens import TokenCounter

DOCTOR_STATES = ('PASS', 'FAIL', 'SKIP', 'UNKNOWN', 'DEGRADED')

def run_doctor(client, style=None, out=None, as_json=False):
    """Live diagnostics against a real deployment: is this endpoint actually usable?

    With `as_json` the same checks are emitted as one machine-readable object instead of
    the human table, so a deployment can be monitored without scraping text.
    """
    style = style or Style(force_color=False)
    out = out if out is not None else sys.stdout
    results = []
    report = {'endpoint': client.base_url, 'model': client.model,
              'client_version': VERSION, 'checks': {}}

    def check(name, state, detail=''):
        if state not in DOCTOR_STATES:
            raise ValueError(f'unknown doctor state {state!r}')
        results.append(state)
        report['checks'][name] = {'state': state, 'detail': detail}
        if as_json:
            return
        mark = {'PASS': style.green('PASS'), 'FAIL': style.red('FAIL'),
                'SKIP': style.yellow('SKIP'), 'UNKNOWN': style.yellow('UNKNOWN'),
                'DEGRADED': style.yellow('DEGRADED')}[state]
        out.write(f'{mark}  {name}' + (style.dim(' — ' + detail) if detail else '') + '\n')
        out.flush()

    def say(text):
        if not as_json:
            out.write(text + '\n')
            out.flush()

    say(style.bold(f'Diagnosing {client.base_url} '
                   f'(key {mask_key(client.api_key)}, model {client.model}, '
                   f'client v{VERSION})\n'))

    try:
        health = client.get_json('/health', timeout=15, root=True)
        check('/health', 'PASS' if isinstance(health, dict) else 'FAIL', json.dumps(health)[:80])
    except BonsaiError as e:
        check('/health', 'FAIL', str(e))
        say(style.red('\nThe endpoint is not reachable — nothing else can be tested. '
                      'If this was a Colab/Kaggle deployment, the runtime (and its '
                      'Quick Tunnel URL) is gone: rerun colab_kaggle_cell.py and use the '
                      'new base URL and key.'))
        return _doctor_exit(report, results, style, out, as_json, client)

    try:
        ids = client.model_ids()
        check('/v1/models', 'PASS' if client.model in ids else 'FAIL', ', '.join(ids) or 'empty')
        if client.model not in ids:
            out.write(style.yellow(f'  note: --model {client.model} is not what the server '
                                   'reports; llama.cpp serves the loaded model anyway.\n'))
    except BonsaiError as e:
        check('/v1/models', 'FAIL', str(e))

    props = client.props()
    ctx = client.context_window()
    check('/props context window', 'PASS' if props else 'SKIP',
          f'{ctx} tokens' + (f', slots {props.get("total_slots")}' if props else ''))

    counter = TokenCounter(client)
    n = counter.count('doctor probe: tokenise this sentence')
    check('/tokenize', 'PASS' if counter.using_server() else 'SKIP',
          f'{n} tokens for a 36-char string' if counter.using_server()
          else 'unavailable — the client will estimate tokens (len/4) instead')

    try:
        wrong = BonsaiClient(client.base_url, 'deliberately-wrong-key', model=client.model,
                             retries=1)
        wrong.models()
        check('bearer auth enforced', 'FAIL', 'a wrong key was accepted')
    except BonsaiAPIError as e:
        check('bearer auth enforced', 'PASS' if e.status in (401, 403) else 'FAIL',
              f'HTTP {e.status}')
    except BonsaiError as e:
        check('bearer auth enforced', 'SKIP', str(e))

    try:
        reply = client.chat([{'role': 'user', 'content': 'Reply with the single word: ok'}],
                            max_tokens=16)
        text = ((reply.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        check('chat completion', 'PASS' if text.strip() else 'FAIL', repr(text[:40]))
    except BonsaiError as e:
        check('chat completion', 'FAIL', str(e))
        hint = e.hint() if isinstance(e, BonsaiAPIError) else ''
        if hint:
            out.write(style.dim('  hint: ' + hint) + '\n')

    try:
        chunks, got = 0, []
        for ev in client.stream_chat([{'role': 'user', 'content': 'Count from 1 to 5.'}],
                                     max_tokens=64):
            if ev['kind'] in ('delta', 'reasoning'):
                chunks += 1
                got.append(ev['text'])
        check('streaming (SSE)', 'PASS' if chunks >= 3 else 'FAIL', f'{chunks} chunks')
    except BonsaiError as e:
        check('streaming (SSE)', 'FAIL', str(e))

    tools = [{'type': 'function', 'function': {
        'name': 'get_weather', 'description': 'Get the current weather for a city',
        'parameters': {'type': 'object',
                       'properties': {'city': {'type': 'string', 'description': 'City name'}},
                       'required': ['city']}}}]
    try:
        reply = client.chat([{'role': 'user',
                              'content': 'What is the weather in Lisbon right now? '
                                         'Answer only by calling the provided tool.'}],
                            tools=tools, max_tokens=512)
        msg = (reply.get('choices') or [{}])[0].get('message') or {}
        finish = (reply.get('choices') or [{}])[0].get('finish_reason')
        if msg.get('tool_calls') or finish == 'tool_calls':
            names = ', '.join((tc.get('function') or {}).get('name', '?')
                              for tc in msg.get('tool_calls') or [])
            check('native tool calling', 'PASS', names)
        else:
            check('native tool calling', 'FAIL',
                  f'no tool_calls (finish_reason={finish}); the server may lack --jinja')
    except BonsaiAPIError as e:
        check('native tool calling', 'SKIP', f'HTTP {e.status}: {e.message[:80]}')
    except BonsaiError as e:
        check('native tool calling', 'SKIP', str(e))

    if client.supports_tools is False:
        out.write(style.dim('  note: the client already fell back to no-tool mode for this '
                            'endpoint.\n'))
    if client.supports_reasoning_effort is False:
        out.write(style.dim('  note: reasoning_effort is not accepted by this build.\n'))

    # Structured output is a capability you can only learn by asking: a server that
    # accepts `response_format` and then ignores it is worse than one that rejects it,
    # because the caller would get prose where it asked for JSON. One probe, three
    # outcomes, all of them reported.
    try:
        structured = client.probe_structured_output()
    except BonsaiError as e:
        check('structured output (response_format)', 'UNKNOWN', f'probe failed: {e}')
    else:
        check('structured output (response_format)', 'PASS' if structured else 'DEGRADED',
              'this server answers in JSON when asked' if structured else
              str(client.caps.facts.get('structured_output_source')
                  or 'not honoured — the client will not send the field'))

    vision = client.probe_vision()
    check('vision (image input)', 'PASS' if vision else 'SKIP',
          'pixels can be sent' if vision else
          'text-only build — the client sends measured image facts instead')

    # ---- reasoning controls: probe what the runtime actually accepts -------------
    for field_name, value, label in (('thinking_budget_tokens', 2048, 'thinking budget'),
                                     ('reasoning_effort', 'medium', 'reasoning_effort')):
        try:
            ok = client.probe_field(field_name, value)
        except BonsaiError as e:
            check(f'reasoning control: {label}', 'UNKNOWN', f'probe failed: {e}')
            continue
        check(f'reasoning control: {label}', 'PASS' if ok else 'DEGRADED',
              'accepted by this build' if ok else
              'rejected with HTTP 400 — thinking stays server-controlled')

    _report_capabilities(client, style, out, as_json, report, check)
    return _doctor_exit(report, results, style, out, as_json, client)

def _report_capabilities(client, style, out, as_json, report, check=None):
    """Print the capability map: supported / unsupported / unknown, with evidence."""
    caps = client.caps
    report['capabilities'] = caps.to_dict()
    report['transport'] = client.transport.stats()
    if as_json:
        return
    out.write('\n' + style.bold('capability map (learned from this endpoint)\n'))
    for name in sorted(caps.fields):
        state = caps.state(name)
        colour = {'supported': style.green, 'unsupported': style.red}.get(state, style.yellow)
        detail = caps.evidence.get(name, 'not tested against this build')
        out.write(f'  {colour(state.ljust(11))} {name}' + style.dim('  — ' + detail) + '\n')
    for key in ('vision', 'vision_source', 'context_window', 'served_models', 'model_path',
                'total_slots', 'version'):
        if key in caps.facts:
            out.write(style.dim(f'  {key}: {caps.facts[key]}') + '\n')
    tr = client.transport.stats()
    out.write(style.dim(f'  transport: {tr["requests"]} requests, '
                        f'{tr["connections_opened"]} connection(s), '
                        f'{tr["reconnects"]} reconnect(s), keep-alive pooled={tr["pooled"]}'
                        ) + '\n')
    out.flush()

def _doctor_exit(report, results, style, out, as_json, client=None):
    counts = {state: results.count(state) for state in DOCTOR_STATES}
    report['summary'] = counts
    report['usable'] = counts['FAIL'] == 0
    if as_json:
        out.write(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
        out.flush()
    else:
        out.write(f'\n{counts["PASS"]} passed, {counts["FAIL"]} failed, '
                  f'{counts["SKIP"]} skipped, {counts["UNKNOWN"]} unknown, '
                  f'{counts["DEGRADED"]} degraded\n')
        out.write(style.green('Endpoint is usable.' if counts['FAIL'] == 0 else
                              'Endpoint has failures — see above.') + '\n')
        out.flush()
    if client is not None:
        client.close()
    return 0 if counts['FAIL'] == 0 else 1
