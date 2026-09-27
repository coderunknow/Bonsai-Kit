"""Measured TTFT / prefill / decode benchmarks."""

from __future__ import annotations

import statistics
import sys
import time

from .conversation import TurnStats
from .errors import BonsaiError
from .style import Style

def sig(value, digits=3):
    """Round to `digits` significant figures.

    A decode rate measured from three samples is not known to six decimal places; printing
    147.123456 tok/s would be a lie about the precision of the measurement.
    """
    if value is None:
        return None
    if value == 0:
        return 0.0
    from math import floor, log10
    magnitude = floor(log10(abs(value)))
    return round(value, -int(magnitude) + (digits - 1))

def summarize(values):
    """median / min / max / p95 over real samples. Empty in, empty out — no invention."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {'n': 0, 'median': None, 'min': None, 'max': None, 'p95': None}
    clean = sorted(clean)
    n = len(clean)

    def pct(p):
        if n == 1:
            return clean[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return clean[idx]

    return {'n': n, 'median': statistics.median(clean), 'min': clean[0], 'max': clean[-1],
            'p95': pct(0.95)}

BENCH_PROMPT_BLOCKS = 40

def run_benchmark(client, samples=3, style=None, out=None, max_tokens=96, warmup=1,
                  progress=None):
    """Measure TTFT, prefill speed, decode speed and total latency against a live endpoint.

    Every number comes from the server's own usage/timings block, or from a monotonic
    clock around a real request. Nothing is estimated, and a field the server did not
    report stays None so the renderer prints `n/a` instead of a guess.

    Sample 1 is the cold start (nothing in the prompt cache). The next `warmup` samples
    are discarded, and the rest are the warm distribution — reported as median/min/max/p95.
    """
    out = out if out is not None else sys.stdout
    prompt = ('Review this Python function and give two concrete improvements.\\n\\n' +
              ('def search(items, target):\\n'
               '    for index, value in enumerate(items):\\n'
               '        if value == target:\\n'
               '            return index\\n'
               '    return -1\\n\\n') * BENCH_PROMPT_BLOCKS)
    messages = [{'role': 'user', 'content': prompt}]
    report = {'endpoint': client.base_url, 'model': client.model, 'samples': samples,
              'warmup': warmup, 'max_tokens': max_tokens, 'cold': None, 'warm': {},
              'errors': [], 'notes': []}
    collected = {'ttft': [], 'prefill': [], 'decode': [], 'latency': [],
                 'prompt_tokens': [], 'completion_tokens': []}
    # i==0 is the cold start (empty prompt cache), then `warmup` discarded samples, then
    # the `samples` that actually form the reported distribution.
    total = samples + warmup + 1
    for i in range(total):
        label = 'cold' if i == 0 else ('warmup' if i <= warmup else f'sample {i - warmup}')
        if progress:
            progress(label, i + 1, total)
        started = time.monotonic()
        stats = TurnStats()
        try:
            with client.stream_chat(messages, max_tokens=max_tokens, temperature=1.0) as stream:
                for ev in stream:
                    if ev['kind'] == 'done':
                        stats.absorb(ev.get('usage'), ev.get('timings'),
                                     ttft=ev.get('ttft'), elapsed=ev.get('elapsed') or 0.0)
        except BonsaiError as e:
            report['errors'].append(f'{label}: {e}')
            continue
        stats.elapsed = time.monotonic() - started
        entry = {'ttft_s': stats.ttft, 'latency_s': stats.elapsed,
                 'prefill_tok_s': stats.prompt_tokens_per_s, 'decode_tok_s': stats.tokens_per_s,
                 'prompt_tokens': stats.prompt_tokens,
                 'completion_tokens': stats.completion_tokens}
        if i == 0:
            report['cold'] = entry
            continue
        if i <= warmup:
            continue
        for key, value in (('ttft', stats.ttft), ('prefill', stats.prompt_tokens_per_s),
                           ('decode', stats.tokens_per_s), ('latency', stats.elapsed),
                           ('prompt_tokens', stats.prompt_tokens),
                           ('completion_tokens', stats.completion_tokens)):
            if value is not None:
                collected[key].append(value)
    report['warm'] = {k: summarize(v) for k, v in collected.items()}
    if not collected['decode']:
        report['notes'].append('no decode rate was measured — the server did not report '
                               'timings and no completion tokens arrived')
    facts = client.caps.facts
    report['context_window'] = facts.get('context_window')
    report['runtime'] = facts.get('version')
    report['transport'] = client.transport.stats()
    return report

def render_benchmark(report, style=None):
    """Human-readable benchmark output. Unmeasured fields are printed as `n/a`."""
    style = style or Style(force_color=False)
    lines = [style.bold('benchmark — measured against ' + report['endpoint'])]
    if report.get('runtime'):
        lines.append(style.dim(f'runtime: {report["runtime"]}'))
    if report.get('context_window'):
        lines.append(style.dim(f'context window: {report["context_window"]} tokens'))

    def row(name, summary, unit, digits=3, scale=1.0):
        if not summary or not summary.get('n'):
            return f'  {name:<22} n/a  (not measured)'
        return (f'  {name:<22} median {sig(summary["median"] * scale, digits)} {unit}'
                f'  |  min {sig(summary["min"] * scale, digits)}'
                f'  |  max {sig(summary["max"] * scale, digits)}'
                f'  |  p95 {sig(summary["p95"] * scale, digits)}'
                f'  |  n={summary["n"]}')

    cold = report.get('cold') or {}
    if cold:
        parts = []
        if cold.get('ttft_s'):
            parts.append(f'ttft {sig(cold["ttft_s"] * 1000)} ms')
        if cold.get('prefill_tok_s'):
            parts.append(f'prefill {sig(cold["prefill_tok_s"])} tok/s')
        if cold.get('decode_tok_s'):
            parts.append(f'decode {sig(cold["decode_tok_s"])} tok/s')
        if cold.get('latency_s'):
            parts.append(f'total {sig(cold["latency_s"])} s')
        lines.append(style.dim('  cold start (first request, empty prompt cache): ' +
                               (' | '.join(parts) if parts else 'n/a')))
    warm = report.get('warm') or {}
    lines.append(style.dim(f'  warm ({warm.get("decode", {}).get("n", 0)} sample(s), '
                           f'{report.get("warmup", 0)} warmup discarded):'))
    lines.append(row('ttft', warm.get('ttft'), 'ms', scale=1000.0))
    lines.append(row('prefill speed', warm.get('prefill'), 'tok/s'))
    lines.append(row('decode speed', warm.get('decode'), 'tok/s'))
    lines.append(row('total latency', warm.get('latency'), 's'))
    pt = warm.get('prompt_tokens') or {}
    ct = warm.get('completion_tokens') or {}
    if pt.get('n'):
        lines.append(style.dim(f'  prompt {sig(pt["median"])} tok, '
                               f'completion {sig(ct.get("median"))} tok per sample'))
    tr = report.get('transport') or {}
    if tr:
        lines.append(style.dim(f'  transport: {tr.get("requests", 0)} requests, '
                               f'{tr.get("connections_opened", 0)} connection(s), '
                               f'{tr.get("reconnects", 0)} reconnect(s)'))
    for note in report.get('notes', []):
        lines.append(style.yellow('  note: ' + note))
    for err in report.get('errors', []):
        lines.append(style.red('  error: ' + str(err)[:200]))
    lines.append(style.dim('  VRAM/RAM are server-side and cannot be measured from the '
                           'client — the deployment cell reports those.'))
    return '\n'.join(lines)
