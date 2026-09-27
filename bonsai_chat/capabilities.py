"""Tri-state capability map: supported / unsupported / unknown, with evidence."""

from __future__ import annotations

class CapabilityMap:
    """What this particular server build actually does, learned not assumed.

    Every entry is one of `supported` / `unsupported` / `unknown`, plus the evidence that
    produced it. `/doctor` prints the whole map so a user can see what is real, what is
    missing, and what was never tested.
    """

    KNOWN_FIELDS = ('reasoning_effort', 'thinking_budget_tokens', 'tools', 'tool_choice',
                    'response_format', 'min_p', 'top_k', 'stream_options')

    def __init__(self):
        self.fields = {k: 'unknown' for k in self.KNOWN_FIELDS}
        self.evidence = {}
        self.facts = {}            # free-form: model ids, context, vision, runtime…

    # ------------------------------------------------------------------
    def mark(self, name, state, evidence=''):
        if state not in ('supported', 'unsupported', 'unknown'):
            raise ValueError(state)
        self.fields.setdefault(name, state)
        self.fields[name] = state
        if evidence:
            self.evidence[name] = evidence

    def supported(self, name):
        """True / False / None(unknown). Callers treat unknown as 'try it'."""
        state = self.fields.get(name, 'unknown')
        return None if state == 'unknown' else state == 'supported'

    def state(self, name):
        return self.fields.get(name, 'unknown')

    def note_400(self, name, message):
        self.mark(name, 'unsupported', f'HTTP 400: {str(message)[:160]}')

    def note_ok(self, name):
        if self.fields.get(name) != 'unsupported':
            self.mark(name, 'supported', 'accepted by the server')

    def set_fact(self, key, value):
        self.facts[key] = value

    def to_dict(self):
        return {'fields': dict(self.fields), 'evidence': dict(self.evidence),
                'facts': dict(self.facts)}
