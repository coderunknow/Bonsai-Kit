"""Thinking budget and reasoning-effort configuration for Bonsai 2."""

from __future__ import annotations

THINK_BUDGETS = {'off': 0, 'low': 512, 'medium': 2048, 'high': 8192, 'max': -1}

THINK_ORDER = ('off', 'low', 'medium', 'high', 'max')

THINK_ALIASES = {'none': 'off', '0': 'off', 'no': 'off', 'disable': 'off', 'disabled': 'off',
                 'minimal': 'low', '512': 'low', 'normal': 'medium', '2048': 'medium',
                 'med': 'medium', '8192': 'high', 'maximum': 'max', 'unlimited': 'max',
                 'xhigh': 'max', 'full': 'max', '-1': 'max'}

EFFORT_VALUES = ('medium', 'xhigh')

EFFORT_ALIASES = {'default': 'xhigh', 'highest': 'xhigh', 'max': 'xhigh', 'maximum': 'xhigh',
                  'high': 'xhigh', 'normal': 'medium', 'med': 'medium', 'balanced': 'medium'}

EFFORT_UNRELIABLE = ('low', 'minimal', 'none')

REASONING_DISPLAYS = ('full', 'compact', 'hidden')

def parse_think_level(raw):
    """Map a user word to a canonical thinking level. Raises ValueError on nonsense."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in THINK_BUDGETS:
        return key
    if key in THINK_ALIASES:
        return THINK_ALIASES[key]
    if key.lstrip('-').isdigit():            # a bare number is an explicit token budget
        return int(key)
    raise ValueError(f'unknown thinking level {raw!r} — use ' + '|'.join(THINK_ORDER)
                     + ' or a token count')

def parse_effort(raw):
    key = str(raw or '').strip().lower()
    if key in EFFORT_VALUES:
        return key
    if key in EFFORT_ALIASES:
        return EFFORT_ALIASES[key]
    if key in EFFORT_UNRELIABLE:
        return key
    raise ValueError(f'unknown reasoning_effort {raw!r} — Bonsai 2 accepts '
                     + '|'.join(EFFORT_VALUES) + ' (low is accepted but documented as a no-op)')

class ReasoningConfig:
    """What to ask the model for, and how to show what comes back.

    Two independent axes, because that is what the runtime offers:
      * `level`  -> a thinking token budget (the Off..Max picker)
      * `effort` -> the template's reasoning_effort (medium / xhigh)
    `display` controls the terminal: `full` prints the thinking trace, `compact` prints a
    one-line indicator plus a token count, `hidden` prints neither.
    """

    def __init__(self, level='medium', effort='medium', display='compact'):
        self.level = parse_think_level(level)
        self.effort = parse_effort(effort) if effort else None
        self.display = display if display in REASONING_DISPLAYS else 'compact'

    # ------------------------------------------------------------------
    def budget(self):
        """The thinking token budget for the current level (int), or None if free-form."""
        if isinstance(self.level, str) and self.level in THINK_BUDGETS:
            return THINK_BUDGETS[self.level]
        try:
            return int(self.level)
        except (TypeError, ValueError):
            return None

    def label(self):
        budget = self.budget()
        shown = 'unlimited' if budget == -1 else ('off' if budget == 0 else f'{budget} tok')
        return f'{self.level} ({shown})'

    def wire(self, caps=None):
        """Request fields for this configuration, filtered by what the server accepts.

        Returns (fields, notes). `notes` explains anything that was dropped, so a user
        who asked for thinking control is never silently ignored.
        """
        fields, notes = {}, []
        budget = self.budget()
        # Tri-state: `caps.supported()` returns None when the field has not been tested
        # against this build yet. Unknown must mean "send it and find out" — only an
        # explicit False (the server answered 400 naming the field) suppresses it.
        if budget is not None:
            if caps is None or caps.supported('thinking_budget_tokens') is not False:
                fields['thinking_budget_tokens'] = budget
            else:
                notes.append('this build rejects thinking_budget_tokens — thinking is '
                             'server-controlled')
        # `caps.supported()` is tri-state: None means untested, so we still send it and
        # let the server's own 400 teach us. Only an explicit False suppresses the field.
        if self.effort and (caps is None or caps.supported('reasoning_effort') is not False):
            fields['reasoning_effort'] = self.effort
        elif self.effort:
            notes.append('this build rejects reasoning_effort — the effort setting is inert')
        return fields, notes

    def describe(self):
        parts = [f'think={self.label()}']
        if self.effort:
            parts.append(f'effort={self.effort}')
        parts.append(f'display={self.display}')
        return ', '.join(parts)

    def to_dict(self):
        return {'level': self.level, 'effort': self.effort, 'display': self.display,
                'budget': self.budget()}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(level=d.get('level', 'medium'), effort=d.get('effort', 'medium'),
                   display=d.get('display', 'compact'))
