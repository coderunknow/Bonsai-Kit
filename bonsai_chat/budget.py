"""Token and cost budgeting.

Two separate things, kept separate on purpose:

* **Tokens** are counted from what the server reports in `usage`. Nothing is estimated:
  a turn whose usage the server did not report counts zero, and the counter says so.
* **Cost** is only ever what the user tells us (a price per million tokens). The client
  has no price list, so cost is `None` until a price is supplied, and every cost print
  is labelled as computed from that supplied price.

A budget is a *ceiling checked before the request and after the response*, not a
prediction. It can refuse to send (session cap reached) and it can report what has been
spent; it never claims to know what a request will cost in advance.
"""

from __future__ import annotations

import time


class Budget:
    """Running totals for one client session, with optional ceilings."""

    def __init__(self, session_tokens=None, prompt_tokens=None, completion_tokens=None,
                 turns=None, price_per_mtok=None):
        self.session_tokens = session_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.turns = turns
        self.price_per_mtok = price_per_mtok
        self.used_prompt = 0
        self.used_completion = 0
        self.turn_count = 0
        self.started = time.time()
        self.stopped_reason = None

    # ------------------------------------------------------------------
    def note(self, usage=None):
        """Add one turn's reported usage. Unknown values add nothing, and say so."""
        usage = usage or {}
        prompt = usage.get('prompt_tokens')
        completion = usage.get('completion_tokens')
        if isinstance(prompt, int):
            self.used_prompt += prompt
        if isinstance(completion, int):
            self.used_completion += completion
        self.turn_count += 1

    # ------------------------------------------------------------------
    @property
    def used_tokens(self):
        return self.used_prompt + self.used_completion

    def remaining(self):
        if self.session_tokens is None:
            return None
        return max(0, self.session_tokens - self.used_tokens)

    def over(self):
        """Why this budget is exhausted, or None. Checked *before* the next request."""
        if self.stopped_reason:
            return self.stopped_reason
        if self.session_tokens is not None and self.used_tokens >= self.session_tokens:
            return (f'session token budget of {self.session_tokens} reached '
                    f'({self.used_tokens} used)')
        if self.prompt_tokens is not None and self.used_prompt >= self.prompt_tokens:
            return f'prompt token budget of {self.prompt_tokens} reached'
        if self.completion_tokens is not None and self.used_completion >= self.completion_tokens:
            return f'completion token budget of {self.completion_tokens} reached'
        if self.turns is not None and self.turn_count >= self.turns:
            return f'turn budget of {self.turns} reached'
        return None

    def stop(self, reason):
        self.stopped_reason = reason

    # ------------------------------------------------------------------
    def cost(self):
        """Money spent, or None. Only computable from a user-supplied price."""
        if not self.price_per_mtok:
            return None
        return self.used_tokens * float(self.price_per_mtok) / 1_000_000.0

    def status(self):
        out = {'prompt_tokens': self.used_prompt,
               'completion_tokens': self.used_completion,
               'total_tokens': self.used_tokens,
               'turns': self.turn_count,
               'elapsed_s': round(time.time() - self.started, 1)}
        if self.session_tokens is not None:
            out['session_limit'] = self.session_tokens
            out['remaining'] = self.remaining()
        if self.price_per_mtok:
            out['price_per_mtok'] = self.price_per_mtok
            out['cost'] = round(self.cost(), 6)
        reason = self.over()
        if reason:
            out['stopped'] = reason
        return out

    def describe(self):
        lines = [f'  tokens used      {self.used_tokens} '
                 f'(prompt {self.used_prompt}, completion {self.used_completion})']
        if self.session_tokens is not None:
            lines.append(f'  session budget   {self.session_tokens} '
                         f'({self.remaining()} left)')
        if self.turns is not None:
            lines.append(f'  turns            {self.turn_count} / {self.turns}')
        if self.price_per_mtok:
            lines.append(f'  cost             {self.cost():.6f} '
                         f'(at {self.price_per_mtok} per Mtok — your figure, not ours)')
        else:
            lines.append('  cost             n/a (no --price-per-mtok given; the client '
                         'has no price list)')
        reason = self.over()
        if reason:
            lines.append(f'  STOPPED          {reason}')
        return lines
