"""Conversation branching: fork, list, switch, delete, rename.

A branch is a named snapshot of the message list plus the point it was taken from.
Switching branches swaps the conversation's live messages, so `/undo`, `/retry`,
trimming and compaction all keep working unchanged — they operate on whatever list is
current and were already written to respect tool-pair integrity.

The tree is a tree and not a stack: forking from an earlier point is allowed, and a
branch keeps a record of where it came from so `/branches` can show the shape. Branches
are stored inside the session file as an extra meta line, which older readers skip, so
session schema 2 keeps loading everywhere.
"""

from __future__ import annotations

import copy
import time

MAIN = 'main'

#: A long session must not be able to grow without limit: every fork deep-copies the
#: messages it is taken from, so the number of branches is capped.
MAX_BRANCHES = 32


class Branch:
    def __init__(self, name, messages=None, parent=None, fork_at=None, branch_id=None):
        self.name = name
        self.messages = list(messages or [])
        self.parent = parent
        self.fork_at = fork_at                 # index in the parent's list, or None
        self.id = branch_id or name
        self.created = time.time()

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'parent': self.parent,
                'fork_at': self.fork_at, 'created': round(self.created, 3),
                'messages': self.messages}

    @classmethod
    def from_dict(cls, data):
        branch = cls(data.get('name') or data.get('id') or MAIN,
                     messages=data.get('messages') or [],
                     parent=data.get('parent'), fork_at=data.get('fork_at'),
                     branch_id=data.get('id'))
        branch.created = data.get('created') or time.time()
        return branch

    def __repr__(self):
        return f'Branch({self.name!r}, {len(self.messages)} messages)'


class BranchTree:
    """The set of branches over one conversation."""

    def __init__(self, conversation, branches=None, current=None):
        self.conversation = conversation
        self.branches = {}
        if branches:
            for data in branches if isinstance(branches, list) else \
                    list((branches or {}).values()):
                if isinstance(data, dict):
                    branch = Branch.from_dict(data)
                    self.branches[branch.name] = branch
        if MAIN not in self.branches:
            self.branches[MAIN] = Branch(MAIN, messages=list(conversation.messages),
                                         parent=None, fork_at=None)
        self.current_name = current if current in self.branches else MAIN
        branch = self.branches[self.current_name]
        conversation.messages = branch.messages

    # ------------------------------------------------------------------
    @classmethod
    def from_conversation(cls, conversation, saved=None):
        return cls(conversation, branches=(saved or {}).get('branches')
                   if isinstance(saved, dict) else saved,
                   current=(saved or {}).get('current') if isinstance(saved, dict) else None)

    @property
    def current(self):
        return self.branches[self.current_name]

    def names(self):
        return list(self.branches)

    # ------------------------------------------------------------------
    def _commit(self):
        """The live message list is the current branch's storage. Keep them identical."""
        self.current.messages = self.conversation.messages

    def fork(self, name, at=None):
        """Create `name` from the current branch, optionally truncated at index `at`."""
        name = str(name or '').strip()
        if not name:
            raise ValueError('a branch needs a name')
        if name in self.branches:
            raise ValueError(f'branch {name!r} already exists')
        if len(self.branches) >= MAX_BRANCHES:
            raise ValueError(f'{MAX_BRANCHES} branches already exist — delete one before '
                             'forking again (each branch holds its own copy of the '
                             'messages it was taken from)')
        self._commit()
        source = self.current.messages
        cut = len(source) if at is None else max(0, min(len(source), int(at)))
        self.branches[name] = Branch(name, messages=copy.deepcopy(source[:cut]),
                                     parent=self.current_name, fork_at=cut)
        return self.switch(name)

    def switch(self, name):
        name = str(name or '').strip()
        if name not in self.branches:
            raise ValueError(f'no branch named {name!r} '
                             f'(known: {", ".join(self.names())})')
        self._commit()                       # save where we were before moving away
        self.current_name = name
        self.conversation.messages = self.branches[name].messages
        return self.branches[name]

    def delete(self, name):
        name = str(name or '').strip()
        if name not in self.branches:
            raise ValueError(f'no branch named {name!r}')
        if name == MAIN:
            raise ValueError(f'the {MAIN} branch cannot be deleted')
        if name == self.current_name:
            parent = self.branches[name].parent or MAIN
            self.switch(parent if parent in self.branches else MAIN)
        del self.branches[name]
        # Re-parent anything that forked from a branch that no longer exists.
        for branch in self.branches.values():
            if branch.parent == name:
                branch.parent = MAIN

    def rename(self, old, new):
        old, new = str(old or '').strip(), str(new or '').strip()
        if old not in self.branches:
            raise ValueError(f'no branch named {old!r}')
        if old == MAIN or new == MAIN:
            raise ValueError(f'the {MAIN} branch cannot be renamed')
        if not new:
            raise ValueError('a branch needs a name')
        if new in self.branches:
            raise ValueError(f'branch {new!r} already exists')
        branch = self.branches.pop(old)
        branch.name = new
        was_current = self.current_name == old
        self.branches[new] = branch
        for other in self.branches.values():
            if other.parent == old:
                other.parent = new
        if was_current:
            self.current_name = new

    # ------------------------------------------------------------------
    def list_rows(self):
        rows = []
        for name in self.names():
            branch = self.branches[name]
            rows.append({'name': name,
                         'messages': len(branch.messages),
                         'parent': branch.parent,
                         'fork_at': branch.fork_at,
                         'current': name == self.current_name})
        return rows

    def describe(self):
        lines = []
        for row in self.list_rows():
            mark = '*' if row['current'] else ' '
            origin = ''
            if row['parent']:
                origin = f"  (from {row['parent']} at message {row['fork_at']})"
            lines.append(f" {mark} {row['name']:<18} {row['messages']:>4} message(s)"
                         f"{origin}")
        return lines

    # ------------------------------------------------------------------
    def to_saved(self):
        self._commit()
        return {'current': self.current_name,
                'branches': [b.to_dict() for b in self.branches.values()]}

    def sync(self):
        """Call after the live list is replaced wholesale (e.g. /load, /clear)."""
        self.current.messages = self.conversation.messages
