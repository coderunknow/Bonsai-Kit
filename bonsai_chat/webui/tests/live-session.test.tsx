/**
 * Live end-to-end session against a REAL `bonsai_chat.py --serve` + mock upstream.
 *
 * Skipped unless BONSAI_LIVE_BASE (and BONSAI_LIVE_ADMIN) point at a running server —
 * the default test run stays offline. Started by evidence/live_session.py.
 *
 * Drives the real React app with real fetch + ReadableStream SSE: connect, stream,
 * stop mid-answer, fail mid-stream, retry — then checks that no key ever reached
 * browser storage.
 */
import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from '../src/App';

const BASE = process.env.BONSAI_LIVE_BASE;
const ADMIN = process.env.BONSAI_LIVE_ADMIN;
const KEY = process.env.BONSAI_LIVE_KEY ?? '';

/**
 * Browsers resolve `/api/...` against the page origin; node's fetch (inside jsdom)
 * cannot. Resolve relative URLs against BASE exactly like a browser would.
 */
function installBrowserFetch(base: string): void {
  const realFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const resolved =
      typeof input === 'string' && input.startsWith('/') ? `${base}${input}` : input;
    return realFetch(resolved as RequestInfo, init);
  }) as typeof fetch;
}

async function arm(scenario: string): Promise<void> {
  await fetch(`${ADMIN}/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario }),
  });
}

const transcript: string[] = [];
function note(line: string) {
  transcript.push(line);
  console.log(`[live] ${line}`);
}

describe.skipIf(!BASE || !ADMIN)('live session against a running --serve', () => {
  it('connects, streams, stops mid-answer, survives a mid-stream failure, retries', async () => {
    if (!BASE || !ADMIN) return;
    installBrowserFetch(BASE);
    await arm('slow-sticky');

    render(<App />);

    // 1. auto-connect: the Python process already holds the key
    await waitFor(
      () => expect(screen.getByTestId('connect-status')).toHaveTextContent(/connected —/),
      { timeout: 20000 },
    );
    const status = screen.getByTestId('connect-status').textContent ?? '';
    note(`connected: ${status.trim()}`);
    expect(status).toContain('ternary-bonsai-2-27b');

    // 2. send a message and watch tokens stream in
    const input = screen.getByLabelText('Message');
    fireEvent.change(input, { target: { value: 'Write a long story about ternary computers.' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(
      () => {
        const draft = screen.getByTestId('draft-content').textContent ?? '';
        expect(draft.replace('▍', '').length).toBeGreaterThan(30);
      },
      { timeout: 20000 },
    );
    const streamingText = (screen.getByTestId('draft-content').textContent ?? '').replace('▍', '');
    note(`streaming draft (${streamingText.length} chars): ${streamingText.slice(0, 120)}…`);
    const reasoningVisible = screen.queryByTestId('draft-reasoning');
    if (reasoningVisible) {
      note(`reasoning block (separate from answer): "${reasoningVisible.textContent}"`);
    }

    // 3. stop mid-answer: partial stays, marked with a measured token count
    fireEvent.click(screen.getByTestId('stop-button'));
    await waitFor(
      () => expect(screen.getByTestId('stopped-note')).toHaveTextContent(
        /\(stopped after \d+ tokens\)/,
      ),
      { timeout: 20000 },
    );
    const stoppedNote = screen.getByTestId('stopped-note').textContent ?? '';
    const messages = screen.getByTestId('messages');
    expect(messages.textContent).toContain(streamingText.slice(0, 40));
    expect(messages.textContent).not.toContain('en.cppreference.com');
    note(`stop honoured: ${stoppedNote}`);

    // 4. force a mid-stream failure on the next turn
    await arm('disconnect-once');
    fireEvent.change(input, {
      target: { value: 'Explain why this request will break halfway.' },
    });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(
      () => expect(screen.getByTestId('error-card')).toHaveTextContent(
        /dropped after \d+ tokens/,
      ),
      { timeout: 20000 },
    );
    const errorText = screen.getByTestId('error-card').textContent ?? '';
    note(`failure rendered: ${errorText.replace(/\s+/g, ' ').slice(0, 200)}`);

    // 5. retry: resends the last user message; wait for the turn to FINISH
    //    (the Stop button disappears only when the stream is idle again)
    fireEvent.click(screen.getByTestId('retry-button'));
    await waitFor(
      () => {
        expect(screen.queryByTestId('error-card')).toBeNull();
        expect(screen.queryByTestId('stop-button')).toBeNull();
        expect(screen.queryByTestId('send-button')).not.toBeNull();
        const all = screen.getByTestId('messages').textContent ?? '';
        expect(all.length).toBeGreaterThan(900);
      },
      { timeout: 30000 },
    );
    const finalAll = screen.getByTestId('messages').textContent ?? '';
    expect(finalAll).not.toContain('connection dropped after');
    // the stopped marker belongs to the first turn only; the retried answer is whole
    expect(screen.queryAllByTestId('stopped-note')).toHaveLength(1);
    expect(screen.queryAllByTestId('failed-note')).toHaveLength(1);
    note(`retry completed: total transcript ${finalAll.length} chars, ` +
      'turn idle with a full answer');

    // 6. browser-visible artifacts carry no key
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
    const configText = await (await fetch(`${BASE}/api/config`)).text();
    if (KEY) expect(configText).not.toContain(KEY);
    expect(configText).not.toContain('"api_key"');
    note('localStorage=0 sessionStorage=0 /api/config carries no api_key');

    console.log(`\n[live-session transcript]\n${transcript.join('\n')}\n`);
  }, 120000);
});
