import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useChat } from '../src/useChat';
import type { ChatRequest, Frame } from '../src/types';

type Sink = (frame: Frame) => void;

/**
 * A streamFn whose promise resolves when a terminal frame (done/error) passes
 * through — so `await send(...)` returns exactly when the turn finalizes. Frames
 * are delivered by the test, one act() at a time, to model the network.
 */
function makeHarness() {
  const calls: ChatRequest[] = [];
  let sink: Sink | null = null;
  let release: (() => void) | null = null;
  const streamFn = vi.fn((request: ChatRequest, onFrame: Sink) => {
    calls.push(request);
    const promise = new Promise<void>((resolve) => {
      release = resolve;
    });
    sink = (frame) => {
      onFrame(frame);
      if (frame.event === 'done' || frame.event === 'error') release?.();
    };
    return promise;
  });
  return {
    calls,
    streamFn,
    emit(frame: Frame) {
      act(() => sink?.(frame));
    },
    release() {
      release?.();
    },
  };
}

const doneFrame = (over: Record<string, unknown> = {}): Frame => ({
  event: 'done',
  data: {
    kind: 'done',
    stopped: false,
    stream_id: 'fixed-id',
    message: { content: 'full answer' },
    usage: { prompt_tokens: 42, completion_tokens: 64, total_tokens: 106 },
    finish_reason: 'stop',
    ...over,
  },
});

/** Start a send/retry without awaiting the stream; returns a promise for later. */
function startSend(thunk: () => Promise<void>): Promise<void> {
  let pending!: Promise<void>;
  act(() => {
    pending = thunk();
  });
  return pending;
}

beforeEach(() => {
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ ok: true, found: true }),
  })) as unknown as typeof fetch;
});

describe('useChat — streaming', () => {
  it('send() opens a stream, appends deltas to the draft, and finalizes on done', async () => {
    const h = makeHarness();
    const { result } = renderHook(() =>
      useChat({ streamFn: h.streamFn, makeStreamId: () => 'fixed-id' }),
    );
    const pending = startSend(() => result.current.send('hello'));
    expect(result.current.state.status).toBe('streaming');
    expect(result.current.state.messages.map((m) => m.role)).toEqual(['user']);
    expect(h.calls[0].stream_id).toBe('fixed-id');
    expect(h.calls[0].messages).toEqual([{ role: 'user', content: 'hello' }]);

    h.emit({ event: 'delta', data: { kind: 'delta', text: 'par' } });
    h.emit({ event: 'delta', data: { kind: 'delta', text: 'tial' } });
    expect(result.current.state.draft?.content).toBe('partial');

    h.emit({
      event: 'usage',
      data: { kind: 'usage', usage: { prompt_tokens: 42, completion_tokens: 64, total_tokens: 106 } },
    });
    h.emit(doneFrame({ message: { content: 'partial answer' } }));
    await act(async () => {
      await pending;
    });
    await waitFor(() => expect(result.current.state.status).toBe('idle'));

    const last = result.current.state.messages.at(-1);
    expect(last?.role).toBe('assistant');
    expect(last?.content).toBe('partial answer');
    expect(last?.stopped).toBeUndefined();
    expect(result.current.state.usage?.completion_tokens).toBe(64);
  });

  it('keeps reasoning deltas out of the answer', async () => {
    const h = makeHarness();
    const { result } = renderHook(() =>
      useChat({ streamFn: h.streamFn, makeStreamId: () => 'fixed-id' }),
    );
    const pending = startSend(() => result.current.send('think'));
    h.emit({ event: 'reasoning', data: { kind: 'reasoning', text: 'pondering… ' } });
    h.emit({ event: 'delta', data: { kind: 'delta', text: 'the answer' } });
    expect(result.current.state.draft?.reasoning).toBe('pondering… ');
    expect(result.current.state.draft?.content).toBe('the answer');
    h.emit(
      doneFrame({
        message: { content: 'the answer', reasoning_content: 'pondering… ' },
      }),
    );
    await act(async () => {
      await pending;
    });
    await waitFor(() => expect(result.current.state.status).toBe('idle'));
    const last = result.current.state.messages.at(-1);
    expect(last?.content).toBe('the answer');
    expect(last?.reasoning).toBe('pondering… ');
  });
});

describe('useChat — stop', () => {
  it('stop() posts /api/stop with the stream id and keeps the partial on done', async () => {
    const h = makeHarness();
    const { result } = renderHook(() =>
      useChat({ streamFn: h.streamFn, makeStreamId: () => 'stop-id' }),
    );
    const pending = startSend(() => result.current.send('story'));
    h.emit({ event: 'delta', data: { kind: 'delta', text: 'once upon' } });

    act(() => result.current.stop());
    expect(result.current.state.status).toBe('stopping');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/stop',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ stream_id: 'stop-id' }) }),
    );

    // the server keeps streaming until it delivers the terminal stopped frame
    h.emit(
      doneFrame({
        stopped: true,
        tokens: 5,
        usage: null,
        message: { content: 'once upon a time' },
      }),
    );
    await act(async () => {
      await pending;
    });
    await waitFor(() => expect(result.current.state.status).toBe('idle'));

    const last = result.current.state.messages.at(-1);
    expect(last?.content).toBe('once upon a time');
    expect(last?.stopped).toBe(true);
    expect(last?.tokens).toBe(5);
    expect(result.current.state.stopped).toBe(true);
  });

  it('stop() is a no-op when idle', () => {
    const h = makeHarness();
    const { result } = renderHook(() => useChat({ streamFn: h.streamFn }));
    act(() => result.current.stop());
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(result.current.state.status).toBe('idle');
  });
});

describe('useChat — failure and retry', () => {
  it('a mid-stream error keeps the partial, marks it failed, and reports tokens', async () => {
    const h = makeHarness();
    const { result } = renderHook(() =>
      useChat({ streamFn: h.streamFn, makeStreamId: () => 'fixed-id' }),
    );
    const pending = startSend(() => result.current.send('long story'));
    h.emit({ event: 'delta', data: { kind: 'delta', text: 'the first bit' } });
    h.emit({
      event: 'error',
      data: { kind: 'mid_stream', error: 'stream ended early', tokens_arrived: 12 },
    });
    await act(async () => {
      await pending;
    });
    await waitFor(() => expect(result.current.state.status).toBe('idle'));

    expect(result.current.state.error?.kind).toBe('mid_stream');
    expect(result.current.state.error?.tokensArrived).toBe(12);
    const kept = result.current.state.messages.at(-1);
    expect(kept?.content).toBe('the first bit');
    expect(kept?.failed).toBe(true);
    expect(kept?.tokens).toBe(12);
  });

  it('retry() resends the last user message as a fresh turn', async () => {
    const h = makeHarness();
    const { result } = renderHook(() =>
      useChat({ streamFn: h.streamFn, makeStreamId: () => 'fixed-id' }),
    );
    const first = startSend(() => result.current.send('question one'));
    h.emit({
      event: 'error',
      data: { kind: 'stall', error: 'no data for 30 seconds', waited_seconds: 30 },
    });
    await act(async () => {
      await first;
    });
    await waitFor(() => expect(result.current.state.status).toBe('idle'));
    expect(result.current.state.error?.kind).toBe('stall');

    const second = startSend(() => result.current.retry());
    expect(h.calls).toHaveLength(2);
    expect(h.calls[1].messages.at(-1)).toEqual({ role: 'user', content: 'question one' });
    expect(result.current.state.status).toBe('streaming');
    expect(result.current.state.error).toBeNull();
    h.emit(doneFrame({ message: { content: 'answer two' } }));
    await act(async () => {
      await second;
    });
    await waitFor(() => expect(result.current.state.status).toBe('idle'));
  });

  it('an ApiError before any frame maps to its kind', async () => {
    const { ApiError } = await import('../src/api');
    const streamFn = vi.fn(async () => {
      throw new ApiError('HTTP 401: invalid api key', 'unauthorized', 401, 'check the key');
    });
    const { result } = renderHook(() => useChat({ streamFn }));
    await act(async () => {
      await result.current.send('hi');
    });
    expect(result.current.state.status).toBe('idle');
    expect(result.current.state.error).toMatchObject({
      kind: 'unauthorized',
      message: 'HTTP 401: invalid api key',
      hint: 'check the key',
    });
    expect(result.current.state.messages.at(-1)?.failed).toBeUndefined();
  });

  it('an empty stream with no terminal frame fails as a connection error', async () => {
    const streamFn = vi.fn(async () => undefined);
    const { result } = renderHook(() => useChat({ streamFn }));
    await act(async () => {
      await result.current.send('hi');
    });
    expect(result.current.state.error?.kind).toBe('connection');
    expect(result.current.state.status).toBe('idle');
  });
});

describe('useChat — guards', () => {
  it('send() is ignored while a stream is running', async () => {
    const h = makeHarness();
    const { result } = renderHook(() =>
      useChat({ streamFn: h.streamFn, makeStreamId: () => 'fixed-id' }),
    );
    startSend(() => result.current.send('first'));
    await act(async () => {
      await result.current.send('second');
    });
    expect(h.calls).toHaveLength(1);
    expect(result.current.state.messages).toHaveLength(1);
    // cleanup: end the still-open stream
    h.emit(doneFrame());
  });
});
