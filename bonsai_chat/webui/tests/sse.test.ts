import { describe, expect, it } from 'vitest';
import { createSSEParser, parseFrameData } from '../src/sse';

const frame = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

describe('SSE parser', () => {
  it('parses a single complete frame', () => {
    const parser = createSSEParser();
    const events = parser.feed(frame('delta', { kind: 'delta', text: 'hi' }));
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('delta');
    expect(JSON.parse(events[0].data)).toEqual({ kind: 'delta', text: 'hi' });
  });

  it('emits nothing for a partial frame and completes it on the next chunk', () => {
    const parser = createSSEParser();
    const whole = frame('delta', { kind: 'delta', text: 'split me' });
    const cut = Math.floor(whole.length / 2);
    expect(parser.feed(whole.slice(0, cut))).toHaveLength(0);
    const events = parser.feed(whole.slice(cut));
    expect(events).toHaveLength(1);
    expect(JSON.parse(events[0].data).text).toBe('split me');
  });

  it('survives a frame split in the middle of the word "event"', () => {
    const parser = createSSEParser();
    expect(parser.feed('eve')).toHaveLength(0);
    expect(parser.feed('nt: delta\nda')).toHaveLength(0);
    const events = parser.feed('ta: {"kind":"delta","text":"x"}\n\n');
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('delta');
    expect(JSON.parse(events[0].data).text).toBe('x');
  });

  it('handles three frames arriving in one chunk', () => {
    const parser = createSSEParser();
    const events = parser.feed(
      frame('delta', { text: 'a' }) + frame('delta', { text: 'b' }) + frame('done', { kind: 'done' }),
    );
    expect(events.map((e) => e.event)).toEqual(['delta', 'delta', 'done']);
  });

  it('ignores keep-alive comments and frames without data, then continues', () => {
    const parser = createSSEParser();
    const events = parser.feed(': keep-alive\n\n' + frame('delta', { text: 'after' }));
    expect(events).toHaveLength(1);
    expect(JSON.parse(events[0].data).text).toBe('after');
  });

  it('parses CRLF frames and multi-line data joined with newlines', () => {
    const parser = createSSEParser();
    const events = parser.feed('event: delta\r\ndata: {"text":\r\ndata: "joined"}\r\n\r\n');
    expect(events).toHaveLength(1);
    expect(events[0].event).toBe('delta');
    expect(events[0].data).toBe('{"text":\n"joined"}');
  });

  it('flush() recovers a trailing frame the stream ended without terminating', () => {
    const parser = createSSEParser();
    expect(parser.feed('event: done\ndata: {"kind":"done"}')).toHaveLength(0);
    const tail = parser.flush();
    expect(tail).toHaveLength(1);
    expect(tail[0].event).toBe('done');
    expect(parser.flush()).toHaveLength(0);
  });

  it('flags malformed JSON data instead of throwing', () => {
    const parsed = parseFrameData({ event: 'delta', data: '{not json' });
    expect(parsed.event).toBe('malformed');
  });
});
