/**
 * Incremental Server-Sent-Events parser.
 *
 * The browser's fetch() hands us arbitrary byte boundaries: one network chunk can
 * contain half a frame, three whole frames, or the middle of `event: delta`. This
 * parser buffers text, splits frames on a blank line (LF or CRLF), and only emits
 * complete frames — exactly the contract the Python proxy writes
 * (`event: X\ndata: {...}\n\n`).
 */

export interface SSEEvent {
  event: string;
  data: string;
}

export interface SSEParser {
  /** Feed a decoded text chunk; returns every frame completed by this chunk. */
  feed(chunk: string): SSEEvent[];
  /** Flush a trailing frame the stream ended without its blank line. */
  flush(): SSEEvent[];
}

function parseFrame(raw: string): SSEEvent | null {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line === '' || line.startsWith(':')) continue; // blank or keep-alive comment
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1); // exactly one leading space
    if (field === 'data') {
      dataLines.push(value);
    } else if (field === 'event') {
      event = value;
    }
    // `id:`, `retry:` and unknown fields are accepted and ignored (per spec)
  }
  if (dataLines.length === 0) return null; // comment-only or fieldless frame
  return { event, data: dataLines.join('\n') };
}

const SEPARATOR = /\r?\n\r?\n/;

export function createSSEParser(): SSEParser {
  let buffer = '';

  return {
    feed(chunk: string): SSEEvent[] {
      if (chunk === '') return [];
      buffer += chunk;
      const out: SSEEvent[] = [];
      let match = SEPARATOR.exec(buffer);
      while (match !== null) {
        const raw = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const frame = parseFrame(raw);
        if (frame) out.push(frame);
        match = SEPARATOR.exec(buffer);
      }
      return out;
    },
    flush(): SSEEvent[] {
      const rest = buffer;
      buffer = '';
      if (rest.trim() === '') return [];
      const frame = parseFrame(rest);
      return frame ? [frame] : [];
    },
  };
}

/** Parse one frame's data field as JSON; malformed payloads become `malformed`. */
export function parseFrameData(frame: SSEEvent): { event: string; data: unknown } {
  try {
    return { event: frame.event, data: JSON.parse(frame.data) };
  } catch {
    return { event: 'malformed', data: frame.data };
  }
}
