/**
 * The browser's only door to the model: same-origin `/api/*` calls.
 *
 * The bearer key is posted once (PUT /api/config) and never read back, stored, or
 * attached to any GET. Nothing here talks to the tunnel directly.
 */

import type { ChatRequest, ErrorKind, Frame } from './types';
import { createSSEParser, parseFrameData } from './sse';

export class ApiError extends Error {
  kind: ErrorKind;
  status: number;
  hint?: string;

  constructor(message: string, kind: ErrorKind, status: number, hint?: string) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.hint = hint;
  }
}

async function errorFromResponse(resp: Response): Promise<ApiError> {
  let body: Record<string, unknown> = {};
  try {
    body = (await resp.json()) as Record<string, unknown>;
  } catch {
    /* non-JSON error body */
  }
  const kind = (body.kind as ErrorKind) || (resp.status === 401 ? 'unauthorized' : 'remote');
  const message =
    typeof body.error === 'string' && body.error ? body.error : `HTTP ${resp.status}`;
  return new ApiError(message, kind, resp.status,
    typeof body.hint === 'string' ? body.hint : undefined);
}

export async function getJSON<T>(path: string): Promise<T> {
  const resp = await fetch(path, { method: 'GET' });
  if (!resp.ok) throw await errorFromResponse(resp);
  return (await resp.json()) as T;
}

export async function putJSON<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw await errorFromResponse(resp);
  return (await resp.json()) as T;
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw await errorFromResponse(resp);
  return (await resp.json()) as T;
}

/**
 * POST /api/chat and stream the SSE reply frame by frame.
 *
 * fetch + ReadableStream, not EventSource: the request must be a POST carrying the
 * stream id, and EventSource can do neither.
 */
export async function streamChat(
  request: ChatRequest,
  onFrame: (frame: Frame) => void,
): Promise<void> {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!resp.ok) throw await errorFromResponse(resp);
  if (!resp.body) {
    throw new ApiError('the response had no body to stream', 'connection', resp.status);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSSEParser();
  let terminal = false;

  const dispatch = (events: ReturnType<typeof parser.feed>) => {
    for (const raw of events) {
      const frame = parseFrameData(raw);
      onFrame(frame);
      if (frame.event === 'done' || frame.event === 'error') terminal = true;
    }
  };

  try {
    while (!terminal) {
      const { done, value } = await reader.read();
      if (done) break;
      dispatch(parser.feed(decoder.decode(value, { stream: true })));
    }
    if (!terminal) dispatch(parser.flush());
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* already closed */
    }
  }
}
