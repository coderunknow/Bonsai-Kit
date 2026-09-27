/**
 * The chat state machine: send → stream → stop/retry, as a plain reducer.
 *
 * No state library, no classes: `useReducer` plus the streaming side effect. The
 * reducer is exported separately from the hook so tests can drive frames directly.
 */

import { useCallback, useEffect, useReducer, useRef } from 'react';
import { ApiError, streamChat as defaultStreamChat, postJSON } from './api';
import type { ChatRequest, DoneData, ErrorKind, Frame, Message, Usage } from './types';

export interface ChatError {
  kind: ErrorKind;
  message: string;
  hint?: string;
  /** How many content tokens arrived before a mid-stream failure. */
  tokensArrived?: number;
  /** How long a stall waited, in seconds. */
  waitedSeconds?: number;
}

export interface ChatState {
  messages: Message[];
  draft: { content: string; reasoning: string } | null;
  status: 'idle' | 'streaming' | 'stopping';
  usage: Usage | null;
  stopped: boolean;
  lastTokens: number | null;
  error: ChatError | null;
  streamId: string | null;
  lastUserText: string | null;
  /** Bumped on every send — lets effects watch "a new turn started". */
  turn: number;
}

export type ChatAction =
  | { type: 'send'; text: string; streamId: string }
  | { type: 'delta'; text: string }
  | { type: 'reasoning'; text: string }
  | { type: 'usage'; usage: Usage }
  | { type: 'done'; data: DoneData }
  | { type: 'fail'; error: ChatError }
  | { type: 'stop' }
  | { type: 'clear-error' };

export const initialChatState: ChatState = {
  messages: [],
  draft: null,
  status: 'idle',
  usage: null,
  stopped: false,
  lastTokens: null,
  error: null,
  streamId: null,
  lastUserText: null,
  turn: 0,
};

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'send':
      return {
        ...state,
        messages: [...state.messages, { role: 'user', content: action.text }],
        draft: { content: '', reasoning: '' },
        status: 'streaming',
        stopped: false,
        error: null,
        streamId: action.streamId,
        lastUserText: action.text,
        turn: state.turn + 1,
      };
    case 'delta':
      if (!state.draft || state.status === 'idle') return state;
      return { ...state, draft: { ...state.draft, content: state.draft.content + action.text } };
    case 'reasoning':
      if (!state.draft || state.status === 'idle') return state;
      return {
        ...state,
        draft: { ...state.draft, reasoning: state.draft.reasoning + action.text },
      };
    case 'usage':
      return { ...state, usage: action.usage };
    case 'done': {
      const draft = state.draft ?? { content: '', reasoning: '' };
      const data = action.data;
      const content = data.message?.content ?? draft.content;
      const reasoning = data.message?.reasoning_content ?? draft.reasoning;
      const finished: Message = {
        role: 'assistant',
        content,
        reasoning: reasoning || null,
        stopped: data.stopped || undefined,
        tokens:
          data.tokens ??
          data.usage?.completion_tokens ??
          null,
      };
      const lastTokens =
        data.tokens ?? data.usage?.completion_tokens ?? state.lastTokens;
      return {
        ...state,
        messages: [...state.messages, finished],
        draft: null,
        status: 'idle',
        stopped: Boolean(data.stopped),
        lastTokens: data.stopped ? lastTokens : null,
        usage: data.usage ?? state.usage,
        error: null,
        streamId: null,
      };
    }
    case 'fail': {
      const draft = state.draft;
      const kept: Message[] = [];
      if (draft && (draft.content || draft.reasoning)) {
        // A partial answer survives a failure: it is shown, marked failed.
        kept.push({
          role: 'assistant',
          content: draft.content,
          reasoning: draft.reasoning || null,
          failed: true,
          tokens: action.error.tokensArrived ?? null,
        });
      }
      return {
        ...state,
        messages: [...state.messages, ...kept],
        draft: null,
        status: 'idle',
        error: action.error,
        streamId: null,
      };
    }
    case 'stop':
      if (state.status !== 'streaming') return state;
      return { ...state, status: 'stopping' };
    case 'clear-error':
      return { ...state, error: null };
    default:
      return state;
  }
}

export function errorFromFrame(data: unknown): ChatError {
  const d = (data ?? {}) as Record<string, unknown>;
  const kind = (typeof d.kind === 'string' ? d.kind : 'remote') as ErrorKind;
  return {
    kind,
    message: typeof d.error === 'string' ? d.error : 'stream failed',
    hint: typeof d.hint === 'string' ? d.hint : undefined,
    tokensArrived: typeof d.tokens_arrived === 'number' ? d.tokens_arrived : undefined,
    waitedSeconds: typeof d.waited_seconds === 'number' ? d.waited_seconds : undefined,
  };
}

export interface UseChatOptions {
  /** Injection seam for tests; defaults to the real fetch-based streamer. */
  streamFn?: (request: ChatRequest, onFrame: (frame: Frame) => void) => Promise<void>;
  /** Sampling + thinking for the next turn, read fresh on every send. */
  getParams?: () => Pick<ChatRequest, 'sampling' | 'thinking'>;
  /** Stream ids are generated here so tests can predict them. */
  makeStreamId?: () => string;
}

export interface UseChatResult {
  state: ChatState;
  send: (text: string) => Promise<void>;
  stop: () => void;
  retry: () => Promise<void>;
  clearError: () => void;
}

export function useChat(options: UseChatOptions = {}): UseChatResult {
  const [state, dispatch] = useReducer(chatReducer, initialChatState);
  const stateRef = useRef(state);
  stateRef.current = state;
  const optionsRef = useRef(options);
  optionsRef.current = options;
  const runningRef = useRef(false);

  const runStream = useCallback(
    async (streamId: string, requestMessages: Array<{ role: Message['role']; content: string }>) => {
      const opts = optionsRef.current;
      const streamFn = opts.streamFn ?? defaultStreamChat;
      const params = opts.getParams?.() ?? {};
      const request: ChatRequest = {
        messages: requestMessages,
        stream_id: streamId,
        ...params,
      };
      runningRef.current = true;
      let finalized = false; // set by done/error frames, not by rendered state
      try {
        await streamFn(request, (frame) => {
          switch (frame.event) {
            case 'delta':
              dispatch({ type: 'delta', text: String((frame.data as { text?: string })?.text ?? '') });
              break;
            case 'reasoning':
              dispatch({ type: 'reasoning', text: String((frame.data as { text?: string })?.text ?? '') });
              break;
            case 'usage':
              dispatch({ type: 'usage', usage: (frame.data as { usage: Usage }).usage });
              break;
            case 'done':
              finalized = true;
              dispatch({ type: 'done', data: frame.data as DoneData });
              break;
            case 'error':
              finalized = true;
              dispatch({ type: 'fail', error: errorFromFrame(frame.data) });
              break;
            default:
              break; // tool_call, malformed: out of scope for v0.7.0
          }
        });
        // The server always ends with done or error; if it vanished without one,
        // surface that as a connection failure instead of a silent hang.
        if (!finalized) {
          dispatch({
            type: 'fail',
            error: { kind: 'connection', message: 'the stream ended without a final frame' },
          });
        }
      } catch (err) {
        if (finalized) return; // already finalized; a late abort is not a new failure
        if (err instanceof ApiError) {
          dispatch({
            type: 'fail',
            error: { kind: err.kind, message: err.message, hint: err.hint },
          });
        } else {
          dispatch({
            type: 'fail',
            error: {
              kind: 'connection',
              message: err instanceof Error ? err.message : String(err),
            },
          });
        }
      } finally {
        runningRef.current = false;
      }
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      const current = stateRef.current;
      if (!trimmed || current.status !== 'idle') return;
      const streamId =
        optionsRef.current.makeStreamId?.() ?? `s-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const requestMessages: Array<{ role: Message['role']; content: string }> = [
        ...current.messages
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({ role: m.role, content: m.content })),
        { role: 'user' as const, content: trimmed },
      ];
      dispatch({ type: 'send', text: trimmed, streamId });
      await runStream(streamId, requestMessages);
    },
    [runStream],
  );

  const stop = useCallback(() => {
    const current = stateRef.current;
    if (current.status !== 'streaming' || !current.streamId) return;
    dispatch({ type: 'stop' });
    // Fire-and-forget: /api/stop is idempotent, and the stream itself delivers the
    // terminal `done {stopped:true}` frame that finalizes the turn.
    void postJSON('/api/stop', { stream_id: current.streamId }).catch(() => undefined);
  }, []);

  const retry = useCallback(async () => {
    const current = stateRef.current;
    if (current.status !== 'idle' || !current.lastUserText) return;
    await send(current.lastUserText);
  }, [send]);

  const clearError = useCallback(() => dispatch({ type: 'clear-error' }), []);

  // Safety net: unmounting with a stream in flight stops it upstream too.
  useEffect(() => {
    return () => {
      const current = stateRef.current;
      if (current.status === 'streaming' && current.streamId) {
        void postJSON('/api/stop', { stream_id: current.streamId }).catch(() => undefined);
      }
    };
  }, []);

  return { state, send, stop, retry, clearError };
}
