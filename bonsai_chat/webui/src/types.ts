/** Shared wire types: everything here mirrors what the Python server sends. */

export type Role = 'user' | 'assistant' | 'system' | 'tool';

export interface Message {
  role: Role;
  content: string;
  /** Reasoning trace of an assistant message, separate from the answer. */
  reasoning?: string | null;
  /** The turn was stopped by the user — kept, not presented as complete. */
  stopped?: boolean;
  /** The turn failed mid-stream — kept so the user can see what arrived. */
  failed?: boolean;
  /** Measured token count for this turn (usage, or the server tokenizer). */
  tokens?: number | null;
}

export type ErrorKind =
  | 'connection'
  | 'unauthorized'
  | 'remote'
  | 'mid_stream'
  | 'stall'
  | 'busy'
  | 'bad_request';

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface DoneData {
  kind: 'done';
  stopped: boolean;
  stream_id: string;
  message: { content?: string | null; reasoning_content?: string | null };
  usage: Usage | null;
  finish_reason: string | null;
  chunks?: number;
  /** Stopped turns: measured token count (usage, or server tokenization). */
  tokens?: number | null;
}

export interface ErrorData {
  kind: ErrorKind;
  error: string;
  status?: number;
  hint?: string;
  waited_seconds?: number;
  tokens_arrived?: number;
  partial_text?: string;
  stream_id?: string;
}

export interface Frame {
  event: string;
  data: unknown;
}

export type CapabilityState = 'supported' | 'unsupported' | 'unknown';

export interface Capabilities {
  fields: Record<string, CapabilityState>;
  evidence: Record<string, string>;
  facts: Record<string, unknown>;
}

export interface Health {
  ok: boolean;
  endpoint: string;
  model: string;
  latency_ms: number | null;
  kind?: ErrorKind;
  error?: string;
}

export interface ConfigInfo {
  path: string | null;
  exists: boolean;
  settings: Record<string, unknown>;
  provenance: Record<string, string>;
  endpoint: string;
  model: string;
  has_api_key: boolean;
}

export interface ChatRequest {
  messages: Array<{ role: Role; content: string }>;
  stream_id: string;
  sampling?: Record<string, number>;
  thinking?: { effort?: string; budget_tokens?: number };
}

/** Per-turn sampling; defaults follow the documented thinking/instruct split. */
export interface Sampling {
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  presence_penalty: number;
  max_tokens: number;
}

export interface Thinking {
  effort: 'medium' | 'xhigh';
  /** 0 = thinking off, N = explicit budget, -1 = unlimited. */
  budget_tokens: number;
}

export const THINKING_DEFAULTS: Sampling & Thinking = {
  temperature: 1.0,
  top_p: 0.95,
  top_k: 20,
  min_p: 0.05,
  presence_penalty: 0,
  max_tokens: 2048,
  effort: 'medium',
  budget_tokens: 2048,
};

export const INSTRUCT_PRESET: Sampling & Thinking = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0.0,
  presence_penalty: 1.5,
  max_tokens: 2048,
  effort: 'medium',
  budget_tokens: 0,
};
