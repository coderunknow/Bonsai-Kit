/**
 * The conversation surface: messages, the streaming draft, reasoning blocks,
 * stop/retry, and one distinct failure treatment per error kind.
 */

import { useEffect, useRef, useState } from 'react';
import type { ChatError, ChatState } from '../useChat';
import type { Message } from '../types';

export function errorTreatment(error: ChatError): { title: string; detail: string } {
  switch (error.kind) {
    case 'connection':
      return {
        title: 'Connection failed',
        detail:
          'The deployment is probably down. Check the Colab/Kaggle runtime and rerun colab_kaggle_cell.py to get a fresh URL, then Connect again.' +
          (error.message ? ` (${error.message})` : ''),
      };
    case 'unauthorized':
      return {
        title: 'API key rejected (401)',
        detail:
          'The endpoint refused the key. Check the API key field in the connect bar — the deployment cell prints the key exactly once, right when the tunnel starts.' +
          (error.hint ? ` ${error.hint}` : ''),
      };
    case 'remote':
      return {
        title: error.hint ? 'Server error' : 'Server error',
        detail:
          (error.hint ? `${error.hint} ` : '') +
          (error.message || 'The endpoint reported an error.') +
          ' Check the runtime: it may be out of VRAM or restarting.',
      };
    case 'mid_stream':
      return {
        title: 'The stream broke mid-answer',
        detail: `The connection dropped after ${error.tokensArrived ?? 0} tokens. What arrived is kept above — retry to resend your last message.`,
      };
    case 'stall':
      return {
        title: 'The server went quiet',
        detail: `No data arrived for ${error.waitedSeconds ?? 0} seconds, so the turn was abandoned. The deployment may be overloaded or hung — retry, or check the runtime.`,
      };
    case 'busy':
      return {
        title: 'Busy — turn refused',
        detail: 'Another stream is already running (streams are refused, never queued silently). Wait for it to finish, then retry.',
      };
    case 'bad_request':
      return {
        title: 'Request rejected',
        detail: error.message || 'The local server rejected the request.',
      };
    default:
      return { title: 'Error', detail: error.message };
  }
}

interface Props {
  state: ChatState;
  onSend: (text: string) => void;
  onStop: () => void;
  onRetry: () => void;
}

function MessageView({ message }: { message: Message }) {
  const [showReasoning, setShowReasoning] = useState(false);
  return (
    <article
      className={`message message-${message.role}${message.stopped ? ' message-stopped' : ''}${
        message.failed ? ' message-failed' : ''
      }`}
      data-testid={`message-${message.role}`}
    >
      <header>
        <span className="role">{message.role}</span>
        {message.stopped && (
          <span className="stopped-note" data-testid="stopped-note">
            (stopped after {message.tokens ?? '?'} tokens)
          </span>
        )}
        {message.failed && (
          <span className="failed-note" data-testid="failed-note">
            (failed{typeof message.tokens === 'number' ? ` after ${message.tokens} tokens` : ''})
          </span>
        )}
      </header>
      {message.reasoning && (
        <div className="reasoning">
          <button
            type="button"
            className="reasoning-toggle"
            aria-expanded={showReasoning}
            onClick={() => setShowReasoning((v) => !v)}
          >
            {showReasoning ? 'Hide reasoning' : 'Show reasoning'}
          </button>
          {showReasoning && (
            <pre className="reasoning-body" data-testid="reasoning-body">
              {message.reasoning}
            </pre>
          )}
        </div>
      )}
      <div className="content">{message.content}</div>
    </article>
  );
}

export function ChatView({ state, onSend, onStop, onRetry }: Props) {
  const [text, setText] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamRegionRef = useRef<HTMLDivElement>(null);
  const wasStreaming = useRef(false);

  const streaming = state.status !== 'idle';

  // Focus management: land back in the input when a turn ends, and keep the
  // streaming region announced via aria-live.
  useEffect(() => {
    if (wasStreaming.current && !streaming) {
      inputRef.current?.focus();
    }
    wasStreaming.current = streaming;
  }, [streaming]);

  const submit = () => {
    const value = text;
    if (!value.trim() || streaming) return;
    setText('');
    onSend(value);
    inputRef.current?.focus();
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    } else if (event.key === 'Escape' && streaming) {
      event.preventDefault();
      onStop();
    }
  };

  const empty = state.messages.length === 0 && !state.draft;

  return (
    <section className="chat" aria-label="Chat">
      <div
        className="messages"
        ref={streamRegionRef}
        aria-live="polite"
        aria-relevant="additions text"
        data-testid="messages"
      >
        {empty && (
          <p className="empty-state" data-testid="empty-state">
            Connected. Send a message to start a conversation with Ternary Bonsai 2.
          </p>
        )}
        {state.messages.map((message, index) => (
          <MessageView key={index} message={message} />
        ))}
        {state.draft && (
          <article className="message message-assistant message-draft" data-testid="message-draft">
            <header>
              <span className="role">assistant</span>
              {state.status === 'stopping' && <span className="stopped-note">stopping…</span>}
            </header>
            {state.draft.reasoning && (
              <div className="reasoning reasoning-live">
                <span className="reasoning-label">thinking…</span>
                <pre className="reasoning-body" data-testid="draft-reasoning">
                  {state.draft.reasoning}
                </pre>
              </div>
            )}
            <div className="content" data-testid="draft-content">
              {state.draft.content}
              {streaming && <span className="cursor" aria-hidden="true">▍</span>}
            </div>
          </article>
        )}
        {state.error && (
          <div className={`error-card error-${state.error.kind}`} role="alert" data-testid="error-card">
            <strong>{errorTreatment(state.error).title}</strong>
            <p>{errorTreatment(state.error).detail}</p>
            <button type="button" onClick={onRetry} data-testid="retry-button">
              Retry (resend last message)
            </button>
          </div>
        )}
      </div>

      <div className="composer">
        <label htmlFor="message-input" className="visually-hidden">
          Message
        </label>
        <textarea
          id="message-input"
          ref={inputRef}
          rows={3}
          value={text}
          placeholder="Send a message — Enter to send, Shift+Enter for a newline, Esc to stop"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Message"
        />
        {streaming ? (
          <button type="button" className="stop-button" onClick={onStop} data-testid="stop-button">
            {state.status === 'stopping' ? 'Stopping…' : 'Stop'}
          </button>
        ) : (
          <button
            type="button"
            className="send-button"
            onClick={submit}
            disabled={!text.trim()}
            data-testid="send-button"
          >
            Send
          </button>
        )}
      </div>
    </section>
  );
}
