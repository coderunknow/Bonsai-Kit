/**
 * Capability-driven settings: every control is gated on what the running build
 * actually supports — `unsupported` disables with a reason, `unknown` stays
 * enabled but says so. Nothing here is hardcoded against the model.
 */

import type { Capabilities, Sampling, Thinking } from '../types';
import { INSTRUCT_PRESET, THINKING_DEFAULTS } from '../types';

interface ControlProps {
  id: string;
  label: string;
  state: 'supported' | 'unsupported' | 'unknown';
  evidence?: string;
  /** 'label' associates by htmlFor; 'span' leaves the child's own name alone. */
  labelElement?: 'label' | 'span';
  children: React.ReactNode;
}

function Control({ id, label, state, evidence, labelElement = 'label', children }: ControlProps) {
  const disabled = state === 'unsupported';
  const reason =
    state === 'unsupported'
      ? evidence || 'this build does not support this control'
      : state === 'unknown'
        ? 'not tested against this build — unknown'
        : '';
  return (
    <div className={`control control-${state}`}>
      {labelElement === 'label' ? (
        <label htmlFor={id}>
          {label}
          <span className={`cap-state cap-${state}`} data-testid={`${id}-state`}>
            {state}
          </span>
        </label>
      ) : (
        <span className="control-label">
          {label}
          <span className={`cap-state cap-${state}`} data-testid={`${id}-state`}>
            {state}
          </span>
        </span>
      )}
      {children}
      {state !== 'supported' && (
        <p className="control-reason" data-testid={`${id}-reason`}>
          {reason}
        </p>
      )}
      {disabled && (
        <span className="visually-hidden" id={`${id}-disabled-note`}>
          disabled: {reason}
        </span>
      )}
    </div>
  );
}

interface Props {
  sampling: Sampling;
  thinking: Thinking;
  capabilities: Capabilities | null;
  contextWindow: number | null;
  promptTokens: number | null;
  onChange: (next: Sampling & Thinking) => void;
  onSave: () => void;
  saveState: 'idle' | 'saving' | 'saved' | 'error';
}

function stateOf(caps: Capabilities | null, field: string): 'supported' | 'unsupported' | 'unknown' {
  return caps?.fields?.[field] ?? 'unknown';
}

export function SettingsPanel({
  sampling,
  thinking,
  capabilities,
  contextWindow,
  promptTokens,
  onChange,
  onSave,
  saveState,
}: Props) {
  const set = (patch: Partial<Sampling & Thinking>) => onChange({ ...sampling, ...thinking, ...patch });

  const visionFact = capabilities?.facts?.['vision'];
  const visionState: 'supported' | 'unsupported' | 'unknown' =
    visionFact === true ? 'supported' : visionFact === false ? 'unsupported' : 'unknown';
  const visionEvidence =
    (capabilities?.facts?.['vision_source'] as string | undefined) ??
    (visionState === 'unsupported' ? 'the running build rejects image input' : undefined);

  const num = (value: number, min?: number, max?: number) => ({
    value: String(value),
    min,
    max,
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
      const parsed = Number(e.target.value);
      if (!Number.isNaN(parsed)) set({ [e.target.id]: parsed } as Partial<Sampling & Thinking>);
    },
  });

  return (
    <section className="settings" aria-label="Settings">
      <div className="settings-row presets">
        <span>Presets:</span>
        <button type="button" onClick={() => onChange({ ...THINKING_DEFAULTS })}>
          Thinking
        </button>
        <button type="button" onClick={() => onChange({ ...INSTRUCT_PRESET })}>
          Instruct
        </button>
        <span className="context-info" data-testid="context-window">
          context:{' '}
          {contextWindow !== null ? `${contextWindow} tokens` : 'unknown'}
          {promptTokens !== null && contextWindow !== null && (
            <> · last prompt: {promptTokens} measured tokens</>
          )}
        </span>
      </div>

      <fieldset>
        <legend>Sampling</legend>
        <Control id="temperature" label="temperature" state="supported">
          <input id="temperature" type="number" step="0.05" {...num(sampling.temperature)} />
        </Control>
        <Control id="top_p" label="top_p" state="supported">
          <input id="top_p" type="number" step="0.01" {...num(sampling.top_p, 0, 1)} />
        </Control>
        <Control id="top_k" label="top_k" state={stateOf(capabilities, 'top_k')}>
          <input id="top_k" type="number" step="1" {...num(sampling.top_k, 0)} />
        </Control>
        <Control id="min_p" label="min_p" state={stateOf(capabilities, 'min_p')}>
          <input id="min_p" type="number" step="0.01" {...num(sampling.min_p, 0, 1)} />
        </Control>
        <Control id="presence_penalty" label="presence_penalty" state="supported">
          <input
            id="presence_penalty"
            type="number"
            step="0.1"
            {...num(sampling.presence_penalty)}
          />
        </Control>
        <Control id="max_tokens" label="max_tokens" state="supported">
          <input id="max_tokens" type="number" step="64" {...num(sampling.max_tokens, 1)} />
        </Control>
      </fieldset>

      <fieldset>
        <legend>Reasoning</legend>
        <Control
          id="effort"
          label="reasoning_effort"
          state={stateOf(capabilities, 'reasoning_effort')}
          evidence={capabilities?.evidence?.['reasoning_effort']}
        >
          <select
            id="effort"
            value={thinking.effort}
            disabled={stateOf(capabilities, 'reasoning_effort') === 'unsupported'}
            onChange={(e) => set({ effort: e.target.value as Thinking['effort'] })}
          >
            <option value="medium">medium</option>
            <option value="xhigh">xhigh</option>
          </select>
        </Control>
        <Control
          id="budget_tokens"
          label="thinking_budget_tokens"
          state={stateOf(capabilities, 'thinking_budget_tokens')}
          evidence={capabilities?.evidence?.['thinking_budget_tokens']}
        >
          <select
            id="budget_tokens"
            value={
              thinking.budget_tokens === 0 || thinking.budget_tokens === -1
                ? String(thinking.budget_tokens)
                : 'custom'
            }
            disabled={stateOf(capabilities, 'thinking_budget_tokens') === 'unsupported'}
            onChange={(e) => {
              const v = e.target.value;
              if (v === 'custom') return;
              set({ budget_tokens: Number(v) });
            }}
          >
            <option value="0">0 — thinking off</option>
            <option value="-1">-1 — unlimited</option>
            <option value="custom">N — custom…</option>
          </select>
          {!(thinking.budget_tokens === 0 || thinking.budget_tokens === -1) && (
            <input
              id="budget_tokens_number"
              aria-label="custom thinking budget in tokens"
              type="number"
              min={1}
              value={thinking.budget_tokens}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!Number.isNaN(v) && v > 0) set({ budget_tokens: v });
              }}
            />
          )}
        </Control>
        <Control
          id="vision"
          label="image input (vision)"
          labelElement="span"
          state={visionState}
          evidence={
            visionState === 'unsupported'
              ? `this build is text-only — ${visionEvidence ?? 'image input is refused'}`
              : visionState === 'unknown'
                ? 'whether this build accepts images is unknown'
                : 'supported by the build'
          }
        >
          <button id="vision" type="button" disabled title="image input ships in v0.7.1">
            Attach image…
          </button>
          <p className="control-reason control-scope">
            image input itself ships in v0.7.1
          </p>
        </Control>
      </fieldset>

      <div className="settings-row">
        <button type="button" onClick={onSave} disabled={saveState === 'saving'}>
          {saveState === 'saving' ? 'Saving…' : 'Save to config'}
        </button>
        <span role="status" aria-live="polite" data-testid="save-state">
          {saveState === 'saved' && 'saved to the CLI config file (0600)'}
          {saveState === 'error' && 'save failed'}
        </span>
      </div>
    </section>
  );
}
