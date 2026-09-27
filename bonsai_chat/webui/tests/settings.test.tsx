import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { SettingsPanel } from '../src/components/SettingsPanel';
import { THINKING_DEFAULTS } from '../src/types';
import type { Capabilities } from '../src/types';

const baseCaps: Capabilities = {
  fields: {
    reasoning_effort: 'supported',
    thinking_budget_tokens: 'supported',
    top_k: 'supported',
    min_p: 'supported',
    tools: 'unknown',
    tool_choice: 'unknown',
    response_format: 'unknown',
    stream_options: 'supported',
  },
  evidence: {},
  facts: { context_window: 8192, vision: true, vision_source: '/props modalities' },
};

function renderPanel(over: Partial<Capabilities> = {}, props: Record<string, unknown> = {}) {
  const onChange = vi.fn();
  const onSave = vi.fn();
  render(
    <SettingsPanel
      sampling={THINKING_DEFAULTS}
      thinking={THINKING_DEFAULTS}
      capabilities={{ ...baseCaps, ...over }}
      contextWindow={8192}
      promptTokens={42}
      onChange={onChange}
      onSave={onSave}
      saveState="idle"
      {...props}
    />,
  );
  return { onChange, onSave };
}

describe('settings panel — capability-driven disabling', () => {
  it('disables reasoning_effort with the evidence as the reason when unsupported', () => {
    renderPanel({
      fields: { ...baseCaps.fields, reasoning_effort: 'unsupported' },
      evidence: { reasoning_effort: "HTTP 400: unknown field 'reasoning_effort'" },
    });
    const select = screen.getByLabelText(/reasoning_effort/);
    expect(select).toBeDisabled();
    expect(screen.getByTestId('effort-reason')).toHaveTextContent(
      "HTTP 400: unknown field 'reasoning_effort'",
    );
    expect(screen.getByTestId('effort-state')).toHaveTextContent('unsupported');
  });

  it('disables thinking_budget_tokens with a reason when unsupported', () => {
    renderPanel({
      fields: { ...baseCaps.fields, thinking_budget_tokens: 'unsupported' },
      evidence: { thinking_budget_tokens: "HTTP 400: invalid value for 'thinking_budget_tokens'" },
    });
    const select = screen.getByLabelText(/thinking_budget_tokens/);
    expect(select).toBeDisabled();
    expect(screen.getByTestId('budget_tokens-reason')).toHaveTextContent('thinking_budget_tokens');
  });

  it('disables vision with a text-only reason when the build reports vision=false', () => {
    renderPanel({ facts: { ...baseCaps.facts, vision: false } });
    const button = screen.getByRole('button', { name: /attach image/i });
    expect(button).toBeDisabled();
    expect(screen.getByTestId('vision-reason')).toHaveTextContent(/text-only/);
    expect(screen.getByTestId('vision-state')).toHaveTextContent('unsupported');
  });

  it('leaves an unknown capability enabled but honestly labeled', () => {
    renderPanel({
      fields: { ...baseCaps.fields, reasoning_effort: 'unknown' },
    });
    const select = screen.getByLabelText(/reasoning_effort/);
    expect(select).not.toBeDisabled();
    expect(screen.getByTestId('effort-state')).toHaveTextContent('unknown');
    expect(screen.getByTestId('effort-reason')).toHaveTextContent(/unknown/);
  });

  it('shows the measured context window and reports edits upward', () => {
    const { onChange } = renderPanel();
    expect(screen.getByTestId('context-window')).toHaveTextContent('8192 tokens');
    expect(screen.getByTestId('context-window')).toHaveTextContent('42 measured tokens');
    fireEvent.change(screen.getByLabelText(/^temperature/), { target: { value: '0.7' } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ temperature: 0.7 }));
  });
});
