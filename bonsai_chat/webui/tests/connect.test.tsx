import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { App } from '../src/App';

interface Call {
  url: string;
  method: string;
  body?: unknown;
}

const HEALTH_OK = {
  ok: true,
  endpoint: 'http://127.0.0.1:9999/v1',
  model: 'ternary-bonsai-2-27b',
  latency_ms: 1.2,
};
const MODELS = { object: 'list', data: [{ id: 'ternary-bonsai-2-27b', object: 'model' }] };
const CAPS = {
  fields: {
    reasoning_effort: 'supported',
    thinking_budget_tokens: 'supported',
    top_k: 'supported',
    min_p: 'supported',
  },
  evidence: {},
  facts: { context_window: 8192, vision: false, vision_source: '/props modalities' },
};

function jsonResponse(payload: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => payload };
}

/** Route-aware fetch mock that records call order. */
function mockFetch(handler: (url: string, init?: RequestInit) => unknown) {
  const calls: Call[] = [];
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, method: init?.method ?? 'GET', body: init?.body });
    return handler(url, init) as ReturnType<typeof jsonResponse>;
  });
  globalThis.fetch = fetchMock as unknown as typeof fetch;
  return { calls, fetchMock };
}

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe('connect flow', () => {
  it('runs health → models → capabilities in order and never persists the key', async () => {
    const { calls } = mockFetch((url) => {
      if (url === '/api/config')
        return jsonResponse({ path: null, exists: false, settings: {}, provenance: {}, endpoint: '', model: 'ternary-bonsai-2-27b', has_api_key: false });
      if (url === '/api/version') return jsonResponse({ version: '0.7.0' });
      if (url === '/api/health') return jsonResponse(HEALTH_OK);
      if (url === '/api/models') return jsonResponse(MODELS);
      if (url === '/api/capabilities') return jsonResponse(CAPS);
      return jsonResponse({}, false, 404);
    });

    render(<App />);
    await waitFor(() => expect(screen.getByTestId('versions')).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText('Endpoint URL'), {
      target: { value: 'http://127.0.0.1:9999/v1' },
    });
    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sekrit-key-123' } });
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() =>
      expect(screen.getByTestId('connect-status')).toHaveTextContent(/connected —/),
    );

    const apiOrder = calls.filter((c) => c.url.startsWith('/api/')).map((c) => c.url);
    const healthAt = apiOrder.indexOf('/api/health');
    const modelsAt = apiOrder.indexOf('/api/models');
    const capsAt = apiOrder.indexOf('/api/capabilities');
    expect(healthAt).toBeGreaterThan(-1);
    expect(healthAt).toBeLessThan(modelsAt);
    expect(modelsAt).toBeLessThan(capsAt);

    // the key went exactly once, into the local server — and left the browser
    const keyPosts = calls.filter(
      (c) => c.url === '/api/config' && String(c.body).includes('sekrit-key-123'),
    );
    expect(keyPosts).toHaveLength(1);
    expect(screen.getByLabelText('API key')).toHaveValue('');
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
    expect(window.location.search).toBe('');

    // capability-driven UI: text-only build disables vision with a reason
    expect(screen.getByRole('button', { name: /attach image/i })).toBeDisabled();
    expect(screen.getByTestId('vision-reason')).toHaveTextContent(/text-only/);
  });

  it('names the failing step when health says the deployment is down', async () => {
    mockFetch((url) => {
      if (url === '/api/config')
        return jsonResponse({ path: null, exists: false, settings: {}, provenance: {}, endpoint: '', model: 'x', has_api_key: false });
      if (url === '/api/version') return jsonResponse({ version: '0.7.0' });
      if (url === '/api/health')
        return jsonResponse({
          ok: false,
          endpoint: '',
          model: 'x',
          latency_ms: null,
          kind: 'connection',
          error: 'cannot reach http://127.0.0.1:1/v1',
        });
      return jsonResponse({}, false, 404);
    });

    render(<App />);
    fireEvent.change(screen.getByLabelText('Endpoint URL'), {
      target: { value: 'http://127.0.0.1:1/v1' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() =>
      expect(screen.getByTestId('connect-status')).toHaveTextContent(/failed at step “health”/),
    );
    expect(screen.getByTestId('connect-status')).toHaveTextContent(/cannot reach/);
    expect(screen.getByTestId('connect-status')).toHaveTextContent(/probably down/);
    expect(screen.getByTestId('disconnected-message')).toBeInTheDocument();
  });

  it('auto-connects from the server-held key without asking the browser for it', async () => {
    const { calls } = mockFetch((url) => {
      if (url === '/api/config')
        return jsonResponse({
          path: '/home/user/.config/bonsai/config.json',
          exists: true,
          settings: {},
          provenance: {},
          endpoint: 'http://127.0.0.1:9999/v1',
          model: 'ternary-bonsai-2-27b',
          has_api_key: true,
        });
      if (url === '/api/version') return jsonResponse({ version: '0.7.0' });
      if (url === '/api/health') return jsonResponse(HEALTH_OK);
      if (url === '/api/models') return jsonResponse(MODELS);
      if (url === '/api/capabilities') return jsonResponse(CAPS);
      return jsonResponse({}, false, 404);
    });

    render(<App />);
    await waitFor(() =>
      expect(screen.getByTestId('connect-status')).toHaveTextContent(/connected —/),
    );
    // no PUT carrying a key: the process already had one
    const keyPosts = calls.filter((c) => c.method === 'PUT');
    expect(keyPosts).toHaveLength(0);
    expect(window.localStorage.length).toBe(0);
  });
});
