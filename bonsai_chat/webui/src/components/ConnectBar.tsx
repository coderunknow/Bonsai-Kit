import { useEffect, useState } from 'react';
import type { ConfigInfo, Health } from '../types';

export interface ConnectStatus {
  status: 'disconnected' | 'connecting' | 'connected' | 'failed';
  step: 'config' | 'health' | 'models' | 'capabilities' | null;
  message?: string;
  kind?: string;
  health?: Health;
}

export function stepLabel(step: ConnectStatus['step']): string {
  switch (step) {
    case 'config':
      return 'sending connection to the local server…';
    case 'health':
      return 'checking /api/health…';
    case 'models':
      return 'checking /api/models…';
    case 'capabilities':
      return 'checking /api/capabilities…';
    default:
      return '';
  }
}

/** Kind-specific guidance, one treatment per error class. */
export function connectHint(kind: string | undefined): string {
  switch (kind) {
    case 'connection':
      return ' The deployment is probably down — check the Colab/Kaggle runtime and rerun colab_kaggle_cell.py for a fresh URL.';
    case 'unauthorized':
      return ' The key was rejected — check the API key field. The cell prints the key exactly once.';
    case 'remote':
      return ' The server answered with an error — see the message above.';
    default:
      return '';
  }
}

interface Props {
  config: ConfigInfo | null;
  connectStatus: ConnectStatus;
  onConnect: (endpoint: string, apiKey: string) => void;
  onDisconnect: () => void;
}

export function ConnectBar({ config, connectStatus, onConnect, onDisconnect }: Props) {
  const [endpoint, setEndpoint] = useState('');
  const [apiKey, setApiKey] = useState('');

  // Prefill the endpoint from the server's config view (never the key: it is not
  // stored anywhere the browser can read it back from).
  useEffect(() => {
    if (config?.endpoint) setEndpoint((prev) => prev || config.endpoint);
  }, [config?.endpoint]);

  const busy = connectStatus.status === 'connecting';
  const connected = connectStatus.status === 'connected';

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (busy || !endpoint.trim()) return;
    onConnect(endpoint.trim(), apiKey);
    setApiKey(''); // the key lives in the Python process from here on
  };

  return (
    <form className="connect-bar" onSubmit={submit} aria-label="Connect to a deployment">
      <label htmlFor="endpoint">Endpoint URL</label>
      <input
        id="endpoint"
        name="endpoint"
        type="url"
        value={endpoint}
        placeholder="https://<tunnel-host>/v1"
        autoComplete="off"
        spellCheck={false}
        disabled={busy || connected}
        onChange={(e) => setEndpoint(e.target.value)}
      />
      <label htmlFor="api-key">API key</label>
      <input
        id="api-key"
        name="api-key"
        type="password"
        value={apiKey}
        placeholder={config?.has_api_key ? '••• (stored server-side)' : 'key from the cell'}
        autoComplete="off"
        disabled={busy || connected}
        onChange={(e) => setApiKey(e.target.value)}
      />
      {connected ? (
        <button type="button" onClick={onDisconnect}>
          Disconnect
        </button>
      ) : (
        <button type="submit" disabled={busy || !endpoint.trim()}>
          Connect
        </button>
      )}
      <span
        className={`connect-status connect-${connectStatus.status}`}
        role="status"
        aria-live="polite"
        data-testid="connect-status"
      >
        {connectStatus.status === 'connecting' && stepLabel(connectStatus.step)}
        {connectStatus.status === 'connected' && connectStatus.health && (
          <>
            connected — {connectStatus.health.model} @ {connectStatus.health.endpoint}
            {connectStatus.health.latency_ms !== null && (
              <> ({connectStatus.health.latency_ms} ms)</>
            )}
          </>
        )}
        {connectStatus.status === 'failed' && (
          <>
            failed at step “{connectStatus.step}”: {connectStatus.message}
            {connectHint(connectStatus.kind)}
          </>
        )}
        {connectStatus.status === 'disconnected' && 'disconnected'}
      </span>
    </form>
  );
}
