import { useCallback, useEffect, useState } from 'react';
import { ApiError, getJSON, putJSON } from './api';
import { ConnectBar, type ConnectStatus } from './components/ConnectBar';
import { ChatView } from './components/ChatView';
import { SettingsPanel } from './components/SettingsPanel';
import { useChat } from './useChat';
import {
  THINKING_DEFAULTS,
  type Capabilities,
  type ConfigInfo,
  type Sampling,
  type Thinking,
} from './types';
import { UI_VERSION } from './version';

interface VersionPayload {
  version: string;
}

export function App() {
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [serverVersion, setServerVersion] = useState<string | null>(null);
  const [connect, setConnect] = useState<ConnectStatus>({ status: 'disconnected', step: null });
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [sampling, setSampling] = useState<Sampling & Thinking>(THINKING_DEFAULTS);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [showSettings, setShowSettings] = useState(true);

  const chat = useChat({
    getParams: () => ({
      sampling: {
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        min_p: sampling.min_p,
        presence_penalty: sampling.presence_penalty,
        max_tokens: sampling.max_tokens,
      },
      thinking: { effort: sampling.effort, budget_tokens: sampling.budget_tokens },
    }),
  });

  /** health → models → capabilities, in that order, with per-step failures. */
  const runChecks = useCallback(async () => {
    setConnect({ status: 'connecting', step: 'health' });
    let health;
    try {
      health = await getJSON<import('./types').Health>('/api/health');
    } catch (err) {
      const e = err as ApiError;
      setConnect({ status: 'failed', step: 'health', message: e.message, kind: e.kind });
      return false;
    }
    if (!health.ok) {
      setConnect({
        status: 'failed',
        step: 'health',
        message: health.error ?? 'endpoint unreachable',
        kind: health.kind,
      });
      return false;
    }
    setConnect({ status: 'connecting', step: 'models' });
    let modelList: Array<{ id?: string }> = [];
    try {
      const payload = await getJSON<{ data?: Array<{ id?: string }> }>('/api/models');
      modelList = payload.data ?? [];
    } catch (err) {
      const e = err as ApiError;
      setConnect({ status: 'failed', step: 'models', message: e.message, kind: e.kind });
      return false;
    }
    setConnect({ status: 'connecting', step: 'capabilities' });
    let caps: Capabilities;
    try {
      caps = await getJSON<Capabilities>('/api/capabilities');
    } catch (err) {
      const e = err as ApiError;
      setConnect({ status: 'failed', step: 'capabilities', message: e.message, kind: e.kind });
      return false;
    }
    setCapabilities(caps);
    setModels(modelList.map((m) => m.id ?? '').filter(Boolean));
    setConnect({ status: 'connected', step: null, health });
    return true;
  }, []);

  // On load: read the server's config view and version; if the process already
  // holds a key (started with --api-key), connect without asking the browser for it.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const info = await getJSON<ConfigInfo>('/api/config');
        if (cancelled) return;
        setConfig(info);
        const stored = info.settings ?? {};
        setSampling((prev) => ({
          ...prev,
          temperature: typeof stored.temperature === 'number' ? stored.temperature : prev.temperature,
          top_p: typeof stored.top_p === 'number' ? stored.top_p : prev.top_p,
          max_tokens: typeof stored.max_tokens === 'number' ? stored.max_tokens : prev.max_tokens,
        }));
        if (info.endpoint && info.has_api_key) {
          await runChecks();
        }
      } catch {
        /* server unreachable: the connect bar stays for manual entry */
      }
      try {
        const v = await getJSON<VersionPayload>('/api/version');
        if (!cancelled) setServerVersion(v.version);
      } catch {
        /* version is advisory only */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runChecks]);

  const onConnect = useCallback(
    async (endpoint: string, apiKey: string) => {
      setConnect({ status: 'connecting', step: 'config' });
      try {
        // The key is posted once and then dropped from component state. It is
        // never written to localStorage/sessionStorage, URLs, or logs.
        await putJSON('/api/config', { endpoint, api_key: apiKey });
      } catch (err) {
        const e = err as ApiError;
        setConnect({ status: 'failed', step: 'config', message: e.message, kind: e.kind });
        return;
      }
      await runChecks();
    },
    [runChecks],
  );

  const onDisconnect = useCallback(() => {
    setConnect({ status: 'disconnected', step: null });
    setCapabilities(null);
  }, []);

  const saveSettings = useCallback(async () => {
    setSaveState('saving');
    try {
      await putJSON('/api/config', {
        settings: {
          temperature: sampling.temperature,
          top_p: sampling.top_p,
          max_tokens: sampling.max_tokens,
        },
      });
      setSaveState('saved');
    } catch {
      setSaveState('error');
    }
  }, [sampling]);

  const connected = connect.status === 'connected';
  const contextWindow =
    typeof capabilities?.facts?.['context_window'] === 'number'
      ? (capabilities.facts['context_window'] as number)
      : null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Bonsai-Kit</h1>
        <span className="versions" data-testid="versions">
          UI v{UI_VERSION}
          {serverVersion && serverVersion !== UI_VERSION && (
            <span className="version-skew" role="alert">
              {' '}
              — client is v{serverVersion}: reload after the server is updated
            </span>
          )}
          {serverVersion && serverVersion === UI_VERSION && <> · client v{serverVersion}</>}
        </span>
      </header>

      <ConnectBar
        config={config}
        connectStatus={connect}
        onConnect={onConnect}
        onDisconnect={onDisconnect}
      />

      {connected && (
        <>
          <div className="settings-toggle">
            <button
              type="button"
              aria-expanded={showSettings}
              onClick={() => setShowSettings((v) => !v)}
            >
              {showSettings ? 'Hide settings' : 'Show settings'}
            </button>
            {models.length > 0 && (
              <span className="model-badge" data-testid="model-badge">
                model: {connect.health?.model ?? models[0]} (server-published: {models.join(', ')})
              </span>
            )}
          </div>
          {showSettings && (
            <SettingsPanel
              sampling={sampling}
              thinking={sampling}
              capabilities={capabilities}
              contextWindow={contextWindow}
              promptTokens={chat.state.usage?.prompt_tokens ?? null}
              onChange={(next) => setSampling(next)}
              onSave={saveSettings}
              saveState={saveState}
            />
          )}
          <ChatView
            state={chat.state}
            onSend={(text) => void chat.send(text)}
            onStop={chat.stop}
            onRetry={() => void chat.retry()}
          />
        </>
      )}
      {!connected && (
        <main className="disconnected-panel">
          <p data-testid="disconnected-message">
            {connect.status === 'failed'
              ? 'Not connected — fix the problem above and try again.'
              : 'Connect to a Bonsai deployment to start chatting. The API key is kept in the local Python process and never sent to this page.'}
          </p>
        </main>
      )}
    </div>
  );
}
