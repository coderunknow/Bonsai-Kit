# Changelog

## v0.6.0 — 2026-09-27

Two headline goals, in priority order: **(1) stability** — make the GPU deployment path
survivable and testable; **(2) feature breadth** — add the missing capability set end to
end.

The model lock, runtime lock, GPU policy, localhost binding, bearer authentication,
session schema 2 and the `--doctor --json` shape are all unchanged. Everything in this
release is additive except two bug fixes, both found by the new harness.

### Stability

- **The deployment path is executed, not just read.** `cell_harness.py` runs the cell's
  real module-level body offline against fakes: `nvidia-smi` with real per-process VRAM
  attribution, a CUDA driver compiled from C and driven through the real `ctypes` path, a
  fake PrismML `llama-server` **executable** (a compiled launcher owns the PID so
  `/proc/<pid>/exe` and `argv[0]` really are `bin/cuda/llama-server`, which is exactly
  what the cell's process-identity checks verify), a fake Hugging Face layer with a
  synthetic 7.21 GB sparse GGUF whose metadata satisfies the identity checks, and a fake
  `cloudflared` whose published URL is backed by a real loopback proxy so public requests
  are real HTTP. 73 end-to-end tests; a full deployment reaches READY in ~1 s.
  The cell gains injectable seams for this — every default is still the real thing.
- **Supervision.** Two independent watchdogs (inference server, tunnel) with health-gated
  liveness rather than "is the PID alive", a generation probe that detects a hang by a
  stall timeout, and restarts bounded in count (3) and rate (per 15 min) with backoff.
  `classify_exit()` tells crash, OOM and hang apart from the server's own log because the
  repair differs: an OOM steps the context down the ladder, a crash restarts where it was.
  A model restart keeps the port, the API key, `state.json` and the tunnel URL, and never
  re-downloads the model; a tunnel restart leaves the server alone. Neither can cascade.
  Past the budget, supervision stops and says so. Every heal is recorded in
  `diagnostics.json`; `heartbeat.json` is rewritten every 60 s.
- **Session longevity.** Rerun reattaches to a verified server (already in v0.5.0) and
  now does so *without* re-downloading the runtime.

### Bug fixes found by the harness

- `plan_to_args()` emitted value-less flags. With `--parallel` advertised, the server was
  launched as `... --parallel -c 8192`, so `-c` was consumed as the flag's value and the
  server refused to start. Only entries carrying a value are emitted now.
- A rerun that adopted a live server re-ran `scripts/download_binaries.sh`, overwriting
  the executable the running server was mapped from (`ETXTBSY`). The runtime is now left
  alone when a verified server is already up.
- Regression tests for both (plus one for a GGUF tensor offset above 4 GiB, which the
  fixture had packed as u32).

### Features — deployment

- **Vision** (`BONSAI_VISION=1`, opt-in): the official `mmproj` from the same repo, same
  SHA-256/size discipline, VRAM accounted before the server starts, `--image-max-tokens`
  reserved from the context budget, `--mmproj` passed only when the build advertises it.
  Refused rather than silently dropped when the build lacks the flag, when the repo
  publishes no projector, or when the fitted context cannot hold weights + projector +
  the image reserve.
- **Multi-slot** (`BONSAI_SLOTS=N`, opt-in): `--parallel N` when advertised, per-slot
  context = context // slots because each slot carries its own KV cache, and the cost
  stated in the output. Single slot stays the default.
- **KV4** (`BONSAI_KV4=1`, opt-in): `--cache-type-k q4_0` when advertised, labelled as a
  memory lever that costs decode speed. Unavailable is reported, never faked.
- `/slots` is polled and reported as `slots_state`, or omitted when the runtime does not
  expose it.

### Features — client

- **Config file + personas** (`config.py`): `--config`, `--save-config`, `--persona`,
  `--list-personas`, `/config`, `/persona`. Precedence, lowest to highest: defaults →
  file → persona → environment → the flags you typed. `explicit_args()` re-parses the
  parser with `argparse.SUPPRESS` defaults, because a default-filled flag would otherwise
  beat the file. `/config` prints where every value came from.
- **Conversation branching** (`branches.py`): `/fork`, `/branches`, `/branch`,
  `/branch del`, `/branch rename`, `--branch`, `--fork`, `--list-branches`. Stored as an
  extra meta line inside the existing session file: schema 2 is unchanged and still loads
  everywhere, because the line is skipped by readers that do not know it.
- **Multi-endpoint** (`endpoints.py`): named endpoints with `api_key_env` (secrets stay in
  the environment), each with its own client, capability map and token cache.
- **MCP client** (`mcp.py`): JSON-RPC 2.0 over stdio, stdlib only, the same `mcpServers`
  shape Claude Desktop uses. Remote tools register into the existing `ToolRegistry` —
  same approval, timeout and output caps, no second tool-calling mechanism.
- **Tools start mid-stream**: `ChatStream` announces a tool call the moment its arguments
  are complete JSON, and the agent runs it on a worker while the model finishes writing
  the turn. `/pretools on|off`.
- **Structured output** (`--json-schema`, `/schema`): proved with a real request before
  use. Honoured, rejected (400) and *ignored* (accepted, then prose) are three outcomes
  and only the first is support; the capability map is corrected to match.
- **Budgets** (`budget.py`): `--budget-tokens`, `--budget-turns`,
  `--budget-prompt-tokens`, `--price-per-mtok`, `/spend`. Counted only from what the
  server reported; cost stays `n/a` until you supply a price.
- **Batch mode** (`--batch`, `--out`): the same client, retry policy and tool loop, one
  conversation per item, a per-item error instead of a crash.
- **Export** (`--export`, `/export`): JSON and self-contained HTML — no CDN, no external
  resources, content escaped.
- **Importable API** (`bonsai_chat.api`): `Bonsai(...)` with `.ask()`, `.stream()`,
  `.tool()`, `.save()`, `.doctor()`; `run_batch()` / `read_batch()` / `write_batch()`.
- **No second web UI.** The PrismML runtime already serves one at the server root; a
  second would be a second renderer and a second auth surface for no new capability.

### Verification

- Tests: **235 → 375**, all offline and deterministic.
  `test_cell_deployment.py` (73), `test_power_features.py` (67), plus the existing
  `test_chat_client` (82), `test_streaming_recovery` (94) and `test_deployment` (59).
- `--selftest`: **20 → 31** checks.
- `mock_bonsai_server.py`: 26 → **30** scenarios (`structured-json`,
  `ignore-response-format`, `reject-response-format`, `early-tool-call`).
- Not verified: the GPU path has still never run against real Bonsai 2 weights on real
  hardware. See the README's "What has not been verified".

## v0.5.0 — 2026-09-27

A substantial release focused on streaming reliability, thinking control, latency,
hardware autotuning and failure recovery. The model lock, runtime lock, GPU policy,
localhost binding and bearer authentication are unchanged.

### Streaming

- New `SSEDecoder`: an incremental parser that survives fragmented network reads, LF and
  CRLF, `:` comment keep-alives, multi-line `data:` fields, unknown SSE fields, and a
  truncated final event. Exactly one leading space is stripped after `data:`, per spec.
- New `ChatStream`: one object owns the HTTP response, the decoder and the accumulator.
  It is a context manager, so a finished, cancelled or abandoned turn always closes the
  response deterministically instead of leaving it to the garbage collector.
- Output coalescing: the renderer buffers and flushes at 24 characters or ~30 Hz instead
  of calling `flush()` per token. Measured on 50 one-character writes: **3** flushes
  instead of 50.
- Streams that end without a usage chunk, without a `finish_reason`, or without a
  `data: [DONE]` sentinel now complete instead of losing the turn.
- A stream that dies mid-generation raises `StreamInterrupted` carrying the partial text.
  The partial answer is shown, reported as partial, and **not** written to history.
- Tool calls are reconstructed from arbitrarily fragmented chunks: split function names,
  split JSON strings, split multi-byte characters, and multiple interleaved calls.
  Incomplete arguments are never executed (`arguments_complete()`).

### Thinking control

- New `/think off|low|medium|high|max` (or an explicit token count) maps 1:1 onto the
  runtime's real `thinking_budget_tokens` request field: 0 / 512 / 2048 / 8192 / unlimited.
  This is the same mechanism the official Bonsai chat UI's picker uses.
- `/effort` now sends `reasoning_effort` with the values the Bonsai 2 chat template
  actually accepts (`medium`, `xhigh`). `low` is documented by PrismML as **not** reducing
  thinking, so it is no longer offered as a speed knob; `high` is accepted as an alias for
  `xhigh` rather than being sent verbatim and ignored.
- `/reasoning full|compact|hidden` controls how the thinking trace is displayed;
  `compact` shows one indicator plus a per-turn reasoning token count.
- Thinking changes take effect on the next request — no server restart, no history reset.
- New `CapabilityMap`: every field is `supported` / `unsupported` / `unknown`, with the
  evidence. A field is only marked unsupported after a real HTTP 400 naming it, and only
  then is it dropped. `unknown` means "send it and find out", never "silently omit it".

### Performance

- Persistent HTTP/1.1 connection with keep-alive and transparent reconnect. Every request
  previously opened a fresh TCP (and, through a tunnel, TLS) connection. Measured on
  loopback in this sandbox, 40 requests each: plain HTTP median 0.57 → 0.27 ms/request;
  TLS median 2.39 → 0.44 ms/request (5.4×). Through a Cloudflare tunnel the saving is
  larger because the handshake is a WAN round trip — **that has not been measured**.
- `/props` is fetched once per session and answers context size, vision modality, runtime
  version and slot count. Token counts stay memoized per unique string.
- Streaming reads use `read1()`. `read(n)` blocks until the buffer is full, which added a
  whole round trip of apparent latency to the end of every stream.
- Retry policy is now explicit about safety. A chat POST whose bytes left the process is
  **never** replayed; the user gets a clear message instead of a duplicated generation.
  GETs and provably-unsent requests are retried with bounded backoff.
- Idle pooled connections are dropped after 3 s, shorter than the server's keep-alive
  window, which avoids the ambiguous "stale socket, did my POST land?" case almost entirely.

### Deployment / hardware autotuning

- `llama-server --help` is parsed to discover which flags this runtime actually supports;
  nothing is passed blind. Tunables: flash attention, ubatch, batch, threads,
  batch-threads, KV cache type, context checkpoints, idle-slot cache, cache RAM,
  reasoning budget, parallel slots.
- Context is chosen from a **measured** per-token KV cost. After the first server start
  the cell reads the KV and compute buffer sizes llama.cpp printed, derives MiB/token, and
  picks the largest tier that provably fits — then restarts once at the larger value if
  the measurement supports it. Bonsai 2 is a hybrid-attention model (~75% linear layers),
  so a fixed bytes-per-token constant would badly underestimate what fits.
- OOM recovery walks a bounded, strictly decreasing context ladder instead of halving
  once. It cannot loop, and it records every step it took.
- Bounded autotuning: at most two extra candidates (larger/smaller ubatch), each a real
  server start and a real request. Unstable candidates are rejected; the winner is cached
  in `tuning.json` against a key of GPU signature + runtime stamp + model + version, so a
  rerun on unchanged hardware does not repeat it. `BONSAI_TUNE=0` opts out.
- Cloudflare tunnel is treated as disposable: `tunnel_status()` classifies
  healthy / dead-process / edge-error (502/503/52x) / auth-error / unreachable, and the
  tunnel is restarted on its own without ever touching the model. Reuse requires the
  stored PID to still be *our* verified `cloudflared` (PIDs are recycled).
- Structured failures (`DeploymentError`, `describe_failure`): what failed, likely cause,
  what was preserved, what the cell did automatically, what the user must do. No more
  60-frame tracebacks for ordinary operational problems.
- `diagnostics.json` is written next to the server logs: model, packing, runtime stamp,
  context, VRAM, RAM, server PID, tunnel state, benchmark, API test results, version.

### Client UX and context

- New commands: `/think`, `/effort`, `/reasoning`, `/stats`, `/speed`, `/compact`,
  `/context`, `/cancel`, `/caps`, `/doctor`, `/bench`. All existing commands still work.
- Context budget now reserves room for the chat template and for the answer being
  generated, so a request can no longer be sent that exactly fills the window.
- `/compact` folds old turns into a system-visible digest instead of dropping them; the
  system message, the last N turns and every tool-call/tool-result pair are preserved, and
  the user is told exactly how many tokens were reclaimed.
- `validate_wire()` checks the outgoing message list for orphaned tool results, unanswered
  tool calls and empty assistant messages before the request is sent.
- Session files carry a schema marker and version, are written atomically with `fsync`,
  and are `0600`. A truncated last line, a garbage line, or a v0.3/v0.4 file without the
  marker all load; bad lines are skipped and reported instead of crashing the client.
  Orphaned tool results in a file are repaired on load.
- `/usage` and `/stats` report measured medians with sample counts; `sig()` rounds to
  three significant figures so a three-sample measurement is never printed as
  `147.123456 tok/s`.

### API compatibility

- `--doctor` gained `UNKNOWN` and `DEGRADED` states and a `--json` mode, and prints the
  whole capability map with evidence. `PASS` is only printed for behaviour that was
  actually exercised against the endpoint.
- `--benchmark` measures cold start, TTFT, prefill speed, decode speed and total latency
  over N samples after discarding warmup, and reports median/min/max/p95. VRAM and RAM are
  reported as not measurable from the client rather than guessed.
- `--json` one-shot output now contains exactly one JSON document, including stats,
  reasoning config and the capability map.

### Mock server

- `mock_bonsai_server.py` can now reproduce 26 deterministic failure modes: slow first
  token, slow stream, fragmented SSE, CRLF, keep-alive comments, multi-line data,
  malformed SSE, partial tool calls, parallel tool calls, no usage, no finish reason, no
  `[DONE]`, reasoning-only, interleaved reasoning, mid-stream disconnect, timeout,
  invalid JSON, 429/500/502/503, context overflow, and per-field rejection.
- Fixed two real bugs in the stub that connection reuse exposed: the chunked body was
  terminated by sending `0\r\n\r\n` *inside* a chunk (a client reading by size blocked
  forever), and a 401 was returned without draining the request body (the next request
  line parsed was the tail of the previous body).

### Tests

- 97 → **235** tests, all passing offline. New `test_streaming_recovery.py` (94 tests)
  covers SSE fragmentation, tool-call reconstruction, cancellation, mid-stream disconnect,
  retry safety, transport pooling, reasoning control, context compaction, session
  corruption, tool budgets, diagnostics and benchmarking. `test_deployment.py` grew from
  15 to 59 tests, covering KV buffer parsing, context selection, the OOM ladder, flag
  planning, tuning-cache invalidation, tunnel classification, PID-recycling rejection and
  structured failures.
- Two pre-existing assertions were updated, both because behaviour intentionally changed
  and both with the reason recorded in the test: session files now carry a `_meta` line,
  and `/effort high` now normalises to `xhigh`. No test was deleted.

### Breaking changes

- `Settings` is a plain class rather than a dataclass; `Settings(effort=…)` and
  `Settings(think=…)` still work, but `dataclasses.asdict()` on it no longer does.
- `Agent.run()` no longer writes a `[cancelled]` placeholder assistant message. A
  cancelled or interrupted turn leaves the history exactly as it was before the turn.
- `Conversation.save()` writes one extra leading `_meta` line.
- `--effort` accepts `none|medium|xhigh` (plus aliases) rather than `none|low|medium|high`.

### Not validated here

No GPU notebook runtime was available, so end-to-end inference against the real 27B model,
the measured-context refinement, the autotune and the OOM ladder were not executed on real
hardware. They are exercised offline against fakes and stubs; the numbers above are
loopback transport measurements and scripted-mock figures, not model throughput.

## v0.3.0 — 2026-09-26

### Added
- **`bonsai_chat.py` — a real chat client for the deployed API** (standard library only;
  `pillow`, `pygments` and `pytesseract` are used when present and skipped when not):
  - interactive multi-turn loop with slash commands (`/help`, `/system`, `/undo`, `/retry`,
    `/reset`, `/image`, `/tools`, `/temp`, `/topp`, `/max-tokens`, `/effort`, `/stream`,
    `/markdown`, `/vision`, `/context`, `/history`, `/usage`, `/save`, `/load`, `/export`),
    multi-line input, Ctrl-C that cancels only the current turn, and JSONL session
    autosave/resume plus Markdown transcript export
  - streaming with the model's `reasoning_content` rendered as a separate dim block and
    per-turn tokens / tok/s / TTFT taken from the server's own usage and timings
  - terminal Markdown rendering: headings, nested and task lists, blockquotes, GFM tables,
    fenced code (Pygments-highlighted when available), inline styles and links; automatic
    plain output when stdout is piped, `--plain` / `--color` to override
  - image handling: PNG/JPEG/GIF/WebP/BMP headers parsed without third-party code,
    Pillow statistics and optional tesseract OCR when installed, downscaling before upload,
    and a `modalities`-aware vision check (`/props`) with a 1x1 PNG probe fallback. Pixels are
    sent only if the server actually has a vision projector; on the default text-only
    deployment the client sends the measured facts and says so explicitly
  - tool calling: sandboxed `read_file`, `list_dir`, `search_text`, `write_file`,
    `run_shell`, `http_get`, `calculator`, `current_time`, `image_inspect`, a decorator for
    registering your own, parallel tool calls, a bounded tool-result loop, and an
    approve/deny/always gate for anything that writes or executes
  - `--doctor` live endpoint diagnostics (health, models, context, auth, chat, streaming,
    native tool calling, vision) and `--selftest` / `--mock` offline modes
  - one-shot (`-p`), piped-stdin and `--json` modes for scripting
- **`mock_bonsai_server.py`** — an offline stub of the PrismML llama.cpp endpoint (SSE with
  `reasoning_content`, tool-call fragments split across chunks, `/props`, `/tokenize`,
  `/health`, bearer auth, text-only image rejection). Scripted replies only: it is not a
  model and runs no inference.
- **`test_chat_client.py`** — 82 offline tests that drive the shipped client over real HTTP
  against that stub: transport/retries, SSE parsing, tool loop and approval, sandbox escape
  refusal, image headers and cards, Markdown rendering, history trimming, REPL commands,
  CLI modes and `--doctor`.
- The deployment cell now prints the chat-client quick start next to the curl example.

### Fixed
- **llama.cpp-native routes were requested under `/v1`**: `/props`, `/tokenize` and
  `/health` live at the server root (only `/v1/*` is OpenAI-compatible), so context
  detection and exact token counting silently 404'd and fell back to defaults/heuristics.
  The client now keeps a separate root URL, matching what `colab_kaggle_cell.py` does for
  `/health`.
- Context window and token counts are cached per session; previously every turn re-fetched
  `/props` and re-tokenized the entire history (one round trip per message per turn).
- History trimming drops whole turns, so a `tool` message can never be sent without the
  `tool_calls` it answers — that combination makes llama.cpp reject the request.
- Attached-image cards were printed twice (once by `/image`, again when the turn was sent).

## v0.2.4 — 2026-09-26

### Fixed
- **Cloudflare Quick Tunnel verification DNS retry**: Remote tunnel API verification in phase [8/8]
  now polls with retries (`verify_tunnel_connectivity`) to accommodate Cloudflare Quick Tunnel
  DNS propagation delay and edge routing initialization. Previously, a single immediate HTTP request
  was executed right after extracting the `trycloudflare.com` URL from the log, failing with
  `RuntimeError: Remote tunnel API verification failed (None: URLError: <urlopen error [Errno -2] Name or service not known>)`.
  The retry loop waits across initial DNS resolution errors (`[Errno -2] Name or service not known`)
  and edge warmup HTTP responses (502/503/52x) until the OpenAI endpoint returns HTTP 200 with the
  verified model alias, or times out cleanly after 90 seconds.
- **Tunnel chat verification resilience**: End-to-end chat completion through the public tunnel
  retries up to 3 times to absorb transient socket drops during Cloudflare edge warmup.
- **Tunnel lifecycle & binary safety**: Terminate stale or unresponsive tunnel processes before
  starting a new tunnel. Download `cloudflared` atomically via a temporary file with size
  validation (≥ 10 MB) to prevent executing corrupted or partial binaries from interrupted downloads.
- **Python 3.13 deprecation warning**: Replaced positional `maxsplit` argument in
  `re.split(r'\s{2,}', s, maxsplit=1)` with keyword argument via top-level `parse_supported_flags()`,
  eliminating `DeprecationWarning: 'maxsplit' is passed as positional argument`.
- **Benchmark usage resilience**: Safely access streaming usage metrics and timings with
  fallbacks (`.get()`) to prevent `KeyError` or `TypeError` if streaming options or usage
  statistics are omitted by the server.
- **Regression test suite**: Added comprehensive offline unit tests covering tunnel DNS retry,
  502 edge response handling, premature process termination detection, invalid model payload rejection,
  and deprecation-warning-free CLI flag parsing (15 tests total).


## v0.2.3 — 2026-09-26

### Fixed
- Rerunning a successful notebook deployment no longer fails the initial 9.5 GiB
  *free* VRAM gate when the previously launched Bonsai server already holds ~9 GiB.
  Recover the running process before checking capacity; verify executable, model,
  bind address, alias, key, authenticated API responses and **per-process** GPU
  residency before reuse. Keep its actual quantization and context instead of
  selecting a different model based on remaining free memory.
- Recover a verified server even after a previous run failed before creating
  `state.json` (e.g. Cloudflare tunnel failure). Persist private, atomic state
  as soon as inference is live so subsequent retries can reuse it.
- Do not adopt/kill unrelated GPU processes or silently override a changed
  `BONSAI_API_KEY`; new deployments still require free VRAM. Require the full
  initial disk headroom only when downloading a new deployment.
- Add offline regression tests for low-free-VRAM reruns, missing/stale state,
  unsafe/unauthenticated/CPU-only processes, duplicate servers, GPU PID memory,
  and private state persistence. GPU notebook integration is still required to
  measure full inference end-to-end.


## v0.2.2 — 2026-09-26

### Fixed
- **Deployment FAIL `unknown model rejected`**: `llama-server` (including the official PrismML
  fork) is a single-model server — `--alias` only controls the model ID *returned* by
  `/v1/models` and chat completions, it does **not** enforce request-time validation.
  The server always serves the loaded model regardless of the `model` field in the request
  (documented behavior: ` -a, --alias STRING set model name aliases, comma-separated (to be
  used by API)` — see `tools/server/README.md` and `man llama-server`). Previous test
  expected `400+` for `model: not-bonsai` and caused `DEPLOYMENT FAILED` even though the
  server was healthy and correctly reported `model: ternary-bonsai-2-27b`.
  Fix accepts either:
  1. Proper `4xx` rejection (strict proxies / future server versions), **or**
  2. `200` with `model == ternary-bonsai-2-27b` (current vanilla behavior — server does not
     impersonate the unknown name). The test now logs the returned model ID for evidence.
  Evidence: deployment log showed `PASS` for all other tests (`/health`, `/v1/models`, auth,
  chat, streaming, tool calling) and `FAIL  unknown model rejected` because `status=200`
  with `model=ternary-bonsai-2-27b`, not `>=400`. The fix makes this case PASS.

## v0.2.1 — 2026-09-26

### Fixed
- GGUF verification now accepts official HF files with generic `general.name='Hf'` and
  architecture `qwen35`. Previous logic required `'bonsai'` in `general.name` and rejected
  the valid official `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.21 GB, SHA-256
  3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1) despite passing SHA-256,
  size, and 26.90B parameter checks. The fix searches all GGUF metadata for `bonsai`,
  `prism.*`/`hadamard` signals and accepts `27B` + `qwen35`/`qwen3` family when SHA-verified,
  preserving strict checks for wrong arch/size while not rejecting the official release.
- Architecture check now accepts `qwen35` and broader `qwen` family prefix (Bonsai 2 is
  Qwen3.8-27B derived), not just `qwen3` prefix strictness.

## v0.2.0 — 2026-09-26

### Fixed
- GGUF verification no longer depends on `gguf-python` recognizing every tensor
  quantization enum. Bonsai's official GGUF uses an extended quantization type
  (including type 142); the deployment now reads standard GGUF metadata and tensor
  dimensions while treating quantization type IDs as opaque, preserving the integrity
  and parameter-count checks without rejecting valid official weights.

## v0.1.0 — 2026-09-26

First tagged release of the single-cell Ternary Bonsai 2 27B deployment.

### Added
- `colab_kaggle_cell.py`: one self-contained Colab/Kaggle cell that deploys the real
  `prism-ml/Ternary-Bonsai-2-27B-gguf` model behind the official PrismML llama.cpp fork
  (`PrismML-Eng/Bonsai-demo` binary mechanism) as an authenticated OpenAI-compatible API
  with a Cloudflare Quick Tunnel.
- Hardware auto-detection for 1× or 2× NVIDIA GPUs (T4-class), VRAM/CUDA/driver/topology
  reporting, and CUDA-driver-level usability checks.
- Model integrity pipeline: official HF SHA-256 + exact size + GGUF metadata checks
  (family, architecture, ~27B parameter count). No substitution, conversion, or re-quantization.
- Measured single-GPU vs dual-GPU layer-split benchmark with automatic selection
  (tensor/row split modes are never used); conservative context tiers with OOM auto-retry.
- Official reasoning defaults (thinking on; `medium` server default, per-request override),
  native `--jinja` tool calling, single-slot prompt-cache-friendly serving, text-only startup
  (no vision projector), and explicit "no speculative decoding" (no official Bonsai 2 drafter).
- End-to-end self-tests: health, models, chat, streaming, model validation, bearer auth,
  invalid/missing key rejection, native tool call, and remote tunnel verification.
  `READY` prints only when all pass.
- Idempotent re-runs: existing healthy server/tunnel/model/runtime are reused.
- `README.md`, `SECURITY.md`, `CHANGELOG.md`, Apache-2.0 `LICENSE` (covers this repo's code only).

### Notes
- GPU integration must be validated in a live notebook runtime; no GPU is available in CI here,
  so no inference numbers are claimed by the repository itself.
