# Ternary Bonsai 2 27B — one-cell Colab/Kaggle API deployment

Deploy the **real** [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(PrismML's 27.36B ternary reasoning model, official GGUF) as an authenticated,
publicly reachable, **OpenAI-compatible HTTP API** — from a **single Python cell**
on Google Colab or Kaggle, running on 1× T4 or 2× T4.

Paste the entire contents of [`colab_kaggle_cell.py`](colab_kaggle_cell.py) into one cell,
select an NVIDIA GPU runtime (T4 or T4×2), enable notebook Internet access, and run.
No second cell, no manual commands, no config files.

## What v0.6.0 adds

Two goals, in this order: **stability** (the GPU deployment path is now executed and
survivable) and **feature breadth** (the capabilities v0.5.0 was missing).

Stability

- **The deployment path is actually executed.** `cell_harness.py` drives the cell's real
  module-level body offline, against a fake GPU: fake `nvidia-smi` with real per-process
  VRAM attribution, a CUDA driver compiled from C and driven through the real `ctypes`
  path, a fake PrismML `llama-server` *executable* (a compiled launcher owns the PID, so
  `/proc/<pid>/exe` and `argv[0]` really are `bin/cuda/llama-server`), a fake Hugging Face
  layer with a synthetic 7.21 GB sparse GGUF, and a fake `cloudflared` backed by a real
  loopback proxy. 73 end-to-end tests run the deployment to READY in about a second and
  then break it on purpose.
- **A supervised deployment.** Health-gated liveness (not "is the PID alive"), hang
  detection through a real generation probe with a stall timeout, and restarts bounded in
  both count (3) and rate (per 15 minutes, with backoff). Crash, OOM and hang are told
  apart from the server's own log, because the repair differs: an OOM steps the context
  down, a crash restarts where it was. The tunnel and the model have separate budgets, so
  neither can cascade into the other. Every heal is logged into `diagnostics.json` and
  `heartbeat.json`.
- **Two real bugs this found.** `plan_to_args()` emitted value-less flags, so
  `--parallel` swallowed `-c 8192` and the server refused to start; and a rerun that
  adopted a live server re-ran `download_binaries.sh`, overwriting the executable it was
  running from (`ETXTBSY`). Both fixed, both with regression tests.

Features (all capability-detected, all opt-in)

- **Vision**, gated on `BONSAI_VISION=1`: the official `mmproj` from the same repo with
  the same SHA-256/size discipline, its VRAM counted before the server starts, and image
  tokens reserved from the context. Refused — not silently dropped — when the build lacks
  `--mmproj`, when no projector is published, or when the fitted context cannot hold
  weights + projector + the image reserve.
- **Multi-slot** (`BONSAI_SLOTS=N`) and **KV4** (`BONSAI_KV4=1`), each refused when the
  build does not advertise the flag, each with its cost stated: a slot carries its own KV
  cache so the per-slot context shrinks, and q4_0 is roughly 3.5× smaller and *slower*.
  Single slot stays the default.
- **Client power features**: a config file with inspectable precedence, persona presets,
  conversation branching, MCP tools through the existing registry, multi-endpoint use with
  a per-endpoint capability map, tools that start the moment their arguments are complete,
  token/turn budgets, batch mode, and JSON/HTML export.

Honesty is unchanged: `unknown` is a first-class diagnostic state, `n/a` is printed
instead of a guess, no benchmark is quoted that was not measured, and `PASS` is never
printed for a check that did not run.

## What the cell does

| Phase | Action |
| --- | --- |
| 1 | Platform detection (Colab vs Kaggle), RAM/disk checks |
| 2 | GPU inspection via `nvidia-smi` + CUDA driver init: count, names, total/free VRAM, compute capability, driver/CUDA version, PCIe topology |
| 3 | Clones the official [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) and runs its `scripts/download_binaries.sh`, which selects the matching **PrismML llama.cpp fork** CUDA build. Open WebUI/MLX/code-interpreter extras are skipped. Refuses CPU fallbacks and non-fork binaries |
| 4 | Downloads only the official language-model GGUF (`PQ2_0` demo default, `PTQ1_0` when free VRAM < ~12 GiB) from `prism-ml/Ternary-Bonsai-2-27B-gguf`, then verifies HF SHA-256 + exact size + GGUF metadata (family name, architecture, ~27B parameter count). No vision projector is downloaded for text-only serving |
| 5 | Discovers the tunable flags from `llama-server --help` itself (nothing is passed blind), picks a context from the **KV cost the server reports** rather than a fixed constant, benchmarks single-GPU vs dual-GPU **layer split** (never tensor/row split) with real Bonsai 2 requests, and runs a bounded autotune of the knobs it found — cached per GPU/runtime so a rerun does not repeat it. OOM steps the context down a bounded ladder and records every step |
| 6 | Starts the PrismML `llama-server` OpenAI API on `127.0.0.1` with `--api-key`, verifies GPU residency via VRAM usage |
| 7 | Real end-to-end tests: `/health`, `/v1/models`, chat completion, streaming, model alias reporting, bearer auth, invalid/missing key rejection, native tool calling (`--jinja`). `READY` is printed only if every test passes |
| 8 | Cloudflare Quick Tunnel exposing only the API port, verified remotely and treated as disposable — restarted on its own if it dies, without ever touching the model. Prints base URL, API key, measured benchmark numbers, recovery steps taken, client examples, and writes `diagnostics.json` |

| 9 | Supervision: two independent watchdogs (inference server and tunnel) watch liveness, probe for hangs with a real generation, and repair what they can within a bounded restart budget. `heartbeat.json` is rewritten every 60 s so a returning session can prove what is still live |

## Using the API

The final output prints the base URL and bearer key. Any OpenAI-compatible client works:

```python
from openai import OpenAI

client = OpenAI(base_url="https://<tunnel-host>/v1", api_key="<API_KEY>")
response = client.chat.completions.create(
    model="ternary-bonsai-2-27b",
    messages=[{"role": "user", "content": "Write a C++ function that reverses a string."}],
)
print(response.choices[0].message.content)
```

Supported: chat completions (streaming and non-streaming), system messages,
`max_tokens`, `temperature`, `top_p`, model alias reporting, bearer auth, and native
tool/function calling. Reasoning (thinking) is **on** by default; the server default
effort is `medium` for interactive coding latency, and a request can ask for stronger
effort via `reasoning_effort`. Sampling defaults follow the official model card
(`temp 1.0, top_p 0.95, top_k 20, min_p 0.05`).

Two thinking controls, both per request, matching what the PrismML runtime actually
accepts:

| Field | Values | Effect |
| --- | --- | --- |
| `thinking_budget_tokens` | `0`, `N`, `-1` | `0` disables thinking, `N` caps the thinking trace at N tokens, `-1` is unlimited |
| `reasoning_effort` | `medium`, `xhigh` | the chat template's effort; `xhigh` is the model default |

`low` reasoning effort is accepted by the template but documented by PrismML as *not*
reducing thinking, so this project does not offer it as a speed knob. Use a token budget
to bound thinking instead.

## Chat client — [`bonsai_chat.py`](bonsai_chat.py)

The cell gives you an API; this is a client worth sitting in front of it. One file,
**standard library only** (no `pip install`), and `pillow` / `pygments` / `pytesseract` are
used automatically when they happen to be installed.

```bash
export BONSAI_BASE_URL=https://<tunnel-host>/v1     # printed by the cell
export BONSAI_API_KEY=<key>                          # printed once by the cell

python3 bonsai_chat.py --doctor   # is this endpoint actually usable? (health → tools → vision)
python3 bonsai_chat.py            # interactive chat loop
python3 bonsai_chat.py -p "Explain why C++ can be fast in 3 sentences."
cat error.log | python3 bonsai_chat.py -p "what is wrong here?"
python3 bonsai_chat.py --image diagram.png -p "describe what this measures"
```

| Feature | What it does |
| --- | --- |
| Chat loop | Multi-turn history, `/undo`, `/retry`, `/reset`, autosave to versioned JSONL, Markdown export, Ctrl-C cancels only the current turn |
| Streaming | Token-by-token output over one persistent connection; fragmented SSE, keep-alives and multi-line `data:` all handled; output coalesced so it flushes ~30×/s instead of once per token; per-turn tokens, tok/s and TTFT from the server's own timings |
| Thinking | `/think off\|low\|medium\|high\|max` sets the thinking budget (0/512/2048/8192/unlimited tokens) and `/effort medium\|xhigh` sets the template effort — live, no restart, no history reset. `/reasoning full\|compact\|hidden` chooses how the trace is shown |
| Markdown | Headings, nested/task lists, blockquotes, GFM tables and fenced code rendered in the terminal (Pygments highlighting when installed); plain text when piped |
| Tools | Sandboxed `read_file`, `list_dir`, `search_text`, `write_file`, `run_shell`, `http_get`, `calculator`, `current_time`, `image_inspect`; parallel calls, tool-result loop, `y/n/always` approval for anything that writes or executes, plus a per-call timeout, a per-turn tool budget and a cap on how much output is fed back |
| Images | `/image path` (or `--image`). With a vision projector the pixels are sent; on the default text-only deployment the client measures the file instead — format, dimensions, aspect, Pillow colour stats, optional OCR — and sends those facts |
| Context | Real token counts from `/tokenize`, a budget that reserves room for the template and the answer, whole-turn trimming that never orphans a `tool` message from its `tool_calls`, and `/compact` to summarise old turns instead of dropping them |
| Resilience | Retries 429/5xx and connect failures, but **never** replays a chat POST that may already be generating; drops `reasoning_effort`/`thinking_budget_tokens`/`tools` only after the build rejects them with a 400; explains a dead Quick Tunnel instead of dumping a traceback |
| Diagnostics | `--doctor [--json]` reports supported / unsupported / unknown / degraded per capability with the evidence behind each; `--benchmark` measures TTFT, prefill and decode; `/stats` and `/caps` in-session |
| Configuration | A config file (`bonsai.json`, `~/.config/bonsai/config.json`, …) plus persona presets. Precedence, lowest to highest: defaults → file → persona → environment → the flags you typed. `/config` prints where every value came from |
| Branching | `/fork <name>`, `/branches`, `/branch <name>`, `/branch del`, `/branch rename` — a branch is a snapshot, stored as an extra line in the same session file, so schema 2 files still load everywhere |
| Endpoints | `/endpoints [name]` switches between named endpoints from the config file. Each gets its own client, capability map and token cache: a capability map belongs to the server it was measured on |
| MCP tools | Any MCP server over stdio (`mcpServers` in the config file, the same shape Claude Desktop uses) registers its tools into the same `ToolRegistry`, with the same approval, timeout and output caps. Colliding names become `server__tool` |
| Structured output | `--json-schema FILE` / `/schema`. Proved with a real request before use: a server that accepts `response_format` and answers in prose anyway is reported as unsupported, and the field is not sent |
| Budgets | `--budget-tokens`, `--budget-turns`, `--price-per-mtok`; `/spend` reports. Counted only from what the server reported; cost stays `n/a` until you supply a price |
| Batch | `--batch file.jsonl --out results.jsonl`: the same client, retry policy and tool loop, one conversation per item, a per-item error instead of a crash |
| Export | `--export out.json` or `out.html` (`/export` in session). The HTML is self-contained: no CDN, no external resources, content escaped |

`/help` lists every command. Useful extras:

```bash
python3 bonsai_chat.py --benchmark          # TTFT, prefill and decode, median/min/max/p95
python3 bonsai_chat.py --doctor --json      # machine-readable capability report
python3 bonsai_chat.py --think off -p "..." # answer without a thinking trace
python3 bonsai_chat.py --save-config --config bonsai.json   # starter config + personas
python3 bonsai_chat.py --config bonsai.json --persona code  # use a preset
python3 bonsai_chat.py --batch jobs.jsonl --out results.jsonl
python3 bonsai_chat.py --session chat.jsonl --fork draft -p "try it another way"
```

Programmatically, the same code path the CLI uses:

```python
from bonsai_chat.api import Bonsai

bot = Bonsai(base_url='http://127.0.0.1:8080/v1', api_key='...')
print(bot.ask('What is 6*7?').text)

@bot.tool('weather', 'Current weather for a city',
          {'type': 'object', 'properties': {'city': {'type': 'string'}},
          'required': ['city']})
def weather(args):
    return 'raining'
```

**No second web UI.** The PrismML runtime already ships a web chat at the server root,
and a second one would be a second renderer and a second auth surface to keep honest for
no capability the runtime does not already have. `--export out.html` covers "I want to
read this later", and `diagnostics.json` / `heartbeat.json` cover "what is running".

Three honest limits. The default deployment is **text-only** (no `mmproj`), so an attached
image is never *seen* unless you deploy with `BONSAI_VISION=1` — the client says so in the
message it sends. `run_shell`/`write_file`/`http_get` are gated behind an approval prompt
unless you pass `--auto-approve`. And the deployment path has been executed end to end
against a fake GPU, offline, in this repository — it has still **not** been run against
real Bonsai 2 weights on real hardware here, so no inference-quality or throughput claim
is made for it.

Try it with no deployment at all — `--mock` and `--selftest` run against
[`mock_bonsai_server.py`](mock_bonsai_server.py), a stub that speaks the same protocol as the
PrismML llama.cpp fork with **scripted replies** (it is not a model and runs no inference):

```bash
python3 bonsai_chat.py --mock     # interactive, scripted answers
python3 bonsai_chat.py --selftest # 31 protocol/behaviour checks
```

Optional environment variables (read from the notebook environment/secrets):

| Variable | Purpose |
| --- | --- |
| `BONSAI_API_KEY` | Use your own bearer key (≥ 32 chars); otherwise a strong random key is generated |
| `HF_TOKEN` / `BONSAI_TOKEN` | Hugging Face token if your network requires one (repo is public) |

## Performance

Measured in this repository's sandbox on **loopback**, 40 requests per configuration, with
`TCP_NODELAY` on the server as real HTTP servers use:

| Transport | one connection per request | persistent connection |
| --- | --- | --- |
| plain HTTP | median 0.57 ms/request | median 0.27 ms/request |
| TLS | median 2.39 ms/request | median 0.44 ms/request (**5.4×**) |

Through a Cloudflare Quick Tunnel the saving is larger, because the TLS handshake is a WAN
round trip rather than a loopback one — **that has not been measured and is not claimed
here**. The only throughput numbers this project prints for the model are the ones the
deployment cell measures live against the running server on your hardware.

`--benchmark` reports median / min / max / p95 over N samples after discarding warmup, and
rounds to three significant figures so a three-sample measurement is never presented as
`147.123456 tok/s`.

## Troubleshooting

| Symptom | What it means | What to do |
| --- | --- | --- |
| `cannot reach <url>` | the notebook runtime or its Quick Tunnel is gone | rerun the cell; it reuses a verified live server instead of restarting the model |
| `HTTP 401` | the bearer key does not match | the cell prints the key once; changing `BONSAI_API_KEY` while a server is live needs the old server stopped first |
| `HTTP 400 … context` | the request exceeded the window | `/compact`, or lower `/max-tokens`; the client reserves room for template + answer |
| `the request reached the server but no reply came back` | the stream was lost after generation started | the client refuses to replay it, so you do not get a duplicated answer — `/retry` if you want it re-run |
| tunnel `edge-error 502/503` | the edge is not routing yet | the cell retries; the model itself is still live on `127.0.0.1` |
| `server died with a CUDA/OOM error` | the context did not fit | the cell steps the context down automatically; free VRAM yourself if it still fails — unrelated GPU processes are never killed |
| `BONSAI_VISION=1 was requested but this PrismML build does not advertise --mmproj` | no multimodal projector in this build | vision stays off on purpose. Unset it, or use a build with the projector |
| `supervision stopped: the restart budget is exhausted` | the server or tunnel kept failing | read `diagnostics.json` — every attempt is logged with what was detected, what was preserved and what changed |
| `refusing to replace it under a live process` | the binary a verified server runs from went missing | stop that server yourself; the cell will not overwrite an executable that is in use |

## Guarantees and limits

- **Model lock.** Only `prism-ml/Ternary-Bonsai-2-27B-gguf` is ever downloaded or served.
  If it cannot be obtained or verified, the cell fails loudly; it never substitutes another model.
- **Official runtime.** Bonsai 2's PTQ1_0/PQ2_0 formats require the PrismML llama.cpp fork
  (stock llama.cpp rejects them). The cell verifies the fork's release stamp and GPU residency.
- **No fakes.** Every printed number (token/s, TTFT, VRAM, health, PASS lines) is measured
  live against the running server. `READY` is never printed without passing end-to-end
  tests, `PASS` is never printed for a check that was not actually run, and a capability
  that was never tested is reported as `unknown` rather than assumed. `--benchmark` prints
  `n/a` for anything it could not measure (including VRAM, which is server-side).
- **No speculative decoding** (no official Bonsai 2 drafter exists). Vision is opt-in and
  off by default because text-only serving saves VRAM; KV4 and multi-slot are opt-in too,
  each refused when the build lacks the flag. All stated explicitly in the output.
- **Bounded supervision.** A watchdog may restart a component at most 3 times per 15
  minutes, then it stops and reports. It only ever touches a process this cell started
  and verified by executable path, command line and liveness.
- **Safe reruns.** A rerun reuses only a verified server from this work directory
  (even if a prior run failed while starting the tunnel). A different running
  GPU job is never stopped or mistaken for Bonsai; if VRAM is insufficient,
  stop that job yourself before starting a new deployment. Changing
  `BONSAI_API_KEY` while a server is live requires stopping the old server first.
- **Session-scoped.** The API and the ephemeral `trycloudflare.com` URL live only while the
  notebook runtime runs. Single-slot serving is tuned for one coding-agent user with
  prompt-cache reuse; it is not a multi-tenant production service.
- First run downloads several GB (runtime + model). Treat notebook output as secret:
  the API key is printed once.

## Verification

Everything below is offline: no GPU, no model, no network, and deterministic.

The deployment cell

```bash
python3 -m unittest -v test_cell_deployment
# 73 tests: the cell's real module-level body run against a fake GPU, to READY and then
# broken on purpose — OOM ladder, measured-context upgrade, autotune cache, dual GPU,
# no GPU, cuInit failure, non-fork stamp, corrupt identity, SHA-256 mismatch, refusal to
# run unauthenticated, reattach, stale PID, corrupt state, a tunnel that never publishes,
# a dead tunnel, a dead model, a hang, bounded restarts, vision on/off/refused,
# multi-slot, KV4
```

`cell_harness.py` builds the fakes. The fake `llama-server` is a compiled launcher that
owns the PID and runs a Python server honouring the real flags, writing llama.cpp's own
startup log (including the `KV buffer size = ... MiB` lines the cell measures), serving
`/health`, `/props`, `/slots`, `/v1/models` and streaming chat, and requiring the bearer
key everywhere. It can be told to boot slowly, OOM above a context, crash after N
requests, hang, or reject a flag. The fake `cloudflared` publishes a real
`*.trycloudflare.com`-shaped URL and backs it with a real loopback proxy, so public
requests are real HTTP.

The chat client

```bash
python -m unittest -v test_chat_client          # 82 tests: transport, streaming, tools,
                                                # images, markdown, history, REPL, CLI, doctor
python -m unittest -v test_streaming_recovery   # 94 tests: SSE fragmentation, cancellation,
                                                # disconnects, context, reasoning, recovery
python -m unittest -v test_power_features       # 67 tests: config precedence, budgets,
                                                # branching, export, endpoints, MCP,
                                                # structured output, batch, the importable API
python -m unittest -v test_deployment           # 59 tests: the cell's helpers and its
                                                # module structure, including that the
                                                # client and cell versions agree
python3 bonsai_chat.py --selftest               # 31 end-to-end checks against the protocol stub
python3 bonsai_chat.py --doctor                 # live diagnostics against a real deployment
```

`test_chat_client` and `test_streaming_recovery` drive the shipped client over real HTTP
against [`mock_bonsai_server.py`](mock_bonsai_server.py), which answers with scripted
content and can reproduce 30 deterministic failure modes (fragmented SSE, mid-stream
disconnect, partial tool calls, 429/5xx, timeouts, context overflow, a server that
ignores `response_format`, …), so what is executed is the code that ships.

```bash
python3 -m unittest discover -s . -p 'test_*.py'
# 375 tests, all offline, no GPU and no model
```

**What has not been verified.** The GPU deployment path has never been run against real
Bonsai 2 weights on real hardware in this repository: what is verified is the decision
logic, the recovery logic and the wire protocol, against fakes. Real-hardware behaviour —
actual VRAM residency, actual decode speed, whether the official `mmproj` file names and
sizes match what the fake advertises, and whether a PrismML build really prints the
`--mmproj` flag the way the tests assume — is unverified and is not claimed here. The
vision, multi-slot and KV4 paths in particular are exercised only against a fake runtime
whose `--help` this repository wrote. Upstream references:
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) (source of truth for running these
models), [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
[PrismML docs](https://docs.prismml.com/download/models).

## License

This repository's code (the cell and docs) is licensed [Apache-2.0](LICENSE).
The Ternary Bonsai 2 27B weights are licensed Apache-2.0 by PrismML under their own terms;
the PrismML runtime binaries are governed by their upstream licenses. Nothing here modifies
or redistributes model weights.

## Web UI (v0.7.0)

Build the committed browser UI with `cd bonsai_chat/webui && npm install && npm run build`.
Run it locally with `python3 bonsai_chat.py --serve --base-url https://your-tunnel/v1 --api-key KEY`.
The Python process is the same-origin proxy: the API key is accepted once by the local
server and never sent to browser JavaScript, URLs, logs, or API responses. The remote
model remains the PrismML Ternary Bonsai deployment. Sessions, tools/MCP visibility,
markdown/export and image input are reserved for v0.7.1.
