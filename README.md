# Ternary Bonsai 2 27B — one-cell Colab/Kaggle API deployment

Deploy the **real** [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(PrismML's 27.36B ternary reasoning model, official GGUF) as an authenticated,
publicly reachable, **OpenAI-compatible HTTP API** — from a **single Python cell**
on Google Colab or Kaggle, running on 1× T4 or 2× T4.

Paste the entire contents of [`colab_kaggle_cell.py`](colab_kaggle_cell.py) into one cell,
select an NVIDIA GPU runtime (T4 or T4×2), enable notebook Internet access, and run.
No second cell, no manual commands, no config files.

## What v0.5.0 adds

- **Reliable streaming.** An incremental SSE parser that survives fragmented reads,
  keep-alive comments and multi-line data; a stream object that always closes its HTTP
  response; and a broken stream that reports its partial text instead of quietly becoming
  a finished answer.
- **Real thinking control.** `/think off|low|medium|high|max` drives the runtime's actual
  `thinking_budget_tokens` field, `/effort medium|xhigh` drives the template's effort, and
  `/reasoning full|compact|hidden` chooses how the trace is shown — live, without a
  restart and without touching history.
- **Lower latency.** One persistent connection instead of a handshake per request,
  coalesced terminal flushing, and reads that return as soon as a chunk arrives.
- **Hardware autotuning.** Flags discovered from `--help`, context chosen from the KV cost
  the server reports, a bounded autotune cached per hardware, and a bounded OOM ladder.
- **Recovery.** Structured failures that say what broke, what was preserved and what to
  do; a tunnel that is restarted on its own without touching the model; sessions that
  survive being killed mid-write.
- **Honesty.** `unknown` is a first-class diagnostic state, `n/a` is printed instead of a
  guess, and no benchmark is quoted that was not measured.

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

`/help` lists every command. Useful extras:

```bash
python3 bonsai_chat.py --benchmark          # TTFT, prefill and decode, median/min/max/p95
python3 bonsai_chat.py --doctor --json      # machine-readable capability report
python3 bonsai_chat.py --think off -p "..." # answer without a thinking trace
```

Two honest limits: the served model is **text-only** (no
`mmproj`), so an attached image is never *seen* unless you deploy with a vision projector —
the client says so in the message it sends; and `run_shell`/`write_file`/`http_get` are
gated behind an approval prompt unless you pass `--auto-approve`.

Try it with no deployment at all — `--mock` and `--selftest` run against
[`mock_bonsai_server.py`](mock_bonsai_server.py), a stub that speaks the same protocol as the
PrismML llama.cpp fork with **scripted replies** (it is not a model and runs no inference):

```bash
python3 bonsai_chat.py --mock     # interactive, scripted answers
python3 bonsai_chat.py --selftest # 20 protocol/behaviour checks
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
- **No speculative decoding** (no official Bonsai 2 drafter exists) and **no vision tower**
  (text-only serving saves VRAM) — both stated explicitly in the output.
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

`python -m py_compile colab_kaggle_cell.py` checks syntax and
`python -m unittest -v test_deployment` runs offline rerun regressions without a GPU. A full integration
run requires a GPU notebook runtime and is not claimed by CI here.

The chat client is verified the same way — offline, no GPU, no model:

```bash
python -m unittest -v test_chat_client          # 82 tests: transport, streaming, tools,
                                                # images, markdown, history, REPL, CLI, doctor
python -m unittest -v test_streaming_recovery   # 94 tests: SSE fragmentation, cancellation,
                                                # disconnects, context, reasoning, recovery
python3 bonsai_chat.py --selftest               # 20 end-to-end checks against the protocol stub
python3 bonsai_chat.py --doctor                 # live diagnostics against a real deployment
```

`test_chat_client` and `test_streaming_recovery` drive the shipped client over real HTTP
against [`mock_bonsai_server.py`](mock_bonsai_server.py), which answers with scripted
content and can reproduce 26 deterministic failure modes (fragmented SSE, mid-stream
disconnect, partial tool calls, 429/5xx, timeouts, context overflow, …), so what is
executed is the code that ships.

```bash
python3 -m unittest test_chat_client test_streaming_recovery test_deployment
# 235 tests, all offline, no GPU and no model
```

End-to-end inference quality against the real 27B model still requires a GPU notebook
runtime and is not claimed here. Upstream references:
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) (source of truth for running these
models), [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
[PrismML docs](https://docs.prismml.com/download/models).

## License

This repository's code (the cell and docs) is licensed [Apache-2.0](LICENSE).
The Ternary Bonsai 2 27B weights are licensed Apache-2.0 by PrismML under their own terms;
the PrismML runtime binaries are governed by their upstream licenses. Nothing here modifies
or redistributes model weights.
