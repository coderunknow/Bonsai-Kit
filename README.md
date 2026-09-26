# Ternary Bonsai 2 27B — one-cell Colab/Kaggle API deployment

Deploy the **real** [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(PrismML's 27.36B ternary reasoning model, official GGUF) as an authenticated,
publicly reachable, **OpenAI-compatible HTTP API** — from a **single Python cell**
on Google Colab or Kaggle, running on 1× T4 or 2× T4.

Paste the entire contents of [`colab_kaggle_cell.py`](colab_kaggle_cell.py) into one cell,
select an NVIDIA GPU runtime (T4 or T4×2), enable notebook Internet access, and run.
No second cell, no manual commands, no config files.

## What the cell does

| Phase | Action |
| --- | --- |
| 1 | Platform detection (Colab vs Kaggle), RAM/disk checks |
| 2 | GPU inspection via `nvidia-smi` + CUDA driver init: count, names, total/free VRAM, compute capability, driver/CUDA version, PCIe topology |
| 3 | Clones the official [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) and runs its `scripts/download_binaries.sh`, which selects the matching **PrismML llama.cpp fork** CUDA build. Open WebUI/MLX/code-interpreter extras are skipped. Refuses CPU fallbacks and non-fork binaries |
| 4 | Downloads only the official language-model GGUF (`PQ2_0` demo default, `PTQ1_0` when free VRAM < ~12 GiB) from `prism-ml/Ternary-Bonsai-2-27B-gguf`, then verifies HF SHA-256 + exact size + GGUF metadata (family name, architecture, ~27B parameter count). No vision projector is downloaded for text-only serving |
| 5 | Benchmarks single-GPU vs dual-GPU **layer split** (never tensor/row split) with real Bonsai 2 requests and selects the measured winner; context tiered conservatively (8K/16K/32K/64K) with automatic OOM retry at half context |
| 6 | Starts the PrismML `llama-server` OpenAI API on `127.0.0.1` with `--api-key`, verifies GPU residency via VRAM usage |
| 7 | Real end-to-end tests: `/health`, `/v1/models`, chat completion, streaming, model alias reporting, bearer auth, invalid/missing key rejection, native tool calling (`--jinja`). `READY` is printed only if every test passes |
| 8 | Cloudflare Quick Tunnel exposing only the API port, verified remotely, then prints base URL, API key, benchmark numbers, and client examples |

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
| Chat loop | Multi-turn history, `/undo`, `/retry`, `/reset`, autosave to JSONL, Markdown export, Ctrl-C cancels only the current turn |
| Streaming | Token-by-token output; the model's `reasoning_content` (thinking) is shown in a separate dim block; per-turn tokens, tok/s and TTFT from the server's own timings |
| Markdown | Headings, nested/task lists, blockquotes, GFM tables and fenced code rendered in the terminal (Pygments highlighting when installed); plain text when piped |
| Tools | Sandboxed `read_file`, `list_dir`, `search_text`, `write_file`, `run_shell`, `http_get`, `calculator`, `current_time`, `image_inspect`; parallel calls, tool-result loop, `y/n/always` approval for anything that writes or executes |
| Images | `/image path` (or `--image`). With a vision projector the pixels are sent; on the default text-only deployment the client measures the file instead — format, dimensions, aspect, Pillow colour stats, optional OCR — and sends those facts |
| Context | Real token counts from `/tokenize`, budgeted trimming that drops whole turns and never orphans a `tool` message from its `tool_calls` |
| Resilience | Retries on 429/5xx and socket drops, drops `reasoning_effort`/`tools` if the build rejects them, explains a dead Quick Tunnel instead of dumping a traceback |

`/help` lists every command. Two honest limits: the served model is **text-only** (no
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

## Guarantees and limits

- **Model lock.** Only `prism-ml/Ternary-Bonsai-2-27B-gguf` is ever downloaded or served.
  If it cannot be obtained or verified, the cell fails loudly; it never substitutes another model.
- **Official runtime.** Bonsai 2's PTQ1_0/PQ2_0 formats require the PrismML llama.cpp fork
  (stock llama.cpp rejects them). The cell verifies the fork's release stamp and GPU residency.
- **No fakes.** Every printed number (token/s, TTFT, VRAM, health, PASS lines) is measured
  live against the running server. `READY` is never printed without passing end-to-end tests.
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
python -m unittest -v test_chat_client   # 82 tests: transport, streaming, tools, images,
                                         # markdown, history, REPL commands, CLI, --doctor
python3 bonsai_chat.py --selftest        # 20 end-to-end checks against the protocol stub
python3 bonsai_chat.py --doctor          # live diagnostics against a real deployment
```

`test_chat_client` drives the shipped client over real HTTP against
[`mock_bonsai_server.py`](mock_bonsai_server.py), which answers with scripted content, so
what is executed is the code that ships. End-to-end inference quality against the real
27B model still requires a GPU notebook runtime and is not claimed here. Upstream references:
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) (source of truth for running these
models), [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
[PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
[PrismML docs](https://docs.prismml.com/download/models).

## License

This repository's code (the cell and docs) is licensed [Apache-2.0](LICENSE).
The Ternary Bonsai 2 27B weights are licensed Apache-2.0 by PrismML under their own terms;
the PrismML runtime binaries are governed by their upstream licenses. Nothing here modifies
or redistributes model weights.
