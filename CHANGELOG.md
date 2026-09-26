# Changelog

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
