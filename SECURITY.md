# Security Policy

## Scope

This repository contains a single deployment cell (and docs). It does not ship model
weights or binaries; both are fetched at run time from their official upstream sources:

- Runtime binaries: `PrismML-Eng/Bonsai-demo` release mechanism (PrismML llama.cpp fork).
- Model weights: `prism-ml/Ternary-Bonsai-2-27B-gguf` on Hugging Face, verified against
  the official LFS SHA-256 (when published) and exact byte size, plus GGUF metadata checks.

## Authentication

The cell refuses to start the public tunnel before the server enforces
`Authorization: Bearer <key>` on every request (`--api-key`). A cryptographically random
key (≥ 32 chars) is generated unless `BONSAI_API_KEY` is supplied. Unauthenticated and
invalid-key requests are verified to be rejected (401/403) before the URL is announced.

## Exposure surface

- The inference server binds to `127.0.0.1` only; it is never bound to `0.0.0.0`.
- The Cloudflare Quick Tunnel forwards only the single API port — no other notebook
  service is exposed.
- `state.json` (key, PIDs, URL) is written with mode `0600` inside the notebook working dir.

## Secrets handling

- The API key is printed once in the notebook output after all tests pass. Notebook output
  is effectively a secret surface: do not share screenshots/logs of a running deployment.
- `HF_TOKEN`/`BONSAI_TOKEN` are read from the environment only and never written to disk
  by this cell.

## Operational caveats

- Quick Tunnel hostnames are ephemeral and rate-limited by Cloudflare; they are suitable
  for interactive use, not for production SLAs.
- The deployment is session-scoped: terminating the notebook kills the API.
- For production, place the same official server behind managed HTTPS ingress with secret
  management, persistent storage, and monitoring; this cell is not a substitute for that.

## Reporting

For vulnerabilities in this cell, open an issue or contact the repository maintainer.
For upstream issues (runtime, kernels, model files), report to
`PrismML-Eng/Bonsai-demo` / `PrismML-Eng/llama.cpp` / the `prism-ml` model repos.
