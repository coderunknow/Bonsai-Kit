# knowledge.md — session memory for the Bonsai-Kit v0.5.0 → v0.6.0 work

A working document for whoever picks this up. The repository is the source of truth for
code; this file is the source of truth for *why* things are the way they are, what was
measured, and what is still unknown.

---

## 0. The task (verbatim intent)

Take `https://github.com/coderunknow/Bonsai-Kit` from **v0.5.0** to **v0.6.0**,
autonomously: inspect, implement, test, fix, leave the repo releasable. Two headline goals
in priority order: **(1) Stability** — make the GPU deployment path survivable and
testable; **(2) Feature breadth** — add the missing capability set end to end.

Hard constraints inherited from v0.5.0 that must keep holding:

- **Model lock**: only `prism-ml/Ternary-Bonsai-2-27B-gguf` (`PQ2_0` 7.21 GB or `PTQ1_0`
  5.95 GB). No substitution, no re-quant, no silent fallback.
- **Runtime lock**: PrismML llama.cpp fork only (stamp must start with `prism`); stock
  llama.cpp refuses PTQ1_0/PQ2_0; never assume a flag — discover from `--help`.
- **GPU policy**: no silent CPU fallback; never kill/adopt/modify a process not verified by
  exe path + cmdline + liveness.
- **Security**: localhost bind only, bearer auth always, keys never logged, state/session
  `0600`, tunnel URLs are ephemeral secrets. New features must not weaken it.
- **Honest diagnostics**: never print PASS for an unrun check; `unknown`/`n/a` are
  first-class; never quote unmeasured tok/s, TTFT, VRAM.
- v0.5.0 CLI flags, session schema 2, and `--doctor --json` shape keep working (additive
  only).
- Single-slot (`--parallel 1`) stays default; multi-slot opt-in with prompt-cache cost
  stated. Vision stays off by default, opt-in, VRAM-accounted before start.
- The cell remains one paste-into-one-cell file and may not import the client package.
- The client becomes a stdlib-only package with `bonsai_chat.py` kept as a working shim
  (`python3 bonsai_chat.py`, `--selftest`, `--mock`, `--doctor`, `--benchmark` all work
  from a bare checkout).
- No new mandatory dependency; `pillow`/`pygments`/`pytesseract` stay optional and degrade
  silently.

Phase plan: **A** stability (A1 harness, A2 watchdog, A3 session longevity) → **B**
features (B1 vision, B2 multi-slot + KV4, B3 client power features, B4 API completeness) →
**C** testing discipline → **D** version/docs. Final report must state plainly what could
**not** be verified; if the GPU path has still never run on real hardware, that goes in
the first line of the report.

## 0.1 User's standing instructions

- "Do not describe what should be done — do it." Stop only for a genuinely external
  blocker; make reasonable decisions autonomously.
- Refactor for correctness, testability and reliability — **not cosmetics**; if moving
  code does not change behaviour or testability, do not move it.
- Never delete a test because it fails. Never reduce the 235 tests.
- Prioritize **wall-clock time** in bash: avoid long/redundant/blocking commands, add
  explicit timeouts, reuse prior outputs, batch independent lightweight checks, stop
  expensive checks once sufficient evidence is obtained.
- Verification discipline: inspect before editing, measure before optimising, test before
  claiming. Before declaring anything done, run the project's own checks over the changed
  code and say in one sentence what was run and what came back, naming the function or
  code path actually executed. A clean exit code with wrong output is a failure.
- No duplicated transport, benchmark or capability-detection code — if a feature appears
  twice under different names, stop and unify.
- Do not say "production ready" unless evidence justifies it.

---

## 1. Where the work stands

**Everything asked for has been implemented and tested. The final report has not yet been
written.** The remaining deliverable is the report itself (section 10 below is a draft).

State at the time of writing:

- Branch `arena/01a0e0ec-bonsai-kit`, pushed. Three reconstructed commits on top of base
  `8f19b9f`: `59160a4` (deployment), `778ec41` (client), `cbe9466` (docs).
- **375 tests, all passing, all offline:** `test_cell_deployment` 73, `test_power_features`
  67, `test_streaming_recovery` 94, `test_chat_client` 82, `test_deployment` 59.
  (Baseline was 235.)
- `python3 bonsai_chat.py --selftest` → **31/31** (was 20).
- `mock_bonsai_server.py` → **30** scenarios (was 26).
- Full deployment through the harness reaches READY in **~0.8 s**.

### 1.1 A git incident worth knowing about

Mid-session the sandbox's `.git` was **re-cloned**, which erased the branch's commit
history (`git reflog` showed `clone:` and `checkout:` as the only entries). The working
tree survived intact. Response: verified the files were functional (ran
`test_power_features` 67 OK, `test_deployment` 59 OK, `--selftest` 31/31), then rebuilt the
history as three commits from the working tree and force-pushed nothing (the branch was
newly pushed: `git push -u origin arena/01a0e0ec-bonsai-kit`). The first reconstructed
commit's message records that the boundaries are reconstructed.

**If the history disappears again: the files are what matter.** Re-verify with the three
commands above and re-commit.

---

## 2. Repository layout

```
colab_kaggle_cell.py        ~1,900 lines. The single paste-into-one-cell deployment. VERSION='0.6.0'.
cell_harness.py             770 lines. CellRun: executes the cell's real body against fakes.
_gguf_writer.py             Synthetic GGUF (sparse via truncate), ~27.36 B params.
_fake_llama_server.py       Behaviour half of the fake PrismML binary (a compiled launcher owns the PID).
_fake_cloudflared.py        Loopback TCP proxy behind a published *.trycloudflare.com URL.
_fake_huggingface_hub.py    Canned HfApi.model_info / hf_hub_download; digest cache.
bonsai_chat/                The client package (27 modules, stdlib only).
bonsai_chat.py              24-line shim so `python3 bonsai_chat.py` still works.
mock_bonsai_server.py       30 deterministic scenarios; must stay importable standalone.
test_cell_deployment.py     73 end-to-end deployment tests.
test_power_features.py      67 tests for the v0.6.0 client features.
test_deployment.py          59 tests (cell AST + helpers).
test_chat_client.py         82 tests.
test_streaming_recovery.py  94 tests.
README.md / CHANGELOG.md / SECURITY.md / LICENSE
```

New client modules in v0.6.0: `config.py`, `budget.py`, `branches.py`, `endpoints.py`,
`export.py`, `mcp.py`, `api.py`.

---

## 3. The harness (A1) — how the deployment is actually executed

```python
from cell_harness import CellRun, T4
with CellRun(gpus=[T4]) as run:
    run.execute(expect_success=False)   # captures the exception in run.error
    run.output / run.state / run.diagnostics / run.ns / run.supervisors
```

Observers: `output`, `state`, `diagnostics`, `ns`, `invocations()`, `contexts_attempted()`,
`arg_values()`, `server_args()`, `tunnel_url()`, `server_pids()`, `servers_started()`,
`tunnel_pid()`, `write_state(dict)`.
Test knobs: `set_server_cfg(**kw)`, `set_tunnel_cfg(**kw)`, `supervisors`,
`supervisor(name)`, `kill_server(sig=None)`, `kill_tunnel()`, `_verified_pid(which)`.
`CellRun(gpus=..., cuda_devices=..., cuda_fail=..., server=..., tunnel=..., hf=...,
env=..., meminfo_gib=..., base=..., gpu_dir=...)`.

GPU fixtures: `T4`, `T4X2`, `L4`, `SMALL`, `MISMATCH`. Config builders: `hf_config`,
`default_model_files`, `server_config`, `tunnel_config`. `requires_compiler` skips tests
when no C compiler is present (the fake CUDA driver is compiled).

### 3.1 Cell seams (unchanged defaults, so the deployment itself is untouched)

`MEMINFO`, `CUDA_LIB`, `PROC_ROOT`, `POLL_INTERVAL`, `STARTUP_TIMEOUT`, `TUNNEL_ATTEMPTS`,
`CLOUDFLARED_MIN_BYTES`, `HTTP`, `URL_OPEN`, `GPU_MEMORY`, `BONSAI_ROOT`.

Seam placement is **order-sensitive**: define a seam before the first place that reads it.
Keep the deliberate asymmetry — `managed_server` / `verify_tunnel_connectivity` keep
`request=http` defaults; only call sites pass `request=HTTP`.

### 3.2 Invariants the harness enforces

- Exactly **one** top-level `try` in the cell (`HarnessSelfTests` asserts it).
- `execute()` **truncates `invocations.jsonl` first**, so `invocations()` describes *this*
  execution. A rerun that reattaches must show none.
- `cleanup()` kills every process this run started, identified by this run's unique paths
  (`/proc` cmdline scan over `base` and the fakes dir, plus the gpu apps dir and the
  invocation log). Without this, leaked fake servers outlive their temp dir and eat the
  machine — 296 leaked processes once pinned memory to 93 MB free and made every test
  hang.

---

## 4. Fake behaviours — the config keys you can drive

### `_fake_llama_server.py`
`oom_above_ctx`, `oom_after` (die of a CUDA OOM N seconds in, evidence in the log),
`boot_delay`, `oom_delay`, `crash_after`, `hang`, `reject_flags`, `reject_fields`,
`extra_flags`, `help_drop` (omit flags from `--help`), `decode_rate`, `prefill_rate`,
`token_delay`, `vram_mib`, `kv_mib_per_token`, `dual_vram`, `dual_factor`, `slots`,
`stamp`.

A background `config_watcher` thread re-reads `FAKE_SERVER_CONFIG` every 0.2 s, so a test
can change `oom_after`, `hang` or `crash_after` **without restarting the server** — real
failures do not arrive conveniently before the process starts.

### `_fake_cloudflared.py`
`fail_immediately`, `no_url`, `publish_delay`, `lifetime`. Honours `FAKE_RUN_DIR`
(falling back to `FAKE_GPU_DIR.parent`) and writes `url→port` into
`run_dir/tunnel-map.json`, which the harness reads from `self.root_path`. The handler's
socket attribute is `self.request` (a `BaseRequestHandler` exposes the client socket as
`self.request`, **not** `self.connection`) — re-check with `grep -n self.connection`
after any edit.

### `_fake_huggingface_hub.py`
`files`, `sha`, `corrupt_identity`, `corrupt_bytes` (rewrites the file post-download at
the same size, so only SHA-256 catches it), `missing`. Digest cache `hf_sha_cache.json`.

### `mock_bonsai_server.py` — 30 scenarios
The original 26 plus v0.6.0's: `structured-json` (honours `response_format`),
`ignore-response-format` (accepts it, answers prose),
`reject-response-format` (400 unknown field), `early-tool-call`.

---

## 5. What was implemented

### 5.1 A2 — supervision (in `colab_kaggle_cell.py`)

Constants: `WATCHDOG_INTERVAL=20`, `STALL_TIMEOUT=180`, `HANG_CHECK_INTERVAL=300`,
`MAX_RESTARTS=3`, `RESTART_WINDOW=900`, `RESTART_BACKOFF`, `HEALTH_CONFIRMATIONS=2`,
`HEARTBEAT_INTERVAL=60`, `OOM_PATTERN`.
Functions/classes: `classify_exit()`, `RestartBudget`, `Supervisor`, `write_heartbeat()`.
Body wiring: `start_server(..., port=None)`; `diag` + `write_diagnostics()` +
`heartbeat_facts()` + `record_heal()` built before phase 7; `server_health`,
`server_probe` (a real streaming generation, stall-bounded), `server_repair` (OOM steps
the context ladder; key/port/state/tunnel preserved), `tunnel_repair` (never touches the
model); `supervisors = [inference-server (probe), cloudflare-tunnel]`, both started at the
end, plus a daemon heartbeat thread.

`last_server_log` is a module global set by `start_server`, so `server_repair` reads the
log the **outgoing** server wrote. Reading `server-watchdog.log` instead meant an OOM was
classified as a crash and the restart repeated the failing context.

### 5.2 B1/B2 — deployment levers

`BONSAI_VISION` (+ `BONSAI_MMPROJ`, `pick_mmproj`, projector download + SHA + `--mmproj` +
`--image-max-tokens`, image reserve inside the context budget), `BONSAI_SLOTS`
(`--parallel N`, ctx//slots, cache cost stated), `BONSAI_KV4` (`--cache-type-k q4_0`,
labelled slower). Helpers added: `sha256_file`, `truthy_env`, `pick_mmproj`,
`image_context_reserve`, `slot_report`, and `choose_context(slots=, extra_mib=)` /
`require_capacity(extra_mib=)`.

### 5.3 B3/B4 — client

Everything in section 2's "new modules". Key design decisions:

- **Precedence** is implemented with `explicit_args(parser, argv)`, which deep-copies the
  parser and sets `action.default = argparse.SUPPRESS` on every action. It must use
  `SUPPRESS` and not a sentinel string: argparse skips `type` conversion for suppressed
  defaults, and a sentinel fed to `type=int` options blows up
  (`--image-max-side: invalid int value: 'SUPPRESS_ME'`).
- **Structured output** is proved with a real request. `probe_structured_output()` sets
  `self._probing_structured = True` around its own call, because `chat()` →
  `_apply_structured()` → `probe_structured_output()` recursed infinitely.
- **`--budget`-style command conflicts**: `/budget` was already the *thinking* budget
  command, so the token/cost report is `/spend`.
- **Branching** is stored as a second `_meta` line inside the existing session file. Only a
  line that *carries* a `schema` may set it, or the branches line downgrades a v2 file to
  schema 1.
- **Forks are persisted immediately** by the CLI (`--fork`) — a branch that only exists in
  memory is a branch the next run cannot find.
- **Branch count is capped at 32** (`MAX_BRANCHES`): each fork deep-copies its messages.
- **No second web UI.** The PrismML runtime already ships one at the server root; a second
  would be a second renderer and a second auth surface for no new capability. Documented
  in the README and CHANGELOG.

---

## 6. Bugs found and fixed (each with a regression test)

1. **`plan_to_args()` emitted value-less flags.** With `--parallel` advertised the server
   was launched as `... --parallel -c 8192`, so `-c` became the flag's value and the server
   refused to start. Only entries carrying a value are emitted. Regression:
   `test_cell_deployment.UnitHelperTests.test_a_value_less_flag_is_never_emitted_and_cannot_swallow_the_next_token`.
2. **A rerun re-ran `download_binaries.sh`**, overwriting the executable a verified server
   was mapped from (`ETXTBSY`). The runtime is left alone when a server is already up.
3. **GGUF tensor data offset packed as u32** (this one was in the *fixture*,
   `_gguf_writer.py:44` — now `<IQ`). Regression:
   `test_a_gguf_offset_above_four_gib_is_read_as_a_u64`.
4. **Fake server never died when its launcher was SIGKILLed**, because the signal handler
   called `httpd.shutdown()` from the same thread that runs `serve_forever()` — a deadlock
   that left a "stopped" server holding its port. Fixed with `prctl(PR_SET_PDEATHSIG)`
   in the launcher's child plus a handler that only releases VRAM and `os._exit(0)`.
5. **The fake server parsed `--api-key -abc...` as a boolean flag** (a urlsafe key can
   start with `-`), silently leaving the server unauthenticated — which made auth tests
   flaky at ~3%. Fixed with `ALWAYS_VALUE_FLAGS`.
6. **The branches meta line downgraded session schema 2 to 1** (see 5.3).
7. **The capability map reported `response_format` as supported** merely because the
   server accepted the field. Now corrected by the probe.

---

## 7. Errors, dead ends and traps

- **Leftover fake processes** once consumed all memory (296 procs, 93 MB free) and made the
  suite hang with `EXIT=137` (SIGKILL, i.e. the OOM killer) and **empty logs** because
  Python buffers stdout when redirected. Always run tests with `python3 -u` when
  redirecting, and check `ps aux | grep -cE "fake_llama|fake_cloud"` after a run.
- `AttributeError: '_Pipe' object has no attribute 'connection'` → it is `self.request`.
- `NameError: name 'supervisors' is not defined` — `heartbeat_facts()`/`write_heartbeat()`
  must be called **after** the `supervisors = [...]` literal.
- `Name or service not known` on tunnel verification — the fake wrote `tunnel-map.json` to
  `run_dir`; the harness read `self.root`. Both now use `self.root_path`.
- `OSError: [Errno 98] Address already in use` on restart — the previous server was still
  alive (see bug 4).
- `TypeError: hf_config() got an unexpected keyword argument 'corrupt_bytes'` — the config
  builder needed the key.
- `RecursionError` in `probe_structured_output` — see 5.3.
- `TypeError: unsupported operand type(s) for +: 'int' and 'dict'` in `--selftest` — a
  local variable named `results` shadowed the check accumulator. Renamed to
  `batch_results`.
- `make_app() got an unexpected keyword argument 'tree'` — the CLI passed new kwargs before
  `make_app` accepted them.
- Test-expectation traps: `'--parallel', '1'` (the argv is joined with spaces, so it is
  `--parallel 1`); `context_recovery` is `[16384, 8192]` (attempts **plus** the successful
  candidate); a cheap KV cost yields `262144`, not `131072`.
- `-DFAKE_SCRIPT="\"…\""` double-quoting is a dead end — use `_with_script()`.
- `self.addCleanup` is a list named `_cleanups` in `CellRun`.
- `dlopen` needs `LD_LIBRARY_PATH` at process start (hence an absolute `CUDA_LIB`).
- Fake `nvidia-smi` must match `--query-gpu=…` (`startswith` + `split('=',1)[1]`).
- **Performance facts**: a full READY run is 0.8 s (the earlier 91 s was a real-DNS
  timeout, not slow fakes). `sha256` of a 6 GB sparse file ≈ 8.4 s (~1.4 s/GB), cached in
  `hf_sha_cache.json`. The suite is 210 s for 375 tests.

---

## 8. Sandbox facts

3 GiB RAM (peaks matter — see §7), 2 cores, ~20 GiB free disk, x86_64, gcc/g++ 12.2.0,
uid 1001, `/proc/self/exe` readable, Python 3.11. Scratch dir `/tmp/split/` (outside the
repo, not persisted): `smoke.py` (CellRun + READY check), `wd1.py` (watchdog debug),
`batch.jsonl`, `cfg.json`, `bonsai.json`, `msg.txt`.

---

## 9. Commands that answer "is it working?"

```bash
python3 -m unittest discover -s . -p 'test_*.py'   # 375 tests, ~210 s
python3 -m unittest test_cell_deployment            # 73 tests, ~128 s
python3 -m unittest test_power_features             # 67 tests, ~14 s
python3 bonsai_chat.py --selftest                   # 31/31
python3 bonsai_chat.py --mock --doctor --json       # 12 checks, shape unchanged
python3 bonsai_chat.py --mock --benchmark --bench-samples 2
python3 bonsai_chat.py --mock --batch file.jsonl --out out.jsonl
ps aux | grep -cE "fake_llama|fake_cloud"           # must be 0 after any run
```

---

## 10. Draft final report (verify the numbers, then deliver)

**First line, as required:** *The GPU deployment path has still never been run against real
Bonsai 2 weights on real hardware in this repository. What is verified is the decision
logic, the recovery logic and the wire protocol, against a fake GPU.*

Then, briefly:

- **Stability**: the cell's real module-level body now runs offline to READY in ~0.8 s
  (`cell_harness.py`), driven by 73 end-to-end tests that then break it on purpose. Two
  watchdogs with health-gated liveness, a stall-bounded generation probe, and restarts
  bounded in count and rate; crash/OOM/hang told apart from the server's own log; model and
  tunnel repairs that cannot cascade; every heal recorded in `diagnostics.json` and
  `heartbeat.json`.
- **Bugs the harness found** (§6, items 1, 2, 3) — with regression tests.
- **Features**: vision, multi-slot, KV4 (all opt-in, capability-detected, cost stated,
  refused rather than faked); client config with inspectable precedence, personas,
  branching, multi-endpoint, MCP, mid-stream tool execution, structured output proved
  before use, budgets, batch, JSON/HTML export, importable API; no second web UI, and why.
- **Verification**: 235 → 375 tests, all offline and deterministic; `--selftest` 20 → 31;
  mock server 26 → 30 scenarios; `test_deployment.FailureReportingTests` (which asserts the
  client and cell versions match) passes.
- **What could not be verified** (say this plainly): real VRAM residency and real decode
  speed; the official `mmproj` file names and sizes the fake advertises; whether a PrismML
  build really prints `--mmproj` in `--help` the way the tests assume; every number the
  deployment prints for real hardware; MCP against a real MCP server (only the bundled
  fake); the tunnel against real Cloudflare (a real loopback proxy stands in); and
  inference quality of the 27B model, which needs a GPU notebook runtime.
- **Suggested next step**: one real T4 run of `colab_kaggle_cell.py`, then check the
  printed diagnostics against what the fakes promised.

---

## 11. If continuing work

- Re-read §7 before touching the fakes; most "mysterious" failures there were one of those
  traps.
- Any new capability must follow the same rule the rest of the code follows: **discover it,
  prove it with a real request, and report `unknown` when it was never tested.**
- Every bug fix gets a regression test (Phase C rule) — and the test count must not go
  down.
- `--doctor --json`, session schema 2 and v0.5.0 CLI flags are compatibility surfaces:
  additive changes only.
