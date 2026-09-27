"""Offline fake-GPU harness for colab_kaggle_cell.py.

v0.5.0 verified the deployment cell with unit tests on its pure helpers plus an AST
check that every name resolves. The cell's *decision logic* — the part that actually
starts processes, chooses a context, climbs the OOM ladder, splits layers across two
GPUs and publishes a tunnel — had never executed anywhere. That was the single largest
stability risk in the project.

This module makes it executable offline. It supplies, all injectable:

  * a fake ``nvidia-smi`` (1xT4, 2xT4, 1xL4, no GPUs, mismatched GPUs, GPUs already
    occupied by *other* work) with real per-process VRAM attribution;
  * a fake CUDA driver (``cuInit`` / ``cuDeviceGetCount``) compiled from C, so the
    driver check runs for real through ``ctypes``;
  * a fake **PrismML llama-server executable**: a compiled launcher that owns the PID
    (so ``/proc/<pid>/exe`` really points at ``bin/cuda/llama-server``, which is what
    the cell's process-identity checks verify) and runs a Python server that honours
    ``--port --api-key --alias -c -m``, writes a realistic startup log including the
    ``KV buffer size = ... MiB`` lines, and serves /health, /props, /v1/models,
    /v1/chat/completions (streaming and not). It can be told to boot slowly, OOM above
    a given context, crash after N requests, hang mid-generation, or reject a flag;
  * a fake Hugging Face layer: canned ``model_info`` with LFS size + sha256, and a
    generated GGUF (sparse, so a 7.21 GB file costs no disk) whose metadata satisfies
    the cell's identity checks;
  * a fake ``cloudflared`` that publishes a ``*.trycloudflare.com`` URL into its log and
    backs it with a real loopback proxy, so "public" requests are real HTTP — and can
    die on cue;
  * a fake ``/proc`` for PID-recycling tests (the pattern test_deployment.py already
    uses).

Runner usage::

    with CellRun(gpus=[T4, T4], oom_above_ctx=16384) as run:
        run.execute()                 # runs the cell's real module-level body
        run.output                    # everything the cell printed
        run.state                     # state.json, if it got that far
        run.diagnostics               # diagnostics.json, if it got that far

Everything is offline and deterministic: no network, no GPU, no model download. The
only build-time requirement is a C compiler for the launcher and the fake libcuda; if
one is unavailable the tests that need it skip with that reason stated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import re
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELL = HERE / 'colab_kaggle_cell.py'

# ---------------------------------------------------------------- GPU profiles

def gpu(index=0, name='Tesla T4', total=15360, used=0, cap='7.5', driver='535.104.05',
        pci='00000000:00:04.0'):
    """One fake GPU row, in the units nvidia-smi -q reports (MiB)."""
    return dict(index=index, name=name, total=total, used=used, cap=cap, driver=driver,
                pci=pci)


T4 = gpu()
T4X2 = [gpu(0), gpu(1, pci='00000000:00:05.0')]
L4 = gpu(name='Tesla L4', total=23034, cap='8.9')
SMALL = gpu(name='NVIDIA GeForce GTX 1650', total=4096, cap='7.5')
MISMATCH = [gpu(0), gpu(1, name='Tesla L4', total=23034, cap='8.9',
                pci='00000000:00:05.0')]

# ---------------------------------------------------------------- C sources

LAUNCHER_C = r'''
/* Fake llama-server / cloudflared launcher.
 *
 * The cell never trusts a PID alone: it verifies /proc/<pid>/exe and argv[0]. A
 * Python script run through a shebang would put the interpreter in /proc/<pid>/exe,
 * so the fake deployment binary has to be a real ELF that keeps its identity while
 * the actual behaviour lives in a Python file. This process owns the PID, forks the
 * interpreter, forwards signals and mirrors the child's exit status.
 */
#include <signal.h>
#include <sys/prctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static pid_t child_pid = 0;

static void forward(int sig) {
    if (child_pid > 0) kill(child_pid, sig);
    _exit(128 + sig);
}

int main(int argc, char **argv) {
    const char *script = FAKE_SCRIPT;
    const char *py = "/usr/bin/env";
    int i, status;
    char **nargv;

    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);
    signal(SIGQUIT, forward);
    fflush(NULL);

    child_pid = fork();
    if (child_pid < 0) { perror("fork"); return 120; }
    if (child_pid == 0) {
        /* llama-server is a single process. If this launcher is killed outright the
         * Python half must die with it, or "kill the server" would leave a live
         * server behind and every recovery test would be meaningless. */
        prctl(PR_SET_PDEATHSIG, SIGTERM);
        nargv = malloc(sizeof(char *) * (argc + 3));
        nargv[0] = (char *)py;
        nargv[1] = "python3";
        nargv[2] = (char *)script;
        for (i = 1; i < argc; i++) nargv[i + 2] = argv[i];
        nargv[argc + 2] = NULL;
        execv(py, nargv);
        execvp("python3", nargv + 1);
        perror("exec");
        _exit(127);
    }
    while (waitpid(child_pid, &status, 0) < 0) {
        if (errno_retry() != 0) { _exit(121); }
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return 128 + WTERMSIG(status);
}
'''

LAUNCHER_C = LAUNCHER_C.replace('errno_retry() != 0', 'errno != EINTR').replace(
    '#include <string.h>', '#include <string.h>\n#include <errno.h>')

CUDA_C = r'''
/* Fake CUDA driver. The cell calls cuInit(0) and cuDeviceGetCount(&n) through
 * ctypes; both must behave like the real thing (return codes, not exceptions).
 */
#include <stdlib.h>
#include <string.h>

int cuInit(unsigned int flags) {
    const char *fail = getenv("FAKE_CUDA_FAIL");
    (void)flags;
    if (fail && strcmp(fail, "init") == 0) return 100;      /* CUDA_ERROR_NO_DEVICE */
    return 0;                                               /* CUDA_SUCCESS */
}

int cuDeviceGetCount(int *count) {
    const char *n = getenv("FAKE_CUDA_DEVICES");
    const char *fail = getenv("FAKE_CUDA_FAIL");
    if (fail && strcmp(fail, "count") == 0) return 100;
    if (count) *count = n ? atoi(n) : 1;
    return 0;
}

int cuDriverGetVersion(int *version) {
    if (version) *version = 12020;
    return 0;
}
'''

# ---------------------------------------------------------------- fake nvidia-smi

NVIDIA_SMI = r'''#!/usr/bin/env python3
"""Fake nvidia-smi. Reads the GPU inventory and the running compute apps from
FAKE_GPU_DIR (config.json + apps/<pid>.json) so the numbers are consistent with the
fake servers that are actually up.
"""
import json, os, sys
from pathlib import Path

root = Path(os.environ.get('FAKE_GPU_DIR', '/tmp/fake-gpu'))
cfg = {}
try:
    cfg = json.loads((root / 'config.json').read_text())
except Exception:
    pass
gpus = cfg.get('gpus', [{'index': 0, 'name': 'Tesla T4', 'total': 15360, 'used': 0,
                         'cap': '7.5', 'driver': '535.104.05', 'pci': '00000000:00:04.0'}])


def live_apps():
    apps = []
    d = root / 'apps'
    if not d.is_dir():
        return apps
    for f in d.glob('*.json'):
        pid = f.stem
        if not (Path('/proc') / pid).exists():
            try:
                f.unlink()
            except OSError:
                pass
            continue
        try:
            apps.append(json.loads(f.read_text()))
        except Exception:
            continue
    return apps


def per_gpu_used():
    used = {}
    for app in live_apps():
        for k, v in (app.get('per_gpu') or {}).items():
            used[int(k)] = used.get(int(k), 0) + int(v)
    return used


args = sys.argv[1:]
if args and args[0] == 'topo' and '-m' in args:
    names = ['GPU%d' % g['index'] for g in gpus] + ['CPU Affinity', 'NUMA Affinity']
    print('\t'.join(names))
    for g in gpus:
        row = ['GPU%d' % g['index']] + ['PHB' if i != g['index'] else 'X ' for i in range(len(gpus))]
        row += ['0-1', '0']
        print('\t'.join(row))
    sys.exit(0)

if any(a.startswith('--query-compute-apps') for a in args):
    fmt = next((a for a in args if a.startswith('--format=')), '')
    if 'noheader' not in fmt:
        print('pid, used_gpu_memory')
    for app in live_apps():
        total = sum(int(v) for v in (app.get('per_gpu') or {}).values())
        print('%s, %d' % (app['pid'], total))
    sys.exit(0)

if any(a.startswith('--query-gpu') for a in args):
    q = next(a for a in args if a.startswith('--query-gpu'))
    cols = (q.split('=', 1)[1] if '=' in q else args[args.index(q) + 1]).split(',')
    fmt = next((a for a in args if a.startswith('--format=')), '')
    used = per_gpu_used()
    if 'noheader' not in fmt:
        print(','.join(cols))
    for g in gpus:
        i = int(g['index'])
        u = int(g.get('used', 0)) + used.get(i, 0)
        vals = {'index': str(i), 'name': g['name'], 'memory.total': str(g['total']),
                'memory.free': str(max(0, int(g['total']) - u)), 'memory.used': str(u),
                'compute_cap': g.get('cap', '7.5'), 'driver_version': g.get('driver', ''),
                'pci.bus_id': g.get('pci', '')}
        print(', '.join(str(vals.get(c, '')) for c in cols))
    sys.exit(0)

print('+-----------------------------------------------------------------------------+')
print('| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |')
print('|-------------------------------+----------------------+----------------------+')
for g in gpus:
    u = int(g.get('used', 0)) + per_gpu_used().get(int(g['index']), 0)
    print('| %-3d %-24s On  | %5d MiB / %5d MiB |' % (int(g['index']), g['name'][:24], u, g['total']))
print('+-----------------------------------------------------------------------------+')
'''

# ---------------------------------------------------------------- fake git

FAKE_GIT = r'''#!/usr/bin/env python3
"""Fake git. `git clone` materialises the Bonsai-demo tree (including the
scripts/download_binaries.sh the cell then runs for real); `git pull` is a no-op.
"""
import os, sys
from pathlib import Path

script = os.environ.get('FAKE_DOWNLOAD_SCRIPT', '')
args = [a for a in sys.argv[1:] if not a.startswith('-')]
if not args:
    sys.exit(0)
verb = args[0]
if verb == 'clone':
    dest = Path(args[-1])
    (dest / '.git').mkdir(parents=True, exist_ok=True)
    (dest / 'scripts').mkdir(parents=True, exist_ok=True)
    body = (
        '#!/bin/sh\n'
        '# Fake Bonsai-demo scripts/download_binaries.sh: installs the PrismML CUDA build\n'
        '# that the harness compiled, exactly where the real script would put it.\n'
        'set -e\n'
        'mkdir -p bin/cuda\n'
        'cp "$FAKE_LLAMA_BIN" bin/cuda/llama-server.new\n'
        'chmod +x bin/cuda/llama-server.new\n'
        'mv -f bin/cuda/llama-server.new bin/cuda/llama-server\n'
        'printf %s "$FAKE_LLAMA_STAMP" > bin/cuda/.llama_release\n'
        'echo "downloaded PrismML llama.cpp CUDA build ($FAKE_LLAMA_STAMP)"\n'
    )
    (dest / 'scripts/download_binaries.sh').write_text(body)
    (dest / 'scripts/download_binaries.sh').chmod(0o755)
    sys.exit(0)
if verb == 'pull':
    sys.exit(0)
sys.exit(0)
'''

# ---------------------------------------------------------------- build

_CACHE = {}


def _compile(dest: Path, source: str, name: str, extra=()):
    src = dest / (name + '.c')
    src.write_text(source)
    cc = os.environ.get('CC') or shutil.which('gcc') or shutil.which('cc')
    if not cc:
        return None
    out = dest / name
    proc = subprocess.run([cc, '-O1', '-w', *extra, '-o', str(out), str(src)],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f'cannot compile {name}: {proc.stderr[:400]}')
    return out


def _with_script(source: str, script: Path) -> str:
    """Bake the interpreter script path into the launcher as a C string literal.

    Passing it with -D would mean fighting two layers of quote escaping (gcc's and
    the shell's); embedding it in the generated source has neither.
    """
    literal = json.dumps(str(script))
    if 'const char *script = FAKE_SCRIPT;' not in source:
        raise ValueError('launcher source has lost its FAKE_SCRIPT anchor')
    return source.replace('const char *script = FAKE_SCRIPT;',
                          'const char *script = %s;' % literal)


def build_fakes(root: Path):
    """Materialise every fake into `root`. Returns a dict of paths."""
    if root in _CACHE:
        return _CACHE[root]
    (root / 'bin').mkdir(parents=True, exist_ok=True)
    (root / 'lib').mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent

    smi = root / 'bin/nvidia-smi'
    smi.write_text(NVIDIA_SMI)
    smi.chmod(0o755)
    git = root / 'bin/git'
    git.write_text(FAKE_GIT)
    git.chmod(0o755)

    libcuda = _compile(root / 'lib', CUDA_C, 'libcuda.so.1', extra=('-shared', '-fPIC'))

    llama_script = root / 'fake_llama_server.py'
    llama_script.write_text((here / '_fake_llama_server.py').read_text())
    tunnel_script = root / 'fake_cloudflared.py'
    tunnel_script.write_text((here / '_fake_cloudflared.py').read_text())
    (root / 'gguf_writer.py').write_text((here / '_gguf_writer.py').read_text())
    (root / 'huggingface_hub.py').write_text((here / '_fake_huggingface_hub.py').read_text())

    llama_bin = cloud_bin = None
    if libcuda:
        llama_bin = _compile(root, _with_script(LAUNCHER_C, llama_script), 'llama-server')
        cloud_bin = _compile(root, _with_script(LAUNCHER_C, tunnel_script), 'cloudflared')
    out = dict(root=root, launcher=llama_bin, libcuda=libcuda, llama_bin=llama_bin,
               cloud_bin=cloud_bin, llama_script=llama_script, tunnel_script=tunnel_script)
    _CACHE[root] = out
    return out


def has_compiler() -> bool:
    return bool(shutil.which('gcc') or shutil.which('cc') or os.environ.get('CC'))


requires_compiler = unittest.skipUnless(has_compiler(),
                                        'a C compiler is required to build the fake '
                                        'llama-server/cloudflared executables')

# ---------------------------------------------------------------- GGUF/model fakes

PTQ1_0_BYTES = 5_950_000_000      # official PTQ1_0 packing, 5.95 GB
PQ2_0_BYTES = 7_210_000_000       # official PQ2_0 packing, 7.21 GB
MMProj_BYTES = 676_000_000        # mmproj HQQ/Q8_0, 0.63 GB


def hf_config(files=None, sha=False, corrupt_identity=False, missing=None,
              corrupt_bytes=False):
    """Configuration for the fake huggingface_hub module.

    `corrupt_bytes` rewrites the file after the download, at the same size, so the
    SHA-256 check is the only thing that can catch it.
    """
    return dict(files=files, sha=bool(sha), corrupt_identity=bool(corrupt_identity),
                missing=list(missing or ()), corrupt_bytes=bool(corrupt_bytes))


def default_model_files(sha=False):
    return {
        'Ternary-Bonsai-2-27B-PQ2_0.gguf': dict(size=PQ2_0_BYTES),
        'Ternary-Bonsai-2-27B-PTQ1_0.gguf': dict(size=PTQ1_0_BYTES),
        'mmproj-Q8_0.gguf': dict(size=MMProj_BYTES),
    }


# ---------------------------------------------------------------- the runner

class CellRun:
    """Execute the cell's real module-level deployment body against the fakes."""

    def __init__(self, gpus=None, cuda_devices=None, cuda_fail=None,
                 server=None, tunnel=None, hf=None, meminfo_gib=(16.0, 12.0),
                 env=None, install_hf=True, ram_total=None, base=None, gpu_dir=None):
        self.tmp = tempfile.TemporaryDirectory(prefix='bonsai-cell-')
        self._cleanups = []
        self.root_path = Path(self.tmp.name)
        # `base` lets a second CellRun target a deployment that is already live, which
        # is how a rerun-with-a-different-key is tested.
        self.base = Path(base) if base else self.root_path / 'content'
        self.base.mkdir(parents=True, exist_ok=True)
        self._owns_base = base is None
        self.root = self.base / 'bonsai2-api'          # where the cell works
        # `gpu_dir` lets a second run see the same GPU inventory the first one is
        # holding, which is what makes "a second cell run adopts the live server" real.
        self.gpu_root = Path(gpu_dir) if gpu_dir else self.root_path / 'gpu'
        (self.gpu_root / 'apps').mkdir(parents=True, exist_ok=True)
        self.fakes = build_fakes(self.root_path / 'fakes')
        self.output_lines = []
        self.error = None
        self.ns = None

        total, avail = meminfo_gib
        meminfo = self.root_path / 'meminfo'
        meminfo.write_text(
            f'MemTotal:       {int(total * 1024 * 1024)} kB\n'
            f'MemFree:        {int(avail * 1024 * 1024)} kB\n'
            f'MemAvailable:   {int(avail * 1024 * 1024)} kB\n')
        self.meminfo = str(meminfo)

        (self.gpu_root / 'config.json').write_text(json.dumps(
            {'gpus': list(gpus if gpus is not None else [T4])}))

        self.cuda_lib = str(self.fakes['libcuda']) if self.fakes.get('libcuda') else 'libcuda.so.1'
        self.server_cfg = dict(server or {})
        self.tunnel_cfg = dict(tunnel or {})
        self.hf_cfg = hf or hf_config(default_model_files())
        (self.root_path / 'server-config.json').write_text(json.dumps(self.server_cfg))
        (self.root_path / 'tunnel-config.json').write_text(json.dumps(self.tunnel_cfg))
        (self.root_path / 'hf-config.json').write_text(json.dumps(self.hf_cfg))

        self.env_overrides = {
            'BONSAI_ROOT': str(self.base),
            'FAKE_GPU_DIR': str(self.gpu_root),
            'FAKE_SERVER_CONFIG': str(self.root_path / 'server-config.json'),
            'FAKE_TUNNEL_CONFIG': str(self.root_path / 'tunnel-config.json'),
            'FAKE_HF_CONFIG': str(self.root_path / 'hf-config.json'),
            'FAKE_LLAMA_BIN': str(self.fakes['llama_bin'] or ''),
            'FAKE_LLAMA_STAMP': self.server_cfg.get('stamp', 'prism-b10658-cuda12'),
            'FAKE_CUDA_DEVICES': str(cuda_devices if cuda_devices is not None
                                     else len(gpus or [T4])),
            'FAKE_HF_SCRIPT_DIR': str(self.fakes['root']),
        }
        if cuda_fail:
            self.env_overrides['FAKE_CUDA_FAIL'] = cuda_fail
        if self.fakes['cloud_bin']:
            self.env_overrides['FAKE_CLOUDFLARED_BIN'] = str(self.fakes['cloud_bin'])
        self.env_overrides['FAKE_RUN_DIR'] = str(self.root_path)
        self.env_overrides.update(env or {})
        self._install_hf = install_hf

    # -- environment -------------------------------------------------
    def __enter__(self):
        self._saved_env = dict(os.environ)
        path = [str(self.fakes['root'] / 'bin')]
        if os.environ.get('PATH'):
            path.append(os.environ['PATH'])
        ld = [str(self.fakes['root'] / 'lib')]
        if os.environ.get('LD_LIBRARY_PATH'):
            ld.append(os.environ['LD_LIBRARY_PATH'])
        os.environ['PATH'] = ':'.join(path)
        os.environ['LD_LIBRARY_PATH'] = ':'.join(ld)
        os.environ.update({k: str(v) for k, v in self.env_overrides.items()})
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self._saved_env)
        self.cleanup()
        return False

    def cleanup(self):
        self.kill_started_processes()
        for fn in reversed(self._cleanups):
            try:
                fn()
            except Exception:
                pass
        self._cleanups.clear()
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    # -- execution ---------------------------------------------------
    def execute(self, expect_success=True):
        """Run the cell's deployment body.

        The invocation log is truncated first, so `invocations()` always describes
        *this* execution: a rerun that reattaches to a live server must show none.
        """
        import ast
        import types
        (self.root_path / 'invocations.jsonl').unlink(missing_ok=True)
        source = CELL.read_text()
        tree = ast.parse(source, filename=str(CELL))
        definitions = ast.Module(body=[n for n in tree.body if not isinstance(n, ast.Try)],
                                 type_ignores=[])
        tries = [n for n in tree.body if isinstance(n, ast.Try)]
        assert len(tries) == 1, 'the cell must keep exactly one top-level try block'

        ns = {'__name__': 'bonsai_cell_under_test'}
        exec(compile(definitions, str(CELL), 'exec'), ns)
        self._apply_seams(ns)
        body = ast.Module(body=[tries[0]], type_ignores=[])
        try:
            exec(compile(body, str(CELL), 'exec'), ns)
        except Exception as exc:                     # noqa: BLE001 - reported, not hidden
            self.error = exc
        self.ns = ns
        if expect_success and self.error is not None:
            raise AssertionError('cell failed: %s\n%s' % (self.error, self.output)) \
                from self.error
        return self

    def _apply_seams(self, ns):
        real_http = ns['http']

        def http_seam(url, key=None, body=None, timeout=60):
            return real_http(self._rewrite(url), key, body, timeout)

        real_open = ns['urllib'].request.urlopen

        def open_seam(req, timeout=None):
            try:
                url = req.full_url
            except AttributeError:
                url = req
            new = self._rewrite(url)
            if new != url:
                req = ns['urllib'].request.Request(
                    new, data=req.data, headers=dict(req.headers), method=req.get_method())
            return real_open(req, timeout=timeout)

        ns['HTTP'] = http_seam
        ns['URL_OPEN'] = open_seam
        ns['MEMINFO'] = self.meminfo
        # dlopen only consults LD_LIBRARY_PATH as it was at process start, so the
        # fake driver is addressed by absolute path rather than by search-path.
        ns['CUDA_LIB'] = self.cuda_lib
        ns['PROC_ROOT'] = ns['Path']('/proc')
        ns['POLL_INTERVAL'] = 0.02
        ns['TUNNEL_ATTEMPTS'] = 60
        ns['CLOUDFLARED_MIN_BYTES'] = 0
        ns['STARTUP_TIMEOUT'] = float(self.server_cfg.get('startup_timeout', 20))
        ns['GPU_MEMORY'] = ns['gpu_process_memory']
        ns['install'] = lambda module, package: None
        ns['log'] = self._log

        if self._install_hf:
            import importlib.util
            path = self.fakes['root'] / 'huggingface_hub.py'
            spec = importlib.util.spec_from_file_location('huggingface_hub', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules['huggingface_hub'] = mod
            self._cleanups.append(lambda: sys.modules.pop('huggingface_hub', None))

        # a fake cloudflared the cell would otherwise download
        fake_cloud = self.fakes['cloud_bin']
        if fake_cloud:
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / 'cloudflared'
            if not target.exists():
                shutil.copy(fake_cloud, target)
            target.chmod(0o755)

    def _log(self, msg=''):
        self.output_lines.append(str(msg))

    def _rewrite(self, url):
        """Map a published *.trycloudflare.com URL onto its loopback proxy."""
        if not isinstance(url, str) or '.trycloudflare.com' not in url:
            return url
        try:
            mapping = json.loads((self.root_path / 'tunnel-map.json').read_text())
        except Exception:
            return url
        for public, port in mapping.items():
            if url.startswith(public):
                return 'http://127.0.0.1:%d%s' % (int(port), url[len(public):])
        return url

    # -- observations ------------------------------------------------
    @property
    def output(self):
        return '\n'.join(self.output_lines)

    @property
    def printed(self):
        return self.output

    def find(self, needle):
        return [ln for ln in self.output_lines if needle in ln]

    @property
    def ready(self):
        return 'READY' in self.output

    @property
    def state(self):
        path = self.root / 'state.json'
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    @property
    def diagnostics(self):
        path = self.root / 'diagnostics.json'
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def write_state(self, state):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'state.json').write_text(json.dumps(state))

    # -- knobs for the tests ------------------------------------------
    def set_server_cfg(self, **kwargs):
        """Rewrite the fake server's behaviour between restarts."""
        cfg = dict(self.server_cfg)
        cfg.update(kwargs)
        self.server_cfg = cfg
        (self.root_path / 'server-config.json').write_text(json.dumps(cfg))
        return cfg

    def set_tunnel_cfg(self, **kwargs):
        cfg = dict(self.tunnel_cfg)
        cfg.update(kwargs)
        self.tunnel_cfg = cfg
        (self.root_path / 'tunnel-config.json').write_text(json.dumps(cfg))
        return cfg

    @property
    def supervisors(self):
        return list((self.ns or {}).get('supervisors') or ())

    def supervisor(self, name):
        for sup in self.supervisors:
            if sup.name == name:
                return sup
        raise AssertionError('no supervisor named %r (have %s)'
                             % (name, [s.name for s in self.supervisors]))

    def _verified_pid(self, which):
        state = self.state or {}
        return state.get(which)

    def kill_server(self, sig=None):
        """SIGKILL the server the cell recorded, after checking it is really ours."""
        import signal
        pid = self._verified_pid('server_pid')
        assert pid, 'no server pid in state.json'
        assert (Path('/proc') / str(pid)).exists(), 'recorded server pid is gone'
        os.kill(int(pid), sig or signal.SIGKILL)
        return int(pid)

    def kill_tunnel(self):
        import signal
        pid = self._verified_pid('tunnel_pid')
        assert pid, 'no tunnel pid in state.json'
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return int(pid)

    def kill_started_processes(self):
        """SIGKILL everything this run started, and only that.

        Every fake is identified by this run's unique paths, so an unrelated process is
        never touched. Without this a leaked fake server outlives its temp directory and
        quietly eats the machine — which is exactly the kind of failure a harness must
        not have.
        """
        import signal
        pids = set()
        apps = self.gpu_root / 'apps'
        if apps.is_dir():
            pids.update(entry.stem for entry in apps.glob('*.json'))
        invocations = self.root_path / 'invocations.jsonl'
        if invocations.is_file():
            for line in invocations.read_text().splitlines():
                try:
                    pids.add(str(int(json.loads(line)['pid'])))
                except (ValueError, KeyError):
                    pass
        markers = [str(self.base).encode(), str(self.fakes['root']).encode()]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                cmdline = (entry / 'cmdline').read_bytes()
            except OSError:
                continue
            if any(m in cmdline for m in markers):
                pids.add(entry.name)
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (OSError, ValueError):
                pass

    def servers_started(self):
        """How many distinct fake llama-server instances the cell launched."""
        return len(self.server_pids())

    def server_pids(self):
        out = []
        d = self.gpu_root / 'apps'
        if d.is_dir():
            for f in sorted(d.glob('*.json')):
                try:
                    out.append(json.loads(f.read_text()))
                except Exception:
                    pass
        return out

    def invocations(self):
        """Every fake server start: pid, argv and the context it was asked for."""
        path = self.root_path / 'invocations.jsonl'
        if not path.is_file():
            return []
        out = []
        for line in path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def contexts_attempted(self):
        """`-c` values the cell asked for, in order."""
        return [inv.get('ctx') for inv in self.invocations()]

    def arg_values(self, flag):
        vals = []
        for inv in self.invocations():
            argv = inv.get('argv') or []
            for i, a in enumerate(argv):
                if a == flag and i + 1 < len(argv):
                    vals.append(argv[i + 1])
        return vals

    def server_args(self):
        return [' '.join(inv['argv']) for inv in self.invocations()]

    def tunnel_url(self):
        import re
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', self.output)
        return m.group(0) if m else None
