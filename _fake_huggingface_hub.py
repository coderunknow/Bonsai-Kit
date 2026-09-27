#!/usr/bin/env python3
"""Fake huggingface_hub, standing in for the one the cell pip-installs.

Provides just the two names the cell imports — ``HfApi`` and ``hf_hub_download``.
``model_info`` returns canned LFS metadata (size, and optionally the real SHA-256 of the
file that will be produced); ``hf_hub_download`` writes a synthetic GGUF whose metadata
satisfies the cell's identity checks. Files are sparse, so a 7.21 GB "download" costs
no disk and no time.

Config comes from FAKE_HF_CONFIG:
    files            {filename: {'size': int}}
    sha              report (and require) a SHA-256 digest
    corrupt_identity write metadata the identity check must reject
    corrupt_bytes    flip bytes after the download so the SHA-256 check fails
    missing          filenames to leave out of the repo listing
"""

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_writer  # noqa: E402


def _config():
    try:
        return json.loads(Path(os.environ['FAKE_HF_CONFIG']).read_text())
    except Exception:
        return {'files': {}}


def _run_dir():
    return Path(os.environ.get('FAKE_GPU_DIR', '/tmp/fake-hf')).parent


def _sha_cache():
    return _run_dir() / 'hf_sha_cache.json'


def _digest(filename, size, corrupt):
    key = '%s|%d|%s' % (filename, size, bool(corrupt))
    cache_path = _sha_cache()
    cache = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except ValueError:
            cache = {}
    if key in cache:
        return cache[key]
    digest, _ = gguf_writer.sha256_of(int(size), corrupt=bool(corrupt))
    cache[key] = digest
    try:
        tmp = cache_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return digest


class Sibling:
    def __init__(self, rfilename, size, lfs=None):
        self.rfilename = rfilename
        self.size = size
        self.lfs = lfs


class ModelInfo:
    def __init__(self, siblings):
        self.siblings = siblings
        self.id = 'prism-ml/Ternary-Bonsai-2-27B-gguf'


class HfApi:
    def model_info(self, repo_id, files_metadata=False, **kw):
        cfg = _config()
        corrupt = bool(cfg.get('corrupt_identity'))
        with_sha = bool(cfg.get('sha'))
        missing = set(cfg.get('missing') or ())
        siblings = []
        for name, spec in (cfg.get('files') or {}).items():
            if name in missing:
                continue
            size = int(spec.get('size', 1_000_000))
            lfs = dict(spec.get('lfs') or {})
            lfs.setdefault('size', size)
            if with_sha and 'sha256' not in lfs:
                lfs['sha256'] = _digest(name, size, corrupt)
            siblings.append(Sibling(name, size, lfs))
        return ModelInfo(siblings)


def hf_hub_download(repo_id, filename, local_dir=None, token=None, **kw):
    cfg = _config()
    files = cfg.get('files') or {}
    spec = files.get(filename)
    if spec is None:
        raise RuntimeError('Entry Not Found: %s in %s' % (filename, repo_id))
    size = int(spec.get('size', 1_000_000))
    dest_dir = Path(local_dir) if local_dir else Path.cwd()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    corrupt = bool(cfg.get('corrupt_identity'))
    gguf_writer.write_gguf(str(dest), size, corrupt=corrupt)
    if cfg.get('corrupt_bytes'):
        # same size, different content: the SHA-256 check must catch this
        with dest.open('r+b') as f:
            f.seek(0)
            f.write(b'GGUF' + b'\x00' * 8)
            f.seek(max(0, size - 16))
            f.write(b'corrupted-tail!!')
        f = dest.open('r+b')
        f.truncate(size)
        f.close()
    return str(dest)
