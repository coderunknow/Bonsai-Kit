"""Build a synthetic GGUF that satisfies the cell's identity checks.

The official file is 5.95 GB (PTQ1_0) or 7.21 GB (PQ2_0). We write the header +
metadata + tensor descriptors and then truncate to the exact size, which produces a
sparse file: full size on stat(), no disk actually consumed. The tensor descriptors
carry the parameter count the cell checks for (~27.36B); quantisation types are
opaque, exactly as the real reader treats them.
"""

import struct

MAGIC = b'GGUF'
VERSION = 3

STR, UINT32, INT32, FLOAT32, BOOL, UINT64 = 8, 4, 5, 6, 7, 10


def _string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def _value(kind, value):
    if kind == STR:
        return _string(value)
    if kind == UINT32:
        return struct.pack('<I', value)
    if kind == INT32:
        return struct.pack('<i', value)
    if kind == FLOAT32:
        return struct.pack('<f', value)
    if kind == BOOL:
        return struct.pack('<B', 1 if value else 0)
    if kind == UINT64:
        return struct.pack('<Q', value)
    raise ValueError('unsupported GGUF value type %r' % kind)


def tensor_record(name, dims, type_id=143, offset=0):
    """A tensor descriptor: name, rank, dims, quant type (opaque), data offset."""
    out = _string(name) + struct.pack('<I', len(dims))
    for d in dims:
        out += struct.pack('<Q', d)
    return out + struct.pack('<IQ', type_id, offset)   # offset is u64 in GGUF


def bonsai_metadata(corrupt=False, arch='qwen35', name='Ternary-Bonsai-2-27B',
                    size_label='27B'):
    """The metadata shape the cell's identity check accepts — or, if `corrupt`, one it
    must reject outright rather than silently substitute."""
    if corrupt:
        return [('general.architecture', STR, 'llama'),
                ('general.name', STR, 'Mistral-7B-Instruct-v0.2'),
                ('general.size_label', STR, '7B'),
                ('general.file_type', UINT32, 2)]
    return [('general.architecture', STR, arch),
            ('general.name', STR, name),
            ('general.size_label', STR, size_label),
            ('general.file_type', UINT32, 143),
            ('general.basename', STR, 'Ternary-Bonsai-2-27B'),
            ('prism.hadamard_transform', BOOL, True),
            ('qwen35.block_count', UINT32, 64),
            ('qwen35.embedding_length', UINT32, 15360),
            ('qwen35.attention.head_count', UINT32, 32),
            ('qwen35.linear_attention_layers', UINT32, 48),
            ('tokenizer.ggml.model', STR, 'gpt2')]


def bonsai_tensors(total_params=27_360_000_000):
    """Tensor descriptors summing to `total_params`, built from plausible shapes."""
    shapes = [('token_embd.weight', (151936, 15360)),
              ('output.weight', (151936, 15360))]
    for i in range(8):
        shapes.append(('blk.%d.attn_q.weight' % i, (15360, 15360)))
    counted = 0
    for _, dims in shapes:
        n = 1
        for d in dims:
            n *= d
        counted += n
    rest = total_params - counted
    if rest > 0:
        shapes.append(('blk.0.ffn_gate.weight', (rest,)))
    return shapes


def write_gguf(path, size, metadata=None, tensors=None, corrupt=False):
    """Write a GGUF header of `size` bytes. Returns the number of header bytes."""
    metadata = metadata if metadata is not None else bonsai_metadata(corrupt=corrupt)
    tensors = tensors if tensors is not None else bonsai_tensors()
    body = b''
    for key, kind, value in metadata:
        body += _string(key) + struct.pack('<I', kind) + _value(kind, value)
    for name, dims in tensors:
        body += tensor_record(name, dims)
    header = MAGIC + struct.pack('<I', VERSION) + struct.pack('<QQ', len(tensors),
                                                              len(metadata)) + body
    with open(path, 'wb') as f:
        f.write(header)
        f.truncate(int(size))
    return len(header)


def sha256_of(size, metadata=None, tensors=None, corrupt=False, tmp=None):
    """The digest of the file `write_gguf` would produce, without keeping it around."""
    import hashlib
    import tempfile
    import os
    metadata = metadata if metadata is not None else bonsai_metadata(corrupt=corrupt)
    tensors = tensors if tensors is not None else bonsai_tensors()
    tmp = tmp or tempfile.mkdtemp(prefix='gguf-sha-')
    p = os.path.join(tmp, 'probe.gguf')
    n = write_gguf(p, size, metadata, tensors)
    digest = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    os.remove(p)
    return digest.hexdigest(), n
