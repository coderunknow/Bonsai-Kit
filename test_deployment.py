"""Offline regression tests for notebook reruns; no CUDA or model download needed."""
import ast
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import Mock

# Execute only definitions, never the notebook's top-level deployment try block.
source = Path(__file__).with_name('colab_kaggle_cell.py')
tree = ast.parse(source.read_text(), filename=str(source))
definitions = ast.Module(body=[n for n in tree.body if not isinstance(n, ast.Try)], type_ignores=[])
ns = {'__name__': 'bonsai_test_definitions'}
exec(compile(definitions, str(source), 'exec'), ns)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'bonsai2-api'
        self.proc = Path(self.temp.name) / 'proc'
        self.proc.mkdir()
        self.binary = self.root / 'Bonsai-demo/bin/cuda/llama-server'
        self.binary.parent.mkdir(parents=True)
        self.binary.touch()
        self.model = self.root / 'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
        self.model.parent.mkdir()
        self.key = 'x' * 32
        self.request = self.good_request
        self.gpu_memory = Mock(return_value=9060)
        self.create_process(5815)

    def create_process(self, pid, binary=None, model=None, key=None, host='127.0.0.1'):
        entry = self.proc / str(pid)
        entry.mkdir()
        binary = binary or self.binary
        (entry / 'exe').symlink_to(binary)
        args = [str(binary), '-m', str(model or self.model), '--alias', ns['ALIAS'],
                '--host', host, '-ngl', '99', '-c', '8192', '--port', '45678',
                '--api-key', key or self.key]
        (entry / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in args) + b'\0')
        return entry

    def good_request(self, url, key, timeout):
        if url.endswith('/health'):
            return 200, {'status': 'ok'}
        if key is None:
            return 401, {}
        return 200, {'data': [{'id': ns['ALIAS']}]}

    def recover(self, state=None):
        return ns['managed_server'](self.root, state or {}, self.request,
                                    self.proc, self.gpu_memory)

    def test_low_free_vram_rerun_with_state_reuses_live_process(self):
        state = dict(server_pid=5815, port=45678, key=self.key, model=self.model.name,
                     ctx=32768, bench={'output': 10}, selected='Single GPU',
                     public='https://demo.trycloudflare.com', tunnel_pid=123)
        found = self.recover(state)
        self.assertEqual(found['server_pid'], 5815)
        self.assertEqual(found['model'], self.model.name)
        self.assertEqual(found['ctx'], 8192)  # live args, not stale state
        self.assertEqual(found['bench'], state['bench'])
        self.assertEqual(found['public'], state['public'])
        ns['require_capacity']([5850], found)  # user's failing T4 rerun
        self.assertEqual(ns['model_band']([5850], found), 'PTQ1_0')
        self.gpu_memory.assert_called_with(5815)

    def test_recover_when_old_run_failed_before_state_was_saved(self):
        found = self.recover()
        self.assertEqual(found['key'], self.key)
        self.assertNotIn('bench', found)
        ns['require_capacity']([5850], found)

    def test_unrelated_or_stale_process_is_not_adopted(self):
        self.assertIsNone(self.recover(dict(server_pid=5815, port=45678,
                                            key='wrong', model=self.model.name)))
        (self.proc / '5815' / 'exe').unlink()
        (self.proc / '5815' / 'exe').symlink_to('/usr/bin/llama-server')
        self.assertIsNone(self.recover())
        with self.assertRaisesRegex(RuntimeError, 'Insufficient free VRAM'):
            ns['require_capacity']([5850], None)
        ns['require_capacity']([9500], None)
        self.assertEqual(ns['model_band']([13000], None), 'PQ2_0')
        self.assertEqual(ns['model_band']([9500], None), 'PTQ1_0')

    def test_reject_wrong_model_or_public_binding(self):
        shutil.rmtree(self.proc / '5815')
        self.create_process(5815, model=Path('/tmp/other.gguf'))
        self.assertIsNone(self.recover())
        shutil.rmtree(self.proc / '5815')
        self.create_process(5815, host='0.0.0.0')
        self.assertIsNone(self.recover())

    def test_reject_unprotected_server_or_cpu_only(self):
        self.request = lambda url, key, timeout: (200, {'status': 'ok'} if url.endswith('/health') else {'data': [{'id': ns['ALIAS']}]})
        self.assertIsNone(self.recover())
        self.request = self.good_request
        self.gpu_memory.return_value = 0
        self.assertIsNone(self.recover())

    def test_reject_multiple_matching_servers_without_killing_anyone(self):
        self.create_process(5816)
        with self.assertRaisesRegex(RuntimeError, 'Multiple Bonsai servers'):
            self.recover()

    def test_gpu_memory_is_attributed_to_pid(self):
        fake_run = Mock(return_value=Mock(stdout='1234, 12000\n5815, 9060\n5815, 2\n'))
        original = ns['run']
        try:
            ns['run'] = fake_run
            self.assertEqual(ns['gpu_process_memory'](5815), 9062)
            self.assertEqual(ns['gpu_process_memory'](9999), 0)
        finally:
            ns['run'] = original

    def test_state_is_atomic_and_private(self):
        state_path = self.root / 'state.json'
        ns['save_state'](state_path, {'key': self.key})
        self.assertEqual(json.loads(state_path.read_text())['key'], self.key)
        self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)
        ns['save_state'](state_path, {'key': 'new'})
        self.assertEqual(json.loads(state_path.read_text()), {'key': 'new'})
        self.assertFalse(state_path.with_name('state.json.tmp').exists())


if __name__ == '__main__':
    unittest.main()
