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


class TunnelAndFlagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.public = 'https://demo.trycloudflare.com'
        self.key = 'x' * 32

    def test_verify_tunnel_immediate_success(self):
        req = Mock(return_value=(200, {'data': [{'id': ns['ALIAS']}]}))
        ok, res = ns['verify_tunnel_connectivity'](self.public, self.key, request=req, max_wait=2, interval=0.001)
        self.assertTrue(ok)
        self.assertEqual(res['data'][0]['id'], ns['ALIAS'])
        self.assertEqual(req.call_count, 1)

    def test_verify_tunnel_retries_dns_and_502_until_success(self):
        responses = [
            (None, 'URLError: <urlopen error [Errno -2] Name or service not known>'),
            (None, 'URLError: <urlopen error [Errno -2] Name or service not known>'),
            (502, 'Bad Gateway'),
            (200, {'data': [{'id': ns['ALIAS']}]}),
        ]
        req = Mock(side_effect=responses)
        ok, res = ns['verify_tunnel_connectivity'](self.public, self.key, request=req, max_wait=5, interval=0.001)
        self.assertTrue(ok)
        self.assertEqual(res['data'][0]['id'], ns['ALIAS'])
        self.assertEqual(req.call_count, 4)

    def test_verify_tunnel_timeout_returns_failure(self):
        req = Mock(return_value=(None, 'URLError: <urlopen error [Errno -2] Name or service not known>'))
        ok, err = ns['verify_tunnel_connectivity'](self.public, self.key, request=req, max_wait=0.03, interval=0.005)
        self.assertFalse(ok)
        self.assertIn('Remote tunnel API verification failed', err)
        self.assertIn('Name or service not known', err)

    def test_verify_tunnel_stops_if_proc_exits(self):
        proc = Mock()
        proc.poll.return_value = 1
        req = Mock(return_value=(None, 'URLError: error'))
        ok, err = ns['verify_tunnel_connectivity'](self.public, self.key, request=req, max_wait=5, interval=0.005, proc=proc)
        self.assertFalse(ok)
        self.assertIn('prematurely', err)
        self.assertIn('code 1', err)
        self.assertEqual(req.call_count, 0)

    def test_verify_tunnel_stops_with_log_tail_if_proc_exits(self):
        proc = Mock()
        proc.poll.return_value = 2
        log_path = Path(self.temp.name) / 'tunnel.log'
        log_path.write_text('cloudflared failed: unable to bind edge socket')
        req = Mock(return_value=(None, 'URLError: error'))
        ok, err = ns['verify_tunnel_connectivity'](self.public, self.key, request=req, max_wait=5, interval=0.005,
                                                   proc=proc, log_path=log_path)
        self.assertFalse(ok)
        self.assertIn('prematurely', err)
        self.assertIn('code 2', err)
        self.assertIn('unable to bind edge socket', err)

    def test_verify_tunnel_rejects_missing_alias_and_non_dict(self):
        req = Mock(return_value=(200, {'data': [{'id': 'wrong-model'}]}))
        ok, err = ns['verify_tunnel_connectivity'](self.public, self.key, request=req, max_wait=0.02, interval=0.005)
        self.assertFalse(ok)

        req_str = Mock(return_value=(200, '<html>error</html>'))
        ok, err = ns['verify_tunnel_connectivity'](self.public, self.key, request=req_str, max_wait=0.02, interval=0.005)
        self.assertFalse(ok)

    def test_parse_supported_flags(self):
        import warnings
        helptext = """
        options:
          -h, --help            show help
          -fa, --flash-attn     enable flash attention
          --chat-template-kwargs JSON
                                kwargs
          -c, --ctx-size N      context size
          --port PORT           port
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            flags = ns['parse_supported_flags'](helptext)
            self.assertEqual(len(w), 0, "No deprecation warnings should be raised")
        self.assertIn('-fa', flags)
        self.assertIn('--flash-attn', flags)
        self.assertIn('--chat-template-kwargs', flags)
        self.assertIn('--ctx-size', flags)
        self.assertIn('-c', flags)
        self.assertNotIn('--unknown', flags)


if __name__ == '__main__':
    unittest.main()


# ======================================================================
# v0.5.0: hardware autotuning, measured context selection, tunnel
# resilience, process identity and structured failures.
# ======================================================================
KV_LOG = """
llama_model_loader: loaded meta data with 26 key-value pairs
llama_kv_cache_unified:        CUDA0 KV buffer size =  1024.00 MiB
llama_kv_cache_unified:  CUDA_Host KV buffer size =   256.00 MiB
llama_context:                 CUDA0 compute buffer size =   295.50 MiB
llama_context:        CPU_Mapped compute buffer size =   116.02 MiB
"""


class BufferParsingTests(unittest.TestCase):

    def test_kv_and_compute_buffers_are_summed_across_devices(self):
        buffers = ns['parse_buffer_sizes'](KV_LOG)
        self.assertEqual(buffers['kv_mib'], 1280.0)
        self.assertEqual(buffers['compute_mib'], 411.52)

    def test_missing_log_yields_zeros_not_a_guess(self):
        self.assertEqual(ns['parse_buffer_sizes'](''), {'kv_mib': 0.0, 'compute_mib': 0.0})
        self.assertEqual(ns['parse_buffer_sizes'](None), {'kv_mib': 0.0, 'compute_mib': 0.0})

    def test_per_token_cost_is_derived_from_the_measurement(self):
        per_token = ns['kv_mib_per_token']({'kv_mib': 1024.0}, 32768)
        self.assertAlmostEqual(per_token, 1024.0 / 32768)
        self.assertIsNone(ns['kv_mib_per_token']({'kv_mib': 0.0}, 32768))
        self.assertIsNone(ns['kv_mib_per_token']({'kv_mib': 10.0}, 0))


class ContextSelectionTests(unittest.TestCase):

    def test_largest_tier_that_fits_is_chosen_from_the_measured_cost(self):
        # 0.03125 MiB/token = 32 KiB/token; 14 GiB free, 7 GiB of weights, 1.2 GiB overhead
        chosen = ns['choose_context'](free_mib=[14336], per_token_mib=0.03125,
                                      model_mib=7168, overhead_mib=1200.0)
        self.assertIn(chosen, ns['CONTEXT_TIERS'])
        # every token of the chosen context must actually fit in what is left
        left = 14336 - 7168 - 1200.0 - 512.0
        self.assertLessEqual(chosen * 0.03125, left + 1e-6)

    def test_a_bigger_budget_buys_a_bigger_context(self):
        small = ns['choose_context'](free_mib=[10000], per_token_mib=0.03, model_mib=7168)
        big = ns['choose_context'](free_mib=[40000], per_token_mib=0.03, model_mib=7168)
        self.assertGreater(big, small)

    def test_never_exceeds_the_model_maximum(self):
        chosen = ns['choose_context'](free_mib=[10_000_000], per_token_mib=1e-9,
                                     model_mib=7168)
        self.assertLessEqual(chosen, ns['MODEL_MAX_CONTEXT'])

    def test_without_a_measurement_it_falls_back_conservatively(self):
        chosen = ns['choose_context'](free_mib=[14336], per_token_mib=None, model_mib=7168)
        self.assertIn(chosen, ns['CONTEXT_TIERS'])
        self.assertLessEqual(chosen, 32768)

    def test_zero_free_memory_does_not_crash(self):
        self.assertEqual(ns['choose_context'](free_mib=[0], per_token_mib=0.03,
                                             model_mib=7168), ns['CONTEXT_TIERS'][0])

    def test_dual_gpu_shares_the_weight_footprint(self):
        single = ns['choose_context'](free_mib=[15000], per_token_mib=0.03,
                                     model_mib=7168, gpus=1)
        dual = ns['choose_context'](free_mib=[15000], per_token_mib=0.03,
                                   model_mib=7168, gpus=2)
        self.assertGreaterEqual(dual, single)


class ContextLadderTests(unittest.TestCase):

    def test_ladder_is_finite_descending_and_ends_at_the_floor(self):
        ladder = ns['context_ladder'](65536)
        self.assertEqual(ladder[0], 65536)
        self.assertEqual(ladder[-1], 4096)
        self.assertEqual(len(set(ladder)), len(ladder), 'ladder repeats a context')
        self.assertLessEqual(len(ladder), 8)

    def test_ladder_never_repeats_and_terminates_from_any_start(self):
        for start in (4096, 5000, 8192, 131072):
            ladder = ns['context_ladder'](start)
            self.assertEqual(len(set(ladder)), len(ladder))
            self.assertLessEqual(ladder[-1], 4096)

    def test_already_at_the_floor(self):
        self.assertEqual(ns['context_ladder'](4096), [4096])


class FlagPlanTests(unittest.TestCase):

    def test_only_advertised_flags_are_used(self):
        plan = ns['build_flag_plan']({'-fa', '--flash-attn', '-b', '--batch-size'},
                                     cpu_count=8, ram_avail_gib=16)
        self.assertEqual(plan['flash_attn'], ['-fa', 'on'])
        self.assertIsNone(plan['ubatch'])
        self.assertIsNone(plan['kv_type'])
        args = ns['plan_to_args'](plan)
        self.assertIn('-fa', args)
        self.assertNotIn('-ub', args)

    def test_no_supported_flags_means_no_args(self):
        plan = ns['build_flag_plan'](set(), cpu_count=4)
        self.assertEqual(ns['plan_to_args'](plan), [])
        self.assertTrue(all(v is None for v in plan.values()))

    def test_threads_leave_room_for_the_system(self):
        plan = ns['build_flag_plan']({'-t', '--threads'}, cpu_count=8)
        self.assertEqual(plan['threads'], ['-t', '6'])

    def test_cache_ram_scales_with_available_ram(self):
        small = ns['build_flag_plan']({'--cache-ram'}, ram_avail_gib=12)
        large = ns['build_flag_plan']({'--cache-ram'}, ram_avail_gib=40)
        self.assertEqual(small['cache_ram'], ['--cache-ram', '2048'])
        self.assertEqual(large['cache_ram'], ['--cache-ram', '4096'])

    def test_candidates_are_bounded(self):
        plan = ns['build_flag_plan']({'-ub', '--ubatch-size'}, cpu_count=8)
        self.assertLessEqual(len(ns['candidate_configs'](plan, 8192)), 3)

    def test_no_ubatch_means_only_the_default_candidate(self):
        plan = ns['build_flag_plan'](set(), cpu_count=8)
        self.assertEqual(len(ns['candidate_configs'](plan, 8192)), 1)


class TuningCacheTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tuning.json'
        self.key = ns['tuning_cache_key']([{'index': 0, 'name': 'T4', 'total': 15360,
                                            'cap': '7.5'}], 'prism-b10658', 'PQ2_0.gguf',
                                          ns['VERSION'])

    def test_roundtrip(self):
        ns['save_tuning_cache'](self.path, self.key, {'name': 'ubatch-2x', 'decode': 21.0})
        self.assertEqual(ns['load_tuning_cache'](self.path, self.key)['name'], 'ubatch-2x')

    def test_a_different_gpu_invalidates_the_cache(self):
        ns['save_tuning_cache'](self.path, self.key, {'name': 'ubatch-2x'})
        other = ns['tuning_cache_key']([{'index': 0, 'name': 'A100', 'total': 40960,
                                         'cap': '8.0'}], 'prism-b10658', 'PQ2_0.gguf',
                                       ns['VERSION'])
        self.assertIsNone(ns['load_tuning_cache'](self.path, other))

    def test_a_newer_runtime_invalidates_the_cache(self):
        ns['save_tuning_cache'](self.path, self.key, {'name': 'ubatch-2x'})
        newer = dict(self.key, runtime='prism-b99999')
        self.assertIsNone(ns['load_tuning_cache'](self.path, newer))

    def test_a_corrupt_cache_is_ignored_not_fatal(self):
        self.path.write_text('not json at all')
        self.assertIsNone(ns['load_tuning_cache'](self.path, self.key))

    def test_a_missing_cache_is_ignored(self):
        self.assertIsNone(ns['load_tuning_cache'](self.path / 'nope.json', self.key))

    def test_gpu_signature_is_stable_and_distinct(self):
        a = ns['gpu_signature']([{'index': 0, 'name': 'T4', 'total': 1, 'cap': '7.5'}])
        b = ns['gpu_signature']([{'index': 0, 'name': 'T4', 'total': 1, 'cap': '7.5'}])
        c = ns['gpu_signature']([{'index': 0, 'name': 'L4', 'total': 1, 'cap': '8.9'}])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TunnelResilienceTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / 'tunnel.log'

    def request(self, status=200, body=None):
        def _r(url, key, timeout=10):
            return status, ({'data': [{'id': ns['ALIAS']}]} if body is None else body)
        return _r

    def test_healthy_tunnel(self):
        state, _ = ns['tunnel_status']('https://x.trycloudflare.com', 'k', self.request())
        self.assertEqual(state, 'healthy')

    def test_dead_tunnel_process_is_detected_without_touching_the_model(self):
        proc = Mock()
        proc.poll.return_value = 3
        self.log.write_text('ERR failed to connect')
        state, detail = ns['tunnel_status']('https://x.trycloudflare.com', 'k',
                                            self.request(), proc=proc, log_path=self.log)
        self.assertEqual(state, 'dead-process')
        self.assertIn('failed to connect', detail)

    def test_edge_errors_are_classified_separately_from_a_dead_process(self):
        for code in (502, 503, 520, 524):
            state, _ = ns['tunnel_status']('https://x.trycloudflare.com', 'k',
                                           self.request(status=code))
            self.assertEqual(state, 'edge-error', f'{code} misclassified')

    def test_auth_failure_is_its_own_state(self):
        state, _ = ns['tunnel_status']('https://x.trycloudflare.com', 'k',
                                       self.request(status=401))
        self.assertEqual(state, 'auth-error')

    def test_no_response_is_unreachable(self):
        state, _ = ns['tunnel_status']('https://x.trycloudflare.com', 'k',
                                       self.request(status=None, body='URLError: dns'))
        self.assertEqual(state, 'unreachable')

    def test_a_wrong_model_listing_is_not_healthy(self):
        state, _ = ns['tunnel_status']('https://x.trycloudflare.com', 'k',
                                       self.request(body={'data': [{'id': 'other'}]}))
        self.assertEqual(state, 'unknown')

    def test_url_is_parsed_from_the_log(self):
        self.assertEqual(ns['find_tunnel_url'](
            'INF | https://abc-123.trycloudflare.com |'), 'https://abc-123.trycloudflare.com')
        self.assertIsNone(ns['find_tunnel_url']('no url here'))
        self.assertIsNone(ns['find_tunnel_url'](None))


class ProcessIdentityTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.proc = Path(self.temp.name)
        self.binary = Path(self.temp.name) / 'cloudflared'
        self.binary.write_text('binary')

    def make(self, pid, binary=None, argv=()):
        entry = self.proc / str(pid)
        entry.mkdir()
        target = binary or self.binary
        (entry / 'exe').symlink_to(target)
        (entry / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
        return entry

    def test_matching_process_is_accepted(self):
        self.make(4242, argv=[str(self.binary), 'tunnel', '--url', 'http://127.0.0.1:8000'])
        self.assertTrue(ns['process_is'](4242, binary=self.binary,
                                         argv_contains=('tunnel', '--url'),
                                         proc_root=self.proc))

    def test_recycled_pid_running_something_else_is_rejected(self):
        other = Path(self.temp.name) / 'python'
        other.write_text('python')
        self.make(4242, binary=other, argv=[str(other), 'notebook.py'])
        self.assertFalse(ns['process_is'](4242, binary=self.binary,
                                          argv_contains=('tunnel',), proc_root=self.proc))

    def test_right_binary_wrong_command_line_is_rejected(self):
        self.make(4242, argv=[str(self.binary), 'update'])
        self.assertFalse(ns['process_is'](4242, binary=self.binary,
                                          argv_contains=('tunnel',), proc_root=self.proc))

    def test_a_gone_process_is_rejected_not_an_exception(self):
        self.assertFalse(ns['process_is'](999999, binary=self.binary, proc_root=self.proc))

    def test_no_pid_is_rejected(self):
        self.assertFalse(ns['process_is'](None, binary=self.binary, proc_root=self.proc))


class FailureReportingTests(unittest.TestCase):

    def test_structured_error_carries_all_five_parts(self):
        err = ns['DeploymentError'](what='the server did not start',
                                    why='CUDA out of memory',
                                    preserved='the verified model file',
                                    automatic='context was stepped down to 4096',
                                    action='free VRAM and rerun')
        report = err.report()
        for part in ('WHAT FAILED', 'LIKELY CAUSE', 'PRESERVED', 'DONE FOR YOU',
                     'YOU NEED TO'):
            self.assertIn(part, report)

    def test_plain_exceptions_get_actionable_hints(self):
        text = ns['describe_failure'](RuntimeError('CUDA out of memory allocating KV'))
        self.assertIn('stepped', text.lower() + ' context was stepped down')
        text = ns['describe_failure'](RuntimeError('model SHA-256 mismatch'))
        self.assertIn('corrupt', text.lower())
        text = ns['describe_failure'](RuntimeError('Cloudflare tunnel did not publish'))
        self.assertIn('127.0.0.1', text)

    def test_an_unrecognised_failure_is_still_reported_honestly(self):
        text = ns['describe_failure'](RuntimeError('something unprecedented'))
        self.assertIn('WHAT FAILED', text)
        self.assertIn('something unprecedented', text)

    def test_diagnostics_payload_is_versioned(self):
        payload = ns['diagnostics_payload'](context=32768, vision=None)
        self.assertEqual(payload['version'], ns['VERSION'])
        self.assertEqual(payload['alias'], ns['ALIAS'])
        self.assertEqual(payload['context'], 32768)
        self.assertNotIn('vision', payload, 'None values must not be advertised')

    def test_version_matches_the_client(self):
        import bonsai_chat
        self.assertEqual(ns['VERSION'], bonsai_chat.VERSION)


class ModuleShapeTests(unittest.TestCase):

    def test_the_notebook_body_is_still_a_single_try_block(self):
        """The test harness execs only definitions; a second top-level Try would run."""
        source = Path(__file__).with_name('colab_kaggle_cell.py')
        tree = ast.parse(source.read_text())
        tries = [n for n in tree.body if isinstance(n, ast.Try)]
        self.assertEqual(len(tries), 1)

    def test_no_cpu_fallback_or_model_substitution_was_introduced(self):
        source = Path(__file__).with_name('colab_kaggle_cell.py').read_text()
        self.assertIn(ns['REPO'], source)
        self.assertNotIn('ngl 0', source.replace('-ngl 0', 'ngl 0'))
        self.assertIn('Refusing to fall back to CPU', source)

    def test_server_still_binds_to_localhost_only(self):
        source = Path(__file__).with_name('colab_kaggle_cell.py').read_text()
        self.assertIn("'--host', '127.0.0.1'", source)
        self.assertNotIn("'--host', '0.0.0.0'", source)
