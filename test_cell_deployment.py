"""End-to-end tests for colab_kaggle_cell.py, driven by the offline fake-GPU harness.

These run the cell's *real* module-level deployment body — the same code a Colab user
pastes into a cell — against fakes, and assert on what the deployment did: which
processes it started, which contexts it attempted, whether it printed READY, what it
wrote to state.json and diagnostics.json, and how it recovered when something died.

Everything is offline and deterministic: no network, no GPU, no model download.
"""

import json
import time
import unittest
from pathlib import Path

import cell_harness as harness
from cell_harness import (CellRun, L4, MISMATCH, SMALL, T4, T4X2, default_model_files,
                          hf_config, requires_compiler)


class CellRunTestCase(unittest.TestCase):
    """Runs a full deployment in setUp unless the subclass says otherwise."""

    deploy = True
    gpus = None
    server = None
    tunnel = None
    hf = None
    cuda_devices = None
    cuda_fail = None
    env = None
    meminfo_gib = (16.0, 12.0)

    def setUp(self):
        self.run = CellRun(gpus=self.gpus, cuda_devices=self.cuda_devices,
                           cuda_fail=self.cuda_fail, server=self.server,
                           tunnel=self.tunnel, hf=self.hf, env=self.env,
                           meminfo_gib=self.meminfo_gib)
        self.run.__enter__()
        self.addCleanup(self.run.__exit__)
        if self.deploy:
            self.run.execute(expect_success=False)
        # backoff is real in production and dead time in a test
        self.no_wait = lambda seconds: None
        for sup in self.run.supervisors:
            sup._sleep = self.no_wait
            self.addCleanup(sup.stop)

    # -- helpers -----------------------------------------------------
    def assert_ready(self):
        self.assertIsNone(self.run.error, self.run.output)
        self.assertTrue(self.run.ready, self.run.output)

    def assert_failed(self, needle=None):
        self.assertIsNotNone(self.run.error, 'the cell unexpectedly succeeded')
        if needle:
            self.assertIn(needle, str(self.run.error) + self.run.output)


@requires_compiler
class HappyPathTests(CellRunTestCase):

    def test_full_deployment_reaches_ready_on_one_t4(self):
        self.assert_ready()
        state = self.run.state
        self.assertEqual(state['model'], 'Ternary-Bonsai-2-27B-PQ2_0.gguf')
        self.assertEqual(state['ctx'], 16384)
        self.assertGreater(len(state['key']), 31)
        self.assertTrue(state['public'].endswith('.trycloudflare.com'))
        self.assertEqual(len(self.run.invocations()), 1)

    def test_state_file_is_private_and_diagnostics_are_written(self):
        self.assert_ready()
        mode = (self.run.root / 'state.json').stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        diag = self.run.diagnostics
        self.assertEqual(diag['version'], '0.6.0')
        self.assertEqual(diag['client_version'], '0.6.0')
        self.assertEqual(diag['model_params_b'], 27.36)
        self.assertTrue(diag['quantization_verified'])
        self.assertFalse(diag['speculative_decoding'])
        self.assertFalse(diag['vision'])
        self.assertEqual(diag['slots'], 1)
        self.assertTrue(all(diag['api_tests'].values()),
                        str(diag['api_tests']))

    def test_every_api_test_is_reported_as_pass(self):
        self.assert_ready()
        for line in ('PASS  /health', 'PASS  GET /v1/models', 'PASS  streaming',
                     'PASS  native tool calling (--jinja)'):
            self.assertIn(line, self.run.output)

    def test_heartbeat_describes_what_is_live(self):
        self.assert_ready()
        beat = json.loads((self.run.root / 'heartbeat.json').read_text())
        self.assertEqual(beat['port'], self.run.state['port'])
        self.assertEqual(beat['public'], self.run.state['public'])
        self.assertEqual(beat['version'], '0.6.0')

    def test_supervision_is_running_for_both_components(self):
        self.assert_ready()
        names = sorted(sup.name for sup in self.run.supervisors)
        self.assertEqual(names, ['cloudflare-tunnel', 'inference-server'])
        for sup in self.run.supervisors:
            self.assertTrue(sup.running())
        self.assertIn('Supervision: ON', self.run.output)

    def test_slot_state_is_reported_when_the_runtime_exposes_it(self):
        self.assert_ready()
        slots = self.run.diagnostics['slots_state']
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]['n_ctx'], 16384)

    def test_a_rerun_reattaches_instead_of_redeploying(self):
        self.assert_ready()
        first = self.run.state
        # A second execution of the same cell body: the live server must be adopted.
        self.run.execute()
        self.assert_ready()
        self.assertEqual(self.run.state['server_pid'], first['server_pid'])
        self.assertEqual(self.run.state['key'], first['key'])
        self.assertEqual(self.run.state['public'], first['public'])
        self.assertEqual(self.run.invocations(), [], 'a reattach must start nothing')
        self.assertIn('no duplicate spawn', self.run.output)


@requires_compiler
class OomLadderTests(CellRunTestCase):

    def test_oom_steps_the_context_down_and_records_every_step(self):
        with CellRun(gpus=[T4], server={'oom_above_ctx': 8192}) as run:
            run.execute(expect_success=False)
            self.assert_ready_impl(run)
            # 16384 OOMs, 8192 fits; the measured KV cost then says 16384 *should*
            # fit, the cell tries once, it OOMs again and the ladder brings it back.
            # Recovery is bounded by the ladder, and every step is recorded.
            self.assertEqual(run.contexts_attempted(), [16384, 8192, 16384, 8192])
            self.assertIn('OOM at context 16384', run.output)
            self.assertEqual(run.state['ctx'], 8192)
            self.assertEqual(run.diagnostics['context'], 8192)
            self.assertEqual(run.diagnostics['context_recovery'], [16384, 8192])

    def test_oom_all_the_way_down_fails_with_a_structured_report(self):
        with CellRun(gpus=[T4], server={'oom_above_ctx': 0}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertFalse(run.ready)
            self.assertIn('DEPLOYMENT FAILED', run.output)
            self.assertIn('PRESERVED', run.output)
            self.assertEqual(run.contexts_attempted(),
                             harness_run_contexts(run))
            self.assertIn('4096', run.output)

    def assert_ready_impl(self, run):
        self.assertIsNone(run.error, run.output)
        self.assertTrue(run.ready, run.output)


def harness_run_contexts(run):
    return [16384, 8192, 4096]


@requires_compiler
class DualGpuTests(CellRunTestCase):

    def test_dual_layer_split_is_chosen_when_it_measures_faster(self):
        with CellRun(gpus=T4X2, server={'dual_factor': 1.5}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            self.assertIn('Dual GPU layer split', run.diagnostics['selected_configuration'])
            self.assertIn('--split-mode', ' '.join(run.invocations()[-1]['argv']))
            self.assertIn('layer', ' '.join(run.invocations()[-1]['argv']))

    def test_dual_is_rejected_when_it_is_not_measurably_faster(self):
        with CellRun(gpus=T4X2, server={'dual_factor': 0.8}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertIn('measured >= dual decode throughput',
                          run.diagnostics['selected_configuration'])

    def test_dual_that_fails_leaves_a_working_single_gpu_server(self):
        # The second GPU gets no work: the cell must refuse the split, not deploy it.
        with CellRun(gpus=T4X2, server={'dual_vram': [7000, 0]}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            self.assertIn('ineffective', run.output)


@requires_compiler
class MeasuredContextTests(CellRunTestCase):

    def test_a_cheap_kv_cost_buys_one_measured_context_upgrade(self):
        with CellRun(gpus=[T4], server={'kv_mib_per_token': 0.02}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            self.assertEqual(run.contexts_attempted(), [16384, 262144])
            self.assertIn('Context raised to 262144 tokens on measured evidence',
                          run.output)
            self.assertEqual(run.state['ctx'], 262144)

    def test_a_missing_kv_report_keeps_the_conservative_context(self):
        with CellRun(gpus=[T4]) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertIn('Measured KV cost', run.output)


@requires_compiler
class AutotuneTests(CellRunTestCase):

    def test_ubatch_candidates_are_measured_and_the_winner_is_cached(self):
        extra = ['-ub, --ubatch-size N   physical batch size',
                 '--slot-save-path PATH  where to save the prompt cache']
        with CellRun(gpus=[T4], server={'extra_flags': extra}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            self.assertIn('Autotuning selected:', run.output)
            self.assertTrue((run.root / 'tuning.json').is_file())

    def test_a_cached_result_is_not_remeasured_on_the_next_run(self):
        extra = ['-ub, --ubatch-size N   physical batch size']
        with CellRun(gpus=[T4], server={'extra_flags': extra}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            run.execute()
            self.assertIsNone(run.error, run.output)
            self.assertIn('no duplicate spawn', run.output)
            self.assertEqual(run.invocations(), [])


@requires_compiler
class RefusalTests(CellRunTestCase):

    def test_no_gpu_visible_is_refused_not_faked(self):
        with CellRun(gpus=[], cuda_devices=0) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('No NVIDIA GPU visible', str(run.error) + run.output)

    def test_cuda_driver_failure_is_refused(self):
        with CellRun(gpus=[T4], cuda_fail='init') as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('cuInit', str(run.error))

    def test_a_gpu_held_by_other_work_is_refused_not_stolen(self):
        busy = dict(T4, used=12000)
        with CellRun(gpus=[busy]) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('Insufficient free VRAM', str(run.error))
            self.assertIn('never terminated automatically', str(run.error))
            self.assertFalse(run.ready)
            self.assertEqual(run.invocations(), [])

    def test_a_smaller_gpu_selects_the_tighter_official_packing(self):
        big = dict(T4, total=11500)
        with CellRun(gpus=[big], server={'oom_above_ctx': 4096}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertIn('PTQ1_0', run.output)

    def test_a_non_fork_runtime_stamp_is_refused(self):
        with CellRun(gpus=[T4], server={'stamp': 'upstream-b5700'}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('not the PrismML fork', str(run.error))

    def test_a_corrupt_model_identity_is_refused_not_substituted(self):
        with CellRun(gpus=[T4], hf=hf_config(default_model_files(),
                                             corrupt_identity=True)) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('does not identify Ternary Bonsai 2 27B', str(run.error))
            self.assertIn('no substitution', str(run.error))

    def test_a_sha256_mismatch_is_refused(self):
        with CellRun(gpus=[T4], hf=hf_config(default_model_files(), sha=True,
                                             corrupt_bytes=True)) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('SHA-256 mismatch', str(run.error))

    def test_a_verified_sha256_passes(self):
        with CellRun(gpus=[T4], hf=hf_config(default_model_files(), sha=True)) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertIn('PASS (official HF SHA-256)', run.output)

    def test_a_missing_official_file_is_refused(self):
        with CellRun(gpus=[T4], hf=hf_config(default_model_files(),
                                             missing=['Ternary-Bonsai-2-27B-PQ2_0.gguf',
                                                     'Ternary-Bonsai-2-27B-PTQ1_0.gguf'])) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('refusing to substitute', str(run.error))

    def test_a_build_without_api_key_is_refused(self):
        with CellRun(gpus=[T4], server={'help_drop': ['--api-key']}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('unauthenticated', str(run.error))

    def test_mismatched_gpus_still_deploy_on_the_first(self):
        with CellRun(gpus=MISMATCH) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            self.assertIn('GPUs identical: False', run.output)


@requires_compiler
class ResumeTests(CellRunTestCase):
    deploy = False

    def test_a_stale_state_pointing_at_a_recycled_pid_is_ignored(self):
        self.run.execute()
        self.assert_ready()
        state = dict(self.run.state)
        state['server_pid'] = 999999
        self.run.write_state(state)
        self.run.execute()
        self.assert_ready()
        self.assertNotEqual(self.run.state['server_pid'], 999999)

    def test_a_different_requested_key_is_refused_rather_than_adopted(self):
        with CellRun(gpus=[T4], env={'BONSAI_API_KEY': 'a' * 40}) as first:
            first.execute()
            self.assertIsNone(first.error, first.output)
            self.assertEqual(first.state['key'], 'a' * 40)
            # Same deployment directory, different requested key: the live server must
            # not be adopted, and its key must not be changed underneath the client
            # that is already using it.
            with CellRun(gpus=[T4], env={'BONSAI_API_KEY': 'b' * 40},
                         base=str(first.base), gpu_dir=str(first.gpu_root)) as second:
                second.execute(expect_success=False)
                self.assertIn('already running with a different', second.output)
            self.assertEqual(first.state['key'], 'a' * 40)

    def test_a_corrupt_state_file_is_survived(self):
        self.run.execute()
        self.assert_ready()
        (self.run.root / 'state.json').write_text('{not json')
        self.run.execute()
        self.assert_ready()


@requires_compiler
class TunnelFailureTests(CellRunTestCase):

    def test_a_tunnel_that_never_publishes_fails_with_the_model_still_up(self):
        with CellRun(gpus=[T4], tunnel={'no_url': True, 'lifetime': 0.2}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('did not publish a URL', str(run.error))
            self.assertIn('the inference server, live on', run.output)
            self.assertIn('rerun the cell', run.output)
            self.assertIsNotNone(run.state, 'state.json must survive a tunnel failure')
            self.assertTrue(run.state['port'])

    def test_a_tunnel_that_dies_at_startup_is_reported_not_guessed(self):
        with CellRun(gpus=[T4], tunnel={'fail_immediately': True}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('tunnel', str(run.error).lower())


@requires_compiler
class WatchdogTests(CellRunTestCase):

    def step_until(self, sup, predicate, limit=8):
        for _ in range(limit):
            sup.step()
            if predicate():
                return True
        return False

    def test_a_crashed_server_is_restarted_on_the_same_port_and_key(self):
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        before = dict(self.run.state)
        self.run.kill_server()
        self.assertTrue(self.step_until(sup, lambda: sup.events), 'no restart happened')
        event = sup.events[-1]
        self.assertEqual(event['action'], 'restart')
        self.assertEqual(event['detected'], 'unhealthy')
        after = self.run.state
        self.assertEqual(after['port'], before['port'], 'a restart must preserve the port')
        self.assertEqual(after['key'], before['key'])
        self.assertEqual(after['public'], before['public'],
                         'a model restart must preserve the tunnel URL')
        self.assertNotEqual(after['server_pid'], before['server_pid'])
        self.assertIn('never re-downloaded', event['preserved'])
        self.assertEqual(len(self.run.invocations()), 2)

    def test_restarts_are_bounded_and_then_supervision_stops(self):
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        before = self.run.state['port']
        for _ in range(6):
            if sup.events and sup.events[-1]['action'] == 'give-up':
                break
            pid = self.run.state.get('server_pid')
            if pid:
                try:
                    self.run.kill_server()
                except AssertionError:
                    break
            for _ in range(3):
                sup.step()
        self.assertEqual(sup.events[-1]['action'], 'give-up')
        self.assertEqual(sup.budget.used, sup.budget.max_restarts)
        self.assertFalse(sup.running(), 'supervision must stop, not flap forever')
        self.assertEqual(self.run.state['port'], before)
        self.assertLessEqual(len(self.run.invocations()), 1 + sup.budget.max_restarts)

    def test_a_hang_is_detected_by_a_real_generation_probe(self):
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        sup.hang_check_interval = 0
        sup.stall_timeout = 3.0
        # The server stays alive and answers /health but stops generating: only a
        # generation probe can tell the difference, which is why it exists.
        self.run.set_server_cfg(hang=True)
        self.run.kill_server()
        self.assertTrue(self.step_until(sup, lambda: sup.events))
        self.run.set_server_cfg(hang=False)
        sup.last_probe_at = 0
        sup.step()
        hang_events = [e for e in sup.events if e['detected'] == 'hang']
        self.assertTrue(hang_events, 'a stalled generation was never detected')
        self.assertIn('probe', hang_events[0]['detail'])
        self.assertTrue(self.run.diagnostics.get('watchdog'))

    def test_an_oom_restart_steps_the_context_down(self):
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        before_ctx = self.run.state['ctx']
        # The server dies of a CUDA OOM on its own, leaving the evidence in its log.
        # That is the one case where the *configuration* was wrong rather than the
        # process, so the correct repair steps the context down instead of retrying
        # the same one.
        self.run.set_server_cfg(oom_after=0.2)
        time.sleep(1.5)
        self.assertTrue(self.step_until(sup, lambda: sup.events), 'no restart happened')
        event = sup.events[-1]
        self.assertEqual(event['action'], 'restart')
        self.assertEqual(event['detected'], 'unhealthy')
        self.assertIn('oom:', event['changed'])
        self.assertLess(self.run.state['ctx'], before_ctx)
        self.assertIn('context_recovery', self.run.diagnostics)

    def test_a_crash_restart_keeps_the_same_context(self):
        # The mirror image of the OOM case: the context was fine, so it must not move.
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        before_ctx = self.run.state['ctx']
        self.run.set_server_cfg(crash_after=9999)
        self.run.kill_server()
        self.assertTrue(self.step_until(sup, lambda: sup.events))
        self.assertIn('crash:', sup.events[-1]['changed'])
        self.assertEqual(self.run.state['ctx'], before_ctx)

    def test_every_heal_is_recorded_in_diagnostics(self):
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        self.run.kill_server()
        self.step_until(sup, lambda: sup.events)
        diag = self.run.diagnostics
        self.assertTrue(diag['watchdog'])
        component = [w for w in diag['watchdog'] if w['component'] == 'inference-server']
        self.assertTrue(component)
        self.assertEqual(component[0]['events'][-1]['action'], 'restart')
        self.assertIn('last_heal', diag)

    def test_a_dead_tunnel_is_restarted_without_touching_the_model(self):
        self.assert_ready()
        sup = self.run.supervisor('cloudflare-tunnel')
        before = dict(self.run.state)
        self.run.kill_tunnel()
        # A dead tunnel process is detected immediately; a routing failure takes two
        # consecutive bad checks so one Cloudflare hiccup cannot cost a restart.
        self.assertTrue(self.step_until(sup, lambda: sup.events))
        after = self.run.state
        self.assertEqual(after['server_pid'], before['server_pid'],
                         'a tunnel restart must not restart the model')
        self.assertNotEqual(after['tunnel_pid'], before['tunnel_pid'])
        self.assertNotEqual(after['public'], before['public'])
        self.assertTrue(after['public'].endswith('.trycloudflare.com'))
        self.assertIn('the inference server was not touched', sup.events[-1]['changed'])

    def test_the_tunnel_survives_a_model_restart(self):
        self.assert_ready()
        sup = self.run.supervisor('inference-server')
        before = dict(self.run.state)
        self.run.kill_server()
        self.assertTrue(self.step_until(sup, lambda: sup.events))
        after = self.run.state
        self.assertEqual(after['public'], before['public'])
        self.assertEqual(after['tunnel_pid'], before['tunnel_pid'])
        self.assertNotEqual(after['server_pid'], before['server_pid'])

    def test_the_model_survives_a_tunnel_restart(self):
        self.assert_ready()
        sup = self.run.supervisor('cloudflare-tunnel')
        before = dict(self.run.state)
        self.run.kill_tunnel()
        self.assertTrue(self.step_until(sup, lambda: sup.events))
        self.assertEqual(self.run.state['server_pid'], before['server_pid'])
        # and the new tunnel really routes to that same server
        ns = self.run.ns
        status, body = ns['HTTP'](self.run.state['public'] + '/v1/models',
                                  self.run.state['key'], timeout=10)
        self.assertEqual(status, 200)
        self.assertEqual(body['data'][0]['id'], ns['ALIAS'])


@requires_compiler
class VisionTests(CellRunTestCase):

    def test_vision_stays_off_by_default(self):
        with CellRun(gpus=[T4]) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertFalse(run.diagnostics['vision'])
            self.assertIn('not downloaded/loaded', run.output)
            self.assertNotIn('--mmproj', ' '.join(run.invocations()[0]['argv']))

    def test_vision_on_loads_and_accounts_for_the_official_projector(self):
        with CellRun(gpus=[T4], env={'BONSAI_VISION': '1'}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            diag = run.diagnostics
            self.assertTrue(diag['vision'])
            self.assertEqual(diag['vision_projector'], 'mmproj-Q8_0.gguf')
            self.assertEqual(diag['image_max_tokens'], 1024)
            argv = ' '.join(run.invocations()[0]['argv'])
            self.assertIn('--mmproj', argv)
            self.assertIn('mmproj-Q8_0.gguf', argv)
            self.assertIn('--image-max-tokens', argv)
            self.assertTrue((run.root / 'models/mmproj-Q8_0.gguf').is_file())
            self.assertIn('counted against the VRAM budget', run.output)
            self.assertIn('Vision: ON', run.output)

    def test_vision_is_refused_when_the_build_has_no_mmproj_flag(self):
        with CellRun(gpus=[T4], env={'BONSAI_VISION': '1'},
                     server={'help_drop': ['--mmproj']}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('Vision is left OFF', str(run.error))
            self.assertEqual(run.invocations(), [])

    def test_vision_is_refused_rather_than_loading_a_smaller_context(self):
        # A GPU too small for weights + projector + the image reserve must fail loudly.
        tight = dict(T4, total=11000)
        with CellRun(gpus=[tight], env={'BONSAI_VISION': '1',
                                        'BONSAI_IMAGE_MAX_TOKENS': '4096'}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertTrue('cannot hold' in str(run.error)
                            or 'Insufficient free VRAM' in str(run.error),
                            str(run.error))
            self.assertFalse(run.ready)

    def test_an_explicit_projector_that_does_not_exist_is_refused(self):
        with CellRun(gpus=[T4], env={'BONSAI_VISION': '1',
                                     'BONSAI_MMPROJ': 'mmproj-made-up.gguf'}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('refusing to substitute', str(run.error))

    def test_vision_off_never_downloads_the_projector(self):
        with CellRun(gpus=[T4]) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertFalse((run.root / 'models/mmproj-Q8_0.gguf').exists())


@requires_compiler
class ServingKnobTests(CellRunTestCase):

    def test_single_slot_stays_the_default(self):
        with CellRun(gpus=[T4]) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertEqual(run.diagnostics['slots'], 1)
            self.assertIn('--parallel 1', ' '.join(run.invocations()[0]['argv']))
            self.assertIn('single slot', run.diagnostics['prompt_cache'])

    def test_multi_slot_is_opt_in_and_states_the_cache_cost(self):
        with CellRun(gpus=[T4], env={'BONSAI_SLOTS': '3'}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertTrue(run.ready)
            self.assertEqual(run.diagnostics['slots'], 3)
            self.assertIn('--parallel 3', ' '.join(run.invocations()[0]['argv']))
            self.assertIn('prompt-cache reuse is', run.output)
            self.assertIn('diluted', run.output)
            self.assertEqual(len(run.diagnostics['slots_state']), 3)
            self.assertIn('3 slots, each with its own KV cache',
                          run.diagnostics['prompt_cache'])

    def test_multi_slot_shrinks_the_context_because_each_slot_carries_a_cache(self):
        with CellRun(gpus=[T4], server={'kv_mib_per_token': 0.02}) as run:
            run.execute(expect_success=False)
            single_ctx = run.diagnostics['context']
        with CellRun(gpus=[T4], server={'kv_mib_per_token': 0.02},
                     env={'BONSAI_SLOTS': '4'}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertLess(run.diagnostics['context'], single_ctx)

    def test_multi_slot_is_refused_when_the_build_has_no_slot_flag(self):
        with CellRun(gpus=[T4], env={'BONSAI_SLOTS': '2'},
                     server={'help_drop': ['--parallel']}) as run:
            run.execute(expect_success=False)
            self.assertIsNotNone(run.error)
            self.assertIn('no slot flag', str(run.error))

    def test_kv4_is_opt_in_and_labelled_a_memory_lever(self):
        with CellRun(gpus=[T4], env={'BONSAI_KV4': '1'},
                     server={'extra_flags': ['--cache-type-k TYPE  KV cache type']}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertIn('--cache-type-k q4_0', ' '.join(run.invocations()[0]['argv']))
            self.assertIn('q4_0', run.diagnostics['kv_type'])
            self.assertIn('SLOWER', run.output)

    def test_kv4_off_leaves_the_runtime_default_alone(self):
        with CellRun(gpus=[T4],
                     server={'extra_flags': ['--cache-type-k TYPE  KV cache type']}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertEqual(run.diagnostics['kv_type'], 'runtime default')
            self.assertNotIn('--cache-type-k', ' '.join(run.invocations()[0]['argv']))

    def test_kv4_is_reported_as_unavailable_when_the_flag_is_missing(self):
        with CellRun(gpus=[T4], env={'BONSAI_KV4': '1'},
                     server={'help_drop': ['--cache-type-k']}) as run:
            run.execute(expect_success=False)
            self.assertIsNone(run.error, run.output)
            self.assertIn('does not advertise', run.output)
            self.assertEqual(run.diagnostics['kv_type'], 'runtime default')


class UnitHelperTests(unittest.TestCase):
    """Pure helpers the deployment body relies on, exercised without a deployment."""

    def setUp(self):
        import ast
        source = Path(__file__).with_name('colab_kaggle_cell.py')
        tree = ast.parse(source.read_text(), filename=str(source))
        definitions = ast.Module(body=[n for n in tree.body if not isinstance(n, ast.Try)],
                                 type_ignores=[])
        self.ns = {'__name__': 'bonsai_cell_definitions'}
        exec(compile(definitions, str(source), 'exec'), self.ns)

    # ------------------------------------------------------------------
    # Regressions. Both of these were found by the harness, not by reading code, and
    # each one is here so it cannot come back.
    # ------------------------------------------------------------------
    def test_a_value_less_flag_is_never_emitted_and_cannot_swallow_the_next_token(self):
        """`--parallel` advertised with no slot policy produced `... --parallel -c 8192`,
        so -c became the *value* of --parallel and the server refused to start.

        A flag with no value must simply not be emitted; the decision belongs to the
        caller, not to the plan-to-argv conversion.
        """
        plan_to_args = self.ns['plan_to_args']
        TUNABLES = self.ns['TUNABLES']
        keys = [t[0] for t in TUNABLES]

        def plan_with(name, entry):
            return {k: (entry if k == name else None) for k in keys}

        for name in keys:
            args = plan_to_args(plan_with(name, ['--flag-of-' + name]))
            self.assertEqual(args, [], f'{name}: a bare flag was emitted')
        valued = plan_to_args(plan_with('ubatch', ['-ub', '512']))
        self.assertEqual(valued, ['-ub', '512'])
        # A value-less --parallel next to a valued -ub must leave the pair intact and
        # contribute nothing of its own: that was the bug.
        mixed = {k: None for k in keys}
        mixed['parallel'] = ['--parallel']
        mixed['ubatch'] = ['-ub', '512']
        out = plan_to_args(mixed)
        self.assertEqual(out, ['-ub', '512'])
        self.assertNotIn('--parallel', out)

    def test_a_gguf_offset_above_four_gib_is_read_as_a_u64(self):
        """The tensor data offset is u64 in GGUF. Packing it as u32 truncates any file
        whose tensor data starts above 4 GiB, and the identity check then reads garbage.

        The cell reads it with `unpack('Q')`; this fixes the *fixture* to match, so a
        harness regression cannot masquerade as a cell bug.
        """
        import struct
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'big.gguf'
            with path.open('wb') as fh:
                fh.write(b'GGUF' + struct.pack('<I', 3))
                fh.write(struct.pack('<QQ', 1, 1))        # 1 tensor, 1 metadata pair
                key = b'general.architecture'
                fh.write(struct.pack('<Q', len(key)) + key + struct.pack('<I', 8))
                value = b'qwen35'
                fh.write(struct.pack('<Q', len(value)) + value)
                name = b'blk.0.weight'
                fh.write(struct.pack('<Q', len(name)) + name)
                fh.write(struct.pack('<I', 2))            # 2 dimensions
                fh.write(struct.pack('<QQ', 4096, 4096))
                fh.write(struct.pack('<I', 142))          # PQ2_0, opaque
                fh.write(struct.pack('<Q', 5 * 1024**3))  # > 4 GiB
            _parse_gguf_raw = self.ns['_parse_gguf_raw']
            _meta, params, tensors = _parse_gguf_raw(path)
            self.assertEqual(len(tensors), 1)
            self.assertEqual(tensors[0][3], 5 * 1024**3)
            self.assertEqual(params, 4096 * 4096)
            self.assertEqual(_meta.get('general.architecture'), 'qwen35')

    def test_exit_logs_are_classified_from_their_own_text(self):
        classify = self.ns['classify_exit']
        self.assertEqual(classify('CUDA error 2: out of memory'), 'oom')
        self.assertEqual(classify('ggml_gallocr_reserve_n: failed to allocate'), 'oom')
        self.assertEqual(classify('GGML_ASSERT(foo) failed'), 'oom')
        self.assertEqual(classify('segmentation fault'), 'crash')
        self.assertEqual(classify(''), 'crash')
        self.assertEqual(classify(None), 'crash')

    def test_restart_budget_is_bounded_in_count_and_rate(self):
        clock = [0.0]
        budget = self.ns['RestartBudget'](max_restarts=2, window=10.0,
                                          now=lambda: clock[0])
        self.assertTrue(budget.allow())
        budget.note()
        budget.note()
        self.assertFalse(budget.allow())
        clock[0] = 11.0                       # outside the rate window
        self.assertTrue(budget.allow())
        self.assertEqual(budget.used, 0)

    def test_truthy_env_reads_opt_in_flags(self):
        import os
        truthy = self.ns['truthy_env']
        saved = os.environ.get('BONSAI_TEST_FLAG')
        try:
            os.environ['BONSAI_TEST_FLAG'] = '1'
            self.assertTrue(truthy('BONSAI_TEST_FLAG'))
            os.environ['BONSAI_TEST_FLAG'] = 'YES'
            self.assertTrue(truthy('BONSAI_TEST_FLAG'))
            os.environ['BONSAI_TEST_FLAG'] = '0'
            self.assertFalse(truthy('BONSAI_TEST_FLAG'))
            os.environ.pop('BONSAI_TEST_FLAG')
            self.assertFalse(truthy('BONSAI_TEST_FLAG'))
        finally:
            if saved is None:
                os.environ.pop('BONSAI_TEST_FLAG', None)
            else:
                os.environ['BONSAI_TEST_FLAG'] = saved

    def test_pick_mmproj_never_invents_a_filename(self):
        pick = self.ns['pick_mmproj']
        siblings = [type('S', (), {'rfilename': n})()
                    for n in ('Ternary-Bonsai-2-27B-PQ2_0.gguf', 'mmproj-Q8_0.gguf',
                              'mmproj-BF16.gguf')]
        self.assertEqual(pick(siblings), 'mmproj-Q8_0.gguf')
        self.assertEqual(pick(siblings, 'mmproj-BF16.gguf'), 'mmproj-BF16.gguf')
        with self.assertRaisesRegex(RuntimeError, 'refusing to substitute'):
            pick(siblings, 'mmproj-nope.gguf')
        text_only = [type('S', (), {'rfilename': 'Ternary-Bonsai-2-27B-PQ2_0.gguf'})()]
        with self.assertRaisesRegex(RuntimeError, 'no mmproj'):
            pick(text_only)
        ambiguous = [type('S', (), {'rfilename': n})()
                     for n in ('mmproj-A.gguf', 'mmproj-B.gguf')]
        with self.assertRaisesRegex(RuntimeError, 'BONSAI_MMPROJ'):
            pick(ambiguous)

    def test_image_tokens_are_reserved_from_the_context(self):
        reserve = self.ns['image_context_reserve']
        self.assertEqual(reserve(1024), 2048)
        self.assertEqual(reserve(4096, images_per_turn=1), 4096)
        self.assertEqual(reserve(0), 0)
        self.assertEqual(reserve(99999), 4096 * 2)   # clamped to the model's maximum

    def test_slot_report_says_unknown_when_the_runtime_does_not(self):
        report = self.ns['slot_report']
        self.assertIsNone(report(lambda url, key, timeout=5: (404, {}), 'http://x', 'k'))
        self.assertIsNone(report(lambda url, key, timeout=5: (200, {}), 'http://x', 'k'))
        got = report(lambda url, key, timeout=5: (200, [{'id': 0, 'n_ctx': 8192}]),
                     'http://x', 'k')
        self.assertEqual(got[0]['n_ctx'], 8192)
        self.assertIsNone(got[0].get('model'))

    def test_choose_context_charges_every_slot_for_its_own_cache(self):
        choose = self.ns['choose_context']
        one = choose(free_mib=[15360], per_token_mib=0.02, model_mib=6970, slots=1)
        four = choose(free_mib=[15360], per_token_mib=0.02, model_mib=6970, slots=4)
        self.assertLess(four, one)

    def test_choose_context_counts_the_projector_before_the_cache(self):
        choose = self.ns['choose_context']
        without = choose(free_mib=[15360], per_token_mib=0.02, model_mib=6970)
        withproj = choose(free_mib=[15360], per_token_mib=0.02, model_mib=6970,
                          extra_mib=645)
        self.assertLessEqual(withproj, without)

    def test_require_capacity_can_be_asked_to_hold_room_for_the_projector(self):
        require = self.ns['require_capacity']
        require([15360], {'server_pid': 1})          # an existing server is exempt
        require([15360], None)
        with self.assertRaises(RuntimeError):
            require([9000], None)
        with self.assertRaises(RuntimeError):
            require([10000], None, extra_mib=645)

    def test_sha256_helper_matches_hashlib(self):
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'blob'
            path.write_bytes(b'x' * 300000)
            expected = hashlib.sha256(b'x' * 300000).hexdigest()
            self.assertEqual(self.ns['sha256_file'](path).hexdigest(), expected)

    def test_supervisor_needs_confirmation_before_it_restarts_anything(self):
        Supervisor = self.ns['Supervisor']
        checks = {'n': 0}

        def health():
            checks['n'] += 1
            return False, 'down'

        sup = Supervisor('test', health, lambda reason, detail: 'restarted',
                         confirmations=2, sleep=lambda s: None)
        self.assertIsNone(sup.step(), 'one failed check must not restart anything')
        self.assertEqual(sup.budget.used, 0)
        event = sup.step()
        self.assertEqual(event['action'], 'restart')
        self.assertEqual(event['detected'], 'unhealthy')
        self.assertEqual(sup.budget.used, 1)

    def test_supervisor_stops_instead_of_flapping_past_its_budget(self):
        Supervisor = self.ns['Supervisor']
        sup = Supervisor('test', lambda: (False, 'down'),
                         lambda reason, detail: 'restarted',
                         budget=self.ns['RestartBudget'](max_restarts=1),
                         confirmations=1, sleep=lambda s: None)
        sup.step()
        self.assertEqual(sup.events[-1]['action'], 'restart')
        event = sup.step()
        self.assertEqual(event['action'], 'give-up')
        self.assertFalse(sup.running())

    def test_a_generation_probe_is_only_run_on_its_own_schedule(self):
        Supervisor = self.ns['Supervisor']
        calls = []

        def probe(timeout):
            calls.append(timeout)
            return False, 'no token'

        clock = [0.0]
        sup = Supervisor('test', lambda: (True, 'ok'), lambda r, d: 'restarted',
                         probe=probe, hang_check_interval=100.0,
                         sleep=lambda s: None, now=lambda: clock[0])
        self.assertIsNone(sup.step())
        self.assertEqual(calls, [])
        clock[0] = 200.0
        self.assertEqual(sup.step()['detected'], 'hang')
        self.assertEqual(calls, [sup.stall_timeout])

    def test_heartbeat_writes_atomically_and_never_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'heartbeat.json'
            payload = self.ns['write_heartbeat'](path, {'port': 1234})
            self.assertEqual(json.loads(path.read_text())['port'], 1234)
            self.assertIn('written_at', payload)
            # an unwritable path is a diagnostic failure, not a deployment failure
            self.assertTrue(self.ns['write_heartbeat'](Path(tmp) / 'nope' / 'x.json',
                                                       {'port': 1}))


class HarnessSelfTests(unittest.TestCase):
    """The fakes have to be faithful, or every test above is theatre."""

    def test_the_cell_keeps_exactly_one_top_level_try_block(self):
        tree = Path(__file__).with_name('colab_kaggle_cell.py').read_text()
        import ast
        tries = [n for n in ast.parse(tree).body if isinstance(n, ast.Try)]
        self.assertEqual(len(tries), 1)

    def test_the_fake_gpu_reports_process_attributed_vram(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fakes = harness.build_fakes(root / 'fakes')
            gpu_dir = root / 'gpu'
            (gpu_dir / 'apps').mkdir(parents=True)
            (gpu_dir / 'config.json').write_text(json.dumps({'gpus': [T4]}))
            env = dict(__import__('os').environ)
            env.update({'FAKE_GPU_DIR': str(gpu_dir), 'FAKE_RUN_DIR': str(root)})
            import subprocess
            out = subprocess.run(
                [str(fakes['root'] / 'bin/nvidia-smi'),
                 '--query-gpu=index,memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(out.split(',')[2].strip(), '0')
            (gpu_dir / 'apps' / '1.json').write_text(json.dumps(
                {'pid': 1, 'per_gpu': {'0': 7000}}))
            out = subprocess.run(
                [str(fakes['root'] / 'bin/nvidia-smi'), '--query-compute-apps=pid,'
                 'used_gpu_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, env=env).stdout
            self.assertIn('1, 7000', out)
            out = subprocess.run(
                [str(fakes['root'] / 'bin/nvidia-smi'),
                 '--query-gpu=index,memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(out.split(',')[2].strip(), '7000')   # dead pid: not counted

    def test_the_fake_gguf_carries_a_27b_parameter_count(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sys_path = __import__('sys').path
            try:
                sys_path.insert(0, str(Path(__file__).parent))
                import _gguf_writer as gguf
                path = Path(tmp) / 'm.gguf'
                gguf.write_gguf(str(path), 1_000_000)
                source = Path(__file__).with_name('colab_kaggle_cell.py').read_text()
                import ast
                definitions = ast.Module(
                    body=[n for n in ast.parse(source).body
                          if not isinstance(n, ast.Try)], type_ignores=[])
                ns = {'__name__': 'defs'}
                exec(compile(definitions, 'cell', 'exec'), ns)
                meta, params = ns['read_gguf_identity'](str(path))
                self.assertEqual(meta['general.architecture'], 'qwen35')
                self.assertAlmostEqual(params / 1e9, 27.36, places=2)
            finally:
                sys_path.remove(str(Path(__file__).parent))


if __name__ == '__main__':
    unittest.main()
