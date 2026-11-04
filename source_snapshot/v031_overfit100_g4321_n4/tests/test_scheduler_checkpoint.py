"""CPU-only state fixtures. No model (including the toy fixture) is updated.

AdamW moments below are SYNTHETIC values, not optimizer-step results. Scheduler
counter advances are state-machine simulations, not reported training updates.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from v031.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from v031.scheduler import UpdateIndexedScheduler, lr_for_update


IDENTITY = {
    "method_version": "V0.31", "dataset_id": "fixture_not_real_data",
    "dataset_manifest_sha256": "fixture_dataset_digest",
    "network_sha256": "fixture_network_digest", "simulator_sha256": "fixture_simulator_digest",
    "code_sha256": "fixture_code_digest",
}
FIXTURE_LABEL = "CPU synthetic optimizer moments; zero actual optimizer updates"


class StateFixture:
    def __init__(self, state):
        self.state = deepcopy(state)

    def state_dict(self):
        return deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = deepcopy(state)


def make_fixture(completed=0):
    network = torch.nn.Linear(3, 2)
    # The remaining settings are TEST FIXTURE values, not a Stage 2 protocol.
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.7, betas=(0.8, 0.95),
                                  eps=1e-7, weight_decay=0.03)
    for index, parameter in enumerate(network.parameters()):
        optimizer.state[parameter] = {
            "step": torch.tensor(float(completed), dtype=torch.float32),
            "exp_avg": torch.full_like(parameter, (index + 1) * 0.125),
            "exp_avg_sq": torch.full_like(parameter, (index + 1) * 0.375),
        }
    scheduler = UpdateIndexedScheduler(optimizer, completed_updates=completed)
    ema = StateFixture({"updates": completed, "decay": 0.9,
                        "integer_counter": torch.tensor(completed, dtype=torch.int64),
                        "initialized": torch.tensor(True, dtype=torch.bool),
                        "shadow": {key: value.detach().float().clone() + 0.5
                                   for key, value in network.state_dict().items()}})
    generator = torch.Generator().manual_seed(189)
    sampler = StateFixture({"epoch": 3, "position": 19, "permutation": torch.randperm(23, generator=generator),
                            "generator_state": generator.get_state(), "completed_paths": completed * 32})
    return network, optimizer, scheduler, ema, sampler


def bits(value):
    return struct.pack("!d", value)


class SchedulerCheckpointTests(unittest.TestCase):
    def setUp(self):
        # Any accidental optimizer update fails the fixture suite immediately.
        self.step_guard = patch.object(torch.optim.AdamW, "step", side_effect=AssertionError(
            "No optimizer.step is allowed in Stage 1 state-fixture checks"))
        self.step_guard.start()
        self.addCleanup(self.step_guard.stop)
        random.seed(231)
        torch.manual_seed(231)
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.seed(231)

    def assertTreeEqual(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual))
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assertTreeEqual(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assertTreeEqual(left, right)
        elif isinstance(expected, float):
            self.assertEqual(bits(actual), bits(expected))
        else:
            self.assertEqual(actual, expected)

    def test_exact_landmarks_and_index_convention(self):
        self.assertEqual(lr_for_update(0), 0.0)
        self.assertAlmostEqual(lr_for_update(1), 2e-6, places=18)
        self.assertEqual(lr_for_update(50), 1e-4)
        self.assertEqual(lr_for_update(1000), 1e-5)
        self.assertEqual(lr_for_update(1001), 1e-5)
        self.assertEqual(lr_for_update(1500), 1e-5)
        _, optimizer, scheduler, _, _ = make_fixture()
        self.assertIsNone(scheduler.last_used_lr)
        self.assertEqual(optimizer.param_groups[0]["lr"], lr_for_update(1))
        scheduler.mark_update_completed(update_index=1)
        self.assertEqual(scheduler.completed_updates, 1)
        self.assertEqual(scheduler.last_used_lr, lr_for_update(1))
        self.assertEqual(optimizer.param_groups[0]["lr"], lr_for_update(2))
        with self.assertRaises(ValueError):
            scheduler.mark_update_completed(update_index=1)
        for value in (-1, True, 1.2):
            with self.assertRaises(ValueError):
                lr_for_update(value)

    def test_500_and_1000_resume_pointwise_bit_exact_to_2000(self):
        _, _, baseline, _, _ = make_fixture()
        trajectory = []
        for update in range(1, 2001):
            trajectory.append(bits(baseline.next_lr))
            baseline.mark_update_completed(update_index=update)
        with tempfile.TemporaryDirectory() as directory:
            for boundary in (500, 1000):
                with self.subTest(boundary=boundary):
                    network, optimizer, scheduler, ema, sampler = make_fixture(boundary)
                    path = Path(directory) / f"fixture-{boundary}.pt"
                    expected_network = deepcopy(network.state_dict())
                    expected_optimizer = deepcopy(optimizer.state_dict())
                    expected_scheduler = deepcopy(scheduler.state_dict())
                    expected_ema = ema.state_dict()
                    expected_sampler = sampler.state_dict()
                    save_checkpoint(path, network=network, optimizer=optimizer, scheduler=scheduler,
                                    ema=ema, sampler=sampler, completed_updates=boundary, identity=IDENTITY,
                                    include_npu_rng=False, fixture_label=FIXTURE_LABEL)
                    expected_draw = (random.random(), torch.rand(11))
                    try:
                        import numpy as np
                    except ImportError:
                        expected_numpy = None
                    else:
                        expected_numpy = np.random.random(7).tolist()
                    network2, optimizer2, scheduler2, ema2, sampler2 = make_fixture()
                    ema2.state = {"deliberately": "different"}
                    sampler2.state = {"position": -1}
                    random.random()
                    torch.rand(17)
                    result = load_checkpoint(path, network=network2, optimizer=optimizer2,
                                             scheduler=scheduler2, ema=ema2, sampler=sampler2,
                                             expected_identity=IDENTITY, allow_fixture=True)
                    self.assertEqual(result["completed_updates"], boundary)
                    self.assertEqual(result["next_update_index"], boundary + 1)
                    self.assertTreeEqual(network2.state_dict(), expected_network)
                    self.assertTreeEqual(optimizer2.state_dict(), expected_optimizer)
                    self.assertTreeEqual(scheduler2.state_dict(), expected_scheduler)
                    self.assertTreeEqual(ema2.state_dict(), expected_ema)
                    self.assertTreeEqual(sampler2.state_dict(), expected_sampler)
                    self.assertTreeEqual((random.random(), torch.rand(11)), expected_draw)
                    if expected_numpy is not None:
                        self.assertTreeEqual(np.random.random(7).tolist(), expected_numpy)
                    after_resume = []
                    for update in range(boundary + 1, 2001):
                        after_resume.append(bits(scheduler2.next_lr))
                        scheduler2.mark_update_completed(update_index=update)
                    self.assertEqual(after_resume, trajectory[boundary:])
                    # Virtual schedule advances did not alter any model/moments.
                    self.assertTreeEqual(network2.state_dict(), expected_network)
                    for parameter, state in optimizer2.state.items():
                        self.assertEqual(state["step"].item(), boundary)

    def test_reject_identity_fixture_counter_and_overwrite(self):
        network, optimizer, scheduler, ema, sampler = make_fixture(500)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            kwargs = dict(network=network, optimizer=optimizer, scheduler=scheduler,
                          ema=ema, sampler=sampler)
            save_kwargs = {**kwargs, "completed_updates": 500, "identity": IDENTITY,
                           "include_npu_rng": False, "fixture_label": FIXTURE_LABEL}
            with self.assertRaises(ValueError):
                save_checkpoint(path, **{**save_kwargs, "completed_updates": 501})
            self.assertFalse(path.exists())
            save_checkpoint(path, **save_kwargs)
            original_bytes = path.read_bytes()
            original_state = deepcopy(network.state_dict())
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, **save_kwargs)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_checkpoint(path, **kwargs, expected_identity={**IDENTITY, "dataset_id": "wrong"},
                                allow_fixture=True)
            with self.assertRaisesRegex(ValueError, "fixture"):
                load_checkpoint(path, **kwargs, expected_identity=IDENTITY)
            self.assertTreeEqual(network.state_dict(), original_state)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_restore_failure_rolls_back_all_mutated_objects(self):
        class RejectForeignSampler(StateFixture):
            def load_state_dict(self, state):
                if state.get("position") == 19:
                    raise ValueError("synthetic sampler rejection")
                super().load_state_dict(state)

        with tempfile.TemporaryDirectory() as directory:
            network, optimizer, scheduler, ema, sampler = make_fixture(500)
            path = Path(directory) / "fixture.pt"
            save_checkpoint(path, network=network, optimizer=optimizer, scheduler=scheduler,
                            ema=ema, sampler=sampler, completed_updates=500, identity=IDENTITY,
                            include_npu_rng=False, fixture_label=FIXTURE_LABEL)
            n2, o2, s2, e2, _ = make_fixture(3)
            p2 = RejectForeignSampler({"position": 7})
            before = [deepcopy(obj.state_dict()) for obj in (n2, o2, s2, e2, p2)]
            rng_before = capture_rng_state(include_npu=False)
            with self.assertRaisesRegex(ValueError, "synthetic sampler rejection"):
                load_checkpoint(path, network=n2, optimizer=o2, scheduler=s2, ema=e2,
                                sampler=p2, expected_identity=IDENTITY, allow_fixture=True)
            for obj, state in zip((n2, o2, s2, e2, p2), before):
                self.assertTreeEqual(obj.state_dict(), state)
            self.assertTreeEqual(capture_rng_state(include_npu=False), rng_before)

    def test_save_rejects_non_fp32_parameters_moments_and_ema_without_file(self):
        with tempfile.TemporaryDirectory() as directory:
            for component in ("network", "exp_avg", "exp_avg_sq", "ema"):
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(component=component, dtype=dtype):
                        network, optimizer, scheduler, ema, sampler = make_fixture(500)
                        if component == "network":
                            network.to(dtype=dtype)
                        elif component == "ema":
                            key = next(iter(ema.state["shadow"]))
                            ema.state["shadow"][key] = ema.state["shadow"][key].to(dtype)
                        else:
                            state = next(iter(optimizer.state.values()))
                            state[component] = state[component].to(dtype)
                        path = Path(directory) / f"invalid-{component}-{dtype}.pt"
                        rng_before = capture_rng_state(include_npu=False)
                        with self.assertRaisesRegex(ValueError, "must remain FP32"):
                            save_checkpoint(path, network=network, optimizer=optimizer,
                                            scheduler=scheduler, ema=ema, sampler=sampler,
                                            completed_updates=500, identity=IDENTITY,
                                            include_npu_rng=False, fixture_label=FIXTURE_LABEL)
                        self.assertFalse(path.exists())
                        self.assertEqual(list(Path(directory).iterdir()), [])
                        self.assertTreeEqual(capture_rng_state(include_npu=False), rng_before)

    def test_load_rejects_serialized_non_fp32_before_implicit_cast(self):
        with tempfile.TemporaryDirectory() as directory:
            network, optimizer, scheduler, ema, sampler = make_fixture(500)
            valid_path = Path(directory) / "valid-fixture.pt"
            save_checkpoint(valid_path, network=network, optimizer=optimizer, scheduler=scheduler,
                            ema=ema, sampler=sampler, completed_updates=500, identity=IDENTITY,
                            include_npu_rng=False, fixture_label=FIXTURE_LABEL)
            valid_payload = torch.load(valid_path, map_location="cpu", weights_only=True)
            for component in ("network", "exp_avg", "exp_avg_sq", "ema"):
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(component=component, dtype=dtype):
                        payload = deepcopy(valid_payload)
                        if component == "network":
                            container = payload["network"]
                            key = next(iter(container))
                        elif component == "ema":
                            container = payload["ema"]["shadow"]
                            key = next(iter(container))
                        else:
                            container = next(iter(payload["optimizer"]["state"].values()))
                            key = component
                        container[key] = container[key].to(dtype)
                        bad_path = Path(directory) / "tampered-fixture.pt"
                        torch.save(payload, bad_path)
                        n2, o2, s2, e2, p2 = make_fixture(3)
                        before = [deepcopy(obj.state_dict()) for obj in (n2, o2, s2, e2, p2)]
                        rng_before = capture_rng_state(include_npu=False)
                        with self.assertRaisesRegex(ValueError, "must remain FP32"):
                            load_checkpoint(bad_path, network=n2, optimizer=o2, scheduler=s2,
                                            ema=e2, sampler=p2, expected_identity=IDENTITY,
                                            allow_fixture=True)
                        for obj, state in zip((n2, o2, s2, e2, p2), before):
                            self.assertTreeEqual(obj.state_dict(), state)
                        self.assertTreeEqual(capture_rng_state(include_npu=False), rng_before)

    def test_load_rejects_reduced_precision_receiver(self):
        with tempfile.TemporaryDirectory() as directory:
            network, optimizer, scheduler, ema, sampler = make_fixture(500)
            path = Path(directory) / "valid-fixture.pt"
            save_checkpoint(path, network=network, optimizer=optimizer, scheduler=scheduler,
                            ema=ema, sampler=sampler, completed_updates=500, identity=IDENTITY,
                            include_npu_rng=False, fixture_label=FIXTURE_LABEL)
            n2, o2, s2, e2, p2 = make_fixture(3)
            n2.to(dtype=torch.bfloat16)
            before = [deepcopy(obj.state_dict()) for obj in (n2, o2, s2, e2, p2)]
            rng_before = capture_rng_state(include_npu=False)
            with self.assertRaisesRegex(ValueError, "receiving network.*must remain FP32"):
                load_checkpoint(path, network=n2, optimizer=o2, scheduler=s2,
                                ema=e2, sampler=p2, expected_identity=IDENTITY, allow_fixture=True)
            for obj, state in zip((n2, o2, s2, e2, p2), before):
                self.assertTreeEqual(obj.state_dict(), state)
            self.assertTreeEqual(capture_rng_state(include_npu=False), rng_before)

    def test_post_restore_dtype_violation_rolls_back(self):
        class DowncastForeignEMA(StateFixture):
            def load_state_dict(self, state):
                super().load_state_dict(state)
                if state["updates"] == 500:
                    self.state["shadow"] = {key: value.bfloat16()
                                            for key, value in self.state["shadow"].items()}

        with tempfile.TemporaryDirectory() as directory:
            network, optimizer, scheduler, ema, sampler = make_fixture(500)
            path = Path(directory) / "valid-fixture.pt"
            save_checkpoint(path, network=network, optimizer=optimizer, scheduler=scheduler,
                            ema=ema, sampler=sampler, completed_updates=500, identity=IDENTITY,
                            include_npu_rng=False, fixture_label=FIXTURE_LABEL)
            n2, o2, s2, e2, p2 = make_fixture(3)
            e2 = DowncastForeignEMA(e2.state_dict())
            before = [deepcopy(obj.state_dict()) for obj in (n2, o2, s2, e2, p2)]
            rng_before = capture_rng_state(include_npu=False)
            with self.assertRaisesRegex(ValueError, "ema.*must remain FP32"):
                load_checkpoint(path, network=n2, optimizer=o2, scheduler=s2,
                                ema=e2, sampler=p2, expected_identity=IDENTITY, allow_fixture=True)
            for obj, state in zip((n2, o2, s2, e2, p2), before):
                self.assertTreeEqual(obj.state_dict(), state)
            self.assertTreeEqual(capture_rng_state(include_npu=False), rng_before)

    def test_optional_npu_rng_adapter_cpu_mock(self):
        class FakeNPU:
            def __init__(self):
                self.states = [torch.tensor([1, 2], dtype=torch.uint8),
                               torch.tensor([7, 9], dtype=torch.uint8)]

            def is_available(self):
                return True

            def device_count(self):
                return len(self.states)

            def get_rng_state_all(self):
                return [item.clone() for item in self.states]

            def set_rng_state_all(self, states):
                self.states = [item.clone() for item in states]

        fake = FakeNPU()
        with patch.object(torch, "npu", fake, create=True):
            saved = capture_rng_state(include_npu=True)
            fake.states = [torch.zeros_like(item) for item in fake.states]
            restore_rng_state(saved)
            self.assertTreeEqual(fake.states, saved["torch_npu"])
        # This proves the adapter contract only, not real NPU RNG restoration.


if __name__ == "__main__":
    unittest.main(verbosity=2)
