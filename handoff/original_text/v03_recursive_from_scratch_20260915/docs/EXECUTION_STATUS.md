# Execution status

- User authorizes the complete V0.3 workflow without stage confirmations, on de36 physical NPUs 0–3 only.
- Next noon is a preference, not a hard deadline. Do not truncate scientific checks just to meet it.
- Earlier 2–4 day estimate withdrawn because update count and total path count were conflated. Estimate at fixed total sampled paths after four-card benchmark.
- Local/remote independent project created. Base V0.1 core/config hashes verified unchanged.
- CPU contract tests passed (initial 12); subsequent additions must be rerun.
- Four-card longest G4.5 K5 microbatch1 global4 acceptance first segment completed: 3 updates, 12 paths, original checkpoint at runs/acceptance_longest_01/checkpoints/update-000003.pt. Warm update ~10.1 seconds, ~14 GiB allocated/card. This is not selected formal batch.
- Acceptance queue 01 launched: resume3→6, fresh uninterrupted6 reference, mixed+uneven tail, real gradient probes, original V0.1 full inference. Outputs artifacts/runtime/acceptance_queue_01/status.json.
- Public AlexNet dependency downloaded from official PyTorch URL, full hash matches original V0.1 cache. Initial internal cache export was rejected by automatic review; no internal payload exported.
- Formal protocol and training not started.
