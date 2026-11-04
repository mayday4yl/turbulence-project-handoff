# V0.2 execution amendment — 2026-08-29

This addendum records explicit user decisions and semantics-preserving execution
optimizations. It does not replace either frozen V0.2 authority document.

## Explicit single-seed decision

The user explicitly removed Seed 2 from this execution on 2026-08-29. Even if
the original Section 12 trigger fires, no Seed-2 B0/B1 confirmation run will be
started. All reported conclusions must be labelled single-seed evidence and
must not claim multi-seed confirmation.

This decision does not change Seed-1 A0, A1, B0/B1, beta, K, frozen-test, metric,
or STOP semantics.

## Approved engineering acceleration boundary

The following optimizations preserve the frozen losses and reductions:

- render fixed-size bridge microbatches and reduce only over `valid_h` rows;
- batch the estimator L_T and L_B simulator calls and split their outputs;
- batch the at-most-three full-L_T severities with zero-weight padding;
- compute the gate-required full-L_T diagnostic during updates 1–200 and
  2801–3000; it remains excluded from optimization;
- probe per-rank microbatch sizes 4, 2, then 1 and select the largest that
  completes one disposable full A1 update; effective batch remains 16;
- execute B1 first and delay B0 until B1 has passed the gates needed to justify
  the matched control; B0 remains mandatory before any final comparative claim,
  uses the identical frozen manifest, and is trained to B1's selected update;
- reuse deterministic beta-independent predictions and K-prefix rollout states
  during later validation, subject to equivalence checks before formal use.

The following suggestions are not adopted because they alter the frozen
scientific execution rather than merely its implementation:

- merging the independent 3k A1 subset pilot into the full-data B1 run;
- adaptively ending the fixed 128-batch A0 audit;
- omitting the matched B0 control or changing the B0/B1 update-alignment rule;
- adding a newly tuned refinement early-stop rule.
