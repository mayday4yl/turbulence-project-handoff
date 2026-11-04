# V0.3: predicted-state recursive training from scratch

This is the new experiment authorized on 2026-09-15. Its only source authority is
the hash-pinned V0.1 at `/data0/yanglin/physical_bridge`. This directory never
imports another V0.3 or V0.2 implementation. The base source and old artifacts
remain untouched.

## Frozen method

- Original DualHeadUNet, PiRN/P2S simulator and full raw realization representation.
- Uniform original train trajectory, independent uniform six-level terminal G.
- Requested K in {0,1,3,5}; original schedule builder; deduplicate full schedule
  tuples within G before applying curriculum probabilities.
- Each path starts with original Y=X_G. Predict clean and xi, supervise full-image
  clean MSE and full-image MSE(F(predicted clean,predicted xi,original G),original Y).
- Next input is F(predicted clean,predicted xi,next G) with the entire graph intact.
- Mean losses within each path, then equal weighting across actual global paths.
- No teacher forcing, detach, training refinement, raw-xi regression, next-state
  GT loss or old GT-clean degradation loss.
- Unchanged V0.1 inference, including original schedule, proximal/prior terms,
  candidate selection, frozen refinement LR 0.03, iteration/early-stop/best rules
  and 16-pixel refinement boundary. Training losses have no boundary crop.
- One freshly randomized formal model, seed fixed before pilots. No old or pilot
  weights may initialize the formal run. No held-out test execution.

## Execution

Only de36 physical NPUs 0,1,2,3 are authorized. Run commands under scripts/guard.py.
Every stage declares a finite budget, and retries count toward the total budget.
Formal configuration is frozen only after fair pilots and trainer acceptance.
The user's next-noon target is a preference, not a hard deadline or permission to
truncate scientific checks. Timing is compared at a fixed total path workload.

Remote root: `/data2/yanglin/v03_recursive_from_scratch_20260915`.
The original successful gradient smoke remains separately archived under
`/data2/yanglin/v01_recursive_gradient_smoke_20260915/attempt3`.
