# N08+O10 stacked v1.0 result: invalid confirmation, no negative inference

## Provenance

- Returned archive: `n08_o10_stacked_v10_results.zip`
- SHA-256: `ff5391547a6411ef98ae7c3b344a05a2d07f16aa4e87b9a8ee0577ddaad380e4`
- Archive size: 81,583,459 bytes
- GPU: NVIDIA RTX 6000 Ada (user-confirmed)
- Measured wall time: 77,579 seconds (21 hours, 32 minutes, 59 seconds)
- Internal cascade time: 77,576.13 seconds
- Premise, calibration, screen, and confirmation ran; the conditional boundary did not run.

## Frozen classification

The v1.0 confirmation is **invalid and inconclusive**, not a valid negative result. It cannot
change the positive N08 v2 result or O10's bounded long-horizon finding. It also cannot support a
stacked-subspace advantage.

Two prespecified apparatus conditions were violated:

1. The exact full-trace reference failed the minimum 20% cell-level learning-headroom gate in
   10 of 96 confirmation cells.
2. Only the stacked candidate was calibrated. It used learning rate 0.001 at every horizon, while
   the exact trace, equal-layer sketch, global SVD control, and replay controls used 0.002.

Because the v1.0 analysis divided several comparisons by
`max(initial_mse - full_trace_mse, 1e-12)`, cells in which the exact trace became worse than the
initial model produced meaningless ratios on the order of millions. The cascade correctly marked
the reference invalid, but its terminal decision string described the candidate as unsupported.
That wording is superseded by the classification above.

## What the run did establish

The structural premise passed. The candidate used 4,160 of 4,352 allocated floats (25.39% of the
16,384-float exact trace) and obtained a matched-budget stacked rank of 13 rather than equal-layer
rank 8. Its premise trace cosine was 0.99981-0.99997 on latent-rank-two tasks and
0.96615-0.97004 on latent-rank-four tasks, without norm inflation.

The three-teacher screen was valid and strong under the v1.0 optimizer assignment:

| Quantity | 8x horizon | 32x horizon |
|---|---:|---:|
| Candidate/full gain retention | 1.0890 | 1.1243 |
| Candidate gain vs global control | 0.0848 | 0.1208 |
| Candidate gain vs equal-layer control | 0.0917 | 0.1267 |
| Candidate gain vs replay | 0.0523 | 0.1327 |
| Candidate gradient cosine | 0.9796 | 0.9792 |

It won 10/12 cells against the global control and 11/12 against both equal-layer and replay.
These observations justify an apparatus repair, but they are not optimizer-fair evidence.

## Confirmation failure localization

All ten invalid exact-reference cells were latent-rank-two tasks:

| Teacher | Invalid horizons and nested replicates | Count |
|---|---|---:|
| 907 | 16x (replicates 0 and 1), 32x (replicates 0 and 1) | 4 |
| 1201 | 8x, 16x, and 32x (both replicates at each horizon) | 6 |

The worst cell was teacher 1201, replicate 0, 32x: initial MSE was 0.00041993 and exact-trace MSE
was 0.00161769, a learning fraction of -2.8523. The candidate MSE was 0.00039024. Across all 96
cells, the candidate descriptively beat the global control in 86, equal-layer in 94, and replay in
96. Those counts are not inferential because the candidate alone received the lower optimizer
rate and the reference gate failed.

The premise geometry reinforces the confound. On latent-rank-two tasks the candidate is nearly
exact, while on latent-rank-four tasks the equal-layer sketch has higher trace cosine than the
candidate. A large training-MSE advantage with a half learning rate therefore cannot be assigned
to representation quality without tuning every method by the same rule.

## Authorized next step

One optimizer/reference apparatus repair is authorized as v1.1:

- tune every method over the same learning-rate grid on the same two unused teachers;
- validate the selected rates on a third unused holdout teacher;
- rerun only the exact trace on the two known failure teachers as a noninferential validity audit;
- proceed to completely fresh screen and confirmation teachers only if the audit passes;
- normalize control comparisons by initial MSE and compute gain retention only after the exact
  reference is valid in every cell.

If the optimizer-fair screen or confirmation is validly negative, the bridge stops permanently
and the project proceeds as an N08-centered bounded paper. No further stacked-mechanism rescue or
control removal is authorized.
