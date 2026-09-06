# Novelty Audit

**Audit date:** 2026-09-05  
**Target venue:** Transactions on Machine Learning Research (TMLR)  
**Scope:** delayed online learning, neural eligibility and influence traces, low-rank gradient/Jacobian approximation, matrix sketching, replay, and delayed conversion modeling.

## Bottom line

The project has a defensible TMLR contribution when presented as an empirical study of **what to retain while labels are pending**. The SVD operation and the general idea of low-rank credit approximation are established techniques. The strongest contribution is the joint experimental framework and its central finding: a delayed-credit representation can reconstruct exact observation-time gradients almost perfectly and still lose to input replay or a cruder sketch in predictive performance and end-to-end throughput.

The literature search did not locate an earlier study that jointly provides all of the following:

1. per-example, label-free observation-time Jacobians retained until asynchronous labels arrive;
2. exact, truncated-SVD, random-sketch, shared/layerwise, full-replay, and equal-byte replay alternatives under one measured state boundary;
3. separate evaluation of credit error, gradient cosine, predictive performance, state bytes, memory, latency, and throughput; and
4. fixed-delay image and naturally delayed conversion streams with leakage-controlled chronological evaluation.

This is a scoped search conclusion, not proof that no related work exists. The manuscript should use wording such as “we study” and “we are unaware of a prior comparison with this combination of state boundary, controls, and measurements.” It should avoid an unqualified priority claim.

## Closest prior work

| Literature | Established contribution | Overlap with this project | Remaining distinction |
|---|---|---|---|
| Joulani, György, and Szepesvári (ICML 2013), [Online Learning under Delayed Feedback](https://proceedings.mlr.press/v28/joulani13.html) | General reductions and regret analysis for delayed online feedback | Formal delayed-feedback setting | Does not study neural observation-time Jacobian storage, rank compression, or matched replay bytes |
| Csaba et al. (NeurIPS 2024), [Label Delay in Online Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d8f5f134febb4bd74d8f79e338de382c-Abstract-Conference.html) | Explicit label-delay protocol and strong rehearsal baseline | Yearbook stream, fixed delays, replay as a central control | Does not retain or compress per-example credit states; the present Yearbook arm uses its released ordering and delay convention |
| Ollivier, Tallec, and Charpiat (2015), [NoBackTrack](https://arxiv.org/abs/1507.07680), and Tallec and Ollivier (2017), [UORO](https://arxiv.org/abs/1702.05043) | Low-memory unbiased approximations to online recurrent gradients | Factorized credit/Jacobian approximation | Approximates the evolving RTRL influence matrix during a sequence; does not queue one credit object per example until an external label arrives |
| Mujika, Meier, and Steger (NeurIPS 2018), [KF-RTRL](https://proceedings.neurips.cc/paper/2018/hash/dba132f6ab6a3e3d17a8d59e82105f4c-Abstract.html), and Benzing et al. (ICML 2019), [Optimal Kronecker-Sum RTRL](https://proceedings.mlr.press/v97/benzing19a.html) | Structured, memory-efficient RTRL approximations with variance analysis | Structured compression and gradient-fidelity questions | Targets temporal derivatives of recurrent state and frequent online updates, rather than asynchronous label queues and replay/recomputation choices |
| Bellec et al. (Nature Communications 2020), [e-prop](https://www.nature.com/articles/s41467-020-17236-y), and Bohnstingl et al. (IEEE TNNLS 2022), [OSTL](https://ieeexplore.ieee.org/document/9736444/) | Local eligibility traces and online temporal-gradient computation | Separates an eligibility signal from a later learning signal | Primarily addresses biologically plausible or online recurrent training; no matched-byte delayed-label storage comparison |
| van Hasselt et al. (AAAI 2021), [Expected Eligibility Traces](https://ojs.aaai.org/index.php/AAAI/article/view/17200) | Expected traces over possible predecessor trajectories | Eligibility representation and credit assignment | Changes which histories receive credit; does not compress per-example neural Jacobians while labels are pending |
| Vogels, Karimireddy, and Jaggi (NeurIPS 2019), [PowerSGD](https://proceedings.neurips.cc/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html) | Practical low-rank gradient compression for distributed communication | Low-rank factorization of gradient-shaped matrices | Compresses already available worker gradients for communication; the delayed-credit setting lacks the label residual when state is encoded |
| Ghashami et al. (SIAM J. Computing 2016), [Frequent Directions](https://epubs.siam.org/doi/10.1137/15M1009718) | Deterministic streaming matrix sketch with approximation guarantees | Candidate sketch family and memory accounting | Generic row-stream matrix sketching; no delayed supervision or learner-level comparison |
| Yi and Wang (2026 preprint), [Certified Low-Rank RTRL](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7070381) | SnAp-1 plus a deterministic two-sided low-rank residual sketch and an online error certificate | Closest algebraic overlap; directly prevents a broad “first low-rank credit sketch” claim | Approximates a dense-RNN influence matrix continuously; the present work studies per-observation credit storage, external label arrival, state-matched replay, and the fidelity/utility separation |
| Chapelle (KDD 2014), [Modeling Delayed Feedback in Display Advertising](https://dl.acm.org/doi/10.1145/2623330.2623634), and Diemert et al. (AdKDD 2017), [Attribution Modeling for Bidding](https://dl.acm.org/doi/10.1145/3124749.3124752) | Conversion-delay modeling and the Criteo attribution data source | Naturally delayed labels and conversion timestamps | Models censoring/attribution or supplies data; does not compare retained credit representations |

## Novelty assessment by component

| Component | Assessment | Reason |
|---|---:|---|
| Applying truncated SVD to a matrix-valued credit object | Low | Direct use of established low-rank approximation |
| Streaming low-rank and shared-right-subspace variants | Moderate | The allocation and state accounting are tailored to layered credit, while related structured RTRL and sketch methods are established |
| Per-example delayed-credit state formulation | Moderate to high | The storage object is fixed before the label residual exists and is released on asynchronous feedback; this creates a different choice from recurrent-gradient approximation |
| Equal-byte comparison with replay/recomputation and random sketching | High | It tests the operational decision a memory-limited delayed learner actually faces |
| Exact-credit audit separated from full predictive comparison | High | It reveals that approximation fidelity and learner utility answer different questions |
| External validation on fixed and naturally variable delays | Moderate to high | The benchmark pairing, chronological controls, and outcome-field exclusions provide useful evidence beyond the synthetic task family |
| Overall TMLR interest | Moderate to high | The study gives a reproducible negative boundary and an actionable lesson even though the primary method is not the strongest learner |

## Claims that the evidence supports

- Rank-four global SVD reduced the exact per-event credit-state representation by 6.3519x in the external adapter while meeting every preregistered exact-credit audit criterion.
- Median gradient cosine ranged from 0.9978 to 0.9999 across the four external benchmark/delay audit cells.
- Predictive behavior stayed within the declared exact-credit tolerance in every audit cell.
- High-fidelity approximation of exact delayed credit did not imply competitive superiority: no paired confirmatory interval favored global SVD at the required conditions.
- Input replay and random sketching are essential controls when assessing compressed delayed-credit states.
- The preferred representation depends on the state boundary, feature size, recomputation cost, model drift during delay, and whether replay is permitted.

## Claims to avoid

- “The first low-rank eligibility trace.”
- “The first memory-efficient online credit-assignment method.”
- “Global SVD solves label delay.”
- “Global SVD is superior to replay, random projection, or Frequent Directions.”
- “Near-perfect gradient cosine guarantees near-oracle delayed learning.”
- “The implementation reduces peak memory or training time.”
- “The Yearbook arm reproduces the full-network results of Csaba et al.”
- “The Criteo result applies to unrestricted commercial deployment.”
- Any claim of generality beyond the tested four-matrix adapter, frozen feature boundaries, delays, and streams.

## Recommended TMLR positioning

**Working title:** *Low-Rank Delayed-Credit States: Gradient Fidelity, Memory Compression, and Competitive Limits*

**Central question:** When labels arrive after the model has moved, is it better to retain an observation-time credit state or to retain inputs and recompute credit later?

**Primary contribution:** A controlled and externally validated comparison that separates representational fidelity from predictive utility under explicit memory budgets.

**Main lesson:** Gradient-faithful compression answers “how closely can we reproduce exact delayed credit?” Replay and sketch controls answer the separate operational question “which retained information produces the best learner?” The experiments show that these answers can diverge.

This positioning fits TMLR’s stated scope for experimental studies that yield insight into learning-system behavior and for applications that expose method strengths and weaknesses. It also aligns with TMLR’s acceptance emphasis on evidence-supported claims and reader interest rather than benchmark state of the art: [TMLR editorial policies](https://jmlr.org/tmlr/editorial-policies.html) and [acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html).

## Remaining novelty risk

The main residual risk is that a reviewer views the work as a straightforward SVD application with a negative benchmark outcome. The paper should make the delayed-credit state boundary mathematically explicit, explain why observation-time credit and label-time recomputation differ after parameter drift, and use the external results to derive practical selection rules. A small theorem or proposition characterizing when the two updates coincide would materially strengthen the paper, although TMLR does not require a theoretical contribution.
