# N08 two-benchmark external-validation cascade v1.1

This archive runs a preregistered external validation of the global low-rank delayed-credit state developed in the N08/O10 project ([DOI 10.5281/zenodo.22217985](https://doi.org/10.5281/zenodo.22217985)). It uses the Yearbook ordering and label-delay convention released with *Label Delay in Online Continual Learning* (NeurIPS 2024), then tests the same credit-state interface on clicked impressions from the Criteo attribution stream with observed conversion times and a declared 30-day maturity rule for non-conversions.

## One-command launch from this repository

```bash
cd external_validation
bash run_external_validation.sh
```

The default is the full five-seed cascade. It downloads only missing source/data, creates or reuses an isolated Python environment, freezes manifests and preprocessing, resumes completed v1.1 trials, logs locally, writes machine-readable tables, and prints `RED`, `YELLOW`, or `GREEN` at the end. No Weights & Biases package, account, API key, or network logger is used.

If a completed v1.0 directory is beside this directory, the launcher automatically reuses its `data/` cache and `.venv`; no benchmark download needs to be repeated. An explicit `DATA_DIR` or `VENV_DIR` always takes priority. The old `results/` directory is never reused because v1.1 is a new protocol.

## Frozen returned result

The full v1.1 run completed validly with terminal status **YELLOW**. Rank-four global SVD passed every exact-credit audit cell at 6.3519x lower per-event state and median gradient cosine from 0.9978 to 0.9999. It did not meet the preregistered competitive requirement against equal-byte replay or random projection. The complete JSON/CSV evidence is committed under [`../results/external_validation_v1.1/`](../results/external_validation_v1.1/), and the root result verifier reproduces the terminal decision.

## Why v1.1 exists

The completed v1.0 run contained promising low-rank geometry, especially on Yearbook and at Criteo rank 8, but its green decision was invalid. It pooled strong Yearbook cosines over weak Criteo cosines, used a Yearbook audit shorter than delays 50 and 100, and counted outcome-defined Criteo delay bins as independent confirmations. Version 1.1 repairs those issues, improves the traced learner's basic trainability, and adds competence checks. See [`docs/V1_0_REVIEW.md`](docs/V1_0_REVIEW.md) for the numerical review.

Two diagnostic modes use the same launcher:

```bash
RUN_MODE=verify bash run_external_validation.sh  # unit and integrity tests
RUN_MODE=smoke  bash run_external_validation.sh  # small synthetic end-to-end run
```

Useful non-scientific overrides are `DATA_DIR`, `RESULTS_DIR`, `VENV_DIR`, `PYTHON_BIN`, `DEVICE`, `FEATURE_BATCH_SIZE`, and `TORCH_INDEX_URL`. The launcher reuses an installed PyTorch when available; otherwise it selects the CUDA 12.4 wheel on a GPU host and the CPU wheel elsewhere. Changing the scientific JSON configuration changes its digest and creates a different protocol; such a run should not be reported as the frozen study.

## What the cascade runs

1. **Source and data integrity.** The launcher checks out the authors' code at commit `5cd6f59e48e8015ecd540f56c08449cb05846214`, uses their `YEARBOOK` loader and within-year ordering, and obtains the published split lists at commit `752c48aa420fd47cd25a5043d8305f3d636ff86c`. The 37,921-image physical archive count is checked before training. The generated manifest records the number referenced by each upstream split, including the paper's separately reported 33,431-image training count.
2. **Frozen Yearbook representation.** A torchvision ResNet-18 with ImageNet-1K V1 weights produces frozen layer-2 maps after a deterministic, record-keyed version of the paper's random resized crop and flip. A fixed projection and 7×7 pooling produce the token stream consumed by the online adapter. This keeps the authors' chronology, batch size 128, and delays 10/50/100 while making the delayed-credit state auditable. The backbone is a frozen featurizer; this package does not claim to reproduce the paper's full-network fine-tuning numbers.
3. **Leakage-controlled Criteo stream.** The launcher downloads the 623 MB Criteo object from Hugging Face at revision `904188a63cbad78bee43cd26ff5ee4ac77903986` and verifies SHA-256 `94ac7a…`. It retains clicked impressions in their published timestamp order. Only `timestamp`, `uid`, `campaign`, `cost`, `time_since_last_click`, and `cat1`–`cat9` enter the feature table. `conversion`, `conversion_timestamp`, `conversion_id`, `attribution`, `click`, `click_pos`, `click_nb`, and `cpo` are forbidden from model input. Conversion fields live in a separate label/schedule table. Positive labels arrive at the observed conversion timestamp; negative labels mature after the dataset's stated 30-day conversion window.
4. **Method-neutral calibration.** Learning rate is selected separately for each benchmark from a four-value grid using development seeds 167 and 181. One selected rate is shared by exact credit, all compressed states, all replay controls, and the oracle. Main inference uses five different frozen seeds.
5. **Audit cascade.** Exact credit and global SVD ranks 2, 4, and 8 run on identical chronological audit windows for all five main seeds. Yearbook uses 16,384 examples, enough to observe updates and later predictions even at delay 100; Criteo uses 524,288 validation clicks. The smallest rank satisfying the frozen performance, per-cell cosine, feedback-exposure, and memory checks advances. If none passes, the strongest budget-eligible rank advances for diagnosis and the status cannot become green.
6. **Full comparison.** The selected global SVD state is compared with full input replay/recomputation, equal-byte replay, a matched-state global random projection, naïve delayed learning with labeled FIFO replay, and an immediate-label oracle. Equal-layer two-sided O10 and shared-right O10 remain secondary analyses and automatically receive the largest ranks that fit the selected global-SVD byte budget.

The online adapter has four trainable 32×32 matrices and a constant homogeneous coordinate that lets those traced matrices learn the class prior. For each binary example, its label-free logit Jacobian is a stack of layer traces, each formed by summing activation/sensitivity outer products over spatial or field tokens. At label arrival, the stored logit residual scales the reconstructed trace. Exact credit stores every matrix. Global SVD stacks layers vertically and stores the leading singular factors. This is the external analogue of the strongest N08 state; the synthetic project's rank-allocation conclusion determines why it is primary here.

## Frozen decision rule

`GREEN` means all of the following passed:

- global SVD is within 1.0 percentage point of exact credit **or** has at most 0.5% relative excess log loss on every matched audit cell;
- the selected rank's median gradient cosine is at least 0.95 in every benchmark/delay audit cell;
- exact credit uses at least three times as many per-event state bytes;
- paired 95% seed-level confidence intervals favor global SVD over the same fixed comparator—equal-byte replay or random projection—at two or more independent conditions.

The gate also requires at least 1,024 predictions after delayed feedback begins in every audit cell, Yearbook oracle balanced accuracy of at least 0.55, and a Criteo oracle relative log-loss improvement of at least 0.5% over a constant predictor. Criteo's delay bins remain descriptive because conversion status helps determine the bin; the complete Criteo test stream supplies one confirmatory condition.

`YELLOW` means the protocol is valid and at least one scientific threshold was missed. `RED` means a validity or integrity check failed, including an untrained oracle, failed download/hash/schema/split check, failed test, or interrupted execution. Smoke/verify mode can print green only as an execution check and sets `claim_evaluated` to false.

## Outputs

All reporting is local under `results/`:

- `final_status.json` and `gate.json`: color and ICLR continuation decision;
- `calibration.json`: frozen development-seed learning-rate selection;
- `REPORT.md`: short human-readable decision;
- `trial_summary.csv`: performance, credit quality, actual peak state bytes, peak RAM/VRAM, update latency, and throughput;
- `paired_intervals.csv`: paired seed-level confidence intervals;
- `trials/*.json`: full metrics, including Criteo delay strata and split-specific results;
- `frozen_protocol.json`, `provenance.json`, and dataset manifests: exact configuration and source hashes;
- `logs/` and `progress.jsonl`: console and structured local logs.

Peak RAM is sampled from the process RSS. Peak VRAM uses PyTorch's allocator counters. State bytes come from the actual retained tensor storages, including fixed sketch matrices and naïve replay memory; model parameters are reported separately. Update latency synchronizes CUDA around feedback processing. Throughput covers the complete prequential pass and feedback flush.

## Interpretation boundary

The Yearbook arm evaluates the method under controlled fixed label delays and the official loader's chronological training stream. The Criteo arm tests naturally variable positive feedback and explicit negative-label maturation. Criteo initialization is retrospective on the fixed historical training interval; natural reveal times govern validation and test updates. Both arms use the same low-rank credit object and binary adapter, which makes credit error comparable across domains. The frozen ResNet representation and fixed Criteo hashing make the experiment reproducible and isolate delayed credit; they do not establish state-of-the-art image or advertising prediction. A yellow or red scientific result remains reportable with its declared meaning and without post-hoc threshold changes.

## Source and dataset terms

- [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d8f5f134febb4bd74d8f79e338de382c-Abstract-Conference.html) and [authors' repository](https://github.com/botcs/label-delay-exp). The authors' repository is fetched at run time and is not redistributed in this archive.
- [Yearbook dating code/splits](https://github.com/katerakelly/yearbook-dating) carry the UC Berkeley educational/research/non-profit notice in their repository. Images download from the URL supplied by the NeurIPS authors and are not redistributed.
- [Criteo attribution dataset](https://huggingface.co/datasets/criteo/criteo-attribution-dataset) is CC BY-NC-SA 4.0 and is downloaded rather than redistributed. Confirm that your use is non-commercial and otherwise complies with its terms.

## Citation

Please cite the N08/O10 release using DOI `10.5281/zenodo.22217985`, the NeurIPS label-delay paper, the original Yearbook paper, and the Criteo attribution paper. Project citation metadata are in the repository-root `CITATION.cff`.
