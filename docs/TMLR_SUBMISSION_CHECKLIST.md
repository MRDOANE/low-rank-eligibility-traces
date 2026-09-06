# TMLR Submission Checklist

This checklist is for the review package. The public GitHub/Zenodo release remains the author-identified archival record.

## Paper framing

- Title the study around delayed-credit state, fidelity, memory, and competitive limits.
- Define the retained object as a per-observation, label-free logit Jacobian fixed before feedback arrives.
- Present global truncated SVD as the strongest tested N08 state, not as a newly invented matrix factorization.
- State the result in two parts: rank-four global SVD is highly faithful and memory-efficient; it does not establish a predictive advantage over equal-byte replay or random projection.
- Explain the operational lesson: gradient reconstruction fidelity and downstream learner utility require separate measurements.
- Cite the closest delayed-feedback, RTRL, eligibility-trace, gradient-compression, sketching, and conversion-delay literature listed in `docs/NOVELTY_AUDIT.md`.

## Anonymization

- Remove author names, affiliations, acknowledgements, ORCID, DOI, and direct GitHub/Zenodo links from the review PDF.
- Refer to the controlled studies and external package without self-identifying phrases.
- Create a separate anonymous supplementary ZIP; do not submit the public GitHub ZIP.
- Remove `CITATION.cff`, `.zenodo.json`, author metadata, badges, and author-specific URLs from that supplement.
- Search the supplement for the author's name, username, ORCID, DOI, absolute workstation paths, and metadata embedded in generated files.

## Evidence and reporting

- Report all four preregistered gate criteria, including the failed competitive criterion.
- Report five-seed paired confidence intervals rather than seed-level significance tests treated as independent trials.
- Distinguish the exact-credit audit subset from the full predictive comparisons.
- Report actual state bytes, peak RAM/VRAM, update latency, throughput, and delay-stratified outcomes.
- Describe the Yearbook adapter scope and avoid implying replication of the original paper's full-network method.
- State that Criteo model features exclude conversion, attribution, conversion time, and other outcome-derived fields.
- Include the calibration protocol, maturity rule for negative Criteo labels, chronological splits, and dataset revisions.
- Discuss the unfavorable throughput result and the limits of the frozen feature/adapter setting.

## Files and format

- Use the official TMLR LaTeX style and submit an anonymized PDF.
- Keep supplementary material at or below TMLR's 100 MB limit and in PDF or ZIP format.
- Include a concise reproducibility statement and environment details.
- Include a broader-impact statement if the final framing raises a significant risk of harm.
- Ensure every table and claim can be traced to the committed trial summaries or verifier scripts.

Official requirements: [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html) and [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html).
