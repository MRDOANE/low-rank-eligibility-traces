# GitHub and Zenodo Release Instructions

## 1. Create the GitHub repository

1. Sign in to GitHub as `MRDOANE`.
2. Select **New repository**.
3. Use the repository name `low-rank-eligibility-traces`.
4. Suggested description: `Low-rank eligibility traces for memory-efficient delayed credit assignment.`
5. Keep the repository **Private** until authorship, employer, manuscript, and license review are complete.
6. Do not initialize it with a README, `.gitignore`, or license. Those files are already in this package.
7. Create the empty repository.

## 2. Populate it with GitHub Desktop

1. In GitHub Desktop, select **File → Clone repository**.
2. Choose `MRDOANE/low-rank-eligibility-traces` and clone it.
3. Extract `low-rank-eligibility-traces-v0.1.0-github.zip` somewhere else on your computer.
4. Open the extracted `low-rank-eligibility-traces` directory.
5. Copy everything **inside** that directory into the cloned GitHub directory. Do not copy the outer wrapper directory.
6. GitHub Desktop should display the new files.
7. Use commit summary `Initial public research repository` and commit directly to `main`.
8. Select **Push origin**.
9. Open the repository on GitHub and confirm that the **Tests** workflow passes.

Suggested GitHub topics: `reinforcement-learning`, `eligibility-traces`, `credit-assignment`, `low-rank`, `memory-compression`, `pytorch`, `reproducibility`.

## 3. Review before release

Check the rendered README, license, `CITATION.cff`, `.zenodo.json`, and Actions status. Keep the large result archives out of the commit history. Their hashes are in `results/PROVENANCE.json`.

When the repository is ready to become public, change its visibility under **Settings → General → Danger Zone → Change repository visibility**.

## 4. Connect the repository to Zenodo

Do this before publishing the first GitHub release:

1. Sign in to Zenodo and link the same GitHub account if it is not already linked.
2. Open the Zenodo profile menu and select **GitHub**.
3. Select **Sync now**.
4. Find `low-rank-eligibility-traces` and enable its integration switch.
5. Refresh and verify that Zenodo shows the repository as enabled.

Zenodo will use `.zenodo.json` for the archived software metadata. When both `.zenodo.json` and `CITATION.cff` exist, Zenodo gives `.zenodo.json` priority.

## 5. Create GitHub release v0.1.0

1. On GitHub, open **Releases → Draft a new release**.
2. Choose or create tag `v0.1.0` targeting `main`.
3. Use release title `Low-Rank Eligibility Traces v0.1.0`.
4. Suggested release text:

   `Initial public research release containing the focused implementation, frozen N08/O10/stacked protocols, tests, compact result summaries, and reproducibility documentation.`

5. Optionally attach the three complete result archives as GitHub release assets:
   - `n08_pareto_v20_results.zip`
   - `o10_horizon_v20_results.zip`
   - `n08_o10_stacked_v11_results.zip`
6. Publish the release.

GitHub release assets are preferable to committing large binary archives. The combined v1.1 archive is close to GitHub's normal 100 MiB per-file repository limit and should never enter Git history.

## 6. Confirm the Zenodo software archive

1. Return to **Zenodo → GitHub**.
2. Open the enabled repository and wait for `v0.1.0` to finish processing.
3. Open the Zenodo record and verify:
   - Resource type: **Software**
   - Title: **Low-Rank Eligibility Traces for Delayed Credit Assignment**
   - Creator: **Michael R. Doane**
   - ORCID: **0009-0003-0521-8981**
   - Version: **0.1.0**
   - Access: **Open**
   - License: **MIT**
4. Record the version-specific DOI assigned to v0.1.0.

Zenodo's GitHub integration archives the tagged repository source. Inspect the record's file list rather than assuming GitHub release assets were copied.

## 7. Optional full-results Zenodo record

If the complete checkpoint archives should be preserved permanently, create a separate Zenodo upload with resource type **Dataset** and title `Full Results for Low-Rank Eligibility Traces for Delayed Credit Assignment, v0.1.0`. Upload the three result archives listed above, use the same creator and ORCID, select open access, and relate it to the software DOI with an `isSupplementTo` relationship. This keeps large checkpoints out of Git while giving them their own immutable DOI.

## 8. Add the DOI to GitHub

After Zenodo supplies the software DOI:

1. Add the following top-level entry to `CITATION.cff`:

   ```yaml
   doi: "10.5281/zenodo.REPLACE_ME"
   ```

2. Add this badge near the top of `README.md`, replacing the placeholder:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.REPLACE_ME.svg)](https://doi.org/10.5281/zenodo.REPLACE_ME)
   ```

3. Commit with summary `Add Zenodo DOI` and push.
4. Use the version-specific DOI when citing the exact v0.1.0 artifact. Use Zenodo's all-versions DOI when referring to the evolving project as a whole.

## Official references

- [Creating a GitHub repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [Adding local code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [Managing GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [Enabling a GitHub repository in Zenodo](https://help.zenodo.org/docs/github/enable-repository/)
- [Archiving a GitHub release in Zenodo](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [Zenodo `.zenodo.json` metadata](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)
