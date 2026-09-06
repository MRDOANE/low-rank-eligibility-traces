# GitHub and Zenodo v0.2.0 Release Instructions

These instructions update the existing GitHub repository and existing Zenodo project using graphical interfaces only. The release adds the complete v1.1 external-validation source and trial-level results. It does not redistribute the Yearbook images or Criteo records.

## Before changing GitHub

1. Download `low-rank-eligibility-traces-v0.2.0-github.zip` and extract it somewhere outside your current GitHub clone.
2. Open GitHub Desktop, choose the existing `MRDOANE/low-rank-eligibility-traces` repository, and select **Fetch origin**.
3. Make sure GitHub Desktop says there are no uncommitted changes. If there are changes, commit them or save a separate copy before continuing.
4. In GitHub Desktop, select **Repository → Show in Finder** on macOS or **Repository → Show in Explorer** on Windows.

## Replace the working files

1. Open the extracted `low-rank-eligibility-traces-v0.2.0` folder.
2. Select everything inside that folder, including `.github`, `.gitignore`, and `.zenodo.json`. Your file browser may require **Show hidden files** to display names beginning with a period.
3. Copy those items into the existing local clone shown by GitHub Desktop. Choose **Replace** for files with matching names and **Merge** for matching folders.
4. Do not copy the outer `low-rank-eligibility-traces-v0.2.0` folder into the clone. Do not delete or replace the clone's hidden `.git` folder.
5. Return to GitHub Desktop. Review the Changes list and spot-check `README.md`, `docs/NOVELTY_AUDIT.md`, `results/external_validation_v1.1`, and `scripts/verify_external_results.py`.

The repository has more than 100 files, so GitHub's browser upload is not appropriate for this update. GitHub Desktop preserves the existing history and handles the full change as one commit.

## Commit and publish v0.2.0

1. In GitHub Desktop, enter the summary **Add external validation and revise claim boundary**.
2. Select **Commit to main** and then **Push origin**.
3. Select **Repository → View on GitHub**.
4. Open the **Actions** tab and wait for the newest **Tests** workflow to pass.
5. On the repository page, open **Releases**, then **Draft a new release**.
6. Choose **Create new tag**, enter `v0.2.0`, and target `main`.
7. Use the release title **Low-Rank Delayed-Credit States v0.2.0**.
8. Paste these release notes:

   > Adds the valid two-benchmark external-validation cascade and its complete trial-level results. Rank-four global SVD passed every exact-credit fidelity and memory criterion, reducing per-event credit state by 6.3519x with median gradient cosine from 0.9978 to 0.9999. The preregistered competitive gate remained YELLOW because no paired confidence interval established the required advantage over equal-byte replay or random projection. This release narrows the claim accordingly and adds a dated novelty audit, external verifier, reproducibility documentation, and TMLR-oriented manuscript outline.

9. Leave **Set as a pre-release** unchecked and select **Publish release**.

The repository already contains all committed trial-level external results. If desired, add the original `n08_external_validation_cascade_v1.1.0_results.zip` as a release asset before publishing; its expected SHA-256 is `868ec7bb45a17b17f95ef26faa229b45cf7330f23f53dc2ac4dff07aff80c37b`. A release asset is useful for preserving the exact original archive, although Zenodo's GitHub integration ordinarily archives the tagged repository source rather than GitHub release assets.

## Archive the new release in Zenodo

The existing project DOI is `10.5281/zenodo.22217985`. First determine whether the GitHub repository is already connected to Zenodo.

### If GitHub integration is already enabled

1. Sign in to Zenodo.
2. Open the profile menu and select **GitHub**.
3. Select `low-rank-eligibility-traces`.
4. Wait for `v0.2.0` to finish processing. Zenodo ingests new GitHub releases automatically after a repository is enabled.
5. Open the DOI shown beside `v0.2.0` and verify the title, creator, ORCID, resource type **Software**, version `0.2.0`, access **Open**, and license **MIT**.
6. Open the record's file list and confirm that the archived source contains `external_validation`, `results/external_validation_v1.1`, and `docs/NOVELTY_AUDIT.md`.

### If GitHub integration is not enabled

1. In Zenodo, open the profile menu and select **GitHub**.
2. Select **Sync now**.
3. Find `low-rank-eligibility-traces` and turn on its integration switch.
4. Return to the repository entry. If `v0.2.0` does not appear automatically, select **Create release**, follow the link to the existing GitHub release, and wait for processing.
5. Open the resulting DOI and perform the metadata and file checks above.

### If the existing DOI was created by manual upload

Use this route only when `10.5281/zenodo.22217985` is not managed by the GitHub integration.

1. Open the existing Zenodo record.
2. Select **New version**. Do not edit or replace the published old version.
3. On GitHub, open the `v0.2.0` release and download **Source code (zip)**.
4. In the new Zenodo draft, remove any inherited old source archive and upload the `v0.2.0` source ZIP.
5. Set version to `0.2.0`; verify the title, creator, ORCID, description, software resource type, open access, and MIT license.
6. Publish the new version and record both DOI values Zenodo displays: the DOI for this exact version and the DOI labeled **Cite all versions**.

Zenodo normally assigns a new version-specific DOI to each release while retaining an all-versions concept DOI for the evolving project. Do not assume that `10.5281/zenodo.22217985` is one or the other: its Zenodo page will label it. Use the version DOI for an immutable artifact citation and the all-versions DOI for the repository as a continuing project.

## DOI metadata check

`CITATION.cff` and the README currently use `10.5281/zenodo.22217985`, the DOI supplied for this project. After Zenodo processes v0.2.0:

- If `10.5281/zenodo.22217985` is labeled **Cite all versions**, leave both files unchanged.
- If it is labeled as the old version only, update both files through GitHub's pencil-shaped **Edit this file** control to the new all-versions DOI, commit the two edits, and leave the already published v0.2.0 tag unchanged. The immutable release still points to the exact source that produced its DOI.
- Record the new version-specific DOI in the paper's artifact or reproducibility section.

## TMLR anonymity warning

The public GitHub and Zenodo records identify the author. Do not link either record in the anonymized TMLR manuscript or upload this author-identified ZIP as review supplementary material. TMLR requires both the PDF and any supplementary ZIP to be anonymized. Prepare a separate anonymous supplement with names, ORCID, GitHub URLs, DOI, and identifying metadata removed; the public repository can be added to the camera-ready version after acceptance.

## Official references

- [Pushing changes with GitHub Desktop](https://docs.github.com/en/desktop/making-changes-in-a-branch/pushing-changes-to-github-from-github-desktop)
- [Managing GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [Enabling a GitHub repository in Zenodo](https://help.zenodo.org/docs/github/enable-repository/)
- [Archiving a GitHub release in Zenodo](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [TMLR author guidelines](https://jmlr.org/tmlr/author-guide.html)
