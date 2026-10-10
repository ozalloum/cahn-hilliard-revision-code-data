# Cahn–Hilliard reproduction code and validation data

Public reproduction materials for **A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**, by Othman H. Y. Zalloum, Palestine Polytechnic University.

This repository is public to support review and reproducibility while the manuscript is under revision. It does not assert that the manuscript has been accepted. The previously submitted Mendeley draft was deleted before public publication; its reserved DOI must not be cited as an available dataset.

## Contents

- `numerics/submitted/`: original reproduction code, requirements and CSV results.
- `numerics/revision_baselines.py` and `numerics/revision_results/`: matched-baseline checks.
- `numerics/validation/`: expanded campaign, settings, seeds, protocols, CSV diagnostics and JSON summaries.
- **Release attachment `Cahn_Hilliard_code_and_validation_data.zip`**: the complete archive, including all model arrays, full fields and trajectory statistics. The 174 NPZ files are distributed in this attachment rather than Git history. Download the attachment from this repository's Releases page.
- `FILE_INVENTORY.json`: all 377 numerical payload files and their checksums.
- `ARCHIVE_SHA256_manifest.json`: checksums for the contents of the complete ZIP, not a manifest of this Git checkout.
- `PACKAGE_SHA256.txt`: SHA-256 checksum of the complete ZIP.

Journal correspondence, cover letters, reviewer responses and manuscript sources are not part of this code-and-data repository. The separate journal revision package remains available locally.

## Reproduction

For full reproduction, download and extract the complete release archive into a new directory. Work from that extracted directory so every required NPZ input is present. The Git checkout alone contains the readable source and tabular results but does not contain the full numerical arrays.

The recorded campaign environment is Python 3.12.14, NumPy 2.3.5 and pandas 3.0.1.

```sh
python -m pip install -r numerics/validation/requirements.txt
python numerics/validation/run_campaign.py --stage verify
python numerics/validation/run_campaign.py --stage all --output-dir results_reproduced
```

Use a new output directory for an independent full rerun. See `numerics/validation/REPRODUCIBILITY.txt` for dependencies, stage ordering and output names. The full campaign is a sustained CPU workload. Timings and floating-point roundoff depend on the machine and software environment.

## Provenance and limitations

The numerical campaign was run on 4–5 October 2026; the complete archive was packaged on 10 October 2026. Repository migration did not rerun the experiments or alter numerical scripts, arrays, CSVs or result JSON files.

The original fitted filter array was not retained. The supplied reconstruction follows the documented protocol and reproduces the archived short-time metrics to floating-point roundoff. It is a reconstruction, not the recovered original array. Original unpadded and dealiased Galerkin spatial formulations are distinguished.

The revised study reports conditional short-time benefits, loss of matched-baseline advantage at long times, failed cases, and comparisons unfavorable to the learned correction. It does not establish a general solver-speed advantage or a general stability theorem. The tested scope is two-dimensional and periodic.

Historical titles and conclusions in the original submitted snapshot are retained for provenance and must be read alongside the revised validation results.

## Access and citation

Code and data are shared under the Creative Commons Attribution 4.0 International license (CC BY 4.0), as approved by the author: https://creativecommons.org/licenses/by/4.0/. External software dependencies retain their own licenses. No DOI is claimed for this GitHub repository. Cite the precise Git commit or release version used for reproduction. Public access does not imply journal acceptance or independent validation of the findings.
