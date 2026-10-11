CAHN-HILLIARD REPRODUCTION CODE AND VALIDATION DATA — v1.0

A Spectral-Residual Correction for Cahn-Hilliard Simulation:
Accuracy, Safeguards and Computational Cost

Extract this complete archive into a new directory. It includes all code,
settings, seeds, CSV results, model arrays, full fields, diagnostics, timing
summaries and the eight final figures.

From the extracted root:
  python -m pip install -r numerics/validation/requirements.txt
  python verify_archive.py
  python numerics/validation/run_campaign.py --stage verify --output-dir verification_results
  python numerics/validation/figures.py

For an independent full campaign, use a new output directory:
  python numerics/validation/run_campaign.py --stage all --output-dir results_reproduced

Read numerics/validation/REPRODUCIBILITY.txt for dependencies and stage details.
Read figures/EXPORT_README.txt for editable TeX sources and PDF/PNG exports.
Spatial coordinates, model time, mobility and interface width are
dimensionless. Wall-clock timings are in seconds.

The original fitted array was not retained. The supplied reconstruction
reproduces archived short-time metrics to roundoff. Original and dealiased
spatial formulations remain distinguished. All saved validation results
and numerical solver implementations are preserved.

SHA256_manifest.json records every other file in this archive.
FILE_INVENTORY.json lists numerical payload files, sizes and SHA-256 hashes.

License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
External dependencies retain their own licenses. Cite the release version:
https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/tag/v1.0
