# Cahn–Hilliard reproduction code and validation data

Materials for **A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**.

## Download

Download **Cahn_Hilliard_code_and_validation_data.zip** from [release v1.0](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/tag/v1.0). It includes all reproduction code, settings, seeds, CSV results, model arrays, full fields, diagnostics, timing summaries and the eight final figures. `PACKAGE_SHA256.txt` provides the archive checksum.

The Git repository includes readable code, result tables and editable figure sources. The complete release ZIP also includes the 174 NPZ files and figure PDF/PNG exports.

## Reproduce

Extract the complete archive into a new directory and run from its root:

```sh
python -m pip install -r numerics/validation/requirements.txt
python verify_archive.py
python numerics/validation/run_campaign.py --stage verify --output-dir verification_results
python numerics/validation/figures.py
```

For an independent full campaign, run `python numerics/validation/run_campaign.py --stage all --output-dir results_reproduced`. Use a new output directory. See `numerics/validation/REPRODUCIBILITY.txt` for environment, stage dependencies and output details, and `figures/EXPORT_README.txt` for figure compilation.

The spatial coordinates, model time, mobility and interface-width parameter are dimensionless; wall-clock timings are in seconds.

The original fitted array was not retained. Its documented reconstruction reproduces the archived short-time metrics to roundoff. Original and dealiased spatial formulations are distinguished in the documentation.

## License and citation

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). External dependencies retain their own licenses. Cite the release version or Git commit used.
