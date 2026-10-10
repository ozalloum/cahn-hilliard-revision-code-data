# Cahn–Hilliard reproduction code and validation data

Materials for **A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**.

## Download

Download `Cahn_Hilliard_code_and_validation_data.zip` from [Releases](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases). It contains the complete code, settings, seeds, CSV results, model arrays, full fields, diagnostics and timing summaries. The accompanying `PACKAGE_SHA256.txt` provides its checksum.

## Reproduce

Extract the complete archive, then run from its root:

```sh
python -m pip install -r numerics/validation/requirements.txt
python numerics/validation/run_campaign.py --stage verify
python numerics/validation/run_campaign.py --stage all --output-dir results_reproduced
```

See `numerics/validation/REPRODUCIBILITY.txt` for the environment, stage ordering and output details. Use a new output directory for a full rerun.

The original fitted array was not retained. The supplied reconstruction reproduces the archived short-time metrics to roundoff. Original and dealiased spatial formulations are distinguished in the documentation.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). External dependencies retain their own licenses. Cite the release version or Git commit used.
