# Cahn–Hilliard reproduction code and validation data

Materials for **A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**.

## Code and data

- **R2:** [Supplementary Material S2](r2/Supplementary_Material_S2/README.md) contains the step-conditioned correction experiment, frozen coefficients, results for 100 held-out fields at four step sizes, and Figure 9. [Download the ZIP](r2/Supplementary_Material_S2.zip?raw=true).
- **Original archive:** [Release v1.0](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/tag/v1.0) contains the complete original reproduction and validation data. Download `Cahn_Hilliard_code_and_validation_data.zip`; the automatically generated source archives omit the full data payload.
- **Artwork:** [Figure 2](r2/artwork/Figure_2.tiff?raw=true) contains the corrected discrete-energy axis label. The original figure sources remain in `figures/`.

The R2 supplement is separate from v1.0. The top-level `numerics/`, `figures/`, archive inventories and `README.txt` describe the original archive.

## Reproduction

After cloning or extracting the repository, verify the R2 files:

```sh
python r2/verify_r2.py
```

Follow the [S2 instructions](r2/Supplementary_Material_S2/README.md) to evaluate the frozen coefficients or refit the models. Use a new output directory for each run. For the original campaign, follow `numerics/validation/REPRODUCIBILITY.txt` in the complete v1.0 archive.

## License and citation

[CC BY 4.0](LICENSE.txt). External dependencies retain their own licenses. Cite the version and commit used; see [CITATION.cff](CITATION.cff).
