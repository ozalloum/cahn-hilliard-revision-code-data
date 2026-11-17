# Cahn–Hilliard reproduction code and validation data

Materials for **A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**.

**Author:** Othman H. Y. Zalloum  
**Affiliation:** Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

## Manuscript and supporting materials

- [Clean R2 manuscript](r2/COMPHY_R2_clean.pdf)
- [Marked R2 manuscript](r2/COMPHY_R2_marked.pdf)
- [Supplementary Material S1: reproduction guide](r2/Supplementary_Material_S1_Access_Guide.pdf)
- [LaTeX sources and figure exports](r2/COMPHY_R2_LaTeX_Source.zip?raw=true)
- [Download all R2 research materials](r2/COMPHY_R2_Research_Materials.zip?raw=true)

The [R2 directory](r2/README.md) provides the current manuscript files and integrity checks.

## Numerical code and validation data

The manuscript uses the numerical code and complete validation data in [release v1.0](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/tag/v1.0).

Download **[Cahn_Hilliard_code_and_validation_data.zip](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/download/v1.0/Cahn_Hilliard_code_and_validation_data.zip)**, approximately 306 MB. GitHub's automatically generated source archives omit the full validation-data payload.

The top-level `numerics/`, `figures/`, archive inventories and `README.txt` document that numerical archive. Its code and results are unchanged.

## Reproduction and integrity

Follow [Supplementary Material S1](r2/Supplementary_Material_S1_Access_Guide.pdf) and `numerics/validation/REPRODUCIBILITY.txt` in the complete numerical archive. Use a separate output directory for an independent campaign.

After cloning or extracting this repository, verify the R2 materials:

```sh
python r2/verify_r2.py
```

## License and citation

[CC BY 4.0](LICENSE.txt). External dependencies retain their own licenses. Cite the version and commit used; see [CITATION.cff](CITATION.cff).
