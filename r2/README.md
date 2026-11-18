# Figures and supplementary material

**A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**

**Author:** Othman H. Y. Zalloum  
**Affiliation:** Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

## Files

- [Supplementary Material S1](Supplementary_Material_S1_Access_Guide.pdf)
- [Figures and supplementary guide](COMPHY_R2_Research_Materials.zip?raw=true)
- [SHA-256 checksums](SHA256SUMS.json)

The ZIP contains S1 and eight figure sources with PDF, PNG and TIFF exports. Figure export instructions are in `figures/EXPORT_README.txt`.

## Numerical reproduction

The numerical code and complete validation data are available in [release v1.0](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/tag/v1.0). Download `Cahn_Hilliard_code_and_validation_data.zip` and follow S1 and the archive's reproduction instructions.

## Integrity

From the repository root, run:

```sh
python r2/verify_r2.py
```

This verifies the published checksums, ZIP integrity and agreement between the supplementary guide and its archived copy. It does not run numerical simulations.
