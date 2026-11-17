# R2 manuscript and supporting materials

**A Spectral-Residual Correction for Cahn–Hilliard Simulation: Accuracy, Safeguards and Computational Cost**

**Author:** Othman H. Y. Zalloum  
**Affiliation:** Department of Applied Mathematics and Physics, Palestine Polytechnic University, Hebron, Palestine

## Files

- [Clean manuscript](COMPHY_R2_clean.pdf)
- [Marked manuscript](COMPHY_R2_marked.pdf)
- [Supplementary Material S1](Supplementary_Material_S1_Access_Guide.pdf)
- [LaTeX sources and figure exports](COMPHY_R2_LaTeX_Source.zip?raw=true)
- [All research materials](COMPHY_R2_Research_Materials.zip?raw=true)
- [SHA-256 checksums](SHA256SUMS.json)

The research-materials ZIP contains both manuscript PDFs, S1, and the complete LaTeX source folder with eight figure sources and PDF, PNG and TIFF exports. The source-only ZIP contains the same source folder. Build instructions are in `BUILD_README.txt` inside the source archive.

## Numerical reproduction

The numerical code and complete validation data are available in [release v1.0](https://github.com/ozalloum/cahn-hilliard-revision-code-data/releases/tag/v1.0). Download the named `Cahn_Hilliard_code_and_validation_data.zip` asset and follow S1 and the archive's reproduction instructions. The code and numerical results are unchanged.

## Integrity

From the repository root, run:

```sh
python r2/verify_r2.py
```

This verifies the published file checksums, ZIP integrity, and agreement between the manuscript downloads and archive contents. It does not run numerical simulations.
