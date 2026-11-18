"""Verify the supplementary guide and figure archive using the standard library."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
for name, expected in manifest.items():
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise SystemExit("Invalid manifest path: " + name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("Checksum mismatch: " + name)

with ZipFile(root / "COMPHY_R2_Research_Materials.zip") as archive:
    bad = archive.testzip()
    if bad:
        raise SystemExit("Archive integrity error: " + bad)
    guide = "Supplementary_Material_S1_Access_Guide.pdf"
    if archive.read(guide) != (root / guide).read_bytes():
        raise SystemExit("Archive content mismatch: " + guide)
    expected = {guide, "figures/EXPORT_README.txt", "figures/export_from_manuscript.py"}
    for number in range(1, 9):
        for extension in ("tex", "pdf", "png"):
            expected.add(f"figures/Figure_{number}.{extension}")
        expected.add(f"figures/TIFF_700dpi/Figure_{number}.tiff")
    if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
        raise SystemExit("Unexpected figure archive contents")

print(f"Verified {len(manifest)} published files and the figure archive.")
