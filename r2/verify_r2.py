"""Verify R2 publication files and archive consistency using the standard library."""
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

with ZipFile(root / "COMPHY_R2_Research_Materials.zip") as research, ZipFile(root / "COMPHY_R2_LaTeX_Source.zip") as source:
    for archive in (research, source):
        bad = archive.testzip()
        if bad:
            raise SystemExit("Archive integrity error: " + bad)
    for name in ("COMPHY_R2_clean.pdf", "COMPHY_R2_marked.pdf", "Supplementary_Material_S1_Access_Guide.pdf"):
        if research.read(name) != (root / name).read_bytes():
            raise SystemExit("Archive content mismatch: " + name)
    source_names = {n.removeprefix("source/") for n in research.namelist() if n.startswith("source/")}
    if source_names != set(source.namelist()):
        raise SystemExit("Source archive file lists differ")
    for name in source_names:
        if source.read(name) != research.read("source/" + name):
            raise SystemExit("Source archive content mismatch: " + name)

print(f"Verified {len(manifest)} R2 files and both research archives.")
