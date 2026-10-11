"""Verify every file listed in the complete archive's SHA-256 manifest."""
from pathlib import Path
import hashlib, json, sys
root=Path(__file__).resolve().parent
manifest=root/"SHA256_manifest.json"
if not manifest.exists():
    sys.exit("Extract the complete release ZIP before running this check.")
expected=json.loads(manifest.read_text(encoding="utf-8"))
failed=[]
for name,digest in expected.items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        failed.append(name); continue
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    if h.hexdigest()!=digest:failed.append(name)
if failed:
    sys.exit("Missing or changed files: " + ", ".join(failed))
print(f"Verified {len(expected)} files: all SHA-256 hashes match.")
