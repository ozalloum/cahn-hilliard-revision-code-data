"""Verify the R2 code, data, artwork and supplementary archive."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import hashlib
import json
import sys


def main():
    root = Path(__file__).resolve().parent
    expected = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    failures = []
    for name, digest in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            failures.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(name)
    if failures:
        sys.exit("Missing or changed files: " + ", ".join(failures))
    prefix = "Supplementary_Material_S2/"
    wanted = {name for name in expected if name.startswith(prefix)}
    with ZipFile(root / "Supplementary_Material_S2.zip") as archive:
        names = archive.namelist()
        if archive.testzip() is not None or len(names) != len(set(names)) or set(names) != wanted:
            sys.exit("Supplementary archive contents do not match the manifest.")
        for name in names:
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                sys.exit("Invalid archive path.")
            if archive.read(name) != (root / name).read_bytes():
                sys.exit("Archive and extracted file differ: " + name)
    print(f"Verified {len(expected)} files and {len(wanted)} supplementary archive members.")


if __name__ == "__main__":
    main()
