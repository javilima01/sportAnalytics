#!/usr/bin/env python3
"""Refresh label/image hashes in a dataset manifest after external label edits.

Usage:
  reconcile_manifest.py <dataset_dir> [<relative_label_path>]

With no label path, sweeps every manifest record. With a label path such as
labels/train/foo.txt, updates only that record. Writers serialize on
<dataset_dir>/.manifest.lock and the manifest is replaced atomically, so this is
safe to run from concurrent recheck workers.
"""

import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    target = sys.argv[2] if len(sys.argv) == 3 else None
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        print(f"reconcile: no manifest at {manifest}", file=sys.stderr)
        return 1

    with (root / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            records = [
                json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
            ]
            changed = 0
            found = target is None
            for record in records:
                if target is not None and record["label"] != target:
                    continue
                found = True
                for key, path in (
                    ("image_sha256", record["image"]),
                    ("label_sha256", record["label"]),
                ):
                    digest = file_hash(root / path)
                    if record[key] != digest:
                        record[key] = digest
                        changed += 1
            if not found:
                print(f"reconcile: no manifest record for {target}", file=sys.stderr)
                return 1
            if changed:
                tmp = manifest.with_name(manifest.name + ".tmp")
                tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
                os.replace(tmp, manifest)
                print(f"reconcile: updated {changed} hash(es) in {manifest.name}")
            else:
                print("reconcile: manifest already current")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
