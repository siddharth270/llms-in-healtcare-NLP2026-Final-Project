"""
Downloads the HealthBench data files into data/raw/.

The four files are public and need no credentials. Checksums are written to
data/manifest.json so the whole team can confirm they trained on the same bytes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from tqdm import tqdm

BASE = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench"

# local name -> remote name
FILES = {
    "meta_eval.jsonl": "2025-05-07-06-14-12_oss_meta_eval.jsonl",
    "eval.jsonl": "2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus.jsonl": "consensus_2025-05-09-20-00-46.jsonl",
    "hard.jsonl": "hard_2025-05-08-21-00-10.jsonl",
}

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
MANIFEST = DATA_DIR / "manifest.json"
CHUNK = 1 << 20


def download(local_name: str, remote_name: str, force: bool) -> dict:
    dest = RAW_DIR / local_name
    url = f"{BASE}/{remote_name}"

    if dest.exists() and not force:
        print(f"{local_name}: already present, hashing")
        digest = hashlib.sha256()
        with dest.open("rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                digest.update(chunk)
        return {
            "remote": remote_name,
            "bytes": dest.stat().st_size,
            "sha256": digest.hexdigest(),
        }

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        digest = hashlib.sha256()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=local_name
        ) as bar:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)

    return {
        "remote": remote_name,
        "bytes": dest.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--only", nargs="*", choices=sorted(FILES), help="subset to fetch")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = args.only or list(FILES)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for local_name in wanted:
        manifest[local_name] = download(local_name, FILES[local_name], args.force)

    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {MANIFEST.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
