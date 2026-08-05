"""
fetch_weights.py
================
Downloads the pretrained CRAFT model weights required by the compression
pipeline into ./pretrained/.

The weights are not stored in the repository (pretrained/ is gitignored —
craft_mlt_25k.pth alone is ~79 MB), so a fresh clone has to fetch them once
before anything that touches text detection will run:

    python fetch_weights.py

Files fetched:
    pretrained/craft_mlt_25k.pth          — CRAFT detector      (~79 MB)
    pretrained/craft_refiner_CTW1500.pth  — RefineNet refiner   (~1.8 MB)

Both are verified against a pinned SHA-256 after download; a file that fails
verification is deleted rather than left in place to fail confusingly later.
Existing files that already match are left alone, so re-running is cheap and
safe. Use --force to re-download regardless.

Usage:
    python fetch_weights.py
    python fetch_weights.py --force
    python fetch_weights.py --outdir ./pretrained
"""

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


# ── Weight sources ────────────────────────────────────────────────────────────
# Mirrored on Hugging Face; both files are byte-identical to the originals
# released by the CRAFT authors (clovaai/CRAFT-pytorch).
WEIGHTS = {
    "craft_mlt_25k.pth": {
        "url": "https://huggingface.co/boomb0om/CRAFT-text-detector/resolve/main/craft_mlt_25k.pth",
        "sha256": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
        "desc": "CRAFT detector",
    },
    "craft_refiner_CTW1500.pth": {
        "url": "https://huggingface.co/boomb0om/CRAFT-text-detector/resolve/main/craft_refiner_CTW1500.pth",
        "sha256": "f7000cd3e9c76f2231b62b32182212203f73c08dfaa12bb16ffb529948a01399",
        "desc": "RefineNet refiner",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download pretrained CRAFT weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outdir", default="./pretrained",
        help="Destination directory for the weights (default: ./pretrained)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if a valid file is already present"
    )
    return parser.parse_args()


def sha256_of(path):
    """Stream a file through SHA-256 so large weights don't land in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest):
    """Download url to dest, showing progress. Returns True on success."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 / total
                        print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB "
                              f"({pct:5.1f}%)", end="", flush=True)
                    else:
                        print(f"\r    {done / 1e6:6.1f} MB", end="", flush=True)
        print()
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"\n    [ERROR] Download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def fetch(name, spec, outdir, force):
    """Fetch and verify a single weight file. Returns True if it ends up valid."""
    dest = outdir / name
    print(f"\n  {name}  ({spec['desc']})")

    if dest.exists() and not force:
        if sha256_of(dest) == spec["sha256"]:
            print("    Already present and verified — skipping.")
            return True
        print("    Present but checksum does not match — re-downloading.")

    if not download(spec["url"], dest):
        return False

    actual = sha256_of(dest)
    if actual != spec["sha256"]:
        print("    [ERROR] Checksum mismatch — deleting corrupt download.")
        print(f"      expected {spec['sha256']}")
        print(f"      actual   {actual}")
        dest.unlink(missing_ok=True)
        return False

    print(f"    Verified ({dest.stat().st_size / 1e6:.1f} MB).")
    return True


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FETCHING PRETRAINED CRAFT WEIGHTS")
    print("=" * 70)
    print(f"  Destination: {outdir.resolve()}")

    failed = [name for name, spec in WEIGHTS.items()
              if not fetch(name, spec, outdir, args.force)]

    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("=" * 70)
        sys.exit(1)
    print("All weights present and verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
