"""Fetch the Franka Emika Panda models (both with and without gripper) 
from MuJoCo Menagerie.

Downloads `panda_nohand.xml`, `panda.xml`, and the matching mesh assets into
`src/mjlab_franka/robots/franka/xmls/`. Safe to re-run (idempotent).

Usage:
    uv run python scripts/fetch_franka.py
"""

from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
from pathlib import Path

MENAGERIE_REF = "main"
TARBALL_URL = (
    f"https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/"
    f"{MENAGERIE_REF}.tar.gz"
)
MENAGERIE_SUBDIR = f"mujoco_menagerie-{MENAGERIE_REF}/franka_emika_panda/"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "src" / "mjlab_franka" / "robots" / "franka" / "xmls"

# Target both XML configurations
WANTED_XMLS = {"panda_nohand.xml", "panda.xml"}


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "assets").mkdir(exist_ok=True)

    # Check if all wanted XMLs and assets are already present
    markers_exist = all((DEST / xml).exists() for xml in WANTED_XMLS)
    if markers_exist and any((DEST / "assets").iterdir()):
        print(f"Franka assets and XMLs already present in {DEST}; skipping download.")
        return 0

    print(f"Downloading {TARBALL_URL} ...")
    with urllib.request.urlopen(TARBALL_URL) as resp:  # noqa: S310 (trusted URL)
        buf = io.BytesIO(resp.read())

    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        extracted = 0
        for member in tar.getmembers():
            if not member.name.startswith(MENAGERIE_SUBDIR):
                continue
            rel = member.name[len(MENAGERIE_SUBDIR) :]
            if not rel:
                continue
            
            # Keep both specified XML files and everything under assets/
            if rel in WANTED_XMLS or rel.startswith("assets/"):
                target = DEST / rel
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tar.extractfile(member)
                if f is None:
                    continue
                target.write_bytes(f.read())
                extracted += 1
                
    print(f"Wrote {extracted} files to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())