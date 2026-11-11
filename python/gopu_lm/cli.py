"""Commande `gopu-lm` : lance le binaire Rust embarque dans le paquet (train / ask / info)."""
import os
import subprocess
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    exe = os.path.join(here, "gopu-lm.exe" if sys.platform == "win32" else "gopu-lm")
    if not os.path.exists(exe):
        sys.exit("binaire gopu-lm introuvable dans %s (reinstalle avec : pip install .)" % here)
    sys.exit(subprocess.call([exe] + sys.argv[1:]))


if __name__ == "__main__":
    main()
