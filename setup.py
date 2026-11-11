#!/usr/bin/env python3
"""Installe tout le framework gopu : compile les bibliotheques Rust (cargo) et installe les paquets Python.

    pip install .          # installation normale
    pip install -e .       # mode developpement (modifs Python prises en compte)
    python3 setup.py build # compilation seule

Prerequis : Rust (https://rustup.rs ; sous Termux : pkg install rust) et Python >= 3.8. Aucune dependance Python.
Ce qui est installe :
  - paquet `gopu_lm` (modele de langage)  + le paquet `gopu` (MLP, orchestrateur) s'il existe dans python/
  - la bibliotheque native libgopu (.so / .dylib / .dll), placee dans chaque paquet
  - le modele par defaut modele.b8n (Model.load_model() sans argument le charge)
  - la commande `gopu-lm` (CLI Rust : train / ask / info), embarquee dans le paquet gopu_lm
"""
import os
import shutil
import subprocess
import sys

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

ROOT = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.join(ROOT, "python")
LIBNAME = {"darwin": "libgopu.dylib", "win32": "gopu.dll"}.get(sys.platform, "libgopu.so")
EXE = ".exe" if sys.platform == "win32" else ""


def packages():
    return find_packages(where=PYDIR)


def build_native():
    """cargo build --release, puis copie de la lib dans chaque paquet Python et du modele dans gopu_lm."""
    if shutil.which("cargo") is None:
        sys.exit("Rust (cargo) est introuvable : installe-le depuis https://rustup.rs (Termux : pkg install rust).")
    print(">>> cargo build --release")
    subprocess.check_call(["cargo", "build", "--release"], cwd=ROOT)
    lib = os.path.join(ROOT, "target", "release", LIBNAME)
    if not os.path.exists(lib):
        sys.exit("Compilation terminee mais %s est introuvable." % lib)
    for pkg in packages():
        d = os.path.join(PYDIR, *pkg.split("."))
        shutil.copy2(lib, os.path.join(d, LIBNAME))
    model = os.path.join(ROOT, "modele.b8n")
    if os.path.exists(model) and "gopu_lm" in packages():
        shutil.copy2(model, os.path.join(PYDIR, "gopu_lm", "modele.b8n"))
    for name in ("gopu-lm", "airtel"):
        cli = os.path.join(ROOT, "target", "release", name + EXE)
        if os.path.exists(cli) and "gopu_lm" in packages():
            shutil.copy2(cli, os.path.join(PYDIR, "gopu_lm", name + EXE))


class BuildPy(build_py):
    def run(self):
        build_native()
        super().run()


class Develop(develop):
    def run(self):
        build_native()
        super().run()


class BinaryDistribution(Distribution):
    """Indique a setuptools que le paquet contient du code natif (roue specifique a la plateforme)."""

    def has_ext_modules(self):
        return True


setup(
    name="gopu",
    version="0.2.0",
    description="gopu : modele de langage (Transformer, BPE, attention, outils) et orchestrateur de sous-agents, en Rust pur",
    long_description=open(os.path.join(ROOT, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "python"},
    packages=packages(),
    package_data={"": [LIBNAME, "modele.b8n", "gopu-lm", "gopu-lm.exe", "airtel", "airtel.exe"]},
    include_package_data=True,
    python_requires=">=3.8",
    cmdclass={"build_py": BuildPy, "develop": Develop},
    distclass=BinaryDistribution,
    entry_points={"console_scripts": ["gopu-lm=gopu_lm.cli:main", "airtel=gopu_lm.airtel:main"]},
    zip_safe=False,
)
