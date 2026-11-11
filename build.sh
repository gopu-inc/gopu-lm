#!/bin/sh
# Compile la bibliotheque native + le demo, lance les tests, copie la lib a cote du paquet Python.
set -e
cd "$(dirname "$0")"
command -v cargo >/dev/null 2>&1 || { echo "Rust absent : https://rustup.rs  (ou: pkg install rust sous Termux)"; exit 1; }
cargo test --release
cargo build --release
case "$(uname -s)" in
  Darwin) LIB=libgopu.dylib ;;
  MINGW*|MSYS*|CYGWIN*) LIB=gopu.dll ;;
  *) LIB=libgopu.so ;;
esac
cp "target/release/$LIB" python/gopu/ 2>/dev/null || true
cp "target/release/$LIB" python/gopu_lm/ 2>/dev/null || true
echo "--- demo Rust ---"
./target/release/gopu-demo
if [ -f modele.b8n ] && command -v python3 >/dev/null 2>&1; then
  echo "--- test gopu-lm (Python) ---"
  python3 python/test_lm.py modele.b8n
fi
if command -v python3 >/dev/null 2>&1; then
  echo "--- test Python ---"
  python3 python/test_gopu.py
fi
