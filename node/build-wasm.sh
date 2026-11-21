#!/bin/sh
# Compiles core.c to core.wasm (committed, so deploying needs no compiler). Needs clang and wasm-ld.
set -eu
cd "$(dirname "$0")"
clang --target=wasm32 -O3 -nostdlib -fno-builtin -mbulk-memory \
  -Wl,--no-entry -Wl,--export=__heap_base -Wl,--strip-all -Wl,-z,stack-size=65536 \
  -o core.wasm core.c
echo "core.wasm: $(wc -c < core.wasm) bytes"
