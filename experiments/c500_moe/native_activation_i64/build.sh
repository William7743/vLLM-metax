#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"
export LD_LIBRARY_PATH="$MACA_PATH/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/opt/mxdriver/lib:${LIBRARY_PATH:-}"
TORCH_INCLUDE=$(/opt/conda/bin/python -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent / "include")')
"$CUCC_PATH/bin/cucc" -std=c++17 -arch=sm_80 -O2 -shared -Xcompiler -fPIC -DUSE_MACA -I"$CUCC_PATH/include" -I"$MACA_PATH/include" -I"$TORCH_INCLUDE" native_silu_i64.cu -o libnative_silu_i64.so
