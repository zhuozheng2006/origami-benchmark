#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/verify_release.py
test "$(uname -m)" = x86_64 || { echo 'Requires Linux x86_64 with AVX2/FMA/BMI2'; exit 1; }
test "$(uname -s)" = Linux
test ! -e external/codim-ipc-directional || { echo 'Existing native installation preserved; remove it explicitly to rebuild.'; exit 1; }
mkdir -p external/codim-ipc-directional
tar -xzf native/codim-directional-source.tar.gz -C external/codim-ipc-directional
cmake -S external/codim-ipc-directional -B external/codim-ipc-directional/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DLINEAR_SOLVER=EIGEN \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix)')" \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE
cmake --build external/codim-ipc-directional/build --target JGSL -j "${BUILD_JOBS:-2}"
