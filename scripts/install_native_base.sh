#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/verify_release.py
test "$(uname -s)" = Linux && test "$(uname -m)" = x86_64
test ! -e external/codim-ipc || { echo 'Existing base backend preserved'; exit 1; }
mkdir -p external/codim-ipc
tar -xzf native/codim-base-source.tar.gz -C external/codim-ipc
# Use bundled, pinned dependencies; never fetch moving upstream branches.
python - <<'PY'
from pathlib import Path
import re
p=Path('external/codim-ipc/Externals/CMakeLists.txt');s=p.read_text()
for name in ['amgcl','Kokkos','Cabana','Kokkos_kernels']:
 pattern=r'download_project\(\s*PROJ '+name+r'\s+.*?^\s*\)'
 repl='set('+name+'_SOURCE_DIR "${CMAKE_BINARY_DIR}/'+name+'-src")\nset('+name+'_BINARY_DIR "${CMAKE_BINARY_DIR}/'+name+'-build")'
 s,n=re.subn(pattern,repl,s,flags=re.S|re.M)
 if n!=1 and 'set('+name+'_SOURCE_DIR' not in s:raise RuntimeError('Unexpected dependency recipe '+name)
p.write_text(s)
PY
cmake -S external/codim-ipc -B external/codim-ipc/build -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DLINEAR_SOLVER=EIGEN \
 -DPYTHON_EXECUTABLE="$(command -v python)" -DCMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix + chr(59) + sys.base_prefix)')" \
 -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE
cmake --build external/codim-ipc/build --target JGSL -j "${BUILD_JOBS:-2}"
