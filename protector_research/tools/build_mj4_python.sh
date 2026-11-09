#!/bin/bash
# Build MuJoCo 4.0 (this fork) and its Python bindings into protector_research/.venv4 for M3 (flex) experiments.
# SKIP_CMAKE=1 reuses an existing build/install of the C library.
set -euo pipefail
# CMAKE_PREFIX_PATH from the ROS workspace points at MuJoCo 3.2.3 in /opt; it must not leak into this build.
unset CMAKE_PREFIX_PATH
ROOT=/home/wang22/engineai/mujoco
PR=$ROOT/protector_research
PREFIX=$PR/.mj4install
if [[ "${SKIP_CMAKE:-0}" != 1 ]]; then
  cmake -S $ROOT -B $ROOT/build -DCMAKE_BUILD_TYPE=Release -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX
  cmake --build $ROOT/build -j 24
  cmake --install $ROOT/build
fi
[[ -x $PR/.venv4/bin/python ]] || python3 -m venv $PR/.venv4
$PR/.venv4/bin/pip install -q --upgrade pip
# make_sdist_requirements.txt uses --require-hashes, which fails on Python 3.10 because `build`
# pulls an unpinned tomli; install the same pins without hashes.
REQ=$PR/runs/make_sdist_requirements_nohash.txt
grep -v -e '--hash' -e '^[[:space:]]*#' $ROOT/python/make_sdist_requirements.txt \
  | sed 's/[[:space:]]*\\$//' | grep -v '^[[:space:]]*$' > $REQ
$PR/.venv4/bin/pip install -q -r $REQ
cd $ROOT/python
PATH=$PR/.venv4/bin:$PATH bash make_sdist.sh
SDIST=$(ls -t dist/mujoco_core-*.tar.gz | head -1)
MUJOCO_PATH=$PREFIX MUJOCO_PLUGIN_PATH=$PREFIX/lib/mujoco_plugin $PR/.venv4/bin/pip install "$SDIST"
$PR/.venv4/bin/pip install -q numpy scipy pandas pyyaml matplotlib
$PR/.venv4/bin/python -c "import mujoco; print('mujoco', mujoco.__version__)"
