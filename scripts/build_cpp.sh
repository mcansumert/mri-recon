#!/bin/bash
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends g++
pip install --no-cache-dir --break-system-packages pybind11
suffix=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
c++ -O3 -shared -std=c++17 -fPIC $(python -m pybind11 --includes) cpp/sense_kernel.cpp -o "cpp/sense_kernel${suffix}"
PYTHONPATH=/work/cpp python scripts/run_cpp_check.py
