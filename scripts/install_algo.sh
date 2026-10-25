#!/usr/bin/env bash
# AGV 项目算法依赖安装（Linux / WSL）
set -e
python3 -m pip install --no-build-isolation -e ./vendor/tsfresh
python3 -m pip install --no-build-isolation -e ./vendor/pyod
python3 -m pip install -e ./vendor/anomalib
python3 scripts/check_algo.py
