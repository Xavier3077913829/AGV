@echo off
REM AGV 项目算法依赖安装（Windows）
setlocal
echo [1/2] 安装 tsfresh / pyod（从 vendor 源码目录）
python -m pip install --no-build-isolation -e .\vendor\tsfresh
python -m pip install --no-build-isolation -e .\vendor\pyod
echo [2/2] 安装 anomalib（依赖较多，耗时较长）
python -m pip install -e .\vendor\anomalib
echo 完成。可用 python scripts\check_algo.py 自检
endlocal
