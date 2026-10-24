# -*- coding: utf-8 -*-
"""算法依赖自检：python scripts/check_algo.py"""
import importlib

LIBS = [
    ("tsfresh", "时序特征自动提取"),
    ("pyod", "异常检测算法库"),
    ("anomalib", "工业缺陷检测（异常检测）"),
]


def main():
    ok = 0
    for name, desc in LIBS:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", "?")
            print(f"[OK]   {name:<10} v{ver:<12} {desc}")
            ok += 1
        except Exception as e:
            print(f"[MISS] {name:<10} {'':13} {desc}  -> {type(e).__name__}: {e}")
    print(f"\n可用 {ok}/{len(LIBS)}。缺失的请运行 scripts/install_algo.bat（Windows）")
    print("提示：即使未安装，项目其余功能仍可正常运行（适配层做了懒加载兜底）")


if __name__ == "__main__":
    main()
