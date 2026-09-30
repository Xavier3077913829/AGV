"""在 PyCharm 中同时启动 Django 和 Vite。

Django: http://127.0.0.1:8000
Vite:   http://127.0.0.1:5173
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    npm_command = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm_command:
        print("未找到 npm，请确认 Node.js 已加入 PATH。", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    spawned: list[subprocess.Popen] = []

    backend = None
    if port_in_use(8000):
        print("Django 已在 8000 端口运行，跳过启动。")
    else:
        backend = subprocess.Popen(
            [sys.executable, "manage.py", "runserver", "127.0.0.1:8000", "--noreload"],
            cwd=ROOT,
            env=env,
        )
        spawned.append(backend)

    frontend = None
    if port_in_use(5173):
        print("Vite 已在 5173 端口运行，跳过启动。")
    else:
        frontend = subprocess.Popen(
            [npm_command, "run", "dev", "--", "--host", "127.0.0.1"],
            cwd=FRONTEND,
            env=env,
            shell=False,
        )
        spawned.append(frontend)

    print("Django: http://127.0.0.1:8000")
    print("Vite:   http://127.0.0.1:5173")
    print("按 Ctrl+C 停止由本启动器创建的服务。")

    try:
        while True:
            running = []
            if backend is not None:
                running.append(backend.poll() is None)
            if frontend is not None:
                running.append(frontend.poll() is None)
            if running and not any(running):
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        for process in spawned:
            if process.poll() is None:
                process.terminate()
        for process in spawned:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
