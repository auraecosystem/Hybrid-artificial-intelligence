#!/usr/bin/env python3
"""Portable, opt-in CMake build driver for KUBU-HAI."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def detect_capabilities() -> dict[str, bool]:
    system = platform.system()
    return {
        "cmake": shutil.which("cmake") is not None,
        "ninja": shutil.which("ninja") is not None,
        "docker": shutil.which("docker") is not None,
        "cuda": shutil.which("nvidia-smi") is not None,
        "metal": system == "Darwin",
        "linux": system == "Linux",
        "macos": system == "Darwin",
    }


def resolve_backend(requested: str, caps: dict[str, bool]) -> str:
    if requested == "auto":
        # CPU remains the safe, deterministic default. GPU is opt-in.
        return "cpu"
    if requested == "cuda" and not caps["cuda"]:
        raise RuntimeError("CUDA requested, but nvidia-smi was not found. Install the NVIDIA driver/toolkit.")
    if requested == "metal" and not caps["metal"]:
        raise RuntimeError("Metal requested on a non-macOS host. Run a native macOS build.")
    return requested


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--build-dir", type=Path, default=Path(".build/kubu-hai"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "metal"), default="auto")
    parser.add_argument("--mode", choices=("native", "docker"), default="native")
    parser.add_argument("--build-type", choices=("Debug", "Release", "RelWithDebInfo"), default="Release")
    parser.add_argument("--target", help="Optional CMake build target")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--image", default="kubu-hai-build:local")
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    source = args.source.resolve()
    build_dir = args.build_dir.resolve()
    if not source.is_dir():
        parser.error(f"source directory does not exist: {source}")
    if not (source / "CMakeLists.txt").is_file():
        parser.error(f"no CMakeLists.txt found in source: {source}")

    caps = detect_capabilities()
    print("Host capabilities:", ", ".join(f"{k}={v}" for k, v in caps.items()))
    try:
        backend = resolve_backend(args.backend, caps)
        if args.mode == "docker":
            if not caps["docker"]:
                raise RuntimeError("Docker mode requested, but docker is not in PATH.")
            if backend == "metal":
                raise RuntimeError("Metal builds must run natively on macOS; Docker mode is unsupported.")
            dockerfile = Path(__file__).with_name("Dockerfile")
            command = [
                "docker", "build", "-f", str(dockerfile), "-t", args.image, str(dockerfile.parent)
            ]
            run(command)
            docker_run = [
                "docker", "run", "--rm",
                "--mount", f"type=bind,src={source},dst=/workspace",
                "--workdir", "/workspace",
                "--env", f"KUBU_HAI_BACKEND={backend}",
                "--env", f"KUBU_HAI_BUILD_TYPE={args.build_type}",
                "--mount", f"type=bind,src={build_dir},dst=/build",
            ]
            if backend == "cuda":
                if not caps["linux"]:
                    raise RuntimeError("CUDA Docker builds require a Linux host with NVIDIA Container Toolkit.")
                docker_run += ["--gpus", "all"]
            docker_run += [args.image, "cmake", "-S", "/workspace", "-B", "/build",
                           "-G", "Ninja", f"-DCMAKE_BUILD_TYPE={args.build_type}",
                           f"-DKUBU_HAI_BACKEND={backend}"]
            run(docker_run)
            build_command = ["docker", "run", "--rm",
                             "--mount", f"type=bind,src={source},dst=/workspace",
                             "--mount", f"type=bind,src={build_dir},dst=/build",
                             "--workdir", "/workspace"]
            if backend == "cuda":
                build_command += ["--gpus", "all"]
            build_command += [args.image, "cmake", "--build", "/build", "--parallel", str(args.jobs)]
            if args.target:
                build_command += ["--target", args.target]
            run(build_command)
        else:
            if not caps["cmake"]:
                raise RuntimeError("CMake was not found in PATH.")
            build_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["KUBU_HAI_BACKEND"] = backend
            env["KUBU_HAI_BUILD_TYPE"] = args.build_type
            configure = ["cmake", "-S", str(source), "-B", str(build_dir),
                         f"-DCMAKE_BUILD_TYPE={args.build_type}",
                         f"-DKUBU_HAI_BACKEND={backend}"]
            if caps["ninja"]:
                configure += ["-G", "Ninja"]
            run(configure, env=env)
            build = ["cmake", "--build", str(build_dir), "--parallel", str(args.jobs)]
            if args.target:
                build += ["--target", args.target]
            run(build, env=env)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Build completed (backend={backend}, mode={args.mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
