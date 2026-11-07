# KUBU-HAI Hybrid Build Foundation

A repository-local, opt-in build entry point for KUBU-HAI. It does not replace or modify existing project build files. It provides a portable CMake/Ninja build driver, capability probes, an isolated Linux toolchain image, and unit tests.

## Requirements

- Python 3.10+
- CMake 3.24+
- Ninja (recommended)
- A C/C++ compiler
- Docker only for container builds

## Native build

From the repository root:

```bash
python3 kubu-hai-build/build.py --source . --build-dir .build/kubu-hai --backend auto
```

Choose `cpu`, `cuda`, or `metal` explicitly with `--backend`. The driver reports detected capabilities; selecting CUDA or Metal does not magically add GPU kernels to a project. The source CMake project must consume `KUBU_HAI_BACKEND` and implement the corresponding backend.

Metal is a native macOS build. Linux containers cannot provide native Apple Metal access.

## Container build

```bash
python3 kubu-hai-build/build.py --source . --mode docker --backend cpu
```

CUDA container builds require a Linux host, NVIDIA driver, NVIDIA Container Toolkit, and a Docker runtime configured for `--gpus all`. The container image is a build environment, not a GPU runtime for macOS.

## Tests

```bash
python3 -m unittest discover -s kubu-hai-build/tests -v
```

## Safety and scope

- No privileged container mode.
- No host Docker socket mount.
- Source is mounted read/write only for the build.
- Build output is kept in the selected build directory.
- No external project is fetched or linked.
- No existing project files are overwritten by this addition.

## Integration contract

The driver configures CMake with:

- `KUBU_HAI_BACKEND=cpu|cuda|metal`
- `KUBU_HAI_BUILD_TYPE=Debug|Release|RelWithDebInfo`

A project must explicitly implement and validate each backend. CPU is the portable baseline.
