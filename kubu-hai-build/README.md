# KUBU-HAI Hybrid Build Foundation

An opt-in build entry point for this repository. It adds capability reporting, a native Flutter/Linux build path, a generic CMake path, an isolated Linux toolchain image, and unit tests. Existing project build files are left untouched.

## This repository's current build

The root `CMakeLists.txt` is the Flutter Linux desktop runner. The driver detects `pubspec.yaml` and invokes Flutter rather than bypassing Flutter's generated CMake rules.

On a Linux host with Flutter installed and Linux desktop dependencies configured:

```bash
python3 kubu-hai-build/build.py --source . --build-dir .build/kubu-hai --build-type Release
```

Install the Flutter SDK and the Flutter Linux desktop prerequisites before building. The driver reports a clear error if Flutter is missing.

## Generic CMake projects

For a non-Flutter source directory containing `CMakeLists.txt`:

```bash
python3 kubu-hai-build/build.py --source path/to/cmake-project --build-dir .build/native --backend cpu
```

The generic path passes `KUBU_HAI_BACKEND` and `KUBU_HAI_BUILD_TYPE` to CMake. The target project must explicitly implement and validate any CUDA or Metal backend. CPU is the safe default.

## Container build

```bash
python3 kubu-hai-build/build.py --source path/to/cmake-project --mode docker --backend cpu
```

Container mode is for generic CMake projects; the supplied image does not bundle Flutter. CUDA container builds require a Linux host, NVIDIA driver, NVIDIA Container Toolkit, and Docker configured for `--gpus all`. Metal builds must run natively on macOS; Linux containers cannot expose native Apple Metal.

## Tests

```shell
python3 -m unittest discover -s kubu-hai-build/tests -v
```

## Safety and scope

- No privileged container mode or host Docker socket mount.
- Source is mounted into the container only for the build.
- Build output is kept in the selected build directory.
- No external project source is fetched or linked.
- No existing project files are overwritten by this addition.

## Important backend limitation

This build driver is infrastructure, not a GPU implementation. It does not create CUDA kernels, Metal shaders, or runtime dispatch by itself. The current Flutter runner's build path is Linux-specific; macOS Metal support requires a separate native macOS runner and actual Metal backend implementation.
