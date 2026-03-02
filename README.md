# Learn scikit-build-core

This repository contains a small set of progressively more complex examples showing how to use **scikit-build-core** to build Python extension modules in C++, CUDA, and CUDA + PyTorch.

The focus is on:

- Using `pybind11` together with scikit-build-core
- Integrating CMake-based builds into Python packaging
- Moving from a pure C++ extension to CUDA, then to PyTorch tensors on the GPU

## Requirements

- Python 3.10+
- A C++17-capable compiler (for example `g++ >= 9`)
- CMake 3.18+
  (scikit-build-core can automatically download a newer CMake if needed)
- CUDA Toolkit (for CUDA examples)
- PyTorch with CUDA support (only required for `example3`)
- Recommended: [`uv`](https://github.com/astral-sh/uv) for managing virtual environments and installs

## Example 1: Pure C++ extension (`example1`)

Basic C++ extension module built with `pybind11` + `scikit-build-core`.

- Layout:
  - `pyproject.toml`: minimal scikit-build-core configuration
  - `CMakeLists.txt`: declares the C++ extension target and install location
  - `example.cpp`: simple C++ functions exposed to Python
  - `main.py`: small script that imports and exercises the extension
- How to build and run:

```bash
cd example1
uv pip install -e .
uv run main.py
```

## Example 2: CUDA vector addition (`example2`)

Adds a CUDA kernel and shows how to build a CUDA-backed extension.

- Highlights:
  - `vadd.cu`: CPU vs CUDA vector addition and timing
  - `CMakeLists.txt`: enables `CUDA` language and uses `pybind11_add_module`
- How to build and run:

```bash
cd example2
uv pip install -e .
uv run main.py
```

## Example 3: CUDA with PyTorch tensors (`example3`)

Shows how to call CUDA kernels directly on `torch::Tensor` objects, sharing GPU memory with PyTorch.

- Highlights:
  - Uses `torch.utils.cmake_prefix_path` to locate Torch’s CMake configuration
  - `find_package(Torch CONFIG REQUIRED)` plus explicit linking to `libtorch_python.so`
  - `vadd.cu` uses `torch::Tensor` and `data_ptr<float>()` to launch a CUDA kernel
- How to build and run:

```bash
cd example3
uv pip install -e .
uv run main.py
```

## Tips for development and debugging

- **CMake / Torch integration**
  - A common pattern is to call Python from CMake (via `execute_process`) and use `torch.utils.cmake_prefix_path` to avoid hard-coding Torch installation paths.
  - If you see `Could not find a package configuration file provided by "Torch"`, it usually means either:
    - Torch is not installed in the build environment, or
    - `CMAKE_PREFIX_PATH` (or the `PATHS/HINTS` passed to `find_package(Torch)`) does not include Torch’s CMake files.

- **CUDA architectures**
  - For a single known GPU (e.g. on a dev machine), you can set a specific `CUDA_ARCHITECTURES` value (such as `86`).
  - For wheels that should run on a range of GPUs, configure multiple architectures (e.g. `70;75;80;86`) instead of relying on implicit defaults.

- **Recommended workflow**
  - Use a dedicated virtual environment per example directory, or a single environment at the repo root, managed with `uv`.
  - Use `uv pip install -e .` for editable installs while developing, so local C++/CUDA changes are immediately visible from Python.
