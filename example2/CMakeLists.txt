cmake_minimum_required(VERSION 3.15)
project(vadd_cuda_library LANGUAGES CXX CUDA)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cuda_ext vadd.cu)

install(TARGETS cuda_ext LIBRARY DESTINATION .)
