#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include <cuda.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

void init_vector(float* arr, int len) {
    for (int i = 0; i < len; i++) {
        arr[i] = i + 1;
    }
}

int cmp_vector(float *left, float* right, int len, float epsilon = 1e-5) {
    int err = 0;
    for (int i = 0; i < len; i++) {
        err += fabs(left[i] - right[i]) > epsilon;
    }
    return err;
}

void vadd_cpu(float* a, float* b, float* c, int len) {
    for (int i = 0; i < len; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vadd_gpu_(float* a, float* b, float* c, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        c[i] = a[i] + b[i];
    }
}

void vadd_gpu(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "input tensors must have the same shape");

    int len = a.numel();
    int nthreads = 32;
    int nblocks = 1 + (len - 1) / nthreads;

    float* A = a.data_ptr<float>();
    float* B = b.data_ptr<float>();
    float* C = c.data_ptr<float>();

    vadd_gpu_<<<nblocks, nthreads>>>(A, B, C, len);
    cudaDeviceSynchronize();
}

void unifiedMemExample(int len) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* D = new float[len];
    cudaMallocManaged(&A, len * sizeof(float));
    cudaMallocManaged(&B, len * sizeof(float));
    cudaMallocManaged(&C, len * sizeof(float));

    init_vector(A, len);
    init_vector(B, len);

    int nthreads = 32;
    int nblocks = 1 + (len - 1) / nthreads;

    auto t1 = std::chrono::high_resolution_clock::now();
    vadd_gpu_<<<nblocks, nthreads>>>(A, B, C, len);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    vadd_cpu(A, B, D, len);
    auto t3 = std::chrono::high_resolution_clock::now();

    int err = cmp_vector(C, D, len);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] D;

    std::chrono::duration<double, std::milli> cpu_time = t3 - t2;
    std::chrono::duration<double, std::milli> gpu_time = t2 - t1;
    std::cout << "Vector length: " << len << std::endl;
    std::cout << "Error count: " << err << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << "ms" << std::endl;
    std::cout << "GPU time: " << gpu_time.count() << "ms" << std::endl;
}

void explicitMemExample(int len) {
    size_t nbytes = len * sizeof(float);
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* D = new float[len];
    cudaMallocHost(&A, nbytes);
    cudaMallocHost(&B, nbytes);
    cudaMallocHost(&C, nbytes);

    init_vector(A, len);
    init_vector(B, len);

    float* _A = nullptr;
    float* _B = nullptr;
    float* _C = nullptr;
    cudaMalloc(&_A, nbytes);
    cudaMalloc(&_B, nbytes);
    cudaMalloc(&_C, nbytes);

    int nthreads = 32;
    int nblocks = 1 + (len - 1) / nthreads;

    auto t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(_A, A, nbytes, cudaMemcpyDefault);
    cudaMemcpy(_B, B, nbytes, cudaMemcpyDefault);
    auto t2 = std::chrono::high_resolution_clock::now();
    vadd_gpu_<<<nblocks, nthreads>>>(_A, _B, _C, len);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(C, _C, nbytes, cudaMemcpyDefault);
    auto t4 = std::chrono::high_resolution_clock::now();
    vadd_cpu(A, B, D, len);
    auto t5 = std::chrono::high_resolution_clock::now();

    int err = cmp_vector(C, D, len);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(_A);
    cudaFree(_B);
    cudaFree(_C);
    delete[] D;

    std::chrono::duration<double, std::milli> cpu_time = t5 - t4;
    std::chrono::duration<double, std::milli> gpu_time_comp = t3 - t2;
    std::chrono::duration<double, std::milli> gpu_time_all = t4 - t1;
    std::cout << "Vector length: " << len << std::endl;
    std::cout << "Error count: " << err << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << "ms" << std::endl;
    std::cout << "GPU time (compute-only): " << gpu_time_comp.count() << "ms" << std::endl;
    std::cout << "GPU time (with memory transfer): " << gpu_time_all.count() << "ms" << std::endl;
}

PYBIND11_MODULE(gpu, m) {
    m.def("unifiedMemExample", &unifiedMemExample, py::arg("len"));
    m.def("explicitMemExample", &explicitMemExample, py::arg("len"));
    m.def("vadd", &vadd_gpu);
}
