#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

float square(float x) { return x * x; }

float sadd_cpu(float a, float b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.def("square", &square);
    m.def("sadd", &sadd_cpu, "a"_a, py::arg("b"));
    m.attr("version") = 1;
    m.attr("hello") = "hello";
    m.attr("word") = py::cast("World");
}
