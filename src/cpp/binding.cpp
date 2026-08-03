#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_mineru_llama_cpp, m) {
    m.def("ping", []() { return "pong"; });
}
