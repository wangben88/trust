#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<LeafScore>(m, "LeafScore")
    .def(py::init<const std::string &>())
    .def("precompute_sum_and_max", &Pet::precompute_sum_and_max)
    .def("sum", &Pet::sum)
    .def("max", &Pet::max);
}