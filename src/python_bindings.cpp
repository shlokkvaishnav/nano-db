#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "../include/core/hnsw.hpp"

namespace py = pybind11;
using namespace nanodb;

PYBIND11_MODULE(nanodb, m) {
    m.doc() = "NanoDB: High-Performance Vector Search Engine (C++ Backend)";

    py::class_<MMapHandler>(m, "MMapHandler")
        .def(py::init<>())
        .def("open_file", &MMapHandler::open_file)
        .def("close_file", &MMapHandler::close_file);

    py::class_<Result>(m, "Result")
        .def_readonly("id", &Result::id)
        .def_readonly("distance", &Result::distance)
        .def_readwrite("metadata", &Result::metadata) // <--- Bind metadata field
        .def("__repr__", [](const Result &r) {
            return "<Result id=" + std::to_string(r.id) + 
                   " dist=" + std::to_string(r.distance) + 
                   " meta='" + r.metadata + "'>";
        });

    py::class_<HNSW>(m, "HNSW")
        // Init now takes optional metadata path
        .def(py::init<MMapHandler&, std::string>(), py::arg("storage"), py::arg("meta_path") = "data/metadata.bin")
        
        // Insert now takes optional metadata string
        .def("insert", &HNSW::insert, "Insert a vector with ID",
             py::arg("vector"), py::arg("id"), py::arg("metadata") = "", 
             py::call_guard<py::gil_scoped_release>())
        
        .def("search", &HNSW::search, "Search for k-nearest neighbors",
             py::arg("query"), py::arg("k") = 5)
             
        .def("get_metadata", &HNSW::get_metadata);
}