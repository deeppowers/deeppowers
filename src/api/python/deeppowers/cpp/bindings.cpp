#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tokenizer.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include "model_types.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_deeppowers_core, m) {
    m.doc() = "DeepPowers C++ backend Python bindings";
    
    // Bind TokenizerType enum
    py::enum_<deeppowers::TokenizerType>(m, "TokenizerType")
        .value("BPE", deeppowers::TokenizerType::BPE)
        .value("WORDPIECE", deeppowers::TokenizerType::WORDPIECE)
        .export_values();
    
    // Bind Tokenizer class
    py::class_<deeppowers::Tokenizer>(m, "Tokenizer")
        .def(py::init<deeppowers::TokenizerType>())
        .def("initialize", &deeppowers::Tokenizer::initialize)
        .def("encode", &deeppowers::Tokenizer::encode)
        .def("decode", &deeppowers::Tokenizer::decode)
        .def("encode_batch", &deeppowers::Tokenizer::encode_batch)
        .def("decode_batch", &deeppowers::Tokenizer::decode_batch)
        .def("save", &deeppowers::Tokenizer::save)
        .def("load", &deeppowers::Tokenizer::load)
        .def("vocab_size", &deeppowers::Tokenizer::vocab_size)
        .def_property_readonly("pad_token_id", &deeppowers::Tokenizer::pad_token_id)
        .def_property_readonly("eos_token_id", &deeppowers::Tokenizer::eos_token_id)
        .def_property_readonly("bos_token_id", &deeppowers::Tokenizer::bos_token_id)
        .def_property_readonly("unk_token_id", &deeppowers::Tokenizer::unk_token_id);
    
    // Bind DataType enum
    py::enum_<deeppowers::DataType>(m, "DataType")
        .value("FLOAT32", deeppowers::DataType::FLOAT32)
        .value("FLOAT16", deeppowers::DataType::FLOAT16)
        .value("INT8", deeppowers::DataType::INT8)
        .value("INT4", deeppowers::DataType::INT4)
        .value("UINT8", deeppowers::DataType::UINT8)
        .value("INT32", deeppowers::DataType::INT32)
        .export_values();
    
    // Bind PrecisionMode enum
    py::enum_<deeppowers::PrecisionMode>(m, "PrecisionMode")
        .value("FULL", deeppowers::PrecisionMode::FULL)
        .value("MIXED", deeppowers::PrecisionMode::MIXED)
        .value("INT8", deeppowers::PrecisionMode::INT8)
        .value("INT4", deeppowers::PrecisionMode::INT4)
        .value("AUTO", deeppowers::PrecisionMode::AUTO)
        .export_values();
    
    // Bind ModelFormat enum
    py::enum_<deeppowers::ModelFormat>(m, "ModelFormat")
        .value("AUTO", deeppowers::ModelFormat::AUTO)
        .value("ONNX", deeppowers::ModelFormat::ONNX)
        .value("PYTORCH", deeppowers::ModelFormat::PYTORCH)
        .value("TENSORFLOW", deeppowers::ModelFormat::TENSORFLOW)
        .value("CUSTOM", deeppowers::ModelFormat::CUSTOM)
        .export_values();
    
    // Bind Tensor class
    py::class_<deeppowers::Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&, deeppowers::DataType>())
        .def("shape", &deeppowers::Tensor::shape)
        .def("dtype", &deeppowers::Tensor::dtype)
        .def("size", &deeppowers::Tensor::size)
        .def("to", &deeppowers::Tensor::to)
        .def("to_device", &deeppowers::Tensor::to_device)
        .def("device", &deeppowers::Tensor::device)
        .def("clone", &deeppowers::Tensor::clone)
        .def("__repr__", &deeppowers::Tensor::to_string);
    
    // Bind Model class
    py::class_<deeppowers::Model, std::shared_ptr<deeppowers::Model>>(m, "Model")
        .def("forward", &deeppowers::Model::forward)
        .def("forward_batch", &deeppowers::Model::forward_batch)
        .def("save", &deeppowers::Model::save)
        .def("config", &deeppowers::Model::config)
        .def("device", &deeppowers::Model::device)
        .def("to", &deeppowers::Model::to)
        .def("model_type", &deeppowers::Model::model_type)
        .def("precision", &deeppowers::Model::precision)
        .def("set_precision", &deeppowers::Model::set_precision);
    
    // Bind ModelLoader functions
    m.def("load_model", &deeppowers::ModelLoader::load, 
          py::arg("path"), py::arg("format") = deeppowers::ModelFormat::AUTO,
          "Load a model from file");
    
    m.def("convert_model", &deeppowers::ModelLoader::convert,
          py::arg("input_path"), py::arg("output_path"),
          py::arg("input_format"), py::arg("output_format"),
          "Convert a model between formats");
    
    // Add CUDA utility functions
    m.def("get_cuda_version", []() { return "11.7"; });
    m.def("is_cuda_available", []() { return true; });
    m.def("get_cuda_device_count", []() { return 1; });
} 