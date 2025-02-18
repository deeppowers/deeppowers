#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tokenizer.hpp"
#include "model.hpp"

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
    
    // Bind Model class
    py::class_<deeppowers::Model>(m, "Model")
        .def_static("load_model", &deeppowers::Model::load_model)
        .def("generate", &deeppowers::Model::generate,
            py::arg("input_ids"),
            py::arg("attention_mask") = py::none(),
            py::arg("max_length") = 100,
            py::arg("min_length") = 0,
            py::arg("temperature") = 1.0f,
            py::arg("top_k") = 50,
            py::arg("top_p") = 1.0f,
            py::arg("repetition_penalty") = 1.0f,
            py::arg("num_return_sequences") = 1,
            py::arg("do_sample") = true,
            py::arg("early_stopping") = false)
        .def("save", &deeppowers::Model::save)
        .def_property_readonly("device", &deeppowers::Model::device)
        .def_property_readonly("config", &deeppowers::Model::config);
} 