#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../../cpp/include/deeppowers.hpp"

namespace py = pybind11;
using namespace deeppowers::api;

PYBIND11_MODULE(_deeppowers, m) {
    m.doc() = "DeepPowers - High Performance Text Generation Library";

    // Version information
    m.def("version", &version, "Get the version of DeepPowers");
    m.def("cuda_version", &cuda_version, "Get the version of CUDA");
    m.def("cuda_available", &cuda_available, "Check if CUDA is available");
    m.def("cuda_device_count", &cuda_device_count, "Get the number of CUDA devices");

    // Factory functions
    m.def("load_model", &load_model, "Load a model from the specified path",
          py::arg("model_path"));
    m.def("list_available_models", &list_available_models, "List all available models");
    m.def("is_model_available", &is_model_available, "Check if a model is available",
          py::arg("model_name"));

    // GenerationConfig
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("model_type", &GenerationConfig::model_type,
                      "Model type (e.g., 'gpt')")
        .def_readwrite("max_tokens", &GenerationConfig::max_tokens,
                      "Maximum tokens to generate")
        .def_readwrite("temperature", &GenerationConfig::temperature,
                      "Sampling temperature")
        .def_readwrite("top_p", &GenerationConfig::top_p,
                      "Nucleus sampling threshold")
        .def_readwrite("top_k", &GenerationConfig::top_k,
                      "Top-k sampling threshold")
        .def_readwrite("stop_tokens", &GenerationConfig::stop_tokens,
                      "Stop sequences")
        .def_readwrite("stream", &GenerationConfig::stream,
                      "Enable streaming generation")
        .def_readwrite("batch_size", &GenerationConfig::batch_size,
                      "Batch size for batch generation");

    // GenerationResult
    py::class_<GenerationResult>(m, "GenerationResult")
        .def(py::init<>())
        .def_readwrite("texts", &GenerationResult::texts,
                      "Generated texts")
        .def_readwrite("logprobs", &GenerationResult::logprobs,
                      "Token log probabilities")
        .def_readwrite("tokens", &GenerationResult::tokens,
                      "Generated tokens")
        .def_readwrite("stop_reasons", &GenerationResult::stop_reasons,
                      "Reasons for stopping")
        .def_readwrite("generation_time", &GenerationResult::generation_time,
                      "Generation time in seconds");

    // Model
    py::class_<Model, std::shared_ptr<Model>>(m, "Model")
        .def(py::init<const std::string&>())
        .def("generate", &Model::generate,
             "Generate text from a prompt",
             py::arg("prompt"),
             py::arg("config") = GenerationConfig())
        .def("generate_stream", &Model::generate_stream,
             "Generate text with streaming output",
             py::arg("prompt"),
             py::arg("callback"),
             py::arg("config") = GenerationConfig())
        .def("generate_batch", &Model::generate_batch,
             "Generate text for multiple prompts in parallel",
             py::arg("prompts"),
             py::arg("config") = GenerationConfig())
        .def_property_readonly("model_type", &Model::model_type,
                             "Get the model type")
        .def_property_readonly("model_path", &Model::model_path,
                             "Get the model path")
        .def_property_readonly("vocab_size", &Model::vocab_size,
                             "Get the vocabulary size")
        .def_property_readonly("max_sequence_length", &Model::max_sequence_length,
                             "Get the maximum sequence length")
        .def_property("device",
                     &Model::device,
                     [](Model& self, const std::string& device) {
                         self.to_device(device);
                     },
                     "Get or set the current device")
        .def("set_config", &Model::set_config,
             "Set a configuration value",
             py::arg("key"),
             py::arg("value"))
        .def("get_config", &Model::get_config,
             "Get a configuration value",
             py::arg("key"));
} 