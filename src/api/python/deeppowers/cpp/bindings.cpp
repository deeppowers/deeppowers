#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "tokenizer.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include "model_types.hpp"
#include "inference/inference_engine.hpp"
#include "inference/inference_optimizer.hpp"

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
    
    // Bind ExecutionMode enum
    py::enum_<deeppowers::ExecutionMode>(m, "ExecutionMode")
        .value("SYNC", deeppowers::ExecutionMode::SYNC)
        .value("ASYNC", deeppowers::ExecutionMode::ASYNC)
        .value("STREAM", deeppowers::ExecutionMode::STREAM)
        .export_values();
    
    // Bind OptimizerType enum
    py::enum_<deeppowers::OptimizerType>(m, "OptimizerType")
        .value("NONE", deeppowers::OptimizerType::NONE)
        .value("FUSION", deeppowers::OptimizerType::FUSION)
        .value("PRUNING", deeppowers::OptimizerType::PRUNING)
        .value("DISTILLATION", deeppowers::OptimizerType::DISTILLATION)
        .value("QUANTIZATION", deeppowers::OptimizerType::QUANTIZATION)
        .value("CACHING", deeppowers::OptimizerType::CACHING)
        .value("AUTO", deeppowers::OptimizerType::AUTO)
        .export_values();
    
    // Bind OptimizationLevel enum
    py::enum_<deeppowers::OptimizationLevel>(m, "OptimizationLevel")
        .value("NONE", deeppowers::OptimizationLevel::NONE)
        .value("O1", deeppowers::OptimizationLevel::O1)
        .value("O2", deeppowers::OptimizationLevel::O2)
        .value("O3", deeppowers::OptimizationLevel::O3)
        .export_values();
    
    // Bind Tensor class
    py::class_<deeppowers::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int64_t>&, deeppowers::DataType>())
        .def("shape", &deeppowers::Tensor::shape)
        .def("data_type", &deeppowers::Tensor::data_type)
        .def("device", &deeppowers::Tensor::device)
        .def("to", &deeppowers::Tensor::to)
        .def("clone", &deeppowers::Tensor::clone)
        .def("reshape", &deeppowers::Tensor::reshape);
    
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
          py::arg("device") = "cpu",
          py::arg("dtype") = deeppowers::DataType::FLOAT32,
          "Load a model from file");
    
    m.def("convert_model", &deeppowers::ModelLoader::convert,
          py::arg("input_path"), py::arg("output_path"),
          py::arg("input_format"), py::arg("output_format"),
          "Convert a model between formats");
    
    // Bind InferenceConfig struct
    py::class_<deeppowers::InferenceConfig>(m, "InferenceConfig")
        .def(py::init<>())
        .def_readwrite("max_length", &deeppowers::InferenceConfig::max_length)
        .def_readwrite("min_length", &deeppowers::InferenceConfig::min_length)
        .def_readwrite("temperature", &deeppowers::InferenceConfig::temperature)
        .def_readwrite("top_k", &deeppowers::InferenceConfig::top_k)
        .def_readwrite("top_p", &deeppowers::InferenceConfig::top_p)
        .def_readwrite("repetition_penalty", &deeppowers::InferenceConfig::repetition_penalty)
        .def_readwrite("num_return_sequences", &deeppowers::InferenceConfig::num_return_sequences)
        .def_readwrite("do_sample", &deeppowers::InferenceConfig::do_sample)
        .def_readwrite("early_stopping", &deeppowers::InferenceConfig::early_stopping)
        .def_readwrite("eos_token_id", &deeppowers::InferenceConfig::eos_token_id)
        .def_readwrite("pad_token_id", &deeppowers::InferenceConfig::pad_token_id)
        .def_readwrite("device", &deeppowers::InferenceConfig::device);
    
    // Bind InferenceResult struct
    py::class_<deeppowers::InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("token_ids", &deeppowers::InferenceResult::token_ids)
        .def_readwrite("logprobs", &deeppowers::InferenceResult::logprobs)
        .def_readwrite("stop_reasons", &deeppowers::InferenceResult::stop_reasons)
        .def_readwrite("generation_time", &deeppowers::InferenceResult::generation_time);
    
    // Bind InferenceEngine class
    py::class_<deeppowers::InferenceEngine>(m, "InferenceEngine")
        .def(py::init<std::shared_ptr<deeppowers::Model>>())
        .def("generate", &deeppowers::InferenceEngine::generate,
             py::arg("input_ids"), py::arg("attention_mask") = std::vector<int>(),
             py::arg("config") = deeppowers::InferenceConfig())
        .def("generate_batch", &deeppowers::InferenceEngine::generate_batch,
             py::arg("batch_input_ids"), py::arg("batch_attention_mask") = std::vector<std::vector<int>>(),
             py::arg("config") = deeppowers::InferenceConfig())
        .def("generate_stream", &deeppowers::InferenceEngine::generate_stream,
             py::arg("input_ids"), py::arg("callback"), 
             py::arg("config") = deeppowers::InferenceConfig())
        .def("prepare", &deeppowers::InferenceEngine::prepare,
             py::arg("config") = deeppowers::InferenceConfig())
        .def("reset", &deeppowers::InferenceEngine::reset)
        .def("model", &deeppowers::InferenceEngine::model)
        .def("set_model", &deeppowers::InferenceEngine::set_model);
    
    // Bind OptimizerConfig struct
    py::class_<deeppowers::OptimizerConfig>(m, "OptimizerConfig")
        .def(py::init<>())
        .def_readwrite("type", &deeppowers::OptimizerConfig::type)
        .def_readwrite("level", &deeppowers::OptimizerConfig::level)
        .def_readwrite("enable_profiling", &deeppowers::OptimizerConfig::enable_profiling)
        .def_readwrite("op_blacklist", &deeppowers::OptimizerConfig::op_blacklist)
        .def_readwrite("parameters", &deeppowers::OptimizerConfig::parameters);
    
    // Bind OptimizerResult struct
    py::class_<deeppowers::OptimizerResult>(m, "OptimizerResult")
        .def(py::init<>())
        .def_readonly("success", &deeppowers::OptimizerResult::success)
        .def_readonly("speedup", &deeppowers::OptimizerResult::speedup)
        .def_readonly("memory_reduction", &deeppowers::OptimizerResult::memory_reduction)
        .def_readonly("accuracy_loss", &deeppowers::OptimizerResult::accuracy_loss)
        .def_readonly("metrics", &deeppowers::OptimizerResult::metrics)
        .def_readonly("error_message", &deeppowers::OptimizerResult::error_message);
    
    // Bind InferenceOptimizer class
    py::class_<deeppowers::InferenceOptimizer>(m, "InferenceOptimizer")
        .def(py::init<const deeppowers::OptimizerConfig&>())
        .def("optimize", &deeppowers::InferenceOptimizer::optimize)
        .def("apply_fusion", &deeppowers::InferenceOptimizer::apply_fusion)
        .def("apply_pruning", &deeppowers::InferenceOptimizer::apply_pruning)
        .def("apply_quantization", &deeppowers::InferenceOptimizer::apply_quantization)
        .def("apply_kv_cache_optimization", &deeppowers::InferenceOptimizer::apply_kv_cache_optimization)
        .def("reset", &deeppowers::InferenceOptimizer::reset)
        .def("set_parameter", &deeppowers::InferenceOptimizer::set_parameter)
        .def("get_parameter", &deeppowers::InferenceOptimizer::get_parameter)
        .def("enable_profiling", &deeppowers::InferenceOptimizer::enable_profiling);
    
    // Add CUDA utility functions
    m.def("get_cuda_version", []() { return "11.7"; });
    m.def("is_cuda_available", []() { return true; });
    m.def("get_cuda_device_count", []() { return 1; });
    
    m.def("optimize_model", [](deeppowers::Model* model, 
                             deeppowers::OptimizerType type = deeppowers::OptimizerType::AUTO,
                             deeppowers::OptimizationLevel level = deeppowers::OptimizationLevel::O1) {
        deeppowers::OptimizerConfig config;
        config.type = type;
        config.level = level;
        
        deeppowers::InferenceOptimizer optimizer(config);
        return optimizer.optimize(model);
    }, py::arg("model"), 
       py::arg("type") = deeppowers::OptimizerType::AUTO,
       py::arg("level") = deeppowers::OptimizationLevel::O1);
} 