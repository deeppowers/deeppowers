# User Guide

This manual will guide users through the basic process of using the engine and cover essential aspects for effective utilization.

**User Manual: DeepPowers Inference Engine**

**Version:** 0.1.0
**Date:** February 5, 2025

---

**1. Introduction**

Welcome to the DeepPowers Inference Engine, a high-performance, multi-hardware acceleration framework designed to efficiently deploy and run Large Language Models (LLMs). This engine is engineered for flexibility and speed, supporting a wide range of hardware platforms including NVIDIA GPUs, AMD GPUs, Apple Silicon, and Intel GPUs.  It leverages advanced optimization techniques and a modular architecture to deliver low latency and high throughput inference, while minimizing operational costs.

**Key Features:**

*   **Multi-Hardware Support:**  Seamlessly runs on NVIDIA, AMD, Apple Silicon, and Intel GPUs, as well as CPUs.
*   **High Performance:**  Optimized for low latency and high throughput through graph optimizations, custom kernels, dynamic scheduling, and efficient memory management.
*   **Flexible Configuration:**  Offers extensive configuration options for hardware selection, optimization levels, quantization, and batching strategies.
*   **Custom Kernel Integration:**  Allows users to integrate custom-optimized kernels for unique algorithmic needs.
*   **Quantization Support:**  Supports various quantization techniques (INT8, INT4, FP16, BF16) to reduce memory footprint and accelerate inference.
*   **Continuous and Dynamic Batching:**  Implements advanced batching strategies to maximize hardware utilization and throughput.
*   **User-Friendly API:**  Provides both C++ and Python APIs for easy integration into diverse applications.
*   **Open and Extensible Architecture:** Modular design allows for easy extension and customization.

**2. Getting Started**

This section guides you through the installation and setup process to get your LLM Inference Engine running.

**2.1 Prerequisites**

Before you begin, ensure you have the following prerequisites installed on your system:

*   **Operating System:**  Linux (Recommended), macOS, Windows (Limited Support - specify if applicable)
*   **CMake:** Version 3.15 or higher.
*   **C++ Compiler:**  C++17 compatible compiler (e.g., GCC, Clang, MSVC).
*   **Python:** Version 3.8 or higher (if using the Python API).
*   **Hardware Drivers:**  Ensure you have the appropriate drivers installed for your target hardware (NVIDIA CUDA Toolkit, AMD ROCm, Apple Metal drivers, Intel oneAPI Toolkit).
*   **Optional Libraries:** (List optional dependencies, e.g., ONNX Runtime, PyTorch if directly used for model parsing)

**2.2 Installation**

The DeepPowers Inference Engine is built from source using CMake. Follow these steps to build and install the engine:

```bash
git clone https://github.com/deeppowers/deeppowers.git
cd deeppowers
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release # Or Debug for debug builds
make -j$(nproc)  # Use all available CPU cores for compilation
sudo make install # Optional: Install to system-wide locations (requires sudo)
```

**Explanation:**

*   `git clone https://github.com/deeppowers/deeppowers.git`: Clones the inference engine repository to your local machine.
*   `mkdir build && cd build`: Creates a `build` directory and navigates into it.  Building in a separate directory is recommended.
*   `cmake .. -DCMAKE_BUILD_TYPE=Release`: Configures the build process using CMake. `-DCMAKE_BUILD_TYPE=Release` specifies a release build with optimizations. You can use `Debug` for debugging.
*   `make -j$(nproc)`: Compiles the code using `make`, utilizing all available CPU cores (`-j$(nproc)`) for faster compilation.
*   `sudo make install`: (Optional) Installs the compiled libraries and headers to system-wide locations (e.g., `/usr/local/lib`, `/usr/local/include`).  Requires administrator privileges.  If you skip this, you will need to set environment variables to use the engine.

**2.3 Setting Environment Variables (If not installed system-wide)**

If you did not run `sudo make install`, you may need to set environment variables so that your applications can find the engine's libraries and headers.  For example, assuming you built in `deeppowers/build`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_inference_engine>/build/lib  # Linux
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:<path_to_inference_engine>/build/lib  # macOS
export PATH=$PATH:<path_to_inference_engine>/build/bin # For executables, if any
```

Replace `<path_to_inference_engine>` with the absolute path to your `deeppowers` directory. Add these lines to your `.bashrc`, `.zshrc`, or equivalent shell configuration file to make them persistent.

**2.4 Verifying Installation (Hello World Example)**

Create a simple C++ program (e.g., `hello_engine.cpp`) to verify the installation:

```cpp
#include <iostream>
#include <inference_engine_api.h> // Replace with the actual API header

int main() {
  try {
    InferenceEngine::Engine engine; // Replace with the actual Engine class name
    std::cout << "LLM Inference Engine initialized successfully!" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing engine: " << e.what() << std::endl;
    return 1;
  }
}
```

Compile and run the example:

```bash
g++ hello_engine.cpp -o hello_engine -I<path_to_inference_engine>/include -L<path_to_inference_engine>/build/lib -linference_engine # Adjust library name
./hello_engine
```

If you see "DeepPowers Inference Engine initialized successfully!", your installation is successful.

**3. Core Concepts**

Understanding these core concepts is essential for using the DeepPowers LLM Inference Engine effectively.

*   **Model:** Represents the pre-trained Large Language Model you want to execute for inference. Models are typically loaded from files in formats like ONNX, or framework-specific formats (if supported).
*   **Configuration:** Defines the settings for the inference engine, including hardware selection, optimization options, quantization parameters, batching strategies, and more. Configurations are typically loaded from configuration files or set programmatically through the API.
*   **Session (or Inference Session):**  An instance of a loaded model configured for inference.  You create a session from a Model and a Configuration. A session manages the execution graph, memory, and kernel execution for a specific model and configuration.
*   **Device:**  Refers to the hardware device where inference will be executed (e.g., "GPU:0" for the first GPU, "CPU" for CPU execution).  The engine automatically manages hardware abstraction through the HAL.
*   **Input Tensor:**  The input data provided to the model for inference. For LLMs, this is typically tokenized and numericalized text input.
*   **Output Tensor:** The output data generated by the model after inference. For LLMs, this usually represents token probabilities or generated tokens.

**4. Using the API**

The DeepPowers Inference Engine provides both C++ and Python APIs. Choose the API that best suits your application development environment.

**4.1 C++ API Usage**

**(Example C++ code snippets - Replace placeholders with actual API names and paths)**

```cpp
#include <inference_engine_api.h> // Replace with actual API header
#include <iostream>
#include <string>
#include <vector>

int main() {
  try {
    // 1. Create Engine instance
    InferenceEngine::Engine engine; // Replace with actual Engine class

    // 2. Load Model from file
    std::string model_path = "path/to/your/model.onnx"; // Replace with your model path
    InferenceEngine::Model model = engine.loadModel(model_path);

    // 3. Create Configuration
    InferenceEngine::Configuration config;
    config.setDeviceType(InferenceEngine::DeviceType::GPU); // Run on GPU
    config.setOptimizationLevel(InferenceEngine::OptimizationLevel::High); // Enable high optimization
    // ... more configuration settings ...

    // 4. Create Inference Session
    InferenceEngine::InferenceSession session = engine.createSession(model, config);

    // 5. Prepare Input Data (Tokenized input IDs)
    std::vector<int> input_ids = {101, 2023, 2003, 1037, 1200, 102}; // Example input IDs
    InferenceEngine::Tensor input_tensor = InferenceEngine::Tensor::create({1, input_ids.size()}, InferenceEngine::DataType::INT32, input_ids.data()); // Example tensor creation

    // 6. Run Inference
    std::vector<InferenceEngine::Tensor> output_tensors = session.run({{"input_ids", input_tensor}}); // Assuming input name is "input_ids"

    // 7. Process Output Tensor
    InferenceEngine::Tensor output_tensor = output_tensors[0]; // Assuming single output
    // ... access output_tensor data, decode tokens, etc. ...

    std::cout << "Inference completed successfully!" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Inference Error: " << e.what() << std::endl;
    return 1;
  }
}
```

**4.2 Python API Usage**

### Basic Usage
```python
import deeppowers as dp

# Method 1: Using Pipeline (Recommended)
# Initialize pipeline with pre-trained model
pipeline = dp.Pipeline.from_pretrained("deepseek-v3")

# Generate text
response = pipeline.generate(
    "Hello, how are you?",
    max_length=50,
    temperature=0.7,
    top_p=0.9
)
print(response)

# Batch processing
responses = pipeline.generate(
    ["Hello!", "How are you?"],
    max_length=50,
    temperature=0.7
)

# Save and load pipeline
pipeline.save("my_pipeline")
loaded_pipeline = dp.Pipeline.load("my_pipeline")

# Method 2: Using Tokenizer and Model separately
# Initialize tokenizer
tokenizer = dp.Tokenizer(model_name="deepseek-v3")  # or use custom vocab
tokenizer.load("path/to/tokenizer.model")

# Initialize model
model = dp.Model.from_pretrained("deepseek-v3")

# Create pipeline manually
pipeline = dp.Pipeline(model=model, tokenizer=tokenizer)
```

### Advanced Usage

#### Custom Tokenizer Training
```python
# Initialize tokenizer with specific type
tokenizer = dp.Tokenizer(tokenizer_type=dp.TokenizerType.WORDPIECE)

# Train on custom data
texts = ["your", "training", "texts"]
tokenizer.train(texts, vocab_size=30000, min_frequency=2)

# Save and load
tokenizer.save("tokenizer.model")
tokenizer.load("tokenizer.model")

# Basic tokenization
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# Batch processing with parallel execution
texts = ["multiple", "texts", "for", "processing"]
tokens_batch = tokenizer.encode_batch(
    texts,
    add_special_tokens=True,
    padding=True,
    max_length=128
)
```

#### Advanced Generation Control
```python
# Configure generation parameters
response = pipeline.generate(
    "Write a story about",
    max_length=200,          # Maximum length of generated text
    min_length=50,           # Minimum length of generated text
    temperature=0.7,         # Controls randomness (higher = more random)
    top_k=50,               # Limits vocabulary to top k tokens
    top_p=0.9,              # Nucleus sampling threshold
    num_return_sequences=3,  # Number of different sequences to generate
    repetition_penalty=1.2   # Penalize repeated tokens
)

# Batch generation with multiple prompts
prompts = [
    "Write a story about",
    "Explain quantum physics",
    "Give me a recipe for"
]
responses = pipeline.generate(
    prompts,
    max_length=100,
    temperature=0.8
)
```


**5. Configuration Options**

The `Configuration` object allows you to fine-tune the inference engine's behavior. Here are some key configuration options:

*   **`device_type` (Enum: `DeviceType.CPU`, `DeviceType.GPU`, `DeviceType.AMD_GPU`, `DeviceType.APPLE_GPU`, `DeviceType.INTEL_GPU`):**  Specifies the target hardware device for inference.  Defaults to `DeviceType.CPU` if not set.
*   **`device_id` (String):**  Specifies a particular device ID if multiple devices of the same type are available (e.g., "GPU:0", "GPU:1").
*   **`optimization_level` (Enum: `OptimizationLevel.None`, `OptimizationLevel.Low`, `OptimizationLevel.Medium`, `OptimizationLevel.High`):**  Controls the level of graph optimization applied by the engine. `High` offers the best performance but might increase compilation time.
*   **`quantization_type` (Enum: `QuantizationType.None`, `QuantizationType.INT8`, `QuantizationType.INT4`, `QuantizationType.FP16`, `QuantizationType.BF16`):**  Enables quantization for model weights and/or activations to reduce memory footprint and accelerate inference. Choose `None` for no quantization.
*   **`batch_size` (Integer):**  Sets the static batch size.  For dynamic batching, this might be a maximum batch size.
*   **`continuous_batching_enabled` (Boolean):** Enables continuous batching for higher throughput in server environments.
*   **`num_threads` (Integer):**  Sets the number of threads to use for CPU inference (and potentially for preprocessing/postprocessing).
*   **`memory_pool_size` (String, e.g., "1GB", "512MB"):**  Sets the size of the memory pool for GPU memory allocation, if applicable.
*   **`custom_kernel_library_path` (String):**  Path to a shared library containing custom-optimized kernels.
*   **(… add more configuration options specific to your engine …)**

Refer to the API documentation for a complete list of configuration options and their descriptions.

**6. Advanced Features**

*   **Custom Kernel Integration:**  For advanced users, the engine allows integration of custom-optimized kernels. This can be beneficial for leveraging unique hardware features or implementing specific algorithmic optimizations.  Refer to the developer documentation for details on how to create and integrate custom kernels.
*   **Quantization Techniques:**  The engine supports various quantization techniques beyond basic INT8, including INT4, FP16, and BF16. Experiment with different quantization types to find the best balance between performance and accuracy for your model. Quantization-aware training workflows can be used to further improve accuracy after quantization.
*   **Dynamic Batching Strategies:**  Explore dynamic batching and continuous batching features to optimize throughput in server scenarios.  Configure batching parameters to balance latency and throughput based on your application's requirements.
*   **Paged Attention (If implemented):**  If your engine implements paged attention, explain how to enable and configure it for efficient handling of long sequences and variable sequence lengths.

**7. Performance Tuning**

*   **Hardware Selection:**  Choose the appropriate hardware device (`device_type`) based on your performance requirements and model size. GPUs generally offer higher performance for LLMs.
*   **Optimization Level:**  Experiment with different `optimization_level` settings. `High` optimization usually provides the best performance but might increase model loading and compilation time.
*   **Quantization:**  Enable quantization (`quantization_type`) to reduce memory footprint and potentially accelerate inference, especially for large models. Carefully evaluate accuracy after quantization.
*   **Batch Size:**  Increase the `batch_size` to improve throughput, especially for GPU inference. However, larger batch sizes can increase latency. Experiment to find the optimal batch size for your workload.
*   **Memory Pool Size:**  Adjust `memory_pool_size` for GPU inference to optimize memory allocation and reduce fragmentation.
*   **Profiling Tools:**  Utilize profiling tools (e.g., NVIDIA Nsight Systems, AMD ROCm Profiler, Intel VTune) to identify performance bottlenecks and guide optimization efforts.

**8. Troubleshooting**

*   **"Engine initialization failed" Error:**
    *   **Possible Cause:**  Incorrect hardware drivers, missing dependencies, or incompatible hardware.
    *   **Solution:**  Verify hardware driver installation, check prerequisites, and ensure your hardware is supported.  Review error logs for more details.
*   **"Model loading failed" Error:**
    *   **Possible Cause:**  Incorrect model path, unsupported model format, or corrupted model file.
    *   **Solution:**  Double-check the model path, ensure the model format is supported by the engine, and verify the model file is not corrupted.
*   **"Out of Memory" Error (GPU):**
    *   **Possible Cause:**  Model too large for GPU memory, large batch size, or insufficient memory pool size.
    *   **Solution:**  Try reducing batch size, enabling quantization, using a smaller model (if possible), or increasing `memory_pool_size` (if applicable).


**9. Examples**

*   **Example 1: Basic Text Generation (C++ and Python)**
*   **Example 2: Question Answering with Context (C++ and Python)**
*   **Example 3: Using Custom Kernels (if applicable)**
*   **Example 4: Performance Benchmarking Script**

for example:
[examples](https://github.com/deeppowers/deeppowers/tree/main/examples)

**10. API Reference**

**(Provide a link to automatically generated API documentation - Doxygen, Sphinx, etc. - if available. Or include a summarized API reference section in the manual itself. )**

*   **(Link to Doxygen/Sphinx generated C++ API documentation)**
*   **(Link to Python API documentation)**
*   **(Or, provide a table summarizing key API classes, functions, and their descriptions directly in the manual)**

**11. FAQ (Frequently Asked Questions)**

**(Example FAQs - Tailor to your engine's specific aspects and potential user questions)**

*   **Q: Which model formats are supported?**
    *   **A:** Currently, the engine supports [List supported formats: ONNX, etc.].  Support for other formats is planned for future releases.
*   **Q: How do I choose the best `optimization_level`?**
    *   **A:** Start with `OptimizationLevel.High` for best performance. If model loading time is a concern, you can try `OptimizationLevel.Medium` or `Low`. `OptimizationLevel.None` disables optimizations and is mainly for debugging.
*   **Q: How do I enable quantization?**
    *   **A:** Set the `quantization_type` configuration option to your desired quantization type (e.g., `QuantizationType.INT8`). Ensure your model is compatible with quantization or perform quantization-aware training.

**12. Contributing

We welcome contributions to the DeepPowers LLM Inference Engine! Please refer to the `CONTRIBUTING.md` file in the repository for guidelines on contributing code, bug reports, feature requests, and documentation improvements.

**13. License**

This LLM Inference Engine is released under the Apache License 2.0. See the `LICENSE` file in the repository for the full license text.