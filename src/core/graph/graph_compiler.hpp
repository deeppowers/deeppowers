#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace deeppowers {

// Graph node types
enum class OpType {
    MATMUL,           // Matrix multiplication
    ATTENTION,        // Attention calculation
    LAYERNORM,        // Layer normalization
    ACTIVATION,       // Activation function
    ELEMENTWISE,      // Element-wise operation
    CONCAT,          // Tensor concatenation
    SPLIT,           // Tensor splitting
    RESHAPE,         // Shape transformation
    TRANSPOSE,       // Transpose
    CUSTOM           // Custom operator
};

// Operator attributes
struct OpAttributes {
    std::string name;                     // Operator name
    OpType type;                          // Operator type
    std::vector<int64_t> input_shapes;    // Input shapes
    std::vector<int64_t> output_shapes;   // Output shapes
    std::unordered_map<std::string, std::string> params;  // Other parameters
};

// Graph node
struct GraphNode {
    OpAttributes attributes;                  // Node attributes
    std::vector<GraphNode*> inputs;          // Input nodes
    std::vector<GraphNode*> outputs;         // Output nodes
    std::unique_ptr<hal::Tensor> output_tensor;  // Output tensor
    bool is_constant = false;                // Whether it's a constant node
    bool is_fused = false;                   // Whether it's fused
    std::string kernel_type;                 // Kernel type
    size_t workspace_size = 0;               // Workspace size
};

//  Compile options
struct CompileOptions {
    bool enable_fusion = true;               // Enable operator fusion
    bool enable_memory_planning = true;      // Enable memory planning
    bool enable_kernel_selection = true;     // Enable kernel selection
    bool enable_constant_folding = true;     // Enable constant folding
    size_t max_workspace_size = 1024*1024*1024;  // Maximum workspace (1GB)
    std::string optimization_level = "O2";   // Optimization level (O0-O3)
};

// Compile statistics
struct CompileStats {
    size_t num_nodes = 0;                    // Number of nodes
    size_t num_fused_nodes = 0;              // Number of fused nodes
    size_t peak_memory = 0;                  // Peak memory
    size_t workspace_size = 0;               // Workspace size
    double compile_time_ms = 0.0;            // Compile time
    std::vector<std::string> optimizations;  // Applied optimizations
};

// Graph compiler class
class GraphCompiler {
public:
    explicit GraphCompiler(hal::Device* device);
    ~GraphCompiler();

    // Compile method
    void compile(const std::vector<GraphNode*>& nodes,
                const CompileOptions& options = CompileOptions());
    
    // Optimization methods
    void optimize_graph();
    void fuse_operators();
    void plan_memory();
    void select_kernels();
    
    // Execution methods
    void execute(const std::unordered_map<std::string, hal::Tensor*>& inputs,
                std::unordered_map<std::string, hal::Tensor*>& outputs);
    
    // Status query
    bool is_compiled() const { return is_compiled_; }
    const CompileStats& get_stats() const { return stats_; }
    
    // Resource management
    void clear();
    size_t get_workspace_size() const;

private:
    // Internal helper methods
    void validate_graph();
    void analyze_dependencies();
    void schedule_nodes();
    void allocate_buffers();
    
    // Optimization helper methods
    bool can_fuse_nodes(const GraphNode* a, const GraphNode* b) const;
    void fuse_node_pair(GraphNode* a, GraphNode* b);
    void eliminate_dead_nodes();
    void fold_constants();
    
    // Memory planning helper methodslanning helper methods
    void analyze_lifetimes();
    void assign_memory_blocks();
    void reuse_buffers();
    
    // Kernel selection helper methods
    void benchmark_kernels();
    void select_best_kernel(GraphNode* node);
    
    // Execution helper methods
    void prepare_workspace();
    void execute_node(const GraphNode* node);
    void sync_execution();
    
    // Member variables
    hal::Device* device_;
    std::vector<std::unique_ptr<GraphNode>> nodes_;
    std::vector<GraphNode*> execution_order_;
    std::unique_ptr<hal::Tensor> workspace_;
    CompileStats stats_;
    bool is_compiled_ = false;
    
    // Memory management
    struct MemoryBlock {
        size_t offset;
        size_t size;
        std::vector<GraphNode*> users;
    };
    std::vector<MemoryBlock> memory_blocks_;
    
    // Kernel cache
    struct KernelInfo {
        std::string type;
        double execution_time;
        size_t workspace_size;
    };
    std::unordered_map<std::string, std::vector<KernelInfo>> kernel_database_;
};

} // namespace deeppowers 