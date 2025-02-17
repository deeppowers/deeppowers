#include "graph_compiler.hpp"
#include <algorithm>
#include <chrono>
#include <queue>
#include <stack>

namespace deeppowers {

GraphCompiler::GraphCompiler(hal::Device* device)
    : device_(device)
    , is_compiled_(false) {
    if (!device_) {
        throw std::runtime_error("Device cannot be null");
    }
}

GraphCompiler::~GraphCompiler() {
    clear();
}

void GraphCompiler::compile(
    const std::vector<GraphNode*>& nodes,
    const CompileOptions& options) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear previous state
    clear();
    
    // Copy nodes
    for (auto node : nodes) {
        nodes_.push_back(std::unique_ptr<GraphNode>(node));
    }
    
    // Validate graph structure
    validate_graph();
    
    // Analyze dependencies
    analyze_dependencies();
    
    // Apply optimizations
    if (options.enable_fusion) {
        fuse_operators();
    }
    if (options.enable_constant_folding) {
        fold_constants();
    }
    
    // Schedule nodes
    schedule_nodes();
    
    // Memory planning
    if (options.enable_memory_planning) {
        plan_memory();
    }
    
    // Select kernels
    if (options.enable_kernel_selection) {
        select_kernels();
    }
    
    // Allocate buffers
    allocate_buffers();
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.compile_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    stats_.num_nodes = nodes_.size();
    
    is_compiled_ = true;
}

void GraphCompiler::optimize_graph() {
    // Apply a series of optimizations
    eliminate_dead_nodes();
    fuse_operators();
    fold_constants();
}

void GraphCompiler::fuse_operators() {
    bool changed;
    do {
        changed = false;
        
        // Iterate through all node pairs, trying to fuse them
        for (size_t i = 0; i < nodes_.size(); ++i) {
            for (size_t j = i + 1; j < nodes_.size(); ++j) {
                if (can_fuse_nodes(nodes_[i].get(), nodes_[j].get())) {
                    fuse_node_pair(nodes_[i].get(), nodes_[j].get());
                    changed = true;
                    stats_.num_fused_nodes++;
                }
            }
        }
    } while (changed);
}

void GraphCompiler::plan_memory() {
    // Analyze tensor lifetimes
    analyze_lifetimes();
    
    // Allocate memory blocks
    assign_memory_blocks();
    
    // Reuse buffers
    reuse_buffers();
}

void GraphCompiler::select_kernels() {
    // Select the best kernel for each node
    for (auto& node : nodes_) {
        if (!node->is_fused) {
            select_best_kernel(node.get());
        }
    }
}

void GraphCompiler::execute(
    const std::unordered_map<std::string, hal::Tensor*>& inputs,
    std::unordered_map<std::string, hal::Tensor*>& outputs) {
    
    if (!is_compiled_) {
        throw std::runtime_error("Graph not compiled");
    }
    
    // Prepare workspace
    prepare_workspace();
    
    // Execute nodes in scheduling order
    for (auto node : execution_order_) {
        execute_node(node);
    }
    
    // Synchronize execution
    sync_execution();
}

void GraphCompiler::clear() {
    nodes_.clear();
    execution_order_.clear();
    workspace_.reset();
    memory_blocks_.clear();
    is_compiled_ = false;
    stats_ = CompileStats();
}

size_t GraphCompiler::get_workspace_size() const {
    return stats_.workspace_size;
}

void GraphCompiler::validate_graph() {
    // Check node validity
    for (const auto& node : nodes_) {
        // Check input and output shapes
        if (node->attributes.input_shapes.empty()) {
            throw std::runtime_error("Node missing input shape: " + node->attributes.name);
        }
        if (node->attributes.output_shapes.empty()) {
            throw std::runtime_error("Node missing output shape:" + node->attributes.name);
        }
        
        // Check connection validity
        for (auto input : node->inputs) {
            if (std::find_if(nodes_.begin(), nodes_.end(),
                [input](const std::unique_ptr<GraphNode>& n) {
                    return n.get() == input;
                }) == nodes_.end()) {
                throw std::runtime_error("Invalid input node connection: " + node->attributes.name);
            }
        }
    }
}

void GraphCompiler::analyze_dependencies() {
    // Build dependency graph
    std::unordered_map<GraphNode*, std::vector<GraphNode*>> deps;
    std::unordered_map<GraphNode*, int> in_degree;
    
    for (const auto& node : nodes_) {
        for (auto input : node->inputs) {
            deps[input].push_back(node.get());
            in_degree[node.get()]++;
        }
    }
    
    // Topological sorting
    std::queue<GraphNode*> q;
    for (const auto& node : nodes_) {
        if (in_degree[node.get()] == 0) {
            q.push(node.get());
        }
    }
    
    while (!q.empty()) {
        auto node = q.front();
        q.pop();
        execution_order_.push_back(node);
        
        for (auto dep : deps[node]) {
            if (--in_degree[dep] == 0) {
                q.push(dep);
            }
        }
    }
    
    if (execution_order_.size() != nodes_.size()) {
            throw std::runtime_error("Cyclic dependency detected in graph");
    }
}

bool GraphCompiler::can_fuse_nodes(const GraphNode* a, const GraphNode* b) const {
    // Check if they can be fused
    if (a->is_fused || b->is_fused) return false;
    
    // Check if operator types are compatible
    if (a->attributes.type == OpType::MATMUL && 
        b->attributes.type == OpType::ACTIVATION) {
        return true;
    }
    
    if (a->attributes.type == OpType::LAYERNORM && 
        b->attributes.type == OpType::ACTIVATION) {
        return true;
    }
    
    return false;
}

void GraphCompiler::fuse_node_pair(GraphNode* a, GraphNode* b) {
    // Create fused node
    auto fused = std::make_unique<GraphNode>();
    fused->attributes.name = a->attributes.name + "_" + b->attributes.name;
    fused->attributes.type = a->attributes.type;
    fused->attributes.input_shapes = a->attributes.input_shapes;
    fused->attributes.output_shapes = b->attributes.output_shapes;
    
    // Update connections
    fused->inputs = a->inputs;
    fused->outputs = b->outputs;
    
    // Mark as fused
    a->is_fused = true;
    b->is_fused = true;
    
    // Add to node list
    nodes_.push_back(std::move(fused));
}

void GraphCompiler::analyze_lifetimes() {
    std::unordered_map<GraphNode*, size_t> first_use;
    std::unordered_map<GraphNode*, size_t> last_use;
    
    // Analyze the lifetime of each tensor
    for (size_t i = 0; i < execution_order_.size(); ++i) {
        auto node = execution_order_[i];
        
        // Record the usage of input tensors
        for (auto input : node->inputs) {
            if (first_use.find(input) == first_use.end()) {
                first_use[input] = i;
            }
            last_use[input] = i;
        }
        
        // Record the usage of output tensors
        first_use[node] = i;
        last_use[node] = i;
    }
    
    // Create memory blocks
    for (const auto& node : nodes_) {
        if (!node->is_fused) {
            MemoryBlock block;
            block.size = node->output_tensor->size_in_bytes();
            block.users.push_back(node.get());
            memory_blocks_.push_back(block);
        }
    }
}

void GraphCompiler::assign_memory_blocks() {
    size_t total_size = 0;
    
    // Sort memory blocks by size
    std::sort(memory_blocks_.begin(), memory_blocks_.end(),
        [](const MemoryBlock& a, const MemoryBlock& b) {
            return a.size > b.size;
        });
    
    // Allocate offsets
    for (auto& block : memory_blocks_) {
        block.offset = total_size;
        total_size += block.size;
    }
    
    stats_.peak_memory = total_size;
}

void GraphCompiler::reuse_buffers() {
    std::vector<MemoryBlock> optimized_blocks;
    std::vector<bool> used(memory_blocks_.size(), false);
    
    // Try to reuse memory blocks
    for (size_t i = 0; i < memory_blocks_.size(); ++i) {
        if (used[i]) continue;
        
        MemoryBlock merged = memory_blocks_[i];
        used[i] = true;
        
        // Find blocks that can be shared
        for (size_t j = i + 1; j < memory_blocks_.size(); ++j) {
            if (used[j]) continue;
            
            bool can_share = true;
            for (auto user1 : merged.users) {
                for (auto user2 : memory_blocks_[j].users) {
                    // Check if lifetimes overlap
                    if (std::find(user1->outputs.begin(), user1->outputs.end(), user2) != user1->outputs.end() ||
                        std::find(user2->outputs.begin(), user2->outputs.end(), user1) != user2->outputs.end()) {
                        can_share = false;
                        break;
                    }
                }
                if (!can_share) break;
            }
            
            if (can_share) {
                merged.users.insert(merged.users.end(),
                    memory_blocks_[j].users.begin(),
                    memory_blocks_[j].users.end());
                used[j] = true;
            }
        }
        
        optimized_blocks.push_back(merged);
    }
    
    memory_blocks_ = std::move(optimized_blocks);
}

void GraphCompiler::benchmark_kernels() {
    // Benchmark each operator type
    for (const auto& node : nodes_) {
        if (node->is_fused) continue;
        
        std::vector<KernelInfo> kernels;
        
        // Test different kernel implementations
        switch (node->attributes.type) {
            case OpType::MATMUL:
                // Test different matrix multiplication algorithms
                kernels.push_back({"cublas", 0.0, 0});
                kernels.push_back({"custom_gemm", 0.0, 0});
                break;
                
            case OpType::ATTENTION:
                // Test different attention implementations
                kernels.push_back({"flash_attention", 0.0, 0});
                kernels.push_back({"standard_attention", 0.0, 0});
                break;
                
            default:
                continue;
        }
        
        // Run benchmark tests
        for (auto& kernel : kernels) {
            // TODO: Implement actual benchmark tests
            kernel.execution_time = 1.0;  // Example value
            kernel.workspace_size = 1024; // Example value
        }
        
        // Save results
        kernel_database_[node->attributes.name] = kernels;
    }
}

void GraphCompiler::select_best_kernel(GraphNode* node) {
    if (node->is_fused) return;
    
    const auto& kernels = kernel_database_[node->attributes.name];
    if (kernels.empty()) return;
    
    // Select the kernel with the shortest execution time
    auto best_kernel = std::min_element(kernels.begin(), kernels.end(),
        [](const KernelInfo& a, const KernelInfo& b) {
            return a.execution_time < b.execution_time;
        });
    
    node->kernel_type = best_kernel->type;
    node->workspace_size = best_kernel->workspace_size;
}

void GraphCompiler::prepare_workspace() {
    // Calculate the required workspace size
    size_t max_workspace = 0;
    for (const auto& node : nodes_) {
        max_workspace = std::max(max_workspace, node->workspace_size);
    }
    
    // Allocate workspace
    if (max_workspace > 0) {
        workspace_ = std::make_unique<hal::Tensor>(
            std::vector<int64_t>{static_cast<int64_t>(max_workspace)},
            hal::DataType::UINT8,
            device_);
    }
    
    stats_.workspace_size = max_workspace;
}

void GraphCompiler::execute_node(const GraphNode* node) {
    if (node->is_fused) return;
    
    // Prepare inputs and outputs
    std::vector<const hal::Tensor*> inputs;
    for (auto input : node->inputs) {
        inputs.push_back(input->output_tensor.get());
    }
    
    // Execute the calculation
    switch (node->attributes.type) {
        case OpType::MATMUL:
            // Execute matrix multiplication
            // TODO: Implement actual calculation
            break;
            
        case OpType::ATTENTION:
            // Execute attention calculation
            // TODO: Implement actual calculation
            break;
            
        case OpType::LAYERNORM:
            // Execute layer normalization
            // TODO: Implement actual calculation
            break;
            
        default:
            throw std::runtime_error("Unsupported operator type");
    }
}

void GraphCompiler::sync_execution() {
    device_->synchronize();
}

} // namespace deeppowers
