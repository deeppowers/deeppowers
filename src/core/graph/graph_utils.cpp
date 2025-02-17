#include "graph_utils.hpp"
#include <algorithm>
#include <queue>
#include <stack>
#include <limits>

namespace deeppowers {

// GraphAnalyzer implementation
std::vector<std::vector<GraphNode*>> GraphAnalyzer::analyze_dependencies(
    const std::vector<GraphNode*>& nodes) {
    
    std::vector<std::vector<GraphNode*>> levels;
    std::unordered_map<GraphNode*, int> in_degree;
    std::unordered_map<GraphNode*, std::vector<GraphNode*>> deps;
    
    // Calculate in-degree
    for (auto node : nodes) {
        for (auto input : node->inputs) {
            deps[input].push_back(node);
            in_degree[node]++;
        }
    }
    
        // Use BFS to traverse by levels
    std::queue<GraphNode*> q;
    for (auto node : nodes) {
        if (in_degree[node] == 0) {
            q.push(node);
        }
    }
    
    while (!q.empty()) {
        int level_size = q.size();
        std::vector<GraphNode*> level;
        
        for (int i = 0; i < level_size; ++i) {
            auto node = q.front();
            q.pop();
            level.push_back(node);
            
            for (auto dep : deps[node]) {
                if (--in_degree[dep] == 0) {
                    q.push(dep);
                }
            }
        }
        
        levels.push_back(level);
    }
    
    return levels;
}

bool GraphAnalyzer::has_cycles(const std::vector<GraphNode*>& nodes) {
    std::unordered_map<GraphNode*, bool> visited;
    std::unordered_map<GraphNode*, bool> in_stack;
    bool has_cycle = false;
    
    for (auto node : nodes) {
        if (!visited[node]) {
            dfs(node, visited, in_stack, has_cycle);
            if (has_cycle) return true;
        }
    }
    
    return false;
}

std::vector<GraphNode*> GraphAnalyzer::topological_sort(
    const std::vector<GraphNode*>& nodes) {
    
    std::vector<GraphNode*> result;
    std::unordered_map<GraphNode*, bool> visited;
    
    for (auto node : nodes) {
        if (!visited[node]) {
            dfs_topological(node, visited, result);
        }
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<GraphNode*> GraphAnalyzer::critical_path(
    const std::vector<GraphNode*>& nodes) {
    
    // Calculate earliest completion time for each node
    std::unordered_map<GraphNode*, double> earliest;
    for (auto node : nodes) {
        earliest[node] = 0.0;
        for (auto input : node->inputs) {
            earliest[node] = std::max(earliest[node],
                earliest[input] + input->workspace_size);
        }
    }
    
    // Calculate latest start time for each node
    std::unordered_map<GraphNode*, double> latest;
    for (auto node : nodes) {
        latest[node] = std::numeric_limits<double>::infinity();
    }
    
    auto sorted = topological_sort(nodes);
    std::reverse(sorted.begin(), sorted.end());
    
    double max_time = 0.0;
    for (auto node : sorted) {
        max_time = std::max(max_time, earliest[node]);
    }
    
    for (auto node : sorted) {
        latest[node] = max_time;
        for (auto output : node->outputs) {
            latest[node] = std::min(latest[node],
                latest[output] - node->workspace_size);
        }
    }
    
    // Find critical path
    std::vector<GraphNode*> path;
    auto current = sorted.front();
    while (current) {
        path.push_back(current);
        
        GraphNode* next = nullptr;
        double min_slack = std::numeric_limits<double>::infinity();
        
        for (auto output : current->outputs) {
            double slack = latest[output] - earliest[current] - current->workspace_size;
            if (slack < min_slack) {
                min_slack = slack;
                next = output;
            }
        }
        
        current = next;
    }
    
    return path;
}

size_t GraphAnalyzer::estimate_peak_memory(
    const std::vector<GraphNode*>& nodes) {
    
    size_t peak = 0;
    size_t current = 0;
    
    auto sorted = topological_sort(nodes);
    std::unordered_map<GraphNode*, int> ref_count;
    
    // Calculate reference count
    for (auto node : nodes) {
        for (auto input : node->inputs) {
            ref_count[input]++;
        }
    }
    
    // Simulate memory usage during execution
    for (auto node : sorted) {
        // Allocate output memory
        current += node->output_tensor->size_in_bytes();
        peak = std::max(peak, current);
        
        // Release unused input memory
        for (auto input : node->inputs) {
            if (--ref_count[input] == 0) {
                current -= input->output_tensor->size_in_bytes();
            }
        }
    }
    
    return peak;
}

void GraphAnalyzer::dfs(
    GraphNode* node,
    std::unordered_map<GraphNode*, bool>& visited,
    std::unordered_map<GraphNode*, bool>& in_stack,
    bool& has_cycle) {
    
    visited[node] = true;
    in_stack[node] = true;
    
    for (auto dep : node->outputs) {
        if (!visited[dep]) {
            dfs(dep, visited, in_stack, has_cycle);
        } else if (in_stack[dep]) {
            has_cycle = true;
            return;
        }
    }
    
    in_stack[node] = false;
}

void GraphAnalyzer::dfs_topological(
    GraphNode* node,
    std::unordered_map<GraphNode*, bool>& visited,
    std::vector<GraphNode*>& result) {
    
    visited[node] = true;
    
    for (auto dep : node->outputs) {
        if (!visited[dep]) {
            dfs_topological(dep, visited, result);
        }
    }
    
    result.push_back(node);
}

// GraphOptimizer implementation
void GraphOptimizer::fold_constants(std::vector<GraphNode*>& nodes) {
    bool changed;
    do {
        changed = false;
        
        for (auto node : nodes) {
            // Check if all inputs are constants
            bool all_constant = true;
            for (auto input : node->inputs) {
                if (!input->is_constant) {
                    all_constant = false;
                    break;
                }
            }
            
            if (all_constant && !node->is_constant) {
                // TODO: Implement constant calculation
                node->is_constant = true;
                changed = true;
            }
        }
    } while (changed);
}

void GraphOptimizer::fuse_operators(std::vector<GraphNode*>& nodes) {
    bool changed;
    do {
        changed = false;
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (can_fuse(nodes[i], nodes[j])) {
                    auto fused = create_fused_node(nodes[i], nodes[j]);
                    nodes[i] = fused;
                    nodes.erase(nodes.begin() + j);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    } while (changed);
}

void GraphOptimizer::eliminate_common_subexpressions(
    std::vector<GraphNode*>& nodes) {
    
    bool changed;
    do {
        changed = false;
        
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (are_nodes_equivalent(nodes[i], nodes[j])) {
                    // Redirect all outputs of j to i
                    for (auto output : nodes[j]->outputs) {
                        for (auto& input : output->inputs) {
                            if (input == nodes[j]) {
                                input = nodes[i];
                            }
                        }
                    }
                    nodes.erase(nodes.begin() + j);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    } while (changed);
}

void GraphOptimizer::optimize_memory_layout(std::vector<GraphNode*>& nodes) {
    // Use MemoryPlanner for memory optimization
    MemoryPlanner::allocate_buffers(nodes);
    MemoryPlanner::reuse_buffers(nodes);
    MemoryPlanner::optimize_workspace(nodes);
}

void GraphOptimizer::optimize_parallelism(std::vector<GraphNode*>& nodes) {
    // Analyze dependencies
    auto levels = GraphAnalyzer::analyze_dependencies(nodes);
    
    // Optimize nodes in each level in parallel
    for (auto& level : levels) {
        // TODO: Implement parallel optimization strategy
    }
}

bool GraphOptimizer::can_fuse(const GraphNode* a, const GraphNode* b) {
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

GraphNode* GraphOptimizer::create_fused_node(GraphNode* a, GraphNode* b) {
    auto fused = new GraphNode();
    fused->attributes.name = a->attributes.name + "_" + b->attributes.name;
    fused->attributes.type = a->attributes.type;
    fused->attributes.input_shapes = a->attributes.input_shapes;
    fused->attributes.output_shapes = b->attributes.output_shapes;
    
    fused->inputs = a->inputs;
    fused->outputs = b->outputs;
    
    return fused;
}

bool GraphOptimizer::are_nodes_equivalent(const GraphNode* a, const GraphNode* b) {
    if (a->attributes.type != b->attributes.type) return false;
    if (a->attributes.input_shapes != b->attributes.input_shapes) return false;
    if (a->attributes.output_shapes != b->attributes.output_shapes) return false;
    
    // Check if inputs are the same
    if (a->inputs.size() != b->inputs.size()) return false;
    for (size_t i = 0; i < a->inputs.size(); ++i) {
        if (a->inputs[i] != b->inputs[i]) return false;
    }
    
    return true;
}

// MemoryPlanner implementation
void MemoryPlanner::allocate_buffers(std::vector<GraphNode*>& nodes) {
    // Analyze tensor lifetimes
    std::unordered_map<GraphNode*, size_t> first_use;
    std::unordered_map<GraphNode*, size_t> last_use;
    analyze_tensor_lifetimes(nodes, first_use, last_use);
    
    // Sort nodes by size
    std::sort(nodes.begin(), nodes.end(),
        [](const GraphNode* a, const GraphNode* b) {
            return a->output_tensor->size_in_bytes() > b->output_tensor->size_in_bytes();
        });
    
    // Allocate memory blocks
    size_t offset = 0;
    for (auto node : nodes) {
        // TODO: Implement actual memory allocation
        offset += node->output_tensor->size_in_bytes();
    }
}

void MemoryPlanner::reuse_buffers(std::vector<GraphNode*>& nodes) {
    std::unordered_map<GraphNode*, size_t> first_use;
    std::unordered_map<GraphNode*, size_t> last_use;
    analyze_tensor_lifetimes(nodes, first_use, last_use);
    
    // Try to reuse memory
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            if (can_share_memory(nodes[i], nodes[j], first_use, last_use)) {
                // TODO: Implement memory reuse
            }
        }
    }
}

void MemoryPlanner::optimize_workspace(std::vector<GraphNode*>& nodes) {
    size_t total_workspace = 0;
    
    // Calculate workspace requirements for each node
    for (auto node : nodes) {
        // TODO: Implement workspace optimization
        total_workspace = std::max(total_workspace, node->workspace_size);
    }
}

size_t MemoryPlanner::estimate_memory_requirements(
    const std::vector<GraphNode*>& nodes) {
    
    return GraphAnalyzer::estimate_peak_memory(nodes);
}

void MemoryPlanner::analyze_tensor_lifetimes(
    const std::vector<GraphNode*>& nodes,
    std::unordered_map<GraphNode*, size_t>& first_use,
    std::unordered_map<GraphNode*, size_t>& last_use) {
    
    auto sorted = GraphAnalyzer::topological_sort(nodes);
    
    for (size_t i = 0; i < sorted.size(); ++i) {
        auto node = sorted[i];
        
        // Record usage of input tensors
        for (auto input : node->inputs) {
            if (first_use.find(input) == first_use.end()) {
                first_use[input] = i;
            }
            last_use[input] = i;
        }
        
        // Record usage of output tensor
        first_use[node] = i;
        last_use[node] = i;
    }
}

bool MemoryPlanner::can_share_memory(
    const GraphNode* a,
    const GraphNode* b,
    const std::unordered_map<GraphNode*, size_t>& first_use,
    const std::unordered_map<GraphNode*, size_t>& last_use) {
    
    // Check if lifetimes overlap
    auto a_first = first_use.at(a);
    auto a_last = last_use.at(a);
    auto b_first = first_use.at(b);
    auto b_last = last_use.at(b);
    
    return (a_last < b_first) || (b_last < a_first);
}

// KernelSelector implementation
std::string KernelSelector::select_best_kernel(
    const GraphNode* node,
    const std::vector<std::string>& available_kernels) {
    
    std::string best_kernel;
    double best_performance = std::numeric_limits<double>::infinity();
    
    for (const auto& kernel : available_kernels) {
        if (is_kernel_compatible(node, kernel)) {
            double perf = estimate_performance(node, kernel);
            if (perf < best_performance) {
                best_performance = perf;
                best_kernel = kernel;
            }
        }
    }
    
    return best_kernel;
}

void KernelSelector::benchmark_kernels(
    const GraphNode* node,
    std::vector<std::pair<std::string, double>>& results) {
    
    results.clear();
    
    // Benchmark each available kernel
    std::vector<std::string> kernels = {
        "cublas", "custom_gemm",  // Matrix multiplication
        "flash_attention", "standard_attention",  // Attention calculation
        "fused_layernorm", "separate_layernorm"  // Layer normalization
    };
    
    for (const auto& kernel : kernels) {
        if (is_kernel_compatible(node, kernel)) {
            double time;
            size_t workspace;
            profile_kernel(node, kernel, time, workspace);
            results.emplace_back(kernel, time);
        }
    }
    
    // Sort by execution time
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
}

double KernelSelector::estimate_performance(
    const GraphNode* node,
    const std::string& kernel_type) {
    
    // TODO: Implement performance estimation
    return 1.0;
}

void KernelSelector::get_launch_config(
    const GraphNode* node,
    const std::string& kernel_type,
    dim3& grid_dim,
    dim3& block_dim,
    size_t& shared_memory) {
    
    // Set launch configuration based on node type and kernel type
    switch (node->attributes.type) {
        case OpType::MATMUL:
            block_dim = dim3(16, 16);
            grid_dim = dim3(
                (node->attributes.output_shapes[0] + block_dim.x - 1) / block_dim.x,
                (node->attributes.output_shapes[1] + block_dim.y - 1) / block_dim.y);
            shared_memory = 2 * block_dim.x * block_dim.y * sizeof(float);
            break;
            
        case OpType::ATTENTION:
            block_dim = dim3(256);
            grid_dim = dim3(
                (node->attributes.output_shapes[2] + block_dim.x - 1) / block_dim.x,
                node->attributes.output_shapes[1],
                node->attributes.output_shapes[0]);
            shared_memory = block_dim.x * sizeof(float) * 3;  // scores, max, sum
            break;
            
        default:
            block_dim = dim3(256);
            grid_dim = dim3(
                (node->output_tensor->size_in_bytes() + block_dim.x - 1) / block_dim.x);
            shared_memory = 0;
            break;
    }
}

bool KernelSelector::is_kernel_compatible(
    const GraphNode* node,
    const std::string& kernel_type) {
    
    // Check if kernel type is compatible with node type
    switch (node->attributes.type) {
        case OpType::MATMUL:
            return kernel_type == "cublas" || kernel_type == "custom_gemm";
            
        case OpType::ATTENTION:
            return kernel_type == "flash_attention" || kernel_type == "standard_attention";
            
        case OpType::LAYERNORM:
            return kernel_type == "fused_layernorm" || kernel_type == "separate_layernorm";
            
        default:
            return false;
    }
}

void KernelSelector::profile_kernel(
    const GraphNode* node,
    const std::string& kernel_type,
    double& execution_time,
    size_t& workspace_size) {
    
    // TODO: Implement actual performance profiling
    execution_time = 1.0;
    workspace_size = 1024;
}

} // namespace deeppowers 