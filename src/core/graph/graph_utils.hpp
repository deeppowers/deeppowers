#pragma once

#include "graph_compiler.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace deeppowers {

// Graph analysis tools
class GraphAnalyzer {
public:
    // Analyze node dependencies
    static std::vector<std::vector<GraphNode*>> analyze_dependencies(
        const std::vector<GraphNode*>& nodes);
        
    // Check for cyclic dependencies
    static bool has_cycles(const std::vector<GraphNode*>& nodes);
    
    // Get topological sort
    static std::vector<GraphNode*> topological_sort(
        const std::vector<GraphNode*>& nodes);
        
    // Calculate critical path
    static std::vector<GraphNode*> critical_path(
        const std::vector<GraphNode*>& nodes);
        
    // Analyze memory usage
    static size_t estimate_peak_memory(
        const std::vector<GraphNode*>& nodes);

private:
    // Helper methods
    static void dfs(GraphNode* node,
                   std::unordered_map<GraphNode*, bool>& visited,
                   std::unordered_map<GraphNode*, bool>& in_stack,
                   bool& has_cycle);
                   
    static void dfs_topological(GraphNode* node,
                              std::unordered_map<GraphNode*, bool>& visited,
                              std::vector<GraphNode*>& result);
};

// Graph optimization tools
class GraphOptimizer {
public:
    // Constant folding
    static void fold_constants(std::vector<GraphNode*>& nodes);
    
    // Operator fusion
    static void fuse_operators(std::vector<GraphNode*>& nodes);
    
    // Eliminate common subexpressions
    static void eliminate_common_subexpressions(std::vector<GraphNode*>& nodes);
    
    // Memory optimization
    static void optimize_memory_layout(std::vector<GraphNode*>& nodes);
    
    // Parallelization optimization
    static void optimize_parallelism(std::vector<GraphNode*>& nodes);

private:
    // Helper methods
    static bool can_fuse(const GraphNode* a, const GraphNode* b);
    static GraphNode* create_fused_node(GraphNode* a, GraphNode* b);
    static bool are_nodes_equivalent(const GraphNode* a, const GraphNode* b);
};

// Memory planning tools
class MemoryPlanner {
public:
    // Allocate memory blocks
    static void allocate_buffers(std::vector<GraphNode*>& nodes);
    
    // Reuse memory
    static void reuse_buffers(std::vector<GraphNode*>& nodes);
    
    // Optimize workspace
    static void optimize_workspace(std::vector<GraphNode*>& nodes);
    
    // Estimate memory requirements
    static size_t estimate_memory_requirements(
        const std::vector<GraphNode*>& nodes);

private:
    // Helper methods
    static void analyze_tensor_lifetimes(
        const std::vector<GraphNode*>& nodes,
        std::unordered_map<GraphNode*, size_t>& first_use,
        std::unordered_map<GraphNode*, size_t>& last_use);
        
    static bool can_share_memory(
        const GraphNode* a,
        const GraphNode* b,
        const std::unordered_map<GraphNode*, size_t>& first_use,
        const std::unordered_map<GraphNode*, size_t>& last_use);
};

// Kernel selection tools
class KernelSelector {
public:
    // Select best kernel
    static std::string select_best_kernel(
        const GraphNode* node,
        const std::vector<std::string>& available_kernels);
        
    // Benchmark kernels
    static void benchmark_kernels(
        const GraphNode* node,
        std::vector<std::pair<std::string, double>>& results);
        
    // Estimate performance
    static double estimate_performance(
        const GraphNode* node,
        const std::string& kernel_type);
        
    // Get kernel configuration
    static void get_launch_config(
        const GraphNode* node,
        const std::string& kernel_type,
        dim3& grid_dim,
        dim3& block_dim,
        size_t& shared_memory);

private:
    // Helper methods
    static bool is_kernel_compatible(
        const GraphNode* node,
        const std::string& kernel_type);
        
    static void profile_kernel(
        const GraphNode* node,
        const std::string& kernel_type,
        double& execution_time,
        size_t& workspace_size);
};

} // namespace deeppowers 