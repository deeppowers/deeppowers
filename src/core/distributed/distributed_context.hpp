#pragma once

#include "../hal/hal.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace deeppowers {

// Distributed computing mode
enum class DistributedMode {
    DATA_PARALLEL,      // Data parallel
    MODEL_PARALLEL,     // Model parallel
    PIPELINE_PARALLEL,  // Pipeline parallel
    HYBRID             // Hybrid parallel
};

// Node role
enum class NodeRole {
    MASTER,    // Master node
    WORKER     // Worker node
};

// Distributed configuration
struct DistributedConfig {
    DistributedMode mode = DistributedMode::DATA_PARALLEL;  // Parallel mode
    size_t world_size = 1;                                  // Total number of nodes
    size_t local_rank = 0;                                  // Local rank
    size_t global_rank = 0;                                // Global rank
    NodeRole role = NodeRole::WORKER;                      // Node role
    std::string master_addr = "localhost";                 // Master node address
    int master_port = 29500;                              // Master node port
    size_t num_gpus_per_node = 1;                         // Number of GPUs per node
    bool enable_grad_sync = true;                         // Whether to enable gradient synchronization
    size_t pipeline_stages = 1;                           // Number of pipeline stages
    size_t micro_batch_size = 1;                          // Micro batch size
};

// Distributed context class
class DistributedContext {
public:
    explicit DistributedContext(const DistributedConfig& config);
    ~DistributedContext();

    // Initialize and clean up
    void initialize();
    void finalize();
    
    // Communication primitives
    void all_reduce(void* data, size_t size, const std::string& name = "");
    void all_gather(void* data, size_t size, const std::string& name = "");
    void broadcast(void* data, size_t size, int root = 0);
    void send(void* data, size_t size, int dst);
    void recv(void* data, size_t size, int src);
    
    // Synchronization primitives
    void barrier();
    void wait_event(const std::string& event);
    void signal_event(const std::string& event);
    
    // Device management
    hal::Device* get_local_device(size_t device_id = 0);
    std::vector<hal::Device*> get_all_devices();
    
    // Status query
    bool is_master() const { return config_.role == NodeRole::MASTER; }
    bool is_worker() const { return config_.role == NodeRole::WORKER; }
    size_t get_world_size() const { return config_.world_size; }
    size_t get_local_rank() const { return config_.local_rank; }
    size_t get_global_rank() const { return config_.global_rank; }
    DistributedMode get_mode() const { return config_.mode; }
    
    // Configuration access and update
    const DistributedConfig& config() const { return config_; }
    void update_config(const DistributedConfig& config);

private:
    // Internal helper methods
    void init_communication();
    void init_devices();
    void create_communicators();
    void setup_pipeline();
    
    // Member variables
    DistributedConfig config_;
    std::vector<std::unique_ptr<hal::Device>> devices_;
    std::unordered_map<std::string, void*> communicators_;
    bool initialized_ = false;
};

} // namespace deeppowers 