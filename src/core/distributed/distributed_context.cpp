#include "distributed_context.hpp"
#include <nccl.h>
#include <mpi.h>
#include <stdexcept>
#include <cuda_runtime.h>

namespace deeppowers {

DistributedContext::DistributedContext(const DistributedConfig& config)
    : config_(config) {
}

DistributedContext::~DistributedContext() {
    if (initialized_) {
        finalize();
    }
}

void DistributedContext::initialize() {
    if (initialized_) return;
    
    // Initialize MPI
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        throw std::runtime_error("MPI implementation does not support MPI_THREAD_MULTIPLE");
    }
    
    // Get MPI information
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Update configuration
    config_.world_size = world_size;
    config_.global_rank = rank;
    config_.local_rank = rank % config_.num_gpus_per_node;
    
    // Initialize devices
    init_devices();
    
    // Initialize communication
    init_communication();
    
    // Create communicators
    create_communicators();
    
    // If it is pipeline parallel mode, set the pipeline
    if (config_.mode == DistributedMode::PIPELINE_PARALLEL ||
        config_.mode == DistributedMode::HYBRID) {
        setup_pipeline();
    }
    
    initialized_ = true;
}

void DistributedContext::finalize() {
    if (!initialized_) return;
    
    // Clean up NCCL communicators
    for (auto& [name, comm] : communicators_) {
        if (comm) {
            ncclCommDestroy(static_cast<ncclComm_t>(comm));
        }
    }
    communicators_.clear();
    
    // Clean up devices
    devices_.clear();
    
    // Terminate MPI
    MPI_Finalize();
    
    initialized_ = false;
}

void DistributedContext::all_reduce(void* data, size_t size, const std::string& name) {
    if (!initialized_) throw std::runtime_error("DistributedContext not initialized");
    
    // Get NCCL communicator
    auto comm_it = communicators_.find(name.empty() ? "default" : name);
    if (comm_it == communicators_.end()) {
        throw std::runtime_error("Communicator not found: " + name);
    }
    ncclComm_t comm = static_cast<ncclComm_t>(comm_it->second);
    
    // Execute all-reduce operation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    ncclAllReduce(data, data, size, ncclFloat32, ncclSum, comm, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void DistributedContext::all_gather(void* data, size_t size, const std::string& name) {
    if (!initialized_) throw std::runtime_error("DistributedContext not initialized");
    
    // Get NCCL communicator
    auto comm_it = communicators_.find(name.empty() ? "default" : name);
    if (comm_it == communicators_.end()) {
        throw std::runtime_error("Communicator not found: " + name);
    }
    ncclComm_t comm = static_cast<ncclComm_t>(comm_it->second);
    
    // Allocate receive buffer
    void* recv_buff;
    cudaMalloc(&recv_buff, size * config_.world_size);
    
    // Execute all-gather operation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    ncclAllGather(data, recv_buff, size, ncclFloat32, comm, stream);
    
    cudaStreamSynchronize(stream);
    cudaMemcpy(data, recv_buff, size * config_.world_size, cudaMemcpyDeviceToDevice);
    
    cudaFree(recv_buff);
    cudaStreamDestroy(stream);
}

void DistributedContext::broadcast(void* data, size_t size, int root) {
    if (!initialized_) throw std::runtime_error("DistributedContext not initialized");
    
    // Get default NCCL communicator
    auto comm_it = communicators_.find("default");
    if (comm_it == communicators_.end()) {
        throw std::runtime_error("Default communicator not found");
    }
    ncclComm_t comm = static_cast<ncclComm_t>(comm_it->second);
    
    // Execute broadcast operation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    ncclBroadcast(data, data, size, ncclFloat32, root, comm, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void DistributedContext::send(void* data, size_t size, int dst) {
    if (!initialized_) throw std::runtime_error("DistributedContext not initialized");
    
    // Get default NCCL communicator
    auto comm_it = communicators_.find("default");
    if (comm_it == communicators_.end()) {
        throw std::runtime_error("Default communicator not found");
    }
    ncclComm_t comm = static_cast<ncclComm_t>(comm_it->second);
    
    // Execute send operation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    ncclSend(data, size, ncclFloat32, dst, comm, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void DistributedContext::recv(void* data, size_t size, int src) {
    if (!initialized_) throw std::runtime_error("DistributedContext not initialized");
    
    // Get default NCCL communicator
    auto comm_it = communicators_.find("default");
    if (comm_it == communicators_.end()) {
        throw std::runtime_error("Default communicator not found");
    }
    ncclComm_t comm = static_cast<ncclComm_t>(comm_it->second);
    
    // Execute recv operation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    ncclRecv(data, size, ncclFloat32, src, comm, stream);
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void DistributedContext::barrier() {
    if (!initialized_) throw std::runtime_error("DistributedContext not initialized");
    MPI_Barrier(MPI_COMM_WORLD);
}

void DistributedContext::wait_event(const std::string& event) {
    // TODO: Implement event waiting mechanism
}

void DistributedContext::signal_event(const std::string& event) {
    // TODO: Implement event signaling mechanism
}

hal::Device* DistributedContext::get_local_device(size_t device_id) {
    if (device_id >= devices_.size()) {
        throw std::runtime_error("Invalid device ID");
    }
    return devices_[device_id].get();
}

std::vector<hal::Device*> DistributedContext::get_all_devices() {
    std::vector<hal::Device*> result;
    for (auto& device : devices_) {
        result.push_back(device.get());
    }
    return result;
}

void DistributedContext::update_config(const DistributedConfig& config) {
    if (initialized_) {
        throw std::runtime_error("Cannot update config while context is initialized");
    }
    config_ = config;
}

void DistributedContext::init_devices() {
    // Get the number of available GPUs
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count < config_.num_gpus_per_node) {
        throw std::runtime_error("Not enough GPUs available");
    }
    
    // Create a device object for each GPU
    for (size_t i = 0; i < config_.num_gpus_per_node; ++i) {
        cudaSetDevice(i);
        devices_.push_back(std::make_unique<hal::Device>(hal::DeviceType::CUDA, i));
    }
    
    // Set the current device to the device corresponding to local_rank
    cudaSetDevice(config_.local_rank);
}

void DistributedContext::init_communication() {
    // Initialize NCCL
    ncclUniqueId nccl_id;
    if (config_.global_rank == 0) {
        ncclGetUniqueId(&nccl_id);
    }
    
    // Broadcast NCCL ID
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Create default NCCL communicator
    ncclComm_t comm;
    ncclCommInitRank(&comm, config_.world_size, nccl_id, config_.global_rank);
    
    communicators_["default"] = comm;
}

void DistributedContext::create_communicators() {
    // Create additional communicators based on parallel mode
    switch (config_.mode) {
        case DistributedMode::MODEL_PARALLEL: {
            // Create model parallel group communicator
            int color = config_.global_rank / (config_.world_size / config_.pipeline_stages);
            MPI_Comm model_comm;
            MPI_Comm_split(MPI_COMM_WORLD, color, config_.global_rank, &model_comm);
            break;
        }
        case DistributedMode::PIPELINE_PARALLEL: {
            // Create pipeline parallel group communicator
            int color = config_.global_rank % config_.pipeline_stages;
            MPI_Comm pipeline_comm;
            MPI_Comm_split(MPI_COMM_WORLD, color, config_.global_rank, &pipeline_comm);
            break;
        }
        case DistributedMode::HYBRID: {
            // Create communicators required for hybrid parallel
            // TODO: Implement hybrid parallel communicators creation
            break;
        }
        default:
            break;
    }
}

void DistributedContext::setup_pipeline() {
    if (config_.pipeline_stages <= 1) return;
    
    // Calculate the number of model shards per stage
    size_t layers_per_stage = config_.num_layers / config_.pipeline_stages;
    size_t extra_layers = config_.num_layers % config_.pipeline_stages;
    
    // Allocate the number of layers per stage
    std::vector<size_t> stage_layers(config_.pipeline_stages, layers_per_stage);
    for (size_t i = 0; i < extra_layers; ++i) {
        stage_layers[i]++;
    }
    
    // Set the pipeline stage information for the current node
    size_t stage_id = config_.global_rank % config_.pipeline_stages;
    size_t start_layer = 0;
    for (size_t i = 0; i < stage_id; ++i) {
        start_layer += stage_layers[i];
    }
    
    // TODO: Store pipeline stage information for subsequent calculations
}

} // namespace deeppowers 