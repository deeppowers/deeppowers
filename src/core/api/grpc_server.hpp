#pragma once

#include "../execution/models/gpt_model.hpp"
#include "../monitoring/monitor.hpp"
#include "../scheduling/scheduler.hpp"
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <thread>
#include <mutex>

namespace deeppowers {

// gRPC server configuration
struct GRPCServerConfig {
    std::string host = "0.0.0.0";     // Server host
    int port = 50051;                 // Server port
    size_t num_threads = 4;           // Number of worker threads
    bool enable_ssl = false;          // Enable TLS
    std::string ssl_cert;             // TLS certificate path
    std::string ssl_key;              // TLS private key path
    size_t max_message_size = 64*1024*1024;  // Maximum message size in bytes
    size_t keepalive_time_ms = 7200000;      // Keepalive time in milliseconds
};

// gRPC service implementation
class GRPCServer {
public:
    explicit GRPCServer(
        std::shared_ptr<GPTModel> model,
        std::shared_ptr<Monitor> monitor,
        std::shared_ptr<Scheduler> scheduler,
        const GRPCServerConfig& config);
    ~GRPCServer();

    // Server lifecycle
    void start();
    void stop();
    bool is_running() const { return running_; }
    
    // Configuration
    const GRPCServerConfig& config() const { return config_; }
    void update_config(const GRPCServerConfig& config);

private:
    // Internal helper methods
    void init_server();
    void init_ssl();
    void setup_service();
    
    // Service implementation methods
    grpc::Status generate(
        grpc::ServerContext* context,
        const GenerateRequest* request,
        GenerateResponse* response);
        
    grpc::Status get_metrics(
        grpc::ServerContext* context,
        const MetricsRequest* request,
        MetricsResponse* response);
        
    grpc::Status get_scheduler_status(
        grpc::ServerContext* context,
        const SchedulerStatusRequest* request,
        SchedulerStatusResponse* response);
    
    // Error handling
    void handle_error(const std::string& error);
    grpc::Status create_error_status(
        grpc::StatusCode code,
        const std::string& message);
    
    // Monitoring
    void record_request_metrics(
        const std::string& method,
        const std::chrono::microseconds& duration);
    
    // Member variables
    std::shared_ptr<GPTModel> model_;
    std::shared_ptr<Monitor> monitor_;
    std::shared_ptr<Scheduler> scheduler_;
    GRPCServerConfig config_;
    bool running_ = false;
    
    // gRPC server
    std::unique_ptr<grpc::Server> server_;
    std::unique_ptr<grpc::ServerBuilder> builder_;
    
    // Service implementation
    class ServiceImpl;
    std::unique_ptr<ServiceImpl> service_;
    
    // Thread management
    std::vector<std::thread> completion_threads_;
    std::mutex server_mutex_;
    
    // Metrics
    struct MethodMetrics {
        size_t request_count = 0;
        double avg_latency_ms = 0.0;
        size_t error_count = 0;
    };
    std::unordered_map<std::string, MethodMetrics> method_metrics_;
    std::mutex metrics_mutex_;
};

} // namespace deeppowers 