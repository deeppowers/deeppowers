#include "grpc_server.hpp"
#include "deeppowers.grpc.pb.h"
#include <chrono>
#include <sstream>

namespace deeppowers {

// Service implementation class
class GRPCServer::ServiceImpl final : public DeepPowers::Service {
public:
    ServiceImpl(GRPCServer* server) : server_(server) {}

    grpc::Status Generate(
        grpc::ServerContext* context,
        const GenerateRequest* request,
        GenerateResponse* response) override {
        return server_->generate(context, request, response);
    }
    
    grpc::Status GetMetrics(
        grpc::ServerContext* context,
        const MetricsRequest* request,
        MetricsResponse* response) override {
        return server_->get_metrics(context, request, response);
    }
    
    grpc::Status GetSchedulerStatus(
        grpc::ServerContext* context,
        const SchedulerStatusRequest* request,
        SchedulerStatusResponse* response) override {
        return server_->get_scheduler_status(context, request, response);
    }

private:
    GRPCServer* server_;
};

GRPCServer::GRPCServer(
    std::shared_ptr<GPTModel> model,
    std::shared_ptr<Monitor> monitor,
    std::shared_ptr<Scheduler> scheduler,
    const GRPCServerConfig& config)
    : model_(model)
    , monitor_(monitor)
    , scheduler_(scheduler)
    , config_(config) {
}

GRPCServer::~GRPCServer() {
    if (running_) {
        stop();
    }
}

void GRPCServer::start() {
    if (running_) return;
    
    // Initialize server
    init_server();
    
    // Initialize SSL if enabled
    if (config_.enable_ssl) {
        init_ssl();
    }
    
    // Setup service
    setup_service();
    
    // Start server
    server_ = builder_->BuildAndStart();
    if (!server_) {
        throw std::runtime_error("Failed to start gRPC server");
    }
    
    running_ = true;
    
    // Start completion queue threads
    completion_threads_.resize(config_.num_threads);
    for (size_t i = 0; i < config_.num_threads; ++i) {
        completion_threads_[i] = std::thread([this]() {
            void* tag;
            bool ok;
            while (running_) {
                if (server_->CompletionQueue()->Next(&tag, &ok)) {
                    if (ok) {
                        // Process completed request
                        auto* call = static_cast<AsyncCall*>(tag);
                        call->Proceed();
                    }
                }
            }
        });
    }
}

void GRPCServer::stop() {
    if (!running_) return;
    
    running_ = false;
    
    // Stop server
    server_->Shutdown();
    
    // Wait for completion threads
    for (auto& thread : completion_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    completion_threads_.clear();
    
    // Cleanup
    service_.reset();
    server_.reset();
    builder_.reset();
}

void GRPCServer::update_config(const GRPCServerConfig& config) {
    if (running_) {
        throw std::runtime_error("Cannot update config while server is running");
    }
    config_ = config;
}

void GRPCServer::init_server() {
    builder_ = std::make_unique<grpc::ServerBuilder>();
    
    // Set server address
    std::string server_address = config_.host + ":" + std::to_string(config_.port);
    builder_->AddListeningPort(server_address, grpc::InsecureServerCredentials());
    
    // Set server options
    builder_->SetMaxMessageSize(config_.max_message_size);
    builder_->SetMaxReceiveMessageSize(config_.max_message_size);
    builder_->SetMaxSendMessageSize(config_.max_message_size);
    
    // Set keepalive options
    grpc::KeepAliveOptions keepalive;
    keepalive.keepalive_time_ms = config_.keepalive_time_ms;
    keepalive.keepalive_timeout_ms = 20000;
    keepalive.keepalive_permit_without_calls = true;
    builder_->SetKeepaliveOptions(keepalive);
}

void GRPCServer::init_ssl() {
    grpc::SslServerCredentialsOptions ssl_opts;
    ssl_opts.pem_key_cert_pairs.push_back({
        config_.ssl_key,    // Private key
        config_.ssl_cert    // Certificate chain
    });
    
    auto creds = grpc::SslServerCredentials(ssl_opts);
    builder_->AddListeningPort(
        config_.host + ":" + std::to_string(config_.port),
        creds);
}

void GRPCServer::setup_service() {
    service_ = std::make_unique<ServiceImpl>(this);
    builder_->RegisterService(service_.get());
}

grpc::Status GRPCServer::generate(
    grpc::ServerContext* context,
    const GenerateRequest* request,
    GenerateResponse* response) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Create model request
        Request model_request(
            request->request_id(),
            request->prompt(),
            RequestPriority::NORMAL);
        
        model_request.mutable_config().max_tokens = request->max_tokens();
        model_request.mutable_config().temperature = request->temperature();
        model_request.mutable_config().top_p = request->top_p();
        
        // Submit request to scheduler
        auto request_ptr = std::make_shared<Request>(model_request);
        scheduler_->submit_request(request_ptr);
        
        // Wait for completion
        while (request_ptr->status() != RequestStatus::COMPLETED &&
               request_ptr->status() != RequestStatus::FAILED) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Check for deadline exceeded
            if (context->IsCancelled()) {
                return create_error_status(
                    grpc::StatusCode::DEADLINE_EXCEEDED,
                    "Request deadline exceeded");
            }
        }
        
        // Check for failure
        if (request_ptr->status() == RequestStatus::FAILED) {
            return create_error_status(
                grpc::StatusCode::INTERNAL,
                request_ptr->result().error_message);
        }
        
        // Set response
        response->set_text(request_ptr->result().generated_texts[0]);
        for (const auto& logprob : request_ptr->result().logprobs) {
            response->add_logprobs(logprob);
        }
        
        // Record metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        record_request_metrics(
            "Generate",
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time));
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        handle_error(e.what());
        return create_error_status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what());
    }
}

grpc::Status GRPCServer::get_metrics(
    grpc::ServerContext* context,
    const MetricsRequest* request,
    MetricsResponse* response) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Get metrics
        auto hardware_metrics = monitor_->get_hardware_metrics();
        auto latency_metrics = monitor_->get_latency_metrics();
        auto throughput_metrics = monitor_->get_throughput_metrics();
        auto error_metrics = monitor_->get_error_metrics();
        
        // Set hardware metrics
        auto* hw = response->mutable_hardware();
        hw->set_gpu_utilization(hardware_metrics.gpu_compute_util);
        hw->set_gpu_memory_used_mb(hardware_metrics.gpu_memory_used_mb);
        hw->set_gpu_temperature(hardware_metrics.gpu_temperature);
        hw->set_host_memory_used_mb(hardware_metrics.host_memory_used_mb);
        
        // Set latency metrics
        auto* latency = response->mutable_latency();
        latency->set_avg_ms(latency_metrics.avg_latency_ms);
        latency->set_p50_ms(latency_metrics.p50_latency_ms);
        latency->set_p90_ms(latency_metrics.p90_latency_ms);
        latency->set_p99_ms(latency_metrics.p99_latency_ms);
        
        // Set throughput metrics
        auto* throughput = response->mutable_throughput();
        throughput->set_requests_per_second(throughput_metrics.requests_per_second);
        throughput->set_tokens_per_second(throughput_metrics.tokens_per_second);
        
        // Set error metrics
        auto* errors = response->mutable_errors();
        errors->set_total_errors(error_metrics.num_errors);
        errors->set_timeout_errors(error_metrics.num_timeouts);
        errors->set_oom_errors(error_metrics.num_oom_errors);
        
        // Record metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        record_request_metrics(
            "GetMetrics",
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time));
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        handle_error(e.what());
        return create_error_status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what());
    }
}

grpc::Status GRPCServer::get_scheduler_status(
    grpc::ServerContext* context,
    const SchedulerStatusRequest* request,
    SchedulerStatusResponse* response) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Get scheduler stats
        const auto& stats = scheduler_->get_stats();
        
        // Set response
        response->set_active_requests(stats.active_requests);
        response->set_total_requests(stats.total_requests);
        response->set_dropped_requests(stats.dropped_requests);
        response->set_avg_latency_ms(stats.avg_latency_ms);
        response->set_avg_throughput(stats.avg_throughput);
        
        // Record metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        record_request_metrics(
            "GetSchedulerStatus",
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time));
        
        return grpc::Status::OK;
        
    } catch (const std::exception& e) {
        handle_error(e.what());
        return create_error_status(
            grpc::StatusCode::INTERNAL,
            std::string("Internal error: ") + e.what());
    }
}

void GRPCServer::handle_error(const std::string& error) {
    // Update error metrics
    monitor_->record_error("grpc_server", error);
    
    // Log error
    std::cerr << "[ERROR] " << error << std::endl;
}

grpc::Status GRPCServer::create_error_status(
    grpc::StatusCode code,
    const std::string& message) {
    
    return grpc::Status(code, message);
}

void GRPCServer::record_request_metrics(
    const std::string& method,
    const std::chrono::microseconds& duration) {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto& metrics = method_metrics_[method];
    metrics.request_count++;
    
    // Update average latency
    double latency_ms = duration.count() / 1000.0;
    metrics.avg_latency_ms = 
        (metrics.avg_latency_ms * (metrics.request_count - 1) + latency_ms) /
        metrics.request_count;
}

} // namespace deeppowers 