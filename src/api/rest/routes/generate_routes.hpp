#pragma once

#include "../../../core/api/http_server.hpp"
#include <httplib.h>

namespace deeppowers {
namespace api {
namespace rest {

// Route configuration for text generation endpoints
struct GenerateRouteConfig {
    std::string base_path = "/api/v1";  // Base path for all generation endpoints
    size_t max_batch_size = 32;         // Maximum batch size for batch generation
    bool enable_streaming = true;        // Enable streaming generation
    size_t stream_chunk_size = 16;      // Number of tokens per stream chunk
};

// Register generation routes
void register_generate_routes(
    httplib::Server& server,
    const GenerateRouteConfig& config = GenerateRouteConfig());

// Route handlers
void handle_generate(const httplib::Request& req, httplib::Response& res);
void handle_generate_stream(const httplib::Request& req, httplib::Response& res);
void handle_generate_batch(const httplib::Request& req, httplib::Response& res);
void handle_generate_async(const httplib::Request& req, httplib::Response& res);

} // namespace rest
} // namespace api
} // namespace deeppowers