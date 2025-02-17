#include "generate_routes.hpp"
#include "../middleware/auth_middleware.hpp"
#include "../middleware/rate_limit_middleware.hpp"
#include "../handlers/generate_handler.hpp"
#include <nlohmann/json.hpp>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;

void register_generate_routes(httplib::Server& server, const GenerateRouteConfig& config) {
    // Single text generation
    server.Post(config.base_path + "/generate", [&](const auto& req, auto& res) {
        // Apply middleware
        if (!auth_middleware(req, res)) return;
        if (!rate_limit_middleware(req, res)) return;
        
        handle_generate(req, res);
    });
    
    // Streaming text generation
    if (config.enable_streaming) {
        server.Post(config.base_path + "/generate/stream", [&](const auto& req, auto& res) {
            // Apply middleware
            if (!auth_middleware(req, res)) return;
            if (!rate_limit_middleware(req, res)) return;
            
            handle_generate_stream(req, res);
        });
    }
    
    // Batch text generation
    server.Post(config.base_path + "/generate/batch", [&](const auto& req, auto& res) {
        // Apply middleware
        if (!auth_middleware(req, res)) return;
        if (!rate_limit_middleware(req, res)) return;
        
        handle_generate_batch(req, res);
    });
    
    // Async text generation
    server.Post(config.base_path + "/generate/async", [&](const auto& req, auto& res) {
        // Apply middleware
        if (!auth_middleware(req, res)) return;
        if (!rate_limit_middleware(req, res)) return;
        
        handle_generate_async(req, res);
    });
    
    // Get generation status
    server.Get(config.base_path + "/generate/status/:request_id", [&](const auto& req, auto& res) {
        // Apply middleware
        if (!auth_middleware(req, res)) return;
        
        // Get request ID from path params
        std::string request_id = req.path_params.at("request_id");
        
        // Get request status
        auto request = scheduler->get_request_status(request_id);
        if (!request) {
            json error = {
                {"error", {
                    {"code", 404},
                    {"message", "Request not found"}
                }}
            };
            res.status = 404;
            res.set_content(error.dump(), "application/json");
            return;
        }
        
        // Return status
        json status = {
            {"request_id", request->id()},
            {"status", to_string(request->status())},
            {"created_at", request->created_time()},
            {"started_at", request->start_time()},
            {"completed_at", request->end_time()},
            {"wait_time_ms", request->wait_time().count()},
            {"processing_time_ms", request->processing_time().count()}
        };
        
        if (request->status() == RequestStatus::COMPLETED) {
            status["result"] = {
                {"text", request->result().generated_texts[0]},
                {"logprobs", request->result().logprobs},
                {"tokens", request->result().top_tokens[0]}
            };
        } else if (request->status() == RequestStatus::FAILED) {
            status["error"] = request->result().error_message;
        }
        
        res.set_content(status.dump(), "application/json");
    });
}

} // namespace rest
} // namespace api
} // namespace deeppowers 