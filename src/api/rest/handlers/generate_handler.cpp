#include "generate_handler.hpp"
#include "../../../core/api/request.hpp"
#include "../../../core/api/response.hpp"
#include "../../../core/scheduler/scheduler.hpp"
#include <nlohmann/json.hpp>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;

void handle_generate(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse request body
        auto body = json::parse(req.body);
        
        // Validate request
        if (!body.contains("prompt")) {
            json error = {
                {"error", {
                    {"code", 400},
                    {"message", "Missing required field: prompt"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }
        
        // Create generation request
        GenerationRequest request;
        request.prompt = body["prompt"].get<std::string>();
        
        // Optional parameters
        if (body.contains("max_tokens")) {
            request.max_tokens = body["max_tokens"].get<int>();
        }
        if (body.contains("temperature")) {
            request.temperature = body["temperature"].get<float>();
        }
        if (body.contains("top_p")) {
            request.top_p = body["top_p"].get<float>();
        }
        if (body.contains("stop")) {
            request.stop = body["stop"].get<std::vector<std::string>>();
        }
        
        // Submit request
        auto result = scheduler->submit_sync(request);
        
        // Return response
        json response = {
            {"text", result.generated_texts[0]},
            {"logprobs", result.logprobs},
            {"tokens", result.top_tokens[0]}
        };
        
        res.set_content(response.dump(), "application/json");
        
    } catch (const json::exception& e) {
        json error = {
            {"error", {
                {"code", 400},
                {"message", "Invalid JSON: " + std::string(e.what())}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"code", 500},
                {"message", "Internal server error: " + std::string(e.what())}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void handle_generate_stream(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse request body
        auto body = json::parse(req.body);
        
        // Validate request
        if (!body.contains("prompt")) {
            json error = {
                {"error", {
                    {"code", 400},
                    {"message", "Missing required field: prompt"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }
        
        // Create generation request
        GenerationRequest request;
        request.prompt = body["prompt"].get<std::string>();
        request.stream = true;
        
        // Optional parameters
        if (body.contains("max_tokens")) {
            request.max_tokens = body["max_tokens"].get<int>();
        }
        if (body.contains("temperature")) {
            request.temperature = body["temperature"].get<float>();
        }
        if (body.contains("top_p")) {
            request.top_p = body["top_p"].get<float>();
        }
        if (body.contains("stop")) {
            request.stop = body["stop"].get<std::vector<std::string>>();
        }
        
        // Set streaming response headers
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        
        // Submit streaming request
        scheduler->submit_stream(request, [&res](const GenerationResult& chunk) {
            json event = {
                {"text", chunk.generated_texts[0]},
                {"logprobs", chunk.logprobs},
                {"tokens", chunk.top_tokens[0]},
                {"finished", chunk.is_finished}
            };
            
            std::string event_data = "data: " + event.dump() + "\n\n";
            res.write(event_data);
            res.flush();
            
            return !chunk.is_finished;
        });
        
    } catch (const json::exception& e) {
        json error = {
            {"error", {
                {"code", 400},
                {"message", "Invalid JSON: " + std::string(e.what())}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"code", 500},
                {"message", "Internal server error: " + std::string(e.what())}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void handle_generate_batch(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse request body
        auto body = json::parse(req.body);
        
        // Validate request
        if (!body.contains("prompts") || !body["prompts"].is_array()) {
            json error = {
                {"error", {
                    {"code", 400},
                    {"message", "Missing or invalid field: prompts (array required)"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }
        
        // Create batch request
        std::vector<GenerationRequest> requests;
        for (const auto& prompt : body["prompts"]) {
            GenerationRequest request;
            request.prompt = prompt.get<std::string>();
            
            // Optional parameters
            if (body.contains("max_tokens")) {
                request.max_tokens = body["max_tokens"].get<int>();
            }
            if (body.contains("temperature")) {
                request.temperature = body["temperature"].get<float>();
            }
            if (body.contains("top_p")) {
                request.top_p = body["top_p"].get<float>();
            }
            if (body.contains("stop")) {
                request.stop = body["stop"].get<std::vector<std::string>>();
            }
            
            requests.push_back(request);
        }
        
        // Submit batch request
        auto results = scheduler->submit_batch(requests);
        
        // Return response
        json response = json::array();
        for (const auto& result : results) {
            response.push_back({
                {"text", result.generated_texts[0]},
                {"logprobs", result.logprobs},
                {"tokens", result.top_tokens[0]}
            });
        }
        
        res.set_content(response.dump(), "application/json");
        
    } catch (const json::exception& e) {
        json error = {
            {"error", {
                {"code", 400},
                {"message", "Invalid JSON: " + std::string(e.what())}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"code", 500},
                {"message", "Internal server error: " + std::string(e.what())}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void handle_generate_async(const httplib::Request& req, httplib::Response& res) {
    try {
        // Parse request body
        auto body = json::parse(req.body);
        
        // Validate request
        if (!body.contains("prompt")) {
            json error = {
                {"error", {
                    {"code", 400},
                    {"message", "Missing required field: prompt"}
                }}
            };
            res.status = 400;
            res.set_content(error.dump(), "application/json");
            return;
        }
        
        // Create generation request
        GenerationRequest request;
        request.prompt = body["prompt"].get<std::string>();
        
        // Optional parameters
        if (body.contains("max_tokens")) {
            request.max_tokens = body["max_tokens"].get<int>();
        }
        if (body.contains("temperature")) {
            request.temperature = body["temperature"].get<float>();
        }
        if (body.contains("top_p")) {
            request.top_p = body["top_p"].get<float>();
        }
        if (body.contains("stop")) {
            request.stop = body["stop"].get<std::vector<std::string>>();
        }
        
        // Submit async request
        auto request_id = scheduler->submit_async(request);
        
        // Return request ID
        json response = {
            {"request_id", request_id}
        };
        
        res.set_content(response.dump(), "application/json");
        
    } catch (const json::exception& e) {
        json error = {
            {"error", {
                {"code", 400},
                {"message", "Invalid JSON: " + std::string(e.what())}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        json error = {
            {"error", {
                {"code", 500},
                {"message", "Internal server error: " + std::string(e.what())}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

} // namespace rest
} // namespace api
} // namespace deeppowers 