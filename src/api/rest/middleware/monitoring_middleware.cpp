#include "monitoring_middleware.hpp"
#include "../../../common/logging.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>

namespace deeppowers {
namespace api {
namespace rest {

using json = nlohmann::json;

// Global instance
std::unique_ptr<MonitoringMiddleware> monitoring_middleware_instance;

MonitoringMiddleware::MonitoringMiddleware(const MonitoringConfig& config)
    : config_(config) {
    // Initialize first time window
    TimeWindow window;
    window.start_time = std::chrono::system_clock::now();
    metrics_windows_.push_back(window);
}

void MonitoringMiddleware::start_request(const httplib::Request& req) {
    if (!config_.enable_metrics) {
        return;
    }

    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Update active requests count
    metrics_windows_.back().metrics.active_requests++;
    
    // Start trace if enabled
    if (config_.enable_tracing) {
        std::lock_guard<std::mutex> trace_lock(traces_mutex_);
        
        // Check trace rate limit
        auto now = std::chrono::system_clock::now();
        auto one_minute_ago = now - std::chrono::minutes(1);
        auto recent_traces = std::count_if(traces_.begin(), traces_.end(),
            [&](const Trace& trace) {
                return trace.start_time >= one_minute_ago;
            });
            
        if (recent_traces < config_.max_traces_per_minute) {
            Trace trace;
            trace.request_id = req.get_header_value("X-Request-ID");
            trace.path = req.path;
            trace.method = req.method;
            trace.start_time = now;
            traces_.push_back(trace);
        }
    }
}

void MonitoringMiddleware::end_request(const httplib::Request& req,
                                     const httplib::Response& res,
                                     const std::chrono::microseconds& duration) {
    if (!config_.enable_metrics) {
        return;
    }

    // Update metrics
    update_metrics(req, res, duration);
    
    // Update trace if exists
    if (config_.enable_tracing) {
        std::lock_guard<std::mutex> trace_lock(traces_mutex_);
        auto request_id = req.get_header_value("X-Request-ID");
        auto it = std::find_if(traces_.begin(), traces_.end(),
            [&](const Trace& trace) {
                return trace.request_id == request_id;
            });
            
        if (it != traces_.end()) {
            it->duration = duration;
            it->status_code = res.status;
            if (res.status >= 400) {
                it->error = res.body;
            }
        }
    }
    
    // Check for alerts
    if (config_.enable_alerts) {
        check_alerts();
    }
    
    // Clean up old data
    cleanup_old_data();
}

void MonitoringMiddleware::start_span(const std::string& request_id,
                                    const std::string& name) {
    if (!config_.enable_tracing) {
        return;
    }

    std::lock_guard<std::mutex> lock(spans_mutex_);
    
    SpanInfo span;
    span.name = name;
    span.start_time = std::chrono::system_clock::now();
    active_spans_[request_id].push_back(span);
}

void MonitoringMiddleware::end_span(const std::string& request_id,
                                  const std::string& name) {
    if (!config_.enable_tracing) {
        return;
    }

    auto now = std::chrono::system_clock::now();
    
    std::lock_guard<std::mutex> spans_lock(spans_mutex_);
    auto spans_it = active_spans_.find(request_id);
    if (spans_it == active_spans_.end()) {
        return;
    }
    
    auto& spans = spans_it->second;
    auto span_it = std::find_if(spans.rbegin(), spans.rend(),
        [&](const SpanInfo& span) {
            return span.name == name;
        });
        
    if (span_it != spans.rend()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            now - span_it->start_time);
            
        std::lock_guard<std::mutex> trace_lock(traces_mutex_);
        auto trace_it = std::find_if(traces_.begin(), traces_.end(),
            [&](const Trace& trace) {
                return trace.request_id == request_id;
            });
            
        if (trace_it != traces_.end()) {
            trace_it->spans.emplace_back(name, duration);
        }
        
        spans.erase((span_it + 1).base());
    }
}

Metrics MonitoringMiddleware::get_current_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_windows_.back().metrics;
}

std::vector<Trace> MonitoringMiddleware::get_recent_traces() const {
    std::lock_guard<std::mutex> lock(traces_mutex_);
    return traces_;
}

std::vector<Alert> MonitoringMiddleware::get_recent_alerts() const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    return alerts_;
}

void MonitoringMiddleware::update_config(const MonitoringConfig& config) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    config_ = config;
}

void MonitoringMiddleware::update_metrics(const httplib::Request& req,
                                        const httplib::Response& res,
                                        const std::chrono::microseconds& duration) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    auto& metrics = metrics_windows_.back().metrics;
    
    // Update request counts
    metrics.total_requests++;
    metrics.active_requests--;
    if (res.status < 400) {
        metrics.successful_requests++;
    } else {
        metrics.failed_requests++;
    }
    
    // Update latency metrics
    double latency_ms = duration.count() / 1000.0;
    metrics.total_latency_ms += latency_ms;
    metrics.min_latency_ms = std::min(metrics.min_latency_ms.load(), latency_ms);
    metrics.max_latency_ms = std::max(metrics.max_latency_ms.load(), latency_ms);
    
    // Update status code metrics
    metrics.status_codes[res.status]++;
    
    // Update endpoint metrics
    auto& endpoint_metrics = metrics.endpoints[req.path];
    endpoint_metrics.request_count++;
    if (res.status >= 400) {
        endpoint_metrics.error_count++;
    }
    endpoint_metrics.total_latency_ms += latency_ms;
}

void MonitoringMiddleware::check_alerts() {
    std::lock_guard<std::mutex> metrics_lock(metrics_mutex_);
    std::lock_guard<std::mutex> alerts_lock(alerts_mutex_);
    
    const auto& metrics = metrics_windows_.back().metrics;
    auto now = std::chrono::system_clock::now();
    
    // Check error rate
    if (metrics.total_requests > 0) {
        double error_rate = static_cast<double>(metrics.failed_requests) /
                           metrics.total_requests;
        if (error_rate > config_.error_threshold) {
            Alert alert;
            alert.type = Alert::Type::ERROR_RATE;
            alert.message = "High error rate detected";
            alert.threshold = config_.error_threshold;
            alert.current_value = error_rate;
            alert.timestamp = now;
            alerts_.push_back(alert);
        }
    }
    
    // Check latency
    if (metrics.total_requests > 0) {
        double avg_latency = metrics.total_latency_ms / metrics.total_requests;
        if (avg_latency > config_.latency_threshold_ms) {
            Alert alert;
            alert.type = Alert::Type::HIGH_LATENCY;
            alert.message = "High average latency detected";
            alert.threshold = config_.latency_threshold_ms;
            alert.current_value = avg_latency;
            alert.timestamp = now;
            alerts_.push_back(alert);
        }
    }
}

void MonitoringMiddleware::cleanup_old_data() {
    auto now = std::chrono::system_clock::now();
    
    // Clean up old metrics windows
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        auto window_cutoff = now - std::chrono::seconds(config_.metrics_window_size);
        metrics_windows_.erase(
            std::remove_if(metrics_windows_.begin(), metrics_windows_.end(),
                [&](const TimeWindow& window) {
                    return window.start_time < window_cutoff;
                }),
            metrics_windows_.end());
            
        // Ensure at least one window exists
        if (metrics_windows_.empty()) {
            TimeWindow window;
            window.start_time = now;
            metrics_windows_.push_back(window);
        }
    }
    
    // Clean up old traces
    {
        std::lock_guard<std::mutex> lock(traces_mutex_);
        auto trace_cutoff = now - std::chrono::minutes(5);  // Keep 5 minutes of traces
        traces_.erase(
            std::remove_if(traces_.begin(), traces_.end(),
                [&](const Trace& trace) {
                    return trace.start_time < trace_cutoff;
                }),
            traces_.end());
    }
    
    // Clean up old alerts
    {
        std::lock_guard<std::mutex> lock(alerts_mutex_);
        auto alert_cutoff = now - std::chrono::hours(24);  // Keep 24 hours of alerts
        alerts_.erase(
            std::remove_if(alerts_.begin(), alerts_.end(),
                [&](const Alert& alert) {
                    return alert.timestamp < alert_cutoff;
                }),
            alerts_.end());
    }
}

void start_request_monitoring(const httplib::Request& req) {
    if (!monitoring_middleware_instance) {
        monitoring_middleware_instance = std::make_unique<MonitoringMiddleware>();
    }
    monitoring_middleware_instance->start_request(req);
}

void end_request_monitoring(const httplib::Request& req,
                          const httplib::Response& res,
                          const std::chrono::microseconds& duration) {
    if (!monitoring_middleware_instance) {
        monitoring_middleware_instance = std::make_unique<MonitoringMiddleware>();
    }
    monitoring_middleware_instance->end_request(req, res, duration);
}

} // namespace rest
} // namespace api
} // namespace deeppowers 