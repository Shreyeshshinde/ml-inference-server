#include "server.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>

#include <json.hpp>

#include "postprocess.h"
#include "preprocess.h"

namespace mis {

namespace {

using nlohmann::json;
using Clock = std::chrono::steady_clock;  // monotonic: never jumps if system time changes

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void send_json(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

void send_error(httplib::Response& res, int status, const std::string& message) {
    send_json(res, status, {{"error", message}});
}

// One log line per request. The mutex stops lines from different threads interleaving.
void log_line(const std::string& line) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    std::cerr << line << "\n";
}

}  // namespace

InferenceServer::InferenceServer(ServerConfig config, ServedModel model,
                                 std::vector<std::string> labels)
    : config_(std::move(config)), model_(std::move(model)), labels_(std::move(labels)) {
    // Fixed-size HTTP thread pool: bounded concurrency instead of a thread per connection.
    const int n = config_.http_threads;
    http_.new_task_queue = [n] { return new httplib::ThreadPool(static_cast<size_t>(n)); };
    // Bodies above this are rejected by httplib with 413 before our handler runs.
    http_.set_payload_max_length(config_.max_body_bytes);
    setup_routes();
}

void InferenceServer::setup_routes() {
    http_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, {{"status", "ok"}});
    });

    http_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics_.to_prometheus(), "text/plain; version=0.0.4");
    });

    // ([^/]+) captures the model name; it arrives as req.matches[1].
    http_.Post(R"(/v1/models/([^/]+)/predict)",
               [this](const httplib::Request& req, httplib::Response& res) {
                   handle_predict(req, res);
               });

    // Last line of defense: any exception that escapes a handler becomes a 500
    // instead of killing the server.
    http_.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            std::string msg = "internal error";
            try {
                if (ep) std::rethrow_exception(ep);
            } catch (const std::exception& e) {
                msg = e.what();
            } catch (...) {
            }
            send_error(res, 500, msg);
        });

    // httplib sets 413 itself for oversized bodies; give it a JSON body like other errors.
    http_.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.body.empty()) {
            send_error(res, res.status, res.status == 413 ? "payload too large" : "error");
        }
    });
}

void InferenceServer::handle_predict(const httplib::Request& req, httplib::Response& res) {
    const auto t_start = Clock::now();
    metrics_.record_request();
    const std::string name = req.matches[1];

    // Helper: count the error, log it, and send it.
    auto fail = [&](int status, const std::string& msg) {
        metrics_.record_error();
        log_line("predict model=" + name + " status=" + std::to_string(status) + " error=" + msg);
        send_error(res, status, msg);
    };

    if (name != model_.config.name) return fail(404, "unknown model: " + name);
    if (!req.has_file("image")) return fail(400, "missing multipart field 'image'");

    int k = 5;
    if (req.has_param("top_k")) {
        try {
            size_t used = 0;
            const std::string raw = req.get_param_value("top_k");
            k = std::stoi(raw, &used);
            if (used != raw.size()) throw std::invalid_argument("trailing characters");
        } catch (const std::exception&) {
            return fail(400, "top_k must be an integer");
        }
        if (k < 1 || k > model_.runner->num_classes()) {
            return fail(400, "top_k out of range");
        }
    }

    // --- Preprocess ---
    const auto t_pre = Clock::now();
    cv::Mat img = decode_image(req.get_file_value("image").content);
    if (img.empty()) return fail(400, "could not decode image");
    std::vector<float> tensor;
    try {
        tensor = preprocess(img, model_.config.preprocess);
    } catch (const std::exception& e) {
        return fail(400, std::string("preprocess failed: ") + e.what());
    }

    // --- Inference ---
    const auto t_inf = Clock::now();
    std::vector<float> logits;
    try {
        const int s = model_.config.preprocess.crop;
        logits = model_.runner->run(tensor.data(), 1, s, s);
    } catch (const std::exception& e) {
        return fail(500, std::string("inference failed: ") + e.what());
    }

    // --- Postprocess ---
    const auto t_post = Clock::now();
    auto probs = softmax(logits.data(), logits.size());
    auto preds = top_k(probs, k, labels_);
    const auto t_end = Clock::now();

    json predictions = json::array();
    for (const auto& p : preds) {
        predictions.push_back({{"class_id", p.class_id}, {"label", p.label}, {"score", p.score}});
    }
    const double total = ms_between(t_start, t_end);
    json body = {
        {"model", model_.config.name},
        {"version", model_.config.version},
        {"predictions", predictions},
        {"batch_size", 1},  // no batching until Phase 3
        {"latency_ms", total},
        {"timings_ms",
         {{"queue_wait", 0.0},  // no queue until Phase 3
          {"preprocess", ms_between(t_pre, t_inf)},
          {"inference", ms_between(t_inf, t_post)},
          {"postprocess", ms_between(t_post, t_end)}}},
    };
    send_json(res, 200, body);

    char line[256];
    std::snprintf(line, sizeof(line),
                  "predict model=%s status=200 batch=1 total_ms=%.2f pre_ms=%.2f inf_ms=%.2f "
                  "post_ms=%.2f",
                  name.c_str(), total, ms_between(t_pre, t_inf), ms_between(t_inf, t_post),
                  ms_between(t_post, t_end));
    log_line(line);
}

bool InferenceServer::listen() { return http_.listen(config_.host, config_.port); }

int InferenceServer::bind_to_any_port(const std::string& host) {
    return http_.bind_to_any_port(host);
}

bool InferenceServer::listen_after_bind() { return http_.listen_after_bind(); }

}  // namespace mis
