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
    : config_(std::move(config)), model_(std::move(model)) {
    batcher_ = std::make_unique<Batcher>(model_.config, model_.runner, std::move(labels),
                                         &metrics_);
    batcher_->start();

    // Fixed-size HTTP thread pool: bounded concurrency instead of a thread per connection.
    // Handler threads mostly sleep on a future while workers do the heavy compute.
    const int n = config_.http_threads;
    http_.new_task_queue = [n] { return new httplib::ThreadPool(static_cast<size_t>(n)); };
    // Bodies above this are rejected by httplib with 413 before our handler runs.
    http_.set_payload_max_length(config_.max_body_bytes);
    setup_routes();
}

InferenceServer::~InferenceServer() {
    http_.stop();
    batcher_->shutdown();
}

void InferenceServer::setup_routes() {
    http_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, {{"status", "ok"}});
    });

    http_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        std::string text = metrics_.to_prometheus();
        text += "# HELP mis_queue_length Requests waiting in the batch queue.\n"
                "# TYPE mis_queue_length gauge\n"
                "mis_queue_length " +
                std::to_string(batcher_->queue_size()) + "\n";
        res.set_content(text, "text/plain; version=0.0.4");
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
    const auto deadline = t_start + std::chrono::milliseconds(model_.config.request_timeout_ms);
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
        if (k < 1 || k > model_.runner->num_classes()) return fail(400, "top_k out of range");
    }

    // Decode on the HTTP thread: it spreads JPEG decoding across many threads and lets
    // us reject garbage with 400 before it ever reaches the queue.
    cv::Mat img = decode_image(req.get_file_value("image").content);
    if (img.empty()) return fail(400, "could not decode image");

    auto fut = batcher_->submit(std::move(img), k, deadline);
    if (!fut) {
        metrics_.record_rejected();
        return fail(503, "queue full, try again later");
    }

    // Sleep until a worker fulfills the promise or our deadline passes.
    if (fut->wait_until(deadline) != std::future_status::ready) {
        metrics_.record_timeout();
        return fail(504, "timed out waiting for inference");
    }

    InferenceResult r;
    try {
        r = fut->get();  // rethrows the worker's exception, if it set one
    } catch (const std::exception& e) {
        return fail(500, std::string("inference failed: ") + e.what());
    }

    json predictions = json::array();
    for (const auto& p : r.predictions) {
        predictions.push_back({{"class_id", p.class_id}, {"label", p.label}, {"score", p.score}});
    }
    const double total = ms_between(t_start, Clock::now());
    json body = {
        {"model", model_.config.name},
        {"version", model_.config.version},
        {"predictions", predictions},
        {"batch_size", r.batch_size},
        {"latency_ms", total},
        {"timings_ms",
         {{"queue_wait", r.queue_wait_ms},
          {"preprocess", r.preprocess_ms},
          {"inference", r.inference_ms},
          {"postprocess", r.postprocess_ms}}},
    };
    send_json(res, 200, body);

    char line[256];
    std::snprintf(line, sizeof(line),
                  "predict model=%s status=200 batch=%d total_ms=%.2f queue_ms=%.2f "
                  "pre_ms=%.2f inf_ms=%.2f post_ms=%.2f",
                  name.c_str(), r.batch_size, total, r.queue_wait_ms, r.preprocess_ms,
                  r.inference_ms, r.postprocess_ms);
    log_line(line);
}

bool InferenceServer::listen() { return http_.listen(config_.host, config_.port); }

int InferenceServer::bind_to_any_port(const std::string& host) {
    return http_.bind_to_any_port(host);
}

bool InferenceServer::listen_after_bind() { return http_.listen_after_bind(); }

}  // namespace mis
