#include "server.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>

#include <json.hpp>

#include "logging.h"
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

InferenceServer::InferenceServer(ServerConfig config, std::shared_ptr<ModelRegistry> registry)
    : config_(std::move(config)), registry_(std::move(registry)) {
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
    registry_->shutdown();
}

void InferenceServer::setup_routes() {
    http_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        send_json(res, 200, {{"status", "ok"}});
    });

    http_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        std::vector<ModelSample> samples;
        for (const auto& name : registry_->names()) {
            const ModelEntry* e = registry_->find(name);
            samples.push_back({name, e->batcher->current_model()->version,
                               e->batcher->queue_size(), e->metrics.snapshot()});
        }
        res.set_content(render_prometheus(samples), "text/plain; version=0.0.4");
    });

    http_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        handle_list_models(res);
    });

    // ([^/]+) captures the model name; it arrives as req.matches[1].
    http_.Post(R"(/v1/models/([^/]+)/predict)",
               [this](const httplib::Request& req, httplib::Response& res) {
                   handle_predict(req, res);
               });

    http_.Post(R"(/v1/models/([^/]+)/reload)",
               [this](const httplib::Request& req, httplib::Response& res) {
                   handle_reload(req, res);
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

void InferenceServer::handle_list_models(httplib::Response& res) {
    json models = json::array();
    for (const auto& name : registry_->names()) {
        const ModelEntry* e = registry_->find(name);
        const auto mv = e->batcher->current_model();
        const ModelConfig& c = e->config;
        models.push_back({
            {"name", name},
            {"version", mv->version},
            {"path", mv->path},
            {"status", e->reloading ? "reloading" : "ready"},
            {"queue_length", e->batcher->queue_size()},
            {"settings",
             {{"input_size", c.preprocess.crop},
              {"resize_short", c.preprocess.resize_short},
              {"mean", c.preprocess.mean},
              {"std", c.preprocess.std},
              {"max_batch_size", c.max_batch_size},
              {"max_wait_ms", c.max_wait_ms},
              {"num_workers", c.num_workers},
              {"intra_op_threads", c.intra_op_threads},
              {"queue_limit", c.queue_limit},
              {"request_timeout_ms", c.request_timeout_ms}}},
        });
    }
    send_json(res, 200, {{"models", models}});
}

void InferenceServer::handle_reload(const httplib::Request& req, httplib::Response& res) {
    const std::string name = req.matches[1];
    // Optional JSON body: {"path": "models/x.onnx", "version": 2}. Both fields optional.
    std::string path;
    std::optional<int> version;
    if (!req.body.empty()) {
        try {
            const json body = json::parse(req.body);
            if (body.contains("path")) path = body.at("path").get<std::string>();
            if (body.contains("version")) version = body.at("version").get<int>();
        } catch (const json::exception& e) {
            return send_error(res, 400, std::string("invalid JSON body: ") + e.what());
        }
    }

    ReloadResult r = registry_->reload(name, path, version);
    if (r.http_status != 200) {
        log_line("reload model=" + name + " status=" + std::to_string(r.http_status) +
                 " error=" + r.error);
        return send_error(res, r.http_status, r.error);
    }
    send_json(res, 200,
              {{"model", name},
               {"old_version", r.old_version},
               {"new_version", r.new_version},
               {"path", r.path},
               {"load_ms", r.load_ms}});
}

void InferenceServer::handle_predict(const httplib::Request& req, httplib::Response& res) {
    const auto t_start = Clock::now();
    const std::string name = req.matches[1];

    ModelEntry* entry = registry_->find(name);
    if (!entry) {
        log_line("predict model=" + name + " status=404 error=unknown model");
        return send_error(res, 404, "unknown model: " + name);
    }
    Metrics& metrics = entry->metrics;
    const ModelConfig& cfg = entry->config;
    const auto deadline = t_start + std::chrono::milliseconds(cfg.request_timeout_ms);
    metrics.record_request();

    // Helper: count the error, log it, and send it.
    auto fail = [&](int status, const std::string& msg) {
        metrics.record_error();
        log_line("predict model=" + name + " status=" + std::to_string(status) + " error=" + msg);
        send_error(res, status, msg);
    };

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
        const auto classes = entry->batcher->current_model()->runner->num_classes();
        if (k < 1 || k > classes) return fail(400, "top_k out of range");
    }

    // Cheap early rejection: if the queue is already full, answer 503 BEFORE spending
    // CPU on JPEG decoding. Under overload, hundreds of rejected requests per second
    // each decoding an image stole the CPU the inference workers needed, and
    // throughput collapsed (measured in Phase 6). This check can race with other
    // threads, which is fine: push() below still enforces the limit exactly.
    auto reject = [&] {
        metrics.record_rejected();
        res.set_header("Retry-After", "1");  // tells well-behaved clients to back off
        fail(503, "queue full, try again later");
    };
    if (entry->batcher->queue_size() >= static_cast<size_t>(cfg.queue_limit)) return reject();

    // Decode on the HTTP thread: it spreads JPEG decoding across many threads and lets
    // us reject garbage with 400 before it ever reaches the queue.
    cv::Mat img = decode_image(req.get_file_value("image").content);
    if (img.empty()) return fail(400, "could not decode image");

    auto fut = entry->batcher->submit(std::move(img), k, deadline);
    if (!fut) return reject();

    // Sleep until a worker fulfills the promise or our deadline passes.
    if (fut->wait_until(deadline) != std::future_status::ready) {
        metrics.record_timeout();
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
        {"model", name},
        {"version", r.model_version},  // the version that actually computed this answer
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

    if (!request_logging()) return;
    char line[256];
    std::snprintf(line, sizeof(line),
                  "predict model=%s v=%d status=200 batch=%d total_ms=%.2f queue_ms=%.2f "
                  "pre_ms=%.2f inf_ms=%.2f post_ms=%.2f",
                  name.c_str(), r.model_version, r.batch_size, total, r.queue_wait_ms,
                  r.preprocess_ms, r.inference_ms, r.postprocess_ms);
    log_line(line);
}

bool InferenceServer::listen() { return http_.listen(config_.host, config_.port); }

int InferenceServer::bind_to_any_port(const std::string& host) {
    return http_.bind_to_any_port(host);
}

bool InferenceServer::listen_after_bind() { return http_.listen_after_bind(); }

}  // namespace mis
