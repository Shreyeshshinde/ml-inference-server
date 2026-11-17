#pragma once

#include <memory>
#include <string>
#include <vector>

#include <httplib.h>

#include "config.h"
#include "metrics.h"
#include "model_runner.h"

namespace mis {

// One loaded model plus its settings.
struct ServedModel {
    ModelConfig config;
    std::shared_ptr<ModelRunner> runner;
};

// HTTP front end. Phase 2: one model, inference runs directly on the HTTP thread.
class InferenceServer {
public:
    InferenceServer(ServerConfig config, ServedModel model, std::vector<std::string> labels);

    // Blocks serving on config.host:config.port until stop() is called.
    bool listen();

    // For tests: bind to a free port (returned), then serve with listen_after_bind().
    int bind_to_any_port(const std::string& host = "127.0.0.1");
    bool listen_after_bind();
    void wait_until_ready() const { http_.wait_until_ready(); }

    // Thread-safe. Stops accepting connections and makes listen() return.
    void stop() { http_.stop(); }

private:
    void setup_routes();
    void handle_predict(const httplib::Request& req, httplib::Response& res);

    ServerConfig config_;
    ServedModel model_;
    std::vector<std::string> labels_;
    Metrics metrics_;
    httplib::Server http_;
};

}  // namespace mis
