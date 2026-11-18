#pragma once

#include <httplib.h>

#include <memory>
#include <string>

#include "config.h"
#include "model_registry.h"

namespace mis {

// HTTP front end. Routes each request to its model's batcher through the registry.
class InferenceServer {
  public:
    InferenceServer(ServerConfig config, std::shared_ptr<ModelRegistry> registry);
    ~InferenceServer();

    // Blocks serving on config.host:config.port until stop() is called.
    bool listen();

    // For tests: bind to a free port (returned), then serve with listen_after_bind().
    int bind_to_any_port(const std::string& host = "127.0.0.1");
    bool listen_after_bind();
    void wait_until_ready() const { http_.wait_until_ready(); }

    // Thread-safe. Stops accepting connections; listen() returns once in-flight
    // HTTP requests have finished (the batchers keep running so they can finish).
    void stop() { http_.stop(); }

    // Second half of graceful shutdown: drain every model's queue and join workers.
    // Call after listen() has returned.
    void drain() { registry_->shutdown(); }

  private:
    void setup_routes();
    void handle_predict(const httplib::Request& req, httplib::Response& res);
    void handle_reload(const httplib::Request& req, httplib::Response& res);
    void handle_list_models(httplib::Response& res);

    ServerConfig config_;
    std::shared_ptr<ModelRegistry> registry_;
    httplib::Server http_;
};

}  // namespace mis
