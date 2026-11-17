#include <csignal>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <thread>

#include "config.h"
#include "postprocess.h"
#include "server.h"
#include "version.h"

// Usage: inference_server [--config config/server.json]
// Paths inside the config are relative to the current directory (run from repo root).
int main(int argc, char** argv) {
    std::string config_path = "config/server.json";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) config_path = argv[++i];
    }

    // Block SIGINT/SIGTERM before any thread starts. Every thread inherits this mask,
    // so the signals stay pending until our dedicated thread picks them up with sigwait().
    // That thread can then call normal functions like server.stop(); a classic signal
    // handler could only call a tiny set of async-signal-safe functions.
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, nullptr);

    try {
        mis::ServerConfig cfg = mis::load_config(config_path);
        auto labels = mis::load_labels(cfg.labels_path);

        // Phase 2 serves a single model: the first one in the config.
        mis::ModelConfig mcfg = cfg.models.front();
        std::cerr << "ONNX Runtime " << mis::onnxruntime_version() << ", OpenCV "
                  << mis::opencv_version() << "\n"
                  << "loading model '" << mcfg.name << "' from " << mcfg.path << "\n";
        auto runner = std::make_shared<mis::ModelRunner>(mcfg.path, mcfg.intra_op_threads);

        mis::InferenceServer server(cfg, {mcfg, runner}, std::move(labels));

        std::thread signal_thread([&] {
            int sig = 0;
            sigwait(&signals, &sig);
            std::cerr << "received signal " << sig << ", shutting down\n";
            server.stop();
        });
        signal_thread.detach();  // process exits right after listen() returns

        std::cerr << "listening on " << cfg.host << ":" << cfg.port << "\n";
        if (!server.listen()) {
            std::cerr << "error: could not listen on " << cfg.host << ":" << cfg.port << "\n";
            return 1;
        }
        std::cerr << "server stopped\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
