#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "batcher.h"
#include "config.h"
#include "metrics.h"

namespace mis {

// Everything that belongs to one served model. Each model has its OWN queue, batcher,
// workers and metrics, so one overloaded model cannot starve another.
struct ModelEntry {
    ModelConfig config;
    Metrics metrics;  // declared before batcher: the batcher points at it
    std::unique_ptr<Batcher> batcher;
    std::mutex reload_mu;  // one reload at a time per model
    std::atomic<bool> reloading{false};
};

struct ReloadResult {
    int http_status = 200;  // 200 ok, 400 bad request/new model unusable, 404, 409 busy
    std::string error;
    int old_version = 0;
    int new_version = 0;
    std::string path;
    double load_ms = 0;
};

// Loads every model listed in the config at startup and owns them for the server's
// lifetime. The set of models never changes after construction, so find() needs no
// lock; only a model's *version* changes (via reload).
class ModelRegistry {
  public:
    // Loads, warms up and starts every model. Throws if any model fails, so a bad
    // config stops the server at startup instead of at the first request.
    ModelRegistry(const ServerConfig& config, Labels labels);
    ~ModelRegistry();

    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    // nullptr if no model has that name (the server answers 404).
    ModelEntry* find(const std::string& name) const;

    // Model names in config order.
    const std::vector<std::string>& names() const { return names_; }

    // Loads a new version in the calling thread while the old one keeps serving,
    // checks it with a warm-up run, then swaps it in atomically.
    // new_path: empty = reload the current file. new_version: empty = current + 1.
    ReloadResult reload(const std::string& name, const std::string& new_path,
                        std::optional<int> new_version);

    // Graceful stop for every model: drain queues, join workers.
    void shutdown();

  private:
    std::shared_ptr<const ModelVersion> load_version(const ModelConfig& cfg,
                                                     const std::string& path, int version);
    bool path_allowed(const std::string& path) const;

    std::string models_dir_;
    Labels labels_;
    std::map<std::string, std::unique_ptr<ModelEntry>> models_;
    std::vector<std::string> names_;
};

}  // namespace mis
