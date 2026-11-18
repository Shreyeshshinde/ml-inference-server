#include "model_registry.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace mis {

namespace fs = std::filesystem;

ModelRegistry::ModelRegistry(const ServerConfig& config, Labels labels)
    : models_dir_(config.models_dir), labels_(std::move(labels)) {
    for (const auto& mcfg : config.models) {
        auto entry = std::make_unique<ModelEntry>();
        entry->config = mcfg;
        auto version = load_version(mcfg, mcfg.path, mcfg.version);
        if (labels_ && !labels_->empty() &&
            static_cast<size_t>(version->runner->num_classes()) != labels_->size()) {
            std::cerr << "warning: model '" << mcfg.name << "' has "
                      << version->runner->num_classes() << " classes but labels file has "
                      << labels_->size() << " lines\n";
        }
        entry->batcher = std::make_unique<Batcher>(mcfg, version, labels_, &entry->metrics);
        entry->batcher->start();
        std::cerr << "loaded model '" << mcfg.name << "' v" << mcfg.version << " from " << mcfg.path
                  << " (workers=" << mcfg.num_workers << ", max_batch_size=" << mcfg.max_batch_size
                  << ")\n";
        names_.push_back(mcfg.name);
        models_.emplace(mcfg.name, std::move(entry));
    }
}

ModelRegistry::~ModelRegistry() {
    shutdown();
}

ModelEntry* ModelRegistry::find(const std::string& name) const {
    auto it = models_.find(name);
    return it == models_.end() ? nullptr : it->second.get();
}

std::shared_ptr<const ModelVersion> ModelRegistry::load_version(const ModelConfig& cfg,
                                                                const std::string& path,
                                                                int version) {
    auto runner = std::make_shared<ModelRunner>(path, cfg.intra_op_threads);
    // Warm-up: one dummy run proves the file works with this model's input size
    // BEFORE it can serve real traffic. It also pays ORT's one-time first-run cost
    // here instead of on a user's request.
    const int s = cfg.preprocess.crop;
    std::vector<float> dummy(static_cast<size_t>(3) * s * s, 0.f);
    runner->run(dummy.data(), 1, s, s);

    auto mv = std::make_shared<ModelVersion>();
    mv->version = version;
    mv->path = path;
    mv->runner = std::move(runner);
    return mv;
}

bool ModelRegistry::path_allowed(const std::string& path) const {
    // Resolve "..", "." and symlinks, then require the file to sit under models_dir.
    // Without this, POST /reload {"path": "/etc/passwd"} would make us open any file.
    std::error_code ec;
    const fs::path root = fs::weakly_canonical(fs::absolute(models_dir_), ec);
    if (ec) return false;
    const fs::path target = fs::weakly_canonical(fs::absolute(path), ec);
    if (ec) return false;
    const fs::path rel = target.lexically_relative(root);
    return !rel.empty() && *rel.begin() != ".." && rel.is_relative();
}

ReloadResult ModelRegistry::reload(const std::string& name, const std::string& new_path,
                                   std::optional<int> new_version) {
    ReloadResult r;
    ModelEntry* entry = find(name);
    if (!entry) {
        r.http_status = 404;
        r.error = "unknown model: " + name;
        return r;
    }

    // try_lock instead of lock: a second reload of the same model fails fast with 409
    // instead of queueing up behind a slow load.
    std::unique_lock<std::mutex> lock(entry->reload_mu, std::try_to_lock);
    if (!lock.owns_lock()) {
        r.http_status = 409;
        r.error = "a reload of this model is already in progress";
        return r;
    }

    auto current = entry->batcher->current_model();
    r.old_version = current->version;
    r.new_version = new_version.value_or(current->version + 1);
    r.path = new_path.empty() ? current->path : new_path;

    if (!new_path.empty() && !path_allowed(new_path)) {
        r.http_status = 400;
        r.error = "path must be inside the models directory '" + models_dir_ + "'";
        return r;
    }
    if (r.new_version <= 0) {
        r.http_status = 400;
        r.error = "version must be > 0";
        return r;
    }

    entry->reloading = true;
    const auto t0 = Clock::now();
    std::shared_ptr<const ModelVersion> next;
    try {
        // Slow part (file read, graph optimization, warm-up). The old version keeps
        // serving every request the whole time.
        next = load_version(entry->config, r.path, r.new_version);
    } catch (const std::exception& e) {
        entry->reloading = false;
        r.http_status = 400;
        r.error = std::string("new model failed to load, still serving v") +
                  std::to_string(r.old_version) + ": " + e.what();
        return r;
    }
    if (next->runner->num_classes() != current->runner->num_classes()) {
        entry->reloading = false;
        r.http_status = 400;
        r.error = "new model has a different number of classes; still serving v" +
                  std::to_string(r.old_version);
        return r;
    }

    // The actual switch: one atomic pointer swap. `old` is our reference to the previous
    // version; when it and every in-flight batch's copy are gone, it is freed.
    auto old = entry->batcher->swap_model(next);
    r.load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    entry->reloading = false;
    std::cerr << "reloaded model '" << name << "' v" << old->version << " -> v" << next->version
              << " from " << r.path << " in " << r.load_ms << " ms\n";
    return r;
}

void ModelRegistry::shutdown() {
    for (auto& [name, entry] : models_) entry->batcher->shutdown();
}

}  // namespace mis
