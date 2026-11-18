#include "config.h"

#include <array>
#include <fstream>
#include <json.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mis {

namespace {

using nlohmann::json;

void require_positive(int value, const std::string& field, const std::string& model) {
    if (value <= 0) {
        throw std::runtime_error("config: model '" + model + "': " + field + " must be > 0");
    }
}

ModelConfig parse_model(const json& j) {
    ModelConfig m;
    m.name = j.at("name").get<std::string>();
    m.path = j.at("path").get<std::string>();
    // .value(key, default) makes a field optional; .at(key) makes it required.
    m.version = j.value("version", 1);
    m.preprocess.resize_short = j.value("resize_short", 256);
    m.preprocess.crop = j.value("input_size", 224);
    if (j.contains("mean")) m.preprocess.mean = j.at("mean").get<std::array<float, 3>>();
    if (j.contains("std")) m.preprocess.std = j.at("std").get<std::array<float, 3>>();
    m.max_batch_size = j.value("max_batch_size", 8);
    m.max_wait_ms = j.value("max_wait_ms", 5);
    m.num_workers = j.value("num_workers", 1);
    m.intra_op_threads = j.value("intra_op_threads", 1);
    m.queue_limit = j.value("queue_limit", 256);
    m.request_timeout_ms = j.value("request_timeout_ms", 5000);

    if (m.name.empty()) throw std::runtime_error("config: model name must not be empty");
    require_positive(m.preprocess.crop, "input_size", m.name);
    require_positive(m.max_batch_size, "max_batch_size", m.name);
    require_positive(m.num_workers, "num_workers", m.name);
    require_positive(m.intra_op_threads, "intra_op_threads", m.name);
    require_positive(m.queue_limit, "queue_limit", m.name);
    require_positive(m.request_timeout_ms, "request_timeout_ms", m.name);
    if (m.max_wait_ms < 0) throw std::runtime_error("config: max_wait_ms must be >= 0");
    if (m.preprocess.resize_short < m.preprocess.crop) {
        throw std::runtime_error("config: model '" + m.name + "': resize_short < input_size");
    }
    for (float s : m.preprocess.std) {
        if (s <= 0.f) throw std::runtime_error("config: model '" + m.name + "': std must be > 0");
    }
    return m;
}

}  // namespace

ServerConfig parse_config(const std::string& json_text) {
    try {
        const json j = json::parse(json_text);
        ServerConfig cfg;
        cfg.host = j.value("host", cfg.host);
        cfg.port = j.value("port", cfg.port);
        cfg.http_threads = j.value("http_threads", cfg.http_threads);
        cfg.max_body_bytes = static_cast<size_t>(j.value("max_body_mb", 10)) * 1024 * 1024;
        cfg.labels_path = j.value("labels_path", cfg.labels_path);
        cfg.models_dir = j.value("models_dir", cfg.models_dir);
        cfg.log_requests = j.value("log_requests", cfg.log_requests);

        std::unordered_set<std::string> seen;
        for (const auto& mj : j.at("models")) {
            ModelConfig m = parse_model(mj);
            if (!seen.insert(m.name).second) {
                throw std::runtime_error("config: duplicate model name '" + m.name + "'");
            }
            cfg.models.push_back(std::move(m));
        }
        if (cfg.models.empty()) throw std::runtime_error("config: no models listed");
        if (cfg.port <= 0 || cfg.port > 65535) throw std::runtime_error("config: bad port");
        if (cfg.http_threads <= 0) throw std::runtime_error("config: http_threads must be > 0");
        return cfg;
    } catch (const json::exception& e) {
        // Re-throw JSON errors (syntax, missing key, wrong type) as one error type.
        throw std::runtime_error(std::string("config: ") + e.what());
    }
}

ServerConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("config: cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parse_config(ss.str());
}

}  // namespace mis
