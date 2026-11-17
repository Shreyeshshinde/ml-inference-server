#pragma once

#include <string>
#include <vector>

#include "preprocess.h"

namespace mis {

// Settings for one served model. Everything model-specific lives here, so adding a
// new ONNX image classifier is a config change, not a code change.
struct ModelConfig {
    std::string name;
    std::string path;
    int version = 1;
    PreprocessConfig preprocess;
    // Batching and capacity settings (used from Phase 3 on).
    int max_batch_size = 8;
    int max_wait_ms = 5;
    int num_workers = 1;
    int intra_op_threads = 1;
    int queue_limit = 256;
    int request_timeout_ms = 5000;
};

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int http_threads = 32;
    size_t max_body_bytes = 10 * 1024 * 1024;
    std::string labels_path = "data/imagenet_labels.txt";
    std::vector<ModelConfig> models;
};

// Parses and validates a server.json file. Throws std::runtime_error with a
// readable message on a missing file, bad JSON or invalid values, so a bad
// config fails at startup instead of at the first request.
ServerConfig load_config(const std::string& path);

// Same, from a JSON string (used by tests).
ServerConfig parse_config(const std::string& json_text);

}  // namespace mis
