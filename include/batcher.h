#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "metrics.h"
#include "model_runner.h"
#include "postprocess.h"
#include "request_queue.h"

namespace mis {

// One loaded version of a model. Immutable once built, so many threads can read it.
// Swapping versions means swapping the shared_ptr that points at one of these.
struct ModelVersion {
    int version = 1;
    std::string path;
    std::shared_ptr<ModelRunner> runner;
};

using Labels = std::shared_ptr<const std::vector<std::string>>;

// What a worker hands back to the waiting HTTP handler.
struct InferenceResult {
    std::vector<Prediction> predictions;
    int model_version = 0;  // the version that actually ran this batch
    int batch_size = 0;     // how many requests shared this model run
    double queue_wait_ms = 0;
    double preprocess_ms = 0;   // whole batch
    double inference_ms = 0;    // whole batch (one model run)
    double postprocess_ms = 0;  // whole batch
};

// One queued request. The promise is the "return address": the worker fulfills it,
// the handler holds the matching future.
struct InferenceRequest {
    cv::Mat image;  // decoded BGR image (decoding happens on the HTTP thread)
    int top_k = 5;
    Clock::time_point deadline;  // after this the handler has given up (504)
    std::promise<InferenceResult> promise;
};

// Dynamic batching for one model: a bounded queue plus num_workers threads that
// pop batches, run the model once per batch, and fulfill each request's promise.
class Batcher {
  public:
    Batcher(ModelConfig config, std::shared_ptr<const ModelVersion> initial, Labels labels,
            Metrics* metrics = nullptr);
    ~Batcher();

    Batcher(const Batcher&) = delete;
    Batcher& operator=(const Batcher&) = delete;

    // Starts the worker threads. Separate from the constructor so tests can fill the
    // queue before any worker runs.
    void start();

    // Enqueues an image. Returns std::nullopt when the queue is full or shut down (503).
    std::optional<std::future<InferenceResult>> submit(cv::Mat image, int top_k,
                                                       Clock::time_point deadline);

    // Zero-downtime reload: atomically replaces the model used by FUTURE batches and
    // returns the previous version. Batches already running keep their own copy of
    // the old shared_ptr, so they finish on the old model; it is freed when the last
    // copy goes away.
    std::shared_ptr<const ModelVersion> swap_model(std::shared_ptr<const ModelVersion> next);

    std::shared_ptr<const ModelVersion> current_model() const {
        return std::atomic_load(&current_);
    }

    // Graceful stop: reject new work, let workers drain the queue, join them.
    // Anything still queued afterwards (e.g. workers were never started) gets an
    // exception so no caller waits forever. Safe to call more than once.
    void shutdown();

    size_t queue_size() const { return queue_.size(); }
    const ModelConfig& config() const { return config_; }

  private:
    using Entry = BoundedQueue<InferenceRequest>::Entry;

    void worker_loop(int worker_id);
    void process_batch(std::vector<Entry>& batch, int worker_id);

    const ModelConfig config_;
    // Read with std::atomic_load and written with std::atomic_exchange ONLY.
    // A plain shared_ptr is not safe to read and write from different threads at once.
    std::shared_ptr<const ModelVersion> current_;
    const Labels labels_;
    Metrics* metrics_;
    BoundedQueue<InferenceRequest> queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> shut_down_{false};
};

}  // namespace mis
