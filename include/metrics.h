#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace mis {

// Plain copy of one model's counters at a moment in time.
struct MetricsSnapshot {
    uint64_t requests = 0, errors = 0, rejected = 0, timeouts = 0, expired = 0;
    uint64_t batch_failures = 0, batches = 0, batched_items = 0;
};

// Lock-free counters for ONE model. std::atomic lets many threads increment without
// a mutex. Latency percentiles are added in Phase 7.
class Metrics {
public:
    void record_request() { inc(requests_); }
    void record_error() { inc(errors_); }
    void record_rejected() { inc(rejected_); }  // 503 queue full
    void record_timeout() { inc(timeouts_); }   // 504
    void record_expired() { inc(expired_); }    // skipped by worker, deadline passed
    void record_batch_failure() { inc(batch_failures_); }
    void record_batch(int size) {
        inc(batches_);
        batched_items_.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
    }

    uint64_t batches() const { return batches_.load(); }
    uint64_t batched_items() const { return batched_items_.load(); }
    MetricsSnapshot snapshot() const;

private:
    static void inc(std::atomic<uint64_t>& c) { c.fetch_add(1, std::memory_order_relaxed); }

    std::atomic<uint64_t> requests_{0};
    std::atomic<uint64_t> errors_{0};
    std::atomic<uint64_t> rejected_{0};
    std::atomic<uint64_t> timeouts_{0};
    std::atomic<uint64_t> expired_{0};
    std::atomic<uint64_t> batch_failures_{0};
    std::atomic<uint64_t> batches_{0};
    std::atomic<uint64_t> batched_items_{0};
};

// Everything /metrics shows for one model.
struct ModelSample {
    std::string model;
    int version = 0;
    size_t queue_length = 0;
    MetricsSnapshot counters;
};

// Prometheus text format: each metric family's HELP/TYPE once, then one sample per
// model with a model="..." label.
std::string render_prometheus(const std::vector<ModelSample>& samples);

}  // namespace mis
