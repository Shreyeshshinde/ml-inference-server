#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace mis {

// Lock-free counters. std::atomic lets many threads increment without a mutex.
// Per-model labels and latency percentiles are added in Phase 7.
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

    // Prometheus text exposition format.
    std::string to_prometheus() const;

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

}  // namespace mis
