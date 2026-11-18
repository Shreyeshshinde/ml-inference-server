#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mis {

// Request latency in two forms:
// 1. A Prometheus histogram (fixed buckets, counts only ever grow). Prometheus can
//    combine histograms from many servers and compute percentiles itself.
// 2. Exact p50/p99 over a sliding window of the most recent requests, for humans
//    reading /metrics directly. A window, not all-time, so a slow minute an hour ago
//    does not hide how the server behaves now.
class LatencyTracker {
  public:
    static constexpr std::array<double, 11> kBucketsMs = {5,   10,   25,   50,   100,  250,
                                                          500, 1000, 2500, 5000, 10000};
    static constexpr size_t kWindow = 2048;

    struct Snapshot {
        std::array<uint64_t, kBucketsMs.size()> cumulative{};  // count with latency <= bucket
        uint64_t count = 0;
        double sum_ms = 0;
        double p50_ms = 0, p99_ms = 0;  // over the last min(count, kWindow) requests
        size_t window = 0;
    };

    void record(double ms);
    Snapshot snapshot() const;

  private:
    // A mutex (not atomics) because one record() updates several fields together.
    // It is held for a few nanoseconds per request, far below the model's cost.
    mutable std::mutex mu_;
    std::array<uint64_t, kBucketsMs.size()> bucket_counts_{};  // per bucket, not cumulative
    uint64_t count_ = 0;
    double sum_ms_ = 0;
    std::array<double, kWindow> ring_{};  // ring buffer: newest overwrites oldest
    size_t next_ = 0;
};

// Nearest-rank percentile (q in (0,1]) of an unsorted list; 0 for an empty list.
double percentile(std::vector<double> values, double q);

// Plain copy of one model's counters at a moment in time.
struct MetricsSnapshot {
    uint64_t requests = 0, errors = 0, rejected = 0, timeouts = 0, expired = 0;
    uint64_t batch_failures = 0, batches = 0, batched_items = 0;
    LatencyTracker::Snapshot latency;
};

// Counters for ONE model. The counters are std::atomic so many threads can increment
// them without a mutex; latency lives in a LatencyTracker.
class Metrics {
  public:
    // End-to-end latency of a successful (200) request.
    void record_latency(double ms) { latency_.record(ms); }
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
    LatencyTracker latency_;
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
