#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace mis {

// Phase 2 placeholder: lock-free counters only. Per-model latency percentiles,
// queue length and batch size are added in Phase 7.
// std::atomic lets many HTTP threads increment the same counter without a mutex.
class Metrics {
public:
    void record_request() { requests_.fetch_add(1, std::memory_order_relaxed); }
    void record_error() { errors_.fetch_add(1, std::memory_order_relaxed); }

    // Prometheus text exposition format.
    std::string to_prometheus() const;

private:
    std::atomic<uint64_t> requests_{0};
    std::atomic<uint64_t> errors_{0};
};

}  // namespace mis
