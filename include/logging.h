#pragma once

#include <atomic>

namespace mis {

// Global switch for per-request and per-batch log lines (errors are always logged).
// Turned off for benchmarks: at hundreds of requests per second, writing a line per
// request costs measurable time.
inline std::atomic<bool>& request_logging_flag() {
    static std::atomic<bool> enabled{true};
    return enabled;
}

inline bool request_logging() {
    return request_logging_flag().load(std::memory_order_relaxed);
}

inline void set_request_logging(bool on) {
    request_logging_flag().store(on);
}

}  // namespace mis
