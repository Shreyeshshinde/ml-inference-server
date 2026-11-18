#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

namespace mis {

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    // Nearest rank: the smallest value with at least q of the data at or below it.
    // nth_element is O(n) and only partially sorts, which is all a percentile needs.
    const size_t rank = static_cast<size_t>(std::ceil(q * values.size()));
    const size_t idx = std::min(values.size() - 1, rank == 0 ? 0 : rank - 1);
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

void LatencyTracker::record(double ms) {
    // First bucket whose upper bound is >= ms; values above the last bound only count
    // toward the implicit +Inf bucket (that is, count_).
    const auto it = std::lower_bound(kBucketsMs.begin(), kBucketsMs.end(), ms);
    std::lock_guard<std::mutex> lock(mu_);
    if (it != kBucketsMs.end()) ++bucket_counts_[it - kBucketsMs.begin()];
    ++count_;
    sum_ms_ += ms;
    ring_[next_] = ms;
    next_ = (next_ + 1) % kWindow;
}

LatencyTracker::Snapshot LatencyTracker::snapshot() const {
    Snapshot s;
    std::vector<double> window;
    {
        // Copy under the lock, compute percentiles after releasing it, so a scrape
        // never blocks request threads while it sorts.
        std::lock_guard<std::mutex> lock(mu_);
        s.count = count_;
        s.sum_ms = sum_ms_;
        uint64_t running = 0;
        for (size_t i = 0; i < kBucketsMs.size(); ++i) {
            running += bucket_counts_[i];
            s.cumulative[i] = running;  // Prometheus buckets are cumulative
        }
        s.window = static_cast<size_t>(std::min<uint64_t>(count_, kWindow));
        window.assign(ring_.begin(), ring_.begin() + s.window);
    }
    // Before the ring wraps, its first `window` slots are exactly the recorded values;
    // after it wraps, all kWindow slots are valid. Order does not matter for percentiles.
    s.p50_ms = percentile(window, 0.50);
    s.p99_ms = percentile(window, 0.99);
    return s;
}

MetricsSnapshot Metrics::snapshot() const {
    MetricsSnapshot s;
    s.requests = requests_.load();
    s.errors = errors_.load();
    s.rejected = rejected_.load();
    s.timeouts = timeouts_.load();
    s.expired = expired_.load();
    s.batch_failures = batch_failures_.load();
    s.batches = batches_.load();
    s.batched_items = batched_items_.load();
    s.latency = latency_.snapshot();
    return s;
}

std::string render_prometheus(const std::vector<ModelSample>& samples) {
    std::ostringstream out;
    // Writes one metric family: header once, then a labeled line per model.
    auto family = [&](const char* name, const char* type, const char* help,
                      const std::function<double(const ModelSample&)>& value) {
        out << "# HELP " << name << " " << help << "\n# TYPE " << name << " " << type << "\n";
        for (const auto& s : samples) {
            out << name << "{model=\"" << s.model << "\"} " << value(s) << "\n";
        }
    };
    using S = const ModelSample&;
    family("mis_requests_total", "counter", "Prediction requests received.",
           [](S s) { return double(s.counters.requests); });
    family("mis_errors_total", "counter", "Prediction requests that returned a non-200 status.",
           [](S s) { return double(s.counters.errors); });
    family("mis_rejected_total", "counter", "Requests rejected with 503 (queue full).",
           [](S s) { return double(s.counters.rejected); });
    family("mis_timeouts_total", "counter", "Requests that returned 504.",
           [](S s) { return double(s.counters.timeouts); });
    family("mis_expired_total", "counter", "Queued requests skipped because their deadline passed.",
           [](S s) { return double(s.counters.expired); });
    family("mis_batch_failures_total", "counter", "Batches whose model run failed.",
           [](S s) { return double(s.counters.batch_failures); });
    family("mis_batches_total", "counter", "Model runs (batches) executed.",
           [](S s) { return double(s.counters.batches); });
    family("mis_batched_requests_total", "counter", "Requests processed inside batches.",
           [](S s) { return double(s.counters.batched_items); });
    family("mis_avg_batch_size", "gauge", "Average requests per model run.", [](S s) {
        return s.counters.batches ? double(s.counters.batched_items) / s.counters.batches : 0.0;
    });
    family("mis_queue_length", "gauge", "Requests waiting in the batch queue.",
           [](S s) { return double(s.queue_length); });
    family("mis_model_version", "gauge", "Model version currently serving.",
           [](S s) { return double(s.version); });
    family("mis_latency_p50_ms", "gauge",
           "Median end-to-end latency of successful requests, last 2048 requests.",
           [](S s) { return s.counters.latency.p50_ms; });
    family("mis_latency_p99_ms", "gauge",
           "99th percentile end-to-end latency of successful requests, last 2048 requests.",
           [](S s) { return s.counters.latency.p99_ms; });

    // Histogram in the standard Prometheus layout: cumulative _bucket{le=...} lines
    // (seconds, by convention), a +Inf bucket equal to _count, then _sum and _count.
    const char* h = "mis_request_latency_seconds";
    out << "# HELP " << h << " End-to-end latency of successful requests.\n"
        << "# TYPE " << h << " histogram\n";
    for (const auto& s : samples) {
        const auto& lat = s.counters.latency;
        const std::string label = "model=\"" + s.model + "\"";
        for (size_t i = 0; i < LatencyTracker::kBucketsMs.size(); ++i) {
            out << h << "_bucket{" << label << ",le=\"" << LatencyTracker::kBucketsMs[i] / 1000.0
                << "\"} " << lat.cumulative[i] << "\n";
        }
        out << h << "_bucket{" << label << ",le=\"+Inf\"} " << lat.count << "\n"
            << h << "_sum{" << label << "} " << lat.sum_ms / 1000.0 << "\n"
            << h << "_count{" << label << "} " << lat.count << "\n";
    }
    return out.str();
}

}  // namespace mis
