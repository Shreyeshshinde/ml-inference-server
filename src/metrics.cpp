#include "metrics.h"

#include <functional>
#include <sstream>

namespace mis {

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
    return out.str();
}

}  // namespace mis
