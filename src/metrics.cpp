#include "metrics.h"

#include <sstream>

namespace mis {

namespace {

void counter(std::ostringstream& out, const char* name, const char* help, uint64_t value) {
    out << "# HELP " << name << " " << help << "\n"
        << "# TYPE " << name << " counter\n"
        << name << " " << value << "\n";
}

}  // namespace

std::string Metrics::to_prometheus() const {
    std::ostringstream out;
    counter(out, "mis_requests_total", "Prediction requests received.", requests_.load());
    counter(out, "mis_errors_total", "Prediction requests that returned a non-200 status.",
            errors_.load());
    counter(out, "mis_rejected_total", "Requests rejected with 503 because the queue was full.",
            rejected_.load());
    counter(out, "mis_timeouts_total", "Requests that returned 504.", timeouts_.load());
    counter(out, "mis_expired_total", "Queued requests skipped because their deadline passed.",
            expired_.load());
    counter(out, "mis_batch_failures_total", "Batches whose model run failed.",
            batch_failures_.load());
    counter(out, "mis_batches_total", "Model runs (batches) executed.", batches_.load());
    counter(out, "mis_batched_requests_total", "Requests processed inside batches.",
            batched_items_.load());
    const uint64_t b = batches_.load();
    out << "# HELP mis_avg_batch_size Average requests per model run.\n"
        << "# TYPE mis_avg_batch_size gauge\n"
        << "mis_avg_batch_size " << (b ? static_cast<double>(batched_items_.load()) / b : 0.0)
        << "\n";
    return out.str();
}

}  // namespace mis
