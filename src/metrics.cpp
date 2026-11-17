#include "metrics.h"

#include <sstream>

namespace mis {

std::string Metrics::to_prometheus() const {
    std::ostringstream out;
    out << "# HELP mis_requests_total Prediction requests received.\n"
        << "# TYPE mis_requests_total counter\n"
        << "mis_requests_total " << requests_.load() << "\n"
        << "# HELP mis_errors_total Prediction requests that returned a non-200 status.\n"
        << "# TYPE mis_errors_total counter\n"
        << "mis_errors_total " << errors_.load() << "\n";
    return out.str();
}

}  // namespace mis
