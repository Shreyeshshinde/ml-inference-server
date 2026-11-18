#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "metrics.h"

TEST(Percentile, NearestRank) {
    std::vector<double> v;
    for (int i = 100; i >= 1; --i) v.push_back(i);  // unsorted input on purpose
    EXPECT_DOUBLE_EQ(mis::percentile(v, 0.50), 50);
    EXPECT_DOUBLE_EQ(mis::percentile(v, 0.99), 99);
    EXPECT_DOUBLE_EQ(mis::percentile(v, 1.00), 100);
    EXPECT_DOUBLE_EQ(mis::percentile({}, 0.5), 0);
    EXPECT_DOUBLE_EQ(mis::percentile({7}, 0.99), 7);
}

TEST(LatencyTracker, HistogramIsCumulativeAndCountsEverything) {
    mis::LatencyTracker t;
    for (double ms : {1.0, 7.0, 7.0, 40.0, 60000.0}) t.record(ms);  // 60 s is above every bucket
    auto s = t.snapshot();
    EXPECT_EQ(s.count, 5u);
    EXPECT_DOUBLE_EQ(s.sum_ms, 60055.0);
    EXPECT_EQ(s.cumulative[0], 1u);      // <= 5 ms
    EXPECT_EQ(s.cumulative[1], 3u);      // <= 10 ms
    EXPECT_EQ(s.cumulative[3], 4u);      // <= 50 ms
    EXPECT_EQ(s.cumulative.back(), 4u);  // 60 s only lands in the implicit +Inf bucket
}

TEST(LatencyTracker, PercentilesUseOnlyTheRecentWindow) {
    mis::LatencyTracker t;
    for (size_t i = 0; i < mis::LatencyTracker::kWindow; ++i) t.record(1000.0);  // old, slow
    for (size_t i = 0; i < mis::LatencyTracker::kWindow; ++i) t.record(10.0);    // recent, fast
    auto s = t.snapshot();
    EXPECT_DOUBLE_EQ(s.p50_ms, 10.0);  // the slow past has rotated out of the window
    EXPECT_DOUBLE_EQ(s.p99_ms, 10.0);
    EXPECT_EQ(s.window, mis::LatencyTracker::kWindow);
    EXPECT_EQ(s.count, 2 * mis::LatencyTracker::kWindow);  // histogram keeps everything
}

TEST(LatencyTracker, ConcurrentRecordsAreAllCounted) {
    mis::LatencyTracker t;
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            for (int j = 0; j < 10000; ++j) t.record(3.0);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(t.snapshot().count, 80000u);
}

TEST(Metrics, PrometheusOutputHasHistogramAndPercentiles) {
    mis::Metrics m;
    m.record_request();
    m.record_latency(42.0);
    const std::string text = mis::render_prometheus({{"resnet50", 1, 0, m.snapshot()}});
    EXPECT_NE(text.find("# TYPE mis_request_latency_seconds histogram"), std::string::npos);
    EXPECT_NE(text.find(R"(mis_request_latency_seconds_bucket{model="resnet50",le="0.05"} 1)"),
              std::string::npos)
        << text;
    EXPECT_NE(text.find(R"(mis_request_latency_seconds_bucket{model="resnet50",le="+Inf"} 1)"),
              std::string::npos);
    EXPECT_NE(text.find(R"(mis_request_latency_seconds_count{model="resnet50"} 1)"),
              std::string::npos);
    EXPECT_NE(text.find(R"(mis_latency_p50_ms{model="resnet50"} 42)"), std::string::npos);
    EXPECT_NE(text.find(R"(mis_requests_total{model="resnet50"} 1)"), std::string::npos);
}
