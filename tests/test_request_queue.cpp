#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "request_queue.h"

using mis::BoundedQueue;
using namespace std::chrono_literals;

TEST(RequestQueue, PushRejectsWhenFull) {
    BoundedQueue<int> q(2);
    EXPECT_TRUE(q.push(1));
    EXPECT_TRUE(q.push(2));
    EXPECT_FALSE(q.push(3));  // backpressure: caller returns 503
    EXPECT_EQ(q.size(), 2u);
}

TEST(RequestQueue, FullBatchReturnsImmediatelyInFifoOrder) {
    BoundedQueue<int> q(100);
    for (int i = 0; i < 10; ++i) q.push(i);
    const auto t0 = mis::Clock::now();
    auto batch = q.pop_batch(4, 1000ms);  // enough items: must not wait the full second
    EXPECT_LT(mis::Clock::now() - t0, 100ms);
    ASSERT_EQ(batch.size(), 4u);
    for (int i = 0; i < 4; ++i) EXPECT_EQ(batch[i].value, i);
    EXPECT_EQ(q.size(), 6u);
}

TEST(RequestQueue, PartialBatchWaitsForMaxWait) {
    BoundedQueue<int> q(100);
    q.push(1);
    q.push(2);
    const auto t0 = mis::Clock::now();
    auto batch = q.pop_batch(8, 50ms);
    const auto waited = mis::Clock::now() - t0;
    EXPECT_EQ(batch.size(), 2u);
    EXPECT_GE(waited, 40ms);   // waited for more items...
    EXPECT_LT(waited, 500ms);  // ...but gave up near the deadline
}

TEST(RequestQueue, DeadlineCountsFromFirstEnqueueNotFromPop) {
    // The first request already waited 60ms in the queue, so with max_wait 50ms the
    // worker must send it right away instead of waiting another 50ms.
    BoundedQueue<int> q(100);
    q.push(1);
    std::this_thread::sleep_for(60ms);
    const auto t0 = mis::Clock::now();
    auto batch = q.pop_batch(8, 50ms);
    EXPECT_LT(mis::Clock::now() - t0, 20ms);
    EXPECT_EQ(batch.size(), 1u);
}

TEST(RequestQueue, ItemsArrivingDuringWaitJoinTheBatch) {
    BoundedQueue<int> q(100);
    std::thread producer([&] {
        for (int i = 0; i < 4; ++i) {
            q.push(i);
            std::this_thread::sleep_for(5ms);
        }
    });
    auto batch = q.pop_batch(4, 500ms);  // fills to 4 well before 500ms
    producer.join();
    EXPECT_EQ(batch.size(), 4u);
}

TEST(RequestQueue, CloseWakesBlockedConsumerAndDrains) {
    BoundedQueue<int> q(100);
    std::thread consumer([&] {
        auto batch = q.pop_batch(4, 10ms);  // blocks: queue is empty
        EXPECT_TRUE(batch.empty());         // woken by close() with nothing left
    });
    std::this_thread::sleep_for(20ms);
    q.close();
    consumer.join();

    BoundedQueue<int> q2(100);
    q2.push(7);
    q2.close();
    EXPECT_FALSE(q2.push(8));             // closed: no new work
    auto rest = q2.pop_batch(4, 1000ms);  // existing work is still handed out...
    ASSERT_EQ(rest.size(), 1u);           // ...without waiting for max_wait
    EXPECT_EQ(rest[0].value, 7);
    EXPECT_TRUE(q2.pop_batch(4, 0ms).empty());  // then the "exit" signal
}

TEST(RequestQueue, ConcurrentProducersAndConsumersLoseNothing) {
    // 4 producers x 5000 items, 3 consumers. Every item must come out exactly once.
    constexpr int kProducers = 4, kPerProducer = 5000, kTotal = kProducers * kPerProducer;
    BoundedQueue<int> q(64);  // small capacity forces producers to hit "full" and retry
    std::vector<std::atomic<int>> seen(kTotal);
    for (auto& s : seen) s = 0;

    std::vector<std::thread> consumers;
    for (int c = 0; c < 3; ++c) {
        consumers.emplace_back([&] {
            while (true) {
                auto batch = q.pop_batch(8, 1ms);
                if (batch.empty()) return;
                for (auto& e : batch) seen[e.value].fetch_add(1);
            }
        });
    }
    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            for (int i = 0; i < kPerProducer; ++i) {
                while (!q.push(p * kPerProducer + i)) std::this_thread::yield();
            }
        });
    }
    for (auto& t : producers) t.join();
    q.close();
    for (auto& t : consumers) t.join();

    for (int i = 0; i < kTotal; ++i) ASSERT_EQ(seen[i].load(), 1) << "item " << i;
}
