#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

namespace mis {

using Clock = std::chrono::steady_clock;  // monotonic: never jumps when system time changes

// Thread-safe bounded FIFO queue with batch pop, used between HTTP handlers (producers)
// and batcher workers (consumers). Generic over T so it can be unit-tested with ints.
template <typename T>
class BoundedQueue {
public:
    struct Entry {
        T value;
        Clock::time_point enqueued_at;  // stamped by push(); drives the batching deadline
    };

    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    // Returns false (and drops `value`) if the queue is full or closed.
    // The caller turns false into HTTP 503: this is the backpressure point.
    bool push(T value) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_ || items_.size() >= capacity_) return false;
            items_.push_back({std::move(value), Clock::now()});
        }
        // notify_all, not notify_one: a worker that is filling a batch must see the
        // new size, and with a handful of workers waking them all is cheap.
        // Notifying after unlocking avoids waking a thread that then blocks on mu_.
        cv_.notify_all();
        return true;
    }

    // Blocks until at least one item exists, then keeps collecting until there are
    // max_n items or max_wait has passed since the FIRST item was enqueued.
    // Returns up to max_n items in FIFO order.
    // Returns an empty vector only when the queue is closed AND fully drained,
    // which tells a worker to exit.
    std::vector<Entry> pop_batch(size_t max_n, std::chrono::milliseconds max_wait) {
        std::unique_lock<std::mutex> lock(mu_);
        while (true) {
            // Phase 1: sleep until there is work or we are shutting down.
            // The predicate is re-checked on every wakeup, so spurious wakeups are harmless.
            cv_.wait(lock, [&] { return !items_.empty() || closed_; });
            if (items_.empty()) return {};  // closed and nothing left

            // Phase 2: wait for the batch to fill, but never past the first item's deadline.
            // wait_until releases the lock while sleeping, so producers can keep pushing.
            const auto deadline = items_.front().enqueued_at + max_wait;
            cv_.wait_until(lock, deadline, [&] { return items_.size() >= max_n || closed_; });

            // Another worker may have taken everything while we slept; start over.
            if (items_.empty()) continue;

            const size_t n = std::min(max_n, items_.size());
            std::vector<Entry> batch;
            batch.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                batch.push_back(std::move(items_.front()));
                items_.pop_front();
            }
            return batch;
        }
    }

    // Stops accepting new items and wakes all waiting workers. Items already queued
    // are still returned by pop_batch (that is the "drain" in graceful shutdown).
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return items_.size();
    }

    size_t capacity() const { return capacity_; }

private:
    const size_t capacity_;
    mutable std::mutex mu_;  // mutable so size() can lock in a const method
    std::condition_variable cv_;
    std::deque<Entry> items_;
    bool closed_ = false;
};

}  // namespace mis
