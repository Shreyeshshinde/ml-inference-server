#include "batcher.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "logging.h"
#include "preprocess.h"

namespace mis {

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

Batcher::Batcher(ModelConfig config, std::shared_ptr<const ModelVersion> initial,
                 Labels labels, Metrics* metrics)
    : config_(std::move(config)),
      current_(std::move(initial)),
      labels_(labels ? std::move(labels) : std::make_shared<const std::vector<std::string>>()),
      metrics_(metrics),
      queue_(static_cast<size_t>(config_.queue_limit)) {
    if (!current_ || !current_->runner) throw std::invalid_argument("Batcher needs a model");
}

Batcher::~Batcher() { shutdown(); }

std::shared_ptr<const ModelVersion> Batcher::swap_model(
    std::shared_ptr<const ModelVersion> next) {
    // One indivisible step: every reader sees either the old pointer or the new one.
    return std::atomic_exchange(&current_, std::move(next));
}

void Batcher::start() {
    for (int i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

std::optional<std::future<InferenceResult>> Batcher::submit(cv::Mat image, int top_k,
                                                            Clock::time_point deadline) {
    InferenceRequest req;
    req.image = std::move(image);
    req.top_k = top_k;
    req.deadline = deadline;
    // Take the future BEFORE moving the promise into the queue.
    std::future<InferenceResult> fut = req.promise.get_future();
    if (!queue_.push(std::move(req))) return std::nullopt;  // full: caller sends 503
    return fut;
}

void Batcher::shutdown() {
    if (shut_down_.exchange(true)) return;  // already done
    queue_.close();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    // Only non-empty if no workers ran. pop_batch on a closed queue never blocks.
    for (auto batch = queue_.pop_batch(1024, std::chrono::milliseconds(0)); !batch.empty();
         batch = queue_.pop_batch(1024, std::chrono::milliseconds(0))) {
        for (auto& e : batch) {
            e.value.promise.set_exception(
                std::make_exception_ptr(std::runtime_error("server shutting down")));
        }
    }
}

void Batcher::worker_loop(int worker_id) {
    const auto max_wait = std::chrono::milliseconds(config_.max_wait_ms);
    const auto max_n = static_cast<size_t>(config_.max_batch_size);
    while (true) {
        auto batch = queue_.pop_batch(max_n, max_wait);
        if (batch.empty()) return;  // queue closed and drained
        process_batch(batch, worker_id);
    }
}

void Batcher::process_batch(std::vector<Entry>& batch, int worker_id) {
    const auto t_start = Clock::now();

    // Drop requests whose handler already timed out (504): computing them is wasted work.
    std::vector<Entry*> live;
    live.reserve(batch.size());
    for (auto& e : batch) {
        if (t_start > e.value.deadline) {
            e.value.promise.set_exception(
                std::make_exception_ptr(std::runtime_error("request expired in queue")));
            if (metrics_) metrics_->record_expired();
        } else {
            live.push_back(&e);
        }
    }
    if (live.empty()) return;

    const int n = static_cast<int>(live.size());
    try {
        // 1. Preprocess every image straight into its slot of one [N,3,H,W] buffer.
        const auto& pcfg = config_.preprocess;
        const size_t per_image = tensor_size(pcfg);
        std::vector<float> input(per_image * n);
        for (int i = 0; i < n; ++i) {
            preprocess_into(live[i]->value.image, pcfg, input.data() + i * per_image);
        }
        const auto t_inf = Clock::now();

        // 2. One model run for the whole batch.
        // Take our own copy of the current version ONCE per batch. If a reload swaps
        // current_ while we run, this copy keeps the old model alive until we finish,
        // and the whole batch is computed by a single, consistent version.
        const std::shared_ptr<const ModelVersion> model = std::atomic_load(&current_);
        std::vector<float> logits = model->runner->run(input.data(), n, pcfg.crop, pcfg.crop);
        const auto t_post = Clock::now();

        // 3. Split the [N, classes] output: row i belongs to request i.
        const size_t classes = static_cast<size_t>(model->runner->num_classes());
        std::vector<InferenceResult> results(n);
        for (int i = 0; i < n; ++i) {
            auto probs = softmax(logits.data() + i * classes, classes);
            results[i].predictions = top_k(probs, live[i]->value.top_k, *labels_);
            results[i].model_version = model->version;
        }
        const auto t_end = Clock::now();

        // 4. Fulfill promises only after everything succeeded, so the catch block below
        // never touches a promise that was already set.
        for (int i = 0; i < n; ++i) {
            auto& r = results[i];
            r.batch_size = n;
            r.queue_wait_ms = ms_between(live[i]->enqueued_at, t_start);
            r.preprocess_ms = ms_between(t_start, t_inf);
            r.inference_ms = ms_between(t_inf, t_post);
            r.postprocess_ms = ms_between(t_post, t_end);
            live[i]->value.promise.set_value(std::move(r));
        }
        if (metrics_) metrics_->record_batch(n);

        if (!request_logging()) return;
        char line[200];
        std::snprintf(line, sizeof(line),
                      "batch model=%s v=%d worker=%d size=%d pre_ms=%.2f inf_ms=%.2f "
                      "post_ms=%.2f\n",
                      config_.name.c_str(), model->version, worker_id, n,
                      ms_between(t_start, t_inf),
                      ms_between(t_inf, t_post), ms_between(t_post, t_end));
        std::cerr << line;  // single << of a full line: no interleaving mid-line
    } catch (...) {
        // Any failure fails the WHOLE batch: every waiting client gets an error
        // instead of hanging until its timeout.
        auto err = std::current_exception();
        for (auto* e : live) e->value.promise.set_exception(err);
        if (metrics_) metrics_->record_batch_failure();
    }
}

}  // namespace mis
