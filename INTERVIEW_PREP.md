# Interview prep: Mini Triton

Every number here comes from the measurements in [README.md](README.md) and `results/`. If a number is not in those files, do not say it.

## Resume bullets

- Built a **C++17 multi-model ML inference server** (ONNX Runtime, cpp-httplib) with dynamic batching, per-model bounded queues and worker pools, 503 backpressure and 504 timeouts. It achieved **1.5 to 2.0x the throughput of a FastAPI + PyTorch baseline** on ResNet-50 at 100 concurrent users (29.8 vs 17.6 images/s).
- Implemented **zero-downtime model hot reload** with an atomic `std::shared_ptr` swap, so in-flight batches finish on the old version: **0 failed requests out of 882** across two live reloads. Load-tested up to 200 concurrent users with **no crashes or 500s**, after finding and fixing an overload collapse where rejected requests still decoded images: throughput had dropped to 110 images per minute.
- Applied **INT8 static quantization**: ResNet-50 became **3.9x smaller** (102 to 26 MB) and about 1.3x faster with only a **0.13-point top-1 drop** on 3,925 Imagenette images. Fixed MobileNetV3's 13-point loss down to 2.7 points by quantizing only Conv/Gemm with percentile calibration. Shipped with 56 GoogleTest tests, AddressSanitizer, Docker and GitHub Actions CI.

Shorter one-line version for a tight resume:

- C++17 ML inference server (ONNX Runtime) with dynamic batching, backpressure and zero-downtime reload; 1.5 to 2.0x a FastAPI + PyTorch baseline, 0 failed requests across live reloads, INT8 ResNet-50 3.9x smaller at -0.13 top-1.

## 2-minute pitch ("Tell me about your project")

> I built a small version of NVIDIA Triton in C++: a server that keeps ML models loaded in memory and serves image classification over HTTP.
>
> The core idea is dynamic batching. Requests go into a bounded queue per model, and worker threads pull them out in groups: they wait for one request, then keep collecting until the batch is full or a few milliseconds have passed since the first one arrived. The worker stacks the images into one tensor, runs ONNX Runtime once, and hands each result back through a promise and future, so every HTTP thread gets exactly its own answer.
>
> Around that I built the production pieces. A full queue returns 503 immediately instead of piling up memory, slow requests get 504, and the server can swap in a new model version with zero downtime: it loads the new model on the side and swaps a shared pointer atomically. In-flight batches keep the old version alive until they finish. Under live traffic, 882 out of 882 requests succeeded across two reloads.
>
> Then I measured everything. Against a FastAPI plus PyTorch baseline on the same CPU cores, the C++ server had 1.5 to 2 times the throughput. Benchmarking also found a real bug: under overload, rejected requests were still decoding their images, which starved the inference workers. Rejecting before decoding fixed it.
>
> One honest finding: on CPU, batching only added about 5 percent, because ONNX Runtime already spreads one image across several cores. That told me batching is mainly a GPU optimization.
>
> I also added INT8 quantization. ResNet-50 got about 4 times smaller with almost no accuracy loss, and MobileNet needed a custom recipe. Everything has unit tests, AddressSanitizer, Docker and CI.

## 20 likely questions

### Design

**1. Why C++ instead of Python?**
To control the serving path: threads, queues, memory layout and copies. Both servers use optimized native kernels for the model math, so the difference comes from everything around the model call: no GIL, preprocessing written straight into the batch buffer, batching, and cheap pointer copies. Measured result: 1.5 to 2.0x the throughput of the FastAPI + PyTorch baseline at 100 users.

**2. What is dynamic batching and why use it?**
Group requests that arrive close together into one model run, so the fixed per-call overhead is shared. My rule: wait for one request, then keep collecting until there are `max_batch_size` requests or `max_wait_ms` has passed since the *first* request was enqueued. On this CPU it only gave about 5% more throughput (27.3 to 28.8 images/s from batch 1 to 8), because ONNX Runtime already parallelizes a single image across 4 threads. On a GPU, where one image leaves most of the device idle, the gain is much larger.

**3. How do batch size and wait time trade throughput against latency?**
- **Bigger batch:** better hardware use, but the whole batch waits for the slowest part, and requests may wait for the batch to fill.
- **Longer wait:** fuller batches under light load, but up to `max_wait_ms` of extra latency for every request.

In my sweep, batch 16 had the worst p99 (4,267 ms vs 3,933 ms at batch 1) with no throughput gain over batch 8. I measure the deadline from the first request's *enqueue* time, so no request waits longer than `max_wait_ms` for a batch to fill.

**4. Why one queue per model instead of one shared queue?**
- **Isolation:** if ResNet-50 is overloaded, only its queue fills and returns 503; MobileNet keeps serving.
- **Different settings:** each model can have its own batch size, wait time and thread budget.
- **Simpler batches:** a shared queue would need a worker to sort a mixed batch by model before running it.

**5. Why did latency at 100 users come out at about 3.4 seconds if one inference takes about 45 ms?**
Little's law: in a closed loop, latency is about users divided by throughput, and 100 / 29.8 is about 3.4 s. Almost all of it is time waiting in the queue. You can't make that faster with a better queue, only with more compute or fewer users per server. That's exactly what the 503 limit and horizontal scaling are for.

### Concurrency

**6. How does your request queue work?**
- **Structure:** a `std::deque` protected by a `std::mutex`, plus a `std::condition_variable`.
- **`push()`:** takes the lock, returns false if the queue is full or closed (the backpressure point), otherwise appends the item with a timestamp, releases the lock, and notifies the workers.
- **`pop_batch()`:**
  1. waits until any item exists;
  2. computes the deadline as the first item's enqueue time plus `max_wait`;
  3. waits again until the batch is full or the deadline passes;
  4. takes up to `max_n` items.

A stress test pushes 20,000 items through 4 producers and 3 consumers and checks that each comes out exactly once.

**7. Mutex vs condition variable: what does each do?**
- **Mutex:** mutual exclusion. Only one thread touches the deque at a time.
- **Condition variable:** lets a thread sleep until something changes, without burning CPU. `wait()` atomically releases the mutex and sleeps, then reacquires the mutex when woken.

Wakeups can be spurious, so I always wait with a predicate (`cv.wait(lock, [&]{ return !items.empty() || closed; })`), which re-checks the real condition. I use `notify_all` because a worker that is filling a batch needs to see every new item, and with a few workers that's cheap.

**8. How do results get back to the right HTTP request?**
Each request carries a `std::promise<InferenceResult>`; the HTTP thread keeps the matching `std::future`. The worker runs the batch, splits row i of the output to request i, and calls `set_value` on each promise. The handler waits with `future.wait_until(deadline)`, so it never blocks forever. A test sends 50 parallel requests with 5 different images and checks that each gets its own correct result.

**9. What happens if the model throws in the middle of a batch?**
The worker catches everything and calls `set_exception` on every promise in that batch, so no client hangs until its timeout; each handler then returns 500. I only call `set_value` after the whole batch has succeeded. That way the catch block never touches a promise that was already set, which would itself throw. A test feeds images of the wrong size so `Run()` throws, and checks that all 6 futures complete with an exception.

**10. How do you avoid oversubscribing the CPU?**
Each worker runs ONNX Runtime with `intra_op_threads` threads, so a model uses `num_workers x intra_op_threads` compute threads. The server sums this over all models and warns at startup if it exceeds the core count. More threads than cores just makes them compete. The Python baseline showed the same effect: 4 PyTorch threads beat 8 and 12 (23.7 vs 21.1 vs 19.6 images/s).

### Zero-downtime reload

**11. How does the reload work?**
Each batcher holds a `std::shared_ptr<const ModelVersion>` (version number, path, ONNX session). A reload goes through four steps:
1. **Load** the new file in the request thread while the old version keeps serving.
2. **Warm up:** run one dummy inference to prove the new model works, and check it has the same number of classes.
3. **Swap** the pointer with `std::atomic_exchange`.
4. **Report** the old and new version.

The slow part (about 2 s for ResNet-50) all happens before the swap, and the swap itself is one atomic pointer exchange.

**12. Why are in-flight requests safe during a swap?**
Each worker calls `std::atomic_load` once per batch and keeps that `shared_ptr` copy until the batch finishes. A batch therefore always runs entirely on one version, and the copy keeps that version alive even if the registry has already switched. The old model is freed automatically when the last copy goes away, which a test verifies with a `weak_ptr`. The response reports the version that actually ran the batch, so the reload test can prove both versions answered with 0 failures.

**13. Why `std::atomic_load` and not just assigning the `shared_ptr`?**
A `shared_ptr` object is not safe to read in one thread while another thread writes it. The reference-count updates are atomic, but the pointer object itself is not. The C++11 free functions `std::atomic_load` / `std::atomic_exchange` make access to the pointer object atomic. In C++20 I would use `std::atomic<std::shared_ptr<T>>`. A mutex around the pointer would also work.

**14. What stops a bad reload from breaking the service?**
- **One reload at a time:** a second reload of the same model gets 409, using `try_lock`.
- **Restricted path:** it must resolve inside `models/` (canonical path check), so a request can't load arbitrary files.
- **Validation before swap:** if loading or the warm-up fails, or the class count differs, the old version keeps serving and the API returns 400.

### Failure handling

**15. How does backpressure work, and what bug did you find?**
A bounded queue: when it is full, `push` returns false and the client gets 503 with `Retry-After: 1`, instead of the server running out of memory.

The bug: under overload, my handler decoded the JPEG *before* checking the queue. Every rejected request still cost a full image decode, so thousands of 503s per second starved the inference workers. Throughput collapsed to 110 images in 60 seconds. I now check queue fullness before decoding. After the fix, the server held about 40 images/s while returning about 40,000 503s per minute at 100 users.

A second bug came from the same kind of testing: httplib's default listen backlog was 5, so the kernel dropped connections in bursts before my code ever ran. I raised it to 1024.

**16. How do timeouts work?**
Each request gets a deadline when it arrives (`request_timeout_ms`). The handler waits on the future only until that deadline, then returns 504. Workers also check deadlines when they take a batch and skip requests that have already expired, so no CPU is spent on answers nobody will read.

**17. How does graceful shutdown work?**
SIGINT/SIGTERM are blocked in all threads at startup, and a dedicated thread waits for them with `sigwait`. That's safer than a signal handler, which may only call a few async-signal-safe functions. On a signal, the order is:
1. Stop the HTTP server. Connections stop being accepted, and in-flight HTTP requests finish while the workers are still running.
2. Close each queue, so no new work can be added.
3. Workers drain what's left in the queues.
4. Join the workers.

`docker stop` follows exactly this path; I checked the logs.

### Scaling

**18. How would you run this on 10 machines?**
The server keeps no per-user state, so the plan is:
- **Load balancer:** put 10 identical copies behind one, routing only to replicas whose `/health` is ok.
- **503 handling:** treat 503 as "try another replica" with backoff.
- **Model files:** keep them in shared storage or the image.
- **Rolling upgrades:** use the reload endpoint, so no replica has to stop.
- **Monitoring:** scrape `/metrics` from all replicas into Prometheus. The latency histogram can be aggregated across machines; per-server p99 values cannot simply be averaged.

**19. How would you autoscale?**
On queue length and p99 latency from `/metrics`, not on CPU. Queue length rises before users feel latency, and with closed-loop traffic, latency is roughly users divided by throughput, so more replicas directly cut queueing time. Scale out when the average queue length stays high or 503s appear; scale in slowly to avoid flapping.

### Quantization

**20. What does INT8 quantization do, and what did it cost?**
It stores each weight and activation as an 8-bit integer plus a scale (`real ≈ scale x (int8 - zero_point)`), so the model is 4x smaller and the CPU can use fast integer instructions (VNNI).

*Static* means activation ranges are measured once, on 200 calibration images from the **train** split, so the validation images stay unseen.

Results:
- **ResNet-50:** 3.9x smaller, about 1.3x faster, top-1 86.83% to 86.70%.
- **MobileNetV3, default recipe:** lost 13 points and ran 3.7x *slower*. Its HardSwish and squeeze-excite ops have no fast INT8 kernels, so the graph filled with quantize/dequantize steps.
- **MobileNetV3, fixed recipe:** quantizing only Conv/Gemm with percentile calibration brought the drop to 2.7 points. It was still slower than FP32, so for a model that small, INT8 mainly buys a smaller file.

## Follow-ups to be ready for

- **"What would you do differently?"**
  - Use an event-loop HTTP server; cpp-httplib holds one thread per open connection.
  - Reject overload before reading the request body.
  - Benchmark on a desktop or cloud VM instead of a laptop. Mine ran in two speed states, about 44 or about 28 images/s, so I only compared runs within one experiment.
- **"How do you know it's correct?"**
  - **ONNX export:** PyTorch vs ONNX outputs agree within 1.4e-5.
  - **C++ vs Python:** top-5 match on 20 of 20 images for all four model files. This caught a rounding bug: `std::round` vs Python's round-half-to-even shifted the center crop by one pixel.
  - **Unit tests:** 56 GoogleTest tests.
  - **Memory:** AddressSanitizer in CI.
- **"What's tested in CI?"** Build with `-Werror`, tests under AddressSanitizer, and a format check. The 24 tests that need model files skip in CI, because the models are not in Git, and run locally.
