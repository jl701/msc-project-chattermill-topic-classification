# Local aspect and sentiment service

This service turns the frozen two-stage Qwen classification workflow into a loopback HTTP API. Each request analyses one complete review against the twelve candidate aspects stored in the selected artifact bundle. The service returns selected aspects and one or at most two sentiments for each selected aspect. The caller does not provide labels or change the taxonomy during a run.

The three configurations are `few_shot` (frozen Qwen for both stages with demonstrations), `qlora` (the selected shared two-task adapter for both stages), and `hybrid` (frozen few-shot detection followed by adapted sentiment). They use their own previously frozen thresholds. A single resident base model and one adapter are serialised by one GPU worker; concurrent HTTP requests enter a bounded queue.

## Prepare and verify

Install the pinned optional environment in `requirements-serving.txt` on a compatible CUDA machine. Artifact preparation requires the author's local dataset and frozen model backup roots. They are not distributed in this public repository. See `scripts/prepare_serving_replay.py --help` and the audit in `artifact_audit.md`.

```powershell
python scripts/check_serving_gpu.py --samples 3 --pair-batch-size 4 --check-frozen-base
python scripts/profile_serving_engine.py --samples 6 --repeats 3
```

Correctness checks compare the original full-grid scorer and decoder with conditional sentiment execution, reduced vocabulary-projection positions, different review batch sizes and an independently loaded frozen base. A completed check records probability differences as well as final-label agreement. Performance profiling uses only the development inputs and cannot be reported as HTTP latency.

## Run a fixed local service

```powershell
python scripts/serve_chattermill.py --artifacts outputs/serving_20261109/artifacts.json --variant hybrid --pair-batch-size 4 --max-batch-tokens 4096 --request-batch-size 1 --queue-size 8 --timeout-s 120 --output-dir outputs/serving_20261109/example_server
```

This example uses the default `--prefix-cache off`. The registered primary
campaign explicitly used `--prefix-cache on`; its latency must not be attributed
to the default uncached command. Prefix reuse is a separately evaluated
configuration: the complete replay changed six of 957 hybrid prediction sets,
even though its overall pair F1 was unchanged. The recorded comparison and
deployment boundaries are documented in [the experiment results](results_20261109.md).

The API binds only to `127.0.0.1`. `GET /health` reports readiness and `GET /metrics` reports runtime counts. `POST /predict` accepts only `request_id` and `text`. For example, this is a synthetic input rather than a dataset record:

```json
{"request_id":"example-001","text":"The app is easy to use and the support team is helpful."}
```

Successful responses contain the same request ID, variant/fold identifiers, selected aspect–sentiment predictions, all twelve aspect scores and diagnostics. Batch-stage timings describe the whole combined batch and must not be mistaken for each constituent request's GPU time. Client end-to-end latency is measured independently.

Invalid inputs return 400 or 413. Queue saturation returns 429. A service deadline returns 504. Inference exceptions return 503 and are retained in local diagnostics. `POST /shutdown` is a local experiment-control endpoint. This implementation is an experimental loopback service, not an internet-facing authenticated deployment.

## Measure and interpret

```powershell
python scripts/benchmark_serving.py --replay outputs/serving_20261109/replay_development.jsonl --output outputs/serving_20261109/example_load --mode closed --requests 30 --concurrency 1 --timeout 130 --expected-variant hybrid
```

Use the registered campaign runner for serial sessions, equipment telemetry, an explicit per-run plan and full draining between workloads. Client timeout must exceed the service deadline. All failed or unsent attempts stay in the denominator. Server `inference_completed` is deliberately separate from successful HTTP responses because work may finish after a client deadline.

Raw reviews, private paths, labels, responses and telemetry stay under ignored `outputs/`. Aggregate reporting must follow `reporting_scope.md`, particularly the distinction between the fixed single-fold replay and the dissertation's twelve-fold held-out mean. Optional prefix reuse is separately registered in `serving_prefix_cache_20261109.json` and is admitted only after its additional correctness checks.
