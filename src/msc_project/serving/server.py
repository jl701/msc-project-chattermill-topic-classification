"""Loopback HTTP API with bounded queue and one serialised GPU worker.

The worker can combine reviews into a request batch. It owns all model calls,
including adapter-state transitions. Logs remain in the caller's local output
directory and must not be committed when they contain row-level predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from typing import Any


@dataclass
class Pending:
    payload: dict[str, str]
    received: float
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    status: int = 200


class Service:
    def __init__(self, engine, *, queue_size: int = 8, request_batch_size: int = 1,
                 batch_wait_ms: float = 0, timeout_s: float = 120,
                 log_path: Path | None = None) -> None:
        if min(queue_size, request_batch_size) < 1 or timeout_s <= 0 or batch_wait_ms < 0:
            raise ValueError('Invalid service capacity or timeout')
        self.engine = engine
        self.queue: queue.Queue[Pending] = queue.Queue(maxsize=queue_size)
        self.request_batch_size = request_batch_size
        self.batch_wait_ms = batch_wait_ms
        self.timeout_s = timeout_s
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.busy = False
        self.runtime_metrics = None
        self.counters = dict(received=0, accepted=0, rejected=0, inference_completed=0,
                             late_inference_completed=0, deadline_expired=0,
                             failed=0, batches=0, log_write_failures=0)
        self.worker = threading.Thread(target=self.run_worker, daemon=True, name='gpu-worker')
        self.worker.start()

    def count(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counters[name] += n

    def log(self, record: dict[str, Any]) -> None:
        if self.log_path:
            try:
                with self.lock, self.log_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            except (OSError, ValueError) as exc:
                # Observability failure must not overwrite already delivered results.
                self.count('log_write_failures')
                print('Worker log failed: ' + type(exc).__name__, file=sys.stderr, flush=True)

    def submit(self, payload: dict[str, str]) -> Pending | None:
        now = time.perf_counter()
        pending = Pending(payload, now, now + self.timeout_s)
        self.count('received')
        try:
            self.queue.put_nowait(pending)
        except queue.Full:
            self.count('rejected')
            return None
        self.count('accepted')
        return pending

    def run_worker(self) -> None:
        while not self.stop.is_set():
            try:
                first = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            stop_collect = time.perf_counter() + self.batch_wait_ms / 1000
            while len(batch) < self.request_batch_size:
                try:
                    remaining = stop_collect - time.perf_counter()
                    item = (self.queue.get(timeout=remaining) if remaining > 0
                            else self.queue.get_nowait())
                    batch.append(item)
                except queue.Empty:
                    break
            live = []
            now = time.perf_counter()
            for item in batch:
                if item.cancelled.is_set() or now >= item.deadline:
                    item.status = 504
                    item.response = {'error': 'request_deadline_exceeded',
                                     'request_id': item.payload['request_id']}
                    item.done.set()
                    self.queue.task_done()
                else:
                    live.append(item)
            if not live:
                continue
            start = time.perf_counter()
            self.busy = True
            self.count('batches')
            try:
                results = self.engine.predict_batch([item.payload for item in live])
                if len(results) != len(live):
                    raise RuntimeError('Engine returned incorrect response count')
                end = time.perf_counter()
                for item, result in zip(live, results):
                    if result.get('request_id') != item.payload['request_id']:
                        raise RuntimeError('Engine response identity mismatch')
                for item, result in zip(live, results):
                    response = dict(result)
                    response['timings_ms'] = {
                        'queue_wait': (start - item.received) * 1000,
                        'worker_batch': (end - start) * 1000,
                        'server_total_to_ready': (end - item.received) * 1000,
                    }
                    response['request_batch_size'] = len(live)
                    item.response = response
                    item.done.set()
                    self.count('inference_completed')
                    if end > item.deadline or item.cancelled.is_set():
                        self.count('late_inference_completed')
                    self.log({'event': 'inference_completed',
                              'elapsed_s': end - self.started,
                              'late_or_cancelled': end > item.deadline or item.cancelled.is_set(),
                              'response': response})
            except Exception as exc:
                self.count('failed', len(live))
                self.log({'event': 'worker_error', 'type': type(exc).__name__,
                          'traceback': traceback.format_exc(),
                          'request_ids': [i.payload['request_id'] for i in live]})
                for item in live:
                    item.status = 503
                    item.response = {'request_id': item.payload['request_id'],
                                     'error': 'inference_failed', 'type': type(exc).__name__}
                    item.done.set()
            finally:
                self.busy = False
                for _ in live:
                    self.queue.task_done()

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            counters = dict(self.counters)
        runtime = self.runtime_metrics() if self.runtime_metrics is not None else {}
        return {**counters, 'queue_depth': self.queue.qsize(), 'worker_busy': self.busy,
                'unfinished_tasks': self.queue.unfinished_tasks,
                'uptime_s': time.perf_counter() - self.started,
                'request_batch_size': self.request_batch_size,
                'batch_wait_ms': self.batch_wait_ms, 'queue_capacity': self.queue.maxsize,
                'success_metric_note': 'Inference completion is not HTTP success; use client-observed outcomes.',
                'runtime': runtime}


def validate_payload(value: Any, max_chars: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError('Request must be a JSON object')
    if set(value) - {'request_id', 'text'}:
        raise ValueError('Only request_id and text are accepted; labels are not API inputs')
    rid = value.get('request_id')
    text = value.get('text')
    if not isinstance(rid, str) or not rid.strip() or len(rid) > 160:
        raise ValueError('request_id must be a nonempty string of at most 160 characters')
    if not isinstance(text, str) or not text.strip() or len(text) > max_chars:
        raise ValueError(f'text must be a nonempty string of at most {max_chars} characters')
    return {'request_id': rid, 'text': text}


def handler_type(service: Service, *, max_chars: int = 20000, max_body_bytes: int = 131072):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            return

        def reply(self, code: int, value: dict[str, Any]) -> None:
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            try:
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            self.close_connection = True

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'ready': service.worker.is_alive(),
                                 'variant': getattr(service.engine, 'variant', 'unknown')})
            elif self.path == '/metrics':
                self.reply(200, service.metrics())
            else:
                self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path == '/shutdown':
                self.reply(200, {'stopping': True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != '/predict':
                self.reply(404, {'error': 'not_found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= max_body_bytes:
                    self.reply(413, {'error': 'invalid_body_size'})
                    return
                self.connection.settimeout(15)
                raw = self.rfile.read(length)
                payload = validate_payload(json.loads(raw.decode('utf-8')), max_chars)
            except (ValueError, UnicodeDecodeError, TimeoutError) as exc:
                self.reply(400, {'error': 'invalid_input', 'detail': str(exc)})
                return
            item = service.submit(payload)
            if item is None:
                self.reply(429, {'error': 'queue_full', 'request_id': payload['request_id']})
                return
            item.done.wait(max(0, item.deadline - time.perf_counter()))
            if not item.done.is_set() or time.perf_counter() > item.deadline:
                item.cancelled.set()
                service.count('deadline_expired')
                self.reply(504, {'error': 'request_deadline_exceeded',
                                 'request_id': payload['request_id']})
            else:
                self.reply(item.status, item.response or {'error': 'missing_result'})
    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--variant', choices=['few_shot', 'qlora', 'hybrid'], required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--pair-batch-size', type=int, default=4)
    parser.add_argument('--max-batch-tokens', type=int, default=4096)
    parser.add_argument('--request-batch-size', type=int, default=1)
    parser.add_argument('--batch-wait-ms', type=float, default=0)
    parser.add_argument('--queue-size', type=int, default=8)
    parser.add_argument('--timeout-s', type=float, default=120)
    parser.add_argument('--logits-mode', choices=['full', 'last_token'], default='last_token')
    parser.add_argument('--prefix-cache', choices=['off', 'on'], default='off')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import importlib.metadata
    import platform
    import torch
    from msc_project.serving.inference import QwenServingEngine

    start = time.perf_counter()
    engine = QwenServingEngine.from_artifacts(
        args.artifacts, variant=args.variant, pair_batch_size=args.pair_batch_size,
        max_batch_tokens=args.max_batch_tokens, logits_mode=args.logits_mode,
        enable_prefix_cache=args.prefix_cache == 'on', device='cuda', local_files_only=True)
    load_seconds = time.perf_counter() - start
    metadata = {'variant': args.variant, 'model_load_s': load_seconds,
                'platform': platform.platform(), 'python': platform.python_version(),
                'device': torch.cuda.get_device_name(),
                'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
                'versions': {n: importlib.metadata.version(n) for n in
                    ['torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'numpy']},
                'arguments': {k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()},
                'engine': engine.metadata(),
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ['server.py', 'inference.py', 'prefix_cache.py']}}
    (args.output_dir / 'server_metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    service = Service(engine, queue_size=args.queue_size,
                      request_batch_size=args.request_batch_size,
                      batch_wait_ms=args.batch_wait_ms, timeout_s=args.timeout_s,
                      log_path=args.output_dir / 'worker_events.jsonl')
    service.runtime_metrics = lambda: {
        'cuda_allocated_bytes': torch.cuda.memory_allocated(),
        'cuda_reserved_bytes': torch.cuda.memory_reserved(),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'scope': 'resident process counters since model load; not isolated per-request peaks',
    }
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_type(service))
    server.daemon_threads = True
    print(json.dumps({'ready': True, 'port': args.port, 'model_load_s': load_seconds}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        service.stop.set()
        service.worker.join(timeout=args.timeout_s + 10)
        server.server_close()
        (args.output_dir / 'server_final_metrics.json').write_text(
            json.dumps(service.metrics(), ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
