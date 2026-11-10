"""Service identity, admission and deadline checks using a synthetic engine."""
import threading
import time

import pytest

from http.server import ThreadingHTTPServer

from msc_project.serving.server import Service, handler_type, validate_payload


class Engine:
    variant = 'synthetic'

    def __init__(self, release=None):
        self.release = release
        self.started = threading.Event()

    def predict_batch(self, rows):
        self.started.set()
        if self.release:
            assert self.release.wait(3)
        return [{'request_id': r['request_id'], 'predictions': []} for r in rows]


def test_labels_cannot_enter_prediction_api():
    with pytest.raises(ValueError):
        validate_payload({'request_id': 'x', 'text': 'sample', 'labels': []}, 100)
    for text in ['', '  ', 42, 'x' * 101]:
        with pytest.raises(ValueError):
            validate_payload({'request_id': 'x', 'text': text}, 100)


def test_worker_preserves_identities_and_bounded_queue():
    release = threading.Event()
    engine = Engine(release)
    service = Service(engine, queue_size=1, request_batch_size=1)
    try:
        first = service.submit({'request_id': 'one', 'text': 'first'})
        assert engine.started.wait(1)
        second = service.submit({'request_id': 'two', 'text': 'second'})
        assert service.submit({'request_id': 'three', 'text': 'third'}) is None
        release.set()
        assert first.done.wait(2) and second.done.wait(2)
        assert first.response['request_id'] == 'one'
        assert second.response['request_id'] == 'two'
        assert service.metrics()['rejected'] == 1
    finally:
        release.set()
        service.stop.set()
        service.worker.join(2)


def test_cancelled_queue_entry_does_not_run():
    release = threading.Event()
    engine = Engine(release)
    service = Service(engine, queue_size=2)
    try:
        first = service.submit({'request_id': 'one', 'text': 'first'})
        assert engine.started.wait(1)
        second = service.submit({'request_id': 'two', 'text': 'second'})
        second.cancelled.set()
        release.set()
        assert first.done.wait(1) and second.done.wait(1)
        assert second.status == 504
        assert service.metrics()['inference_completed'] == 1
    finally:
        release.set()
        service.stop.set()
        service.worker.join(2)


def test_actual_http_input_contract_without_model_work():
    from run_serving_study import probe_api_contract

    service = Service(Engine())
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_type(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = probe_api_contract(f'http://127.0.0.1:{server.server_port}')
        assert result['gpu_requests_admitted'] == 0
        assert all(item['pass'] for item in result['checks'])
    finally:
        server.shutdown()
        server.server_close()
        service.stop.set()
        service.worker.join(2)


def test_logging_failure_does_not_replace_completed_response(tmp_path):
    service = Service(Engine(), log_path=tmp_path)
    try:
        pending = service.submit({'request_id': 'logging-case', 'text': 'synthetic'})
        assert pending.done.wait(2)
        deadline = time.monotonic() + 2
        while service.queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pending.status == 200
        assert pending.response['request_id'] == 'logging-case'
        assert service.metrics()['log_write_failures'] == 1
        assert service.metrics()['failed'] == 0
    finally:
        service.stop.set()
        service.worker.join(2)


@pytest.mark.parametrize("declared,body", [(131073,b"x"), (10**12,b"")])
def test_oversize_slow_or_incomplete_body_rejection_is_time_bounded(declared,body):
    import socket
    service=Service(Engine())
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_type(service))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with socket.create_connection(('127.0.0.1',server.server_port),timeout=2) as client:
            request=(f'POST /predict HTTP/1.1\r\nHost: localhost\r\nContent-Length: {declared}\r\n\r\n').encode()+body
            started=time.monotonic()
            client.sendall(request)
            reply=client.recv(4096)
            assert b' 413 ' in reply.split(b'\r\n',1)[0]
            assert time.monotonic()-started < 1.5
        assert service.metrics()['accepted'] == 0
    finally:
        server.shutdown()
        server.server_close()
        service.stop.set()
        service.worker.join(2)
