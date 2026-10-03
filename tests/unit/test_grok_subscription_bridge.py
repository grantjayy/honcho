from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request

import pytest

from scripts import grok_subscription_bridge as bridge


class UpstreamHandler(BaseHTTPRequestHandler):
    requests: list[tuple[dict[str, str], bytes]] = []
    response_body = b'{"choices":[{"message":{"tool_calls":[{"id":"x"}]}}]}'
    stream = False
    delay = 0

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        type(self).requests.append((dict(self.headers), body))
        if self.delay:
            import time

            time.sleep(self.delay)
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/event-stream" if self.stream else "application/json"
        )
        self.end_headers()
        if self.stream:
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.wfile.write(self.response_body)

    def log_message(self, *_args: Any) -> None:
        pass


@pytest.fixture
def running_bridge(monkeypatch: pytest.MonkeyPatch):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    UpstreamHandler.requests = []
    UpstreamHandler.stream = False
    UpstreamHandler.delay = 0
    token = iter(("subscription-one", "subscription-two"))
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")

    def resolver():
        return next(token), "https://cli-chat-proxy.grok.com"

    def transport(request, timeout):
        synthetic = request.full_url.replace(
            "https://cli-chat-proxy.grok.com",
            f"http://127.0.0.1:{upstream.server_port}",
        )
        request.full_url = synthetic
        return bridge.urlopen_transport(request, timeout)

    server = bridge.make_server(
        "127.0.0.1",
        0,
        resolver,
        timeout=0.5,
        transport=transport,
        default_headers=lambda: {"User-Agent": "hermes-test"},
    )
    threads = [
        threading.Thread(target=upstream.serve_forever),
        threading.Thread(target=server.serve_forever),
    ]
    for thread in threads:
        thread.start()
    yield server, upstream
    server.shutdown()
    upstream.shutdown()
    for thread in threads:
        thread.join()


def request(
    server: ThreadingHTTPServer,
    path: str,
    body: bytes = b'{"model":"grok-4.3"}',
    auth: str | None = "bridge-secret",
):
    import http.client

    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    if auth is not None:
        headers["Authorization"] = f"Bearer {auth}"
    connection.request("POST", path, body, headers)
    response = connection.getresponse()
    result = response.status, response.getheaders(), response.read()
    connection.close()
    return result


def test_auth_route_and_model_validation(running_bridge) -> None:
    server, _ = running_bridge
    assert request(server, "/v1/chat/completions", auth=None)[0] == 401
    assert request(server, "/v1/chat/completions", auth="wrong")[0] == 401
    assert request(server, "/other")[0] == 404
    assert (
        request(server, "/v1/chat/completions", b'{"model":"grok-4.3-mini"}')[0] == 400
    )


def test_forwards_json_and_fresh_subscription_credentials(running_bridge) -> None:
    server, upstream = running_bridge
    body = b'{"model":"grok-4.3","tools":[{"type":"function"}],"stream":false}'
    first = request(server, "/v1/chat/completions", body)
    second = request(server, "/v1/chat/completions", body)
    assert first[0] == second[0] == 200
    assert first[2] == second[2] == UpstreamHandler.response_body
    assert [headers["Authorization"] for headers, _ in UpstreamHandler.requests] == [
        "Bearer subscription-one",
        "Bearer subscription-two",
    ]
    assert all(
        headers["Host"] == f"127.0.0.1:{upstream.server_port}"
        for headers, _ in UpstreamHandler.requests
    )
    assert all(
        headers["User-Agent"] == "hermes-test"
        for headers, _ in UpstreamHandler.requests
    )
    assert all(received == body for _, received in UpstreamHandler.requests)


def test_preserves_structured_tool_call_response(running_bridge) -> None:
    server, _ = running_bridge
    body = (
        b'{"model":"grok-4.3","stream":false,"reasoning_effort":"medium",'
        b'"response_format":{"type":"json_schema","json_schema":{"name":"x"}},'
        b'"tools":[{"type":"function","function":{"name":"lookup",'
        b'"parameters":{"type":"object"}}}]}'
    )
    expected = (
        b'{"choices":[{"message":{"tool_calls":[{"id":"call_1",'
        b'"function":{"name":"lookup","arguments":"{\\"ok\\":true}"}}]}}],'
        b'"structured":{"ok":true},"usage":{"total_tokens":7}}'
    )
    UpstreamHandler.response_body = expected
    try:
        status, _, response_body = request(server, "/v1/chat/completions", body)
        assert status == 200
        assert response_body == expected
        assert UpstreamHandler.requests[-1][1] == body
    finally:
        UpstreamHandler.response_body = (
            b'{"choices":[{"message":{"tool_calls":[{"id":"x"}]}}]}'
        )


def test_streaming_is_forwarded(running_bridge) -> None:
    server, _ = running_bridge
    UpstreamHandler.stream = True
    status, headers, body = request(
        server, "/v1/chat/completions", b'{"model":"grok-4.3","stream":true}'
    )
    assert status == 200
    assert dict(headers)["Content-Type"] == "text/event-stream"
    assert b"data: [DONE]" in body


def test_streaming_delivers_event_before_upstream_finishes(running_bridge) -> None:
    server, _ = running_bridge
    released = threading.Event()
    UpstreamHandler.stream = True

    original_do_post = UpstreamHandler.do_POST

    def gated_post(self) -> None:
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
        self.wfile.flush()
        released.wait(2)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    UpstreamHandler.do_POST = gated_post
    connection = socket.create_connection(server.server_address, timeout=2)
    try:
        body = b'{"model":"grok-4.3","stream":true}'
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Authorization: Bearer bridge-secret\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        received = b""
        while b'data: {"choices"' not in received:
            received += connection.recv(4096)
        assert b"data: [DONE]" not in received
        released.set()
        while b"data: [DONE]" not in received:
            received += connection.recv(4096)
        assert b"data: [DONE]" in received
    finally:
        released.set()
        connection.close()
        UpstreamHandler.do_POST = original_do_post


def test_midstream_failure_closes_without_second_response(monkeypatch) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")

    def transport(_request, _timeout):
        def body():
            yield b"data: first\n\n"
            raise TimeoutError("synthetic timeout")

        return bridge.UpstreamResponse(
            200, [("Content-Type", "text/event-stream")], body()
        )

    server = bridge.make_server(
        "127.0.0.1",
        0,
        lambda: ("subscription", "https://cli-chat-proxy.grok.com"),
        transport=transport,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        connection = socket.create_connection(server.server_address, timeout=2)
        body = b'{"model":"grok-4.3","stream":true}'
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Authorization: Bearer bridge-secret\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        chunks = []
        while True:
            chunk = connection.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        result = b"".join(chunks)
        assert result.count(b"HTTP/") == 1
        assert b"data: first" in result
        assert b"upstream unavailable" not in result
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_timeout_fails_closed_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    calls = 0

    def resolver():
        nonlocal calls
        calls += 1
        return ("subscription", "https://cli-chat-proxy.grok.com")

    transport_calls = 0

    def transport(_request, _timeout):
        nonlocal transport_calls
        transport_calls += 1
        raise TimeoutError("synthetic timeout")

    server = bridge.make_server("127.0.0.1", 0, resolver, transport=transport)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        status, _, body = request(server, "/v1/chat/completions")
        assert status == 502
        assert body == b'{"error":{"message":"upstream unavailable"}}'
        assert calls == transport_calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_redirect_is_not_followed() -> None:
    target_hit = False

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(307)
            self.send_header("Location", "http://127.0.0.1:1/target")
            self.end_headers()

        def log_message(self, *_args: Any) -> None:
            pass

    redirect_server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=redirect_server.serve_forever)
    thread.start()
    try:
        request_to_redirect = Request(
            f"http://127.0.0.1:{redirect_server.server_port}/redirect",
            data=b"{}",
            method="POST",
        )
        with pytest.raises(RuntimeError, match="upstream unavailable"):
            bridge.urlopen_transport(request_to_redirect, 0.5)
        assert not target_hit
    finally:
        redirect_server.shutdown()
        redirect_server.server_close()
        thread.join()


def test_health_is_public_and_secret_free(running_bridge) -> None:
    import http.client

    server, _ = running_bridge
    connection = http.client.HTTPConnection(*server.server_address)
    connection.request("GET", "/health")
    response = connection.getresponse()
    body = response.read()
    assert response.status == 200
    assert b"bridge-secret" not in body
    assert json.loads(body) == {"status": "ok"}
    connection.close()


def test_unconfigured_secret_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROK_BRIDGE_TOKEN", raising=False)
    server = bridge.make_server(
        "127.0.0.1",
        0,
        lambda: ("secret", "https://cli-chat-proxy.grok.com"),
        timeout=0.5,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert request(server, "/v1/chat/completions")[0] == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_rejects_unapproved_upstream_without_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    transport_called = False

    def transport(_request, _timeout):
        nonlocal transport_called
        transport_called = True
        raise AssertionError("transport must not run")

    server = bridge.make_server(
        "127.0.0.1",
        0,
        lambda: ("subscription", "http://attacker.test"),
        transport=transport,
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert request(server, "/v1/chat/completions")[0] == 502
        assert not transport_called
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_rejects_oversized_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    monkeypatch.setattr(bridge, "MAX_BODY_BYTES", 32)
    server = bridge.make_server("127.0.0.1", 0, lambda: None)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        body = b"x" * 33
        assert request(server, "/v1/chat/completions", body)[0] == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "body",
    [b"", b"[]", b"null", b"not-json", b'{"model":"grok-4.3"'],
)
def test_invalid_bodies_fail_before_resolver(monkeypatch, body) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    called = False

    def resolver():
        nonlocal called
        called = True
        return ("subscription", "https://cli-chat-proxy.grok.com")

    server = bridge.make_server("127.0.0.1", 0, resolver)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert request(server, "/v1/chat/completions", body)[0] == 400
        assert not called
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_exact_body_limit_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    monkeypatch.setattr(bridge, "MAX_BODY_BYTES", 32)
    called = False

    def resolver():
        nonlocal called
        called = True
        return ("subscription", "https://cli-chat-proxy.grok.com")

    server = bridge.make_server("127.0.0.1", 0, resolver)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        body = b'{"model":"grok-4.3"}'
        assert len(body) <= 32
        assert request(server, "/v1/chat/completions", body)[0] == 502
        assert called
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("length, expected_status", [(0, 400), (-1, 413), (33, 413)])
def test_body_length_bounds_reject_without_resolver(
    monkeypatch, length, expected_status
) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    monkeypatch.setattr(bridge, "MAX_BODY_BYTES", 32)
    called = False

    def resolver():
        nonlocal called
        called = True
        return ("subscription", "https://cli-chat-proxy.grok.com")

    server = bridge.make_server("127.0.0.1", 0, resolver)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        import http.client

        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request(
            "POST",
            "/v1/chat/completions",
            b"x" * max(length, 0),
            {
                "Authorization": "Bearer bridge-secret",
                "Content-Length": str(length),
            },
        )
        response = connection.getresponse()
        assert response.status == expected_status
        response.read()
        connection.close()
        assert not called
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_truncated_body_fails_without_resolver(monkeypatch) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")
    called = False

    def resolver():
        nonlocal called
        called = True
        return ("subscription", "https://cli-chat-proxy.grok.com")

    server = bridge.make_server("127.0.0.1", 0, resolver)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    connection = socket.create_connection(server.server_address, timeout=2)
    try:
        connection.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Authorization: Bearer bridge-secret\r\nContent-Length: 20\r\n\r\n{}"
        )
        connection.shutdown(socket.SHUT_WR)
        result = connection.recv(4096)
        assert b"400 Bad Request" in result
        assert not called
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_resolver_exception_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_BRIDGE_TOKEN", "bridge-secret")

    def resolver():
        raise RuntimeError("secret token must not escape")

    server = bridge.make_server("127.0.0.1", 0, resolver)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        status, _, body = request(server, "/v1/chat/completions")
        assert status == 503
        assert b"secret token" not in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
