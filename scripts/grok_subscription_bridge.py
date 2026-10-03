from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_BODY_BYTES = 2 * 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 60.0
APPROVED_HOST = "cli-chat-proxy.grok.com"
APPROVED_HOSTS = {APPROVED_HOST, "api.x.ai"}


class UpstreamResponse(NamedTuple):
    status: int
    headers: list[tuple[str, str]]
    body: Iterable[bytes]
    close: Callable[[], None] = lambda: None


Resolver = Callable[[], tuple[str, str] | None]
Headers = Callable[[], dict[str, str]]
Transport = Callable[[Request, float], UpstreamResponse]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def urlopen_transport(request: Request, timeout: float) -> UpstreamResponse:
    opener = build_opener(NoRedirect)
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        error.close()
        raise RuntimeError("upstream unavailable") from error
    except (OSError, URLError) as error:
        raise RuntimeError("upstream unavailable") from error

    def chunks() -> Iterable[bytes]:
        try:
            while True:
                chunk = (
                    response.read1(65536)
                    if hasattr(response, "read1")
                    else response.read(65536)
                )
                if not chunk:
                    return
                yield chunk
        finally:
            response.close()

    return UpstreamResponse(
        response.status,
        list(response.headers.items()),
        chunks(),
        response.close,
    )


def validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in APPROVED_HOSTS
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/", "/v1")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid upstream")
    return f"https://{parsed.hostname}/v1"


def make_server(
    host: str,
    port: int,
    resolver: Resolver,
    *,
    timeout: float = UPSTREAM_TIMEOUT_SECONDS,
    transport: Transport = urlopen_transport,
    default_headers: Headers | None = None,
) -> ThreadingHTTPServer:
    default_headers = default_headers or (lambda: {})

    class BridgeHandler(BaseHTTPRequestHandler):
        server_version = "GrokSubscriptionBridge"

        def log_message(self, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._json_error(404, "not found")
                return
            self._send_headers(200, "application/json")
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._json_error(404, "not found")
                return
            if not os.environ.get("GROK_BRIDGE_TOKEN"):
                self._json_error(503, "bridge unavailable")
                return
            if not self._authorized():
                self._json_error(401, "unauthorized")
                return
            length = self.headers.get("Content-Length")
            try:
                body_length = int(length or "-1")
            except ValueError:
                body_length = -1
            if body_length < 0 or body_length > MAX_BODY_BYTES:
                self._json_error(413, "request too large")
                return
            body = self.rfile.read(body_length)
            if len(body) != body_length:
                self._json_error(400, "invalid request")
                return
            try:
                payload = json.loads(body)
            except (ValueError, TypeError):
                self._json_error(400, "invalid JSON")
                return
            if not isinstance(payload, dict) or payload.get("model") != "grok-4.3":
                self._json_error(400, "unsupported model")
                return
            try:
                resolved = resolver()
            except Exception:
                self._json_error(503, "subscription unavailable")
                return
            if resolved is None:
                self._json_error(503, "subscription unavailable")
                return
            upstream_token, base_url = resolved
            try:
                upstream_url = validate_base_url(base_url) + "/chat/completions"
                headers = dict(default_headers())
                headers.update(
                    {
                        "Authorization": f"Bearer {upstream_token}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                        if payload.get("stream")
                        else "application/json",
                    }
                )
                request = Request(
                    upstream_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                response = transport(request, timeout)
                self._forward(response)
            except (ValueError, RuntimeError, OSError):
                self._json_error(502, "upstream unavailable")

        def _authorized(self) -> bool:
            presented = self.headers.get("Authorization", "")
            expected = "Bearer " + os.environ.get("GROK_BRIDGE_TOKEN", "")
            return bool(expected != "Bearer ") and hmac.compare_digest(
                presented, expected
            )

        def _send_headers(
            self, status: int, content_type: str, content_length: int | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if content_length is not None:
                self.send_header("Content-Length", str(content_length))
            self.send_header("Connection", "close")
            self.end_headers()

        def _json_error(self, status: int, message: str) -> None:
            body = json.dumps(
                {"error": {"message": message}}, separators=(",", ":")
            ).encode()
            self._send_headers(status, "application/json", len(body))
            self.wfile.write(body)

        def _forward(self, response: UpstreamResponse) -> None:
            try:
                try:
                    self.send_response(response.status)
                    allowed = {
                        "content-type",
                        "content-length",
                        "cache-control",
                        "x-request-id",
                    }
                    for name, value in response.headers:
                        if name.lower() in allowed:
                            self.send_header(name, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                except (OSError, RuntimeError):
                    self.close_connection = True
                    return
                try:
                    for chunk in response.body:
                        if chunk:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (OSError, TimeoutError):
                    self.close_connection = True
            finally:
                close = getattr(response.body, "close", None)
                if close is not None:
                    close()
                response.close()

    return ThreadingHTTPServer((host, port), BridgeHandler)


def load_hermes(source_path: str) -> tuple[Resolver, Headers]:
    sys.path.insert(0, source_path)
    from agent.auxiliary_client import _resolve_xai_oauth_for_aux
    from tools.xai_http import hermes_xai_default_headers

    return _resolve_xai_oauth_for_aux, hermes_xai_default_headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    resolver, default_headers = load_hermes(args.hermes_source_path)
    server = make_server(
        args.host, args.port, resolver, default_headers=default_headers
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
