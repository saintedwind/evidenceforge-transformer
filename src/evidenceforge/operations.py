"""Bound request bodies and log operational metadata without document contents."""

import asyncio
import json
import logging
import time
from uuid import uuid4

from starlette.responses import JSONResponse

logger = logging.getLogger("evidenceforge.audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False


class RequestLimits:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        request_id = uuid4().hex
        status = 500

        async def respond(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode())
                ]
            await send(message)

        try:
            body = bytearray()
            while True:
                remaining = 10 - (time.monotonic() - started)
                try:
                    message = await asyncio.wait_for(receive(), timeout=max(0.001, remaining))
                except TimeoutError:
                    return await JSONResponse({"detail": "request body timed out"}, 408)(
                        scope, receive, respond
                    )
                if message["type"] == "http.disconnect":
                    status = 499
                    return
                body.extend(message.get("body", b""))
                if len(body) > 16 * 1024:
                    return await JSONResponse({"detail": "request body exceeds 16 KiB"}, 413)(
                        scope, receive, respond
                    )
                if not message.get("more_body", False):
                    break
            consumed = False

            async def replay():
                nonlocal consumed
                if consumed:
                    return await receive()
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, replay, respond)
        finally:
            path = scope.get("path")
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "route": path
                        if path in {"/query", "/ingest", "/health", "/ready"}
                        else "other",
                        "status": status,
                        "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    }
                )
            )
