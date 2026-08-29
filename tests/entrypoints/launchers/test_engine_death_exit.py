# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The API server must fail the process when the engine dies.

Exiting 0 after an engine crash makes the crash indistinguishable from a clean
shutdown to a container supervisor, which turns a crash loop into a pod that
merely looks restarted.
"""

import asyncio
import os
import signal
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm.entrypoints.launchers.launcher import NoSignalServer, serve_http


class _StubEngine:
    """Minimal EngineClient surface used by the shutdown path."""

    def __init__(self, *, errored: bool):
        self.errored = errored
        self.is_running = not errored
        self.vllm_config = SimpleNamespace(shutdown_timeout=0)

    def shutdown(self, timeout=None):
        # A deliberate shutdown stops the output handler, which is what makes a
        # drained engine report itself as errored.
        self.errored = True
        self.is_running = False


class _NullLifespan:
    async def shutdown(self):
        pass


def _app(*, errored: bool) -> FastAPI:
    app = FastAPI()
    app.state.engine_client = _StubEngine(errored=errored)
    return app


def _stub_serve(monkeypatch, body=None):
    """Make ``server.serve()`` return as it does once ``should_exit`` is set."""

    async def _serve(self, sockets=None):
        # Set up the state that the real serve() would leave behind for
        # shutdown(), which the signal path calls.
        self.servers = []
        self.lifespan = _NullLifespan()
        if body is not None:
            await body()

    monkeypatch.setattr(NoSignalServer, "serve", _serve)


@pytest.mark.asyncio
async def test_serve_http_raises_when_engine_died(monkeypatch):
    _stub_serve(monkeypatch)
    with pytest.raises(RuntimeError, match="engine died"):
        await serve_http(_app(errored=True), sock=None, host="127.0.0.1", port=0)


@pytest.mark.asyncio
async def test_serve_http_returns_when_engine_healthy(monkeypatch):
    _stub_serve(monkeypatch)
    await (await serve_http(_app(errored=False), sock=None, host="127.0.0.1", port=0))


@pytest.mark.asyncio
async def test_serve_http_does_not_raise_on_requested_shutdown(monkeypatch):
    """A drained engine after SIGTERM is a clean exit, not a crash."""

    async def _signal_then_return():
        os.kill(os.getpid(), signal.SIGTERM)
        # Let the loop's signal handler and the shutdown task run.
        await asyncio.sleep(0.05)

    _stub_serve(monkeypatch, body=_signal_then_return)
    app = _app(errored=False)
    await (await serve_http(app, sock=None, host="127.0.0.1", port=0))
    # The drain really did mark the engine errored; the signal is what keeps
    # that from being reported as a crash.
    assert app.state.engine_client.errored
