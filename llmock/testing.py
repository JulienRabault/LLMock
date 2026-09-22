"""Run a real LLMock server inside the current process.

The server listens on a real socket, in a background thread, so clients go
through a genuine HTTP stack: connection drops, stalls and truncated streams
behave exactly as they would against a remote provider. An in-memory ASGI
transport cannot reproduce those.

    with LLMockServer() as server:
        client = OpenAI(base_url=server.base_url("openai"), api_key="test")
        server.state.scenarios.add(Fail(429))
        ...
"""

from __future__ import annotations

import socket
import threading
import time
from types import TracebackType

from llmock.chaos import ChaosSettings
from llmock.simulation import MockResponseSettings
from llmock.state import LLMockState

__all__ = ["BASE_PATHS", "LLMockServer"]

#: Base URL path per provider, as the provider SDKs expect it.
BASE_PATHS = {
    "openai": "/v1",
    "anthropic": "/anthropic",
    "gemini": "/gemini",
    "mistral": "/mistral/v1",
    "cohere": "/cohere",
    "groq": "/groq/openai/v1",
    "together": "/together/v1",
    "perplexity": "/perplexity/v1",
    "ai21": "/ai21/v1",
    "xai": "/xai/v1",
}

_STARTUP_TIMEOUT = 10.0


class LLMockServer:
    """A real LLMock HTTP server in a daemon thread.

    Binds to an OS-chosen free port by default. The socket is bound before
    uvicorn starts, so there is no window in which another process can take
    the port.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        chaos: ChaosSettings | None = None,
        responses: MockResponseSettings | None = None,
    ) -> None:
        # Imported lazily: this module is loaded by the pytest plugin in every
        # test session, and should cost nothing when no test uses it.
        from llmock.main import create_app

        self.app = create_app(
            chaos=chaos or ChaosSettings(),
            responses=responses or MockResponseSettings(),
        )
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> LLMockServer:
        if self._thread is not None:
            raise RuntimeError("LLMockServer is already running")
        import uvicorn

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        self._socket = sock
        self._port = sock.getsockname()[1]

        config = uvicorn.Config(self.app, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [sock]},
            name=f"llmock-server-{self._port}",
            daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("LLMock server thread exited during startup")
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f"LLMock server did not start within {_STARTUP_TIMEOUT}s")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=_STARTUP_TIMEOUT)
        if self._socket is not None:
            self._socket.close()
        self._server = None
        self._thread = None
        self._socket = None

    def __enter__(self) -> LLMockServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # -- access ---------------------------------------------------------------

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        """Root URL, e.g. ``http://127.0.0.1:53412``."""
        return f"http://{self._host}:{self._port}"

    def base_url(self, provider: str = "openai") -> str:
        """The ``base_url`` to hand a provider's SDK."""
        try:
            return self.url + BASE_PATHS[provider]
        except KeyError:
            known = ", ".join(sorted(BASE_PATHS))
            raise ValueError(f"Unknown provider {provider!r}; expected one of {known}") from None

    @property
    def state(self) -> LLMockState:
        return self.app.state.llmock
