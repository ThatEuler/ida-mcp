import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from ida_mcp import mcp as mcp_api


def test_stdio_protocol_and_eof_shutdown() -> None:
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                # Exercise lines larger than the input buffer as well.
                "clientInfo": {"name": "x" * 20000, "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ida_mcp.cli import main; raise SystemExit(main(['stdio', '--agent=test']))",
        ],
        input=b"".join(json.dumps(request).encode() + b"\n" for request in requests),
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    responses = {
        response["id"]: response
        for response in map(json.loads, result.stdout.splitlines())
    }
    assert responses[1]["result"]["serverInfo"]["name"] == "ida"
    assert any(
        tool["name"] == "open_database" for tool in responses[2]["result"]["tools"]
    )


def test_stopping_stdio_input_releases_a_pending_read() -> None:
    reading = threading.Event()
    release = threading.Event()
    shutdown = Mock()
    results: list[bytes] = []

    class Input(BytesIO):
        def readline(self, size: int | None = -1, /) -> bytes:
            reading.set()
            assert release.wait(timeout=5)
            return super().readline(size)

    stdin = mcp_api._ShutdownOnEOFInput(Input(b"late input\n"), shutdown)
    consumer = threading.Thread(target=lambda: results.append(stdin.readline()))
    consumer.start()
    try:
        assert reading.wait(timeout=5)
        stdin.stop()
        consumer.join(timeout=5)
        assert not consumer.is_alive()
        assert results == [b""]
        assert stdin.readline() == b""
        shutdown.assert_not_called()
    finally:
        stdin.stop()
        release.set()
        consumer.join(timeout=5)


def test_stdio_input_preserves_sized_reads_and_errors() -> None:
    stdin = mcp_api._ShutdownOnEOFInput(BytesIO(b"hello\nworld\n"), Mock())
    try:
        assert stdin.readline(2) == b"he"
        assert stdin.readline() == b"llo\n"
        assert stdin.readline() == b"world\n"
        assert stdin.readline() == b""
    finally:
        stdin.stop()

    stream = BytesIO()
    stream.close()
    stdin = mcp_api._ShutdownOnEOFInput(stream, Mock())
    try:
        with pytest.raises(ValueError, match="closed file"):
            stdin.readline()
    finally:
        stdin.stop()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_stdio_restores_signal_handlers_on_failure(monkeypatch, failure) -> None:
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    manager = Mock()
    monkeypatch.setattr(mcp_api, "DATABASE_MANAGER", manager)
    monkeypatch.setattr(mcp_api, "_start_mcp_trace", Mock())

    async def fail(**kwargs):
        raise failure()

    monkeypatch.setattr(mcp_api.mcp, "stdio_async", fail)
    stdin = mcp_api._ShutdownOnEOFInput(BytesIO(), Mock())
    with pytest.raises(failure):
        asyncio.run(mcp_api._serve_stdio_async(stdin, None, None))

    manager.shutdown.assert_called_once_with()
    assert {signum: signal.getsignal(signum) for signum in previous} == previous


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process signals")
@pytest.mark.parametrize(
    "scenario",
    [
        "lifecycle-lock",
        "trace-lock",
        "blocked-stdin",
        "shutdown-signal",
        "eof",
        "eof-signal",
    ],
)
def test_stdio_signal_shutdown_does_not_deadlock(tmp_path: Path, scenario: str) -> None:
    # Run in a subprocess so a regression cannot deadlock the pytest process.
    # Keep stdin open throughout: terminating the transport must also release
    # asyncio's executor thread waiting for its next input line.
    program = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import threading
        from pathlib import Path
        from types import SimpleNamespace

        from ida_mcp import mcp as api
        from ida_nexus.manager import DatabaseManager

        scenario, marker_path = sys.argv[1:]

        class SignalLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.signalled = False

            def __enter__(self):
                self.lock.acquire()
                if not self.signalled:
                    self.signalled = True
                    signal.raise_signal(signal.SIGTERM)
                return self

            def __exit__(self, *args):
                self.lock.release()

        class Manager(DatabaseManager):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.release_lock = threading.Lock()
                self.released = False

            def shutdown(self):
                with self.release_lock:
                    if self.released:
                        return
                    if scenario in {"shutdown-signal", "eof-signal"}:
                        # A second signal must not re-enter lease release.
                        os.kill(os.getpid(), signal.SIGTERM)
                        if scenario == "shutdown-signal":
                            os.kill(os.getpid(), signal.SIGINT)
                    super().shutdown()
                    self.released = True
                    Path(marker_path).write_text("released")

        if scenario == "lifecycle-lock":
            api._TRACE_LIFECYCLE_LOCK = SignalLock()
        elif scenario == "trace-lock":
            api.TRACE._lock = SignalLock()
        elif scenario in {"blocked-stdin", "shutdown-signal"}:
            stream = sys.stdin.buffer

            class Input:
                signalled = False

                def start_signal(self):
                    if not self.signalled:
                        self.signalled = True
                        timer = threading.Timer(
                            0.05, lambda: os.kill(os.getpid(), signal.SIGTERM)
                        )
                        timer.start()

                def fileno(self):
                    self.start_signal()
                    return stream.fileno()

                def readline(self, size=-1):
                    self.start_signal()
                    return stream.readline(size)

            sys.stdin = SimpleNamespace(buffer=Input())

        api.serve_stdio(database_manager_class=Manager, agent="test")
        """
    )
    marker = tmp_path / "released"
    with subprocess.Popen(
        [sys.executable, "-c", program, scenario, str(marker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            if scenario.startswith("eof"):
                stdout, stderr = process.communicate(timeout=5)
            else:
                process.wait(timeout=5)
                stdout, stderr = process.communicate()
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"{scenario} deadlocked: {stdout!r}, {stderr!r}")

    expected = 0 if scenario == "eof" else 128 + signal.SIGTERM
    assert process.returncode == expected, (stdout, stderr)
    assert marker.read_text() == "released"
