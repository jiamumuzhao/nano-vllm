import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from nanovllm.engine.async_engine import RequestOutput
from nanovllm.entrypoints.openai.api_server import _stream_chat_completion, _stream_completion, create_app
from nanovllm.sampling_params import SamplingParams


class FakeAsyncEngine:
    def __init__(self):
        self.engine = SimpleNamespace(tokenizer=SimpleNamespace(
            encode=lambda text: list(range(len(text))),
            apply_chat_template=lambda messages, tokenize=False, add_generation_prompt=True: "chat",
        ))
        self.states = {}
        self.accept = True
        self.started = False
        self.closed = False
        self.unavailable = False
        self.engine_error = None

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.closed = True

    def can_accept_request(self, count=1):
        return self.accept

    def is_unavailable(self):
        return self.unavailable

    def get_request_status(self, request_id):
        return self.states.get(request_id)

    async def cancel(self, request_id, reason="cancelled"):
        state = self.states.get(request_id)
        if state is None:
            return None
        if state["status"] not in {"finished", "failed", "cancelled"}:
            state.update(status="cancelled", finish_reason=reason)
        return state

    async def generate(self, prompt, sampling_params, request_id=None):
        self.states[request_id] = {"request_id": request_id, "status": "finished", "finish_reason": "stop"}
        yield RequestOutput(request_id, "ok", [1], "ok", [1], True, "stop", "finished")


def test_request_status_cancel_and_completion_api():
    engine = FakeAsyncEngine()
    app = create_app(engine, "fake-model")
    with TestClient(app) as client:
        assert client.get("/v1/requests/missing").status_code == 404
        response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1})
        assert response.status_code == 200
        request_id = response.json()["id"]
        assert request_id.startswith("cmpl-")
        assert client.get(f"/v1/requests/{request_id}").json()["finish_reason"] == "stop"
        cancelled = client.delete(f"/v1/requests/{request_id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "finished"
        assert client.delete("/v1/requests/missing").status_code == 404


def test_queue_full_maps_to_429_and_stream_has_done():
    engine = FakeAsyncEngine()
    app = create_app(engine, "fake-model")
    with TestClient(app) as client:
        engine.accept = False
        response = client.post("/v1/completions", json={"prompt": "hi"})
        assert response.status_code == 429
        engine.accept = True
        with client.stream("POST", "/v1/completions", json={"prompt": "hi", "stream": True}) as response:
            body = "".join(response.iter_text())
        assert response.status_code == 200
        assert "[DONE]" in body


def test_chat_completion_nonstream_success_and_request_status():
    engine = FakeAsyncEngine()
    app = create_app(engine, "fake-model")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1, "temperature": 1.0},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["finish_reason"] == "stop"
        request_id = body["id"]
        assert request_id.startswith("chatcmpl-")
        status = client.get(f"/v1/requests/{request_id}").json()
        assert status["status"] == "finished"
        assert status["finish_reason"] == "stop"


def test_chat_completion_streaming_success_and_done():
    engine = FakeAsyncEngine()
    app = create_app(engine, "fake-model")
    with TestClient(app) as client:
        with client.stream(
            "POST", "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            response.read()
            body = response.text
        records = [line[6:] for line in body.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
        assert records
        chunks = [json.loads(record) for record in records]
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        ids = {chunk["id"] for chunk in chunks}
        assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
        assert "data: [DONE]" in body


def test_engine_fatal_maps_to_503():
    engine = FakeAsyncEngine()
    engine.unavailable = True
    engine.engine_error = "engine-level failure: worker exited"
    app = create_app(engine, "fake-model")
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 503
    assert "engine-level failure" in response.json()["detail"]


def test_stream_cancel_runs_request_cleanup():
    class SlowEngine(FakeAsyncEngine):
        def __init__(self):
            super().__init__()
            self.cancelled = False
            self.stop = asyncio.Event()
            self.entered = asyncio.Event()

        async def cancel(self, request_id, reason="cancelled"):
            self.cancelled = True
            return {"request_id": request_id, "status": "cancelled", "finish_reason": reason}

        async def generate(self, prompt, sampling_params, request_id=None):
            self.states[request_id] = {"request_id": request_id, "status": "decode", "finish_reason": None}
            yield RequestOutput(request_id, "x", [1], "x", [1], False, None, "decode")
            self.entered.set()
            await self.stop.wait()

    async def scenario():
        engine = SlowEngine()
        stream = _stream_completion(engine, [1], SamplingParams(max_tokens=2), "fake", 0, "req")
        first = await stream.__anext__()
        assert "req" in first
        task = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(engine.entered.wait(), 1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream.aclose()
        assert engine.cancelled

    asyncio.run(scenario())


def test_chat_stream_cancel_runs_request_cleanup():
    class SlowEngine(FakeAsyncEngine):
        def __init__(self):
            super().__init__()
            self.cancelled = False
            self.stop = asyncio.Event()
            self.entered = asyncio.Event()

        async def cancel(self, request_id, reason="cancelled"):
            self.cancelled = True
            self.states[request_id] = {"request_id": request_id, "status": "cancelled", "finish_reason": reason}
            return self.states[request_id]

        async def generate(self, prompt, sampling_params, request_id=None):
            self.states[request_id] = {"request_id": request_id, "status": "decode", "finish_reason": None}
            yield RequestOutput(request_id, "x", [1], "x", [1], False, None, "decode")
            self.entered.set()
            await self.stop.wait()

    async def scenario():
        engine = SlowEngine()
        stream = _stream_chat_completion(
            engine, "chat", SamplingParams(max_tokens=2), "fake", 0, "chatcmpl-test"
        )
        first = await stream.__anext__()
        assert "chatcmpl-test" in first
        task = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(engine.entered.wait(), 1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream.aclose()
        assert engine.cancelled
        assert engine.states["chatcmpl-test"]["status"] == "cancelled"

    asyncio.run(scenario())


def test_malformed_requests_return_precise_client_errors():
    engine = FakeAsyncEngine()
    app = create_app(engine, "fake-model")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
        assert response.status_code == 400
        assert "messages must not be empty" in response.json()["detail"]

        response = client.post("/v1/completions", json={"prompt": []})
        assert response.status_code == 400
        assert "prompt must not be empty" in response.json()["detail"]

        response = client.post("/v1/completions", json={"prompt": ["a", "b"], "stream": True})
        assert response.status_code == 400
        assert "streaming currently supports exactly one prompt" in response.json()["detail"]


def test_schema_validation_errors_are_422_not_500():
    engine = FakeAsyncEngine()
    app = create_app(engine, "fake-model")
    cases = [
        ("/v1/completions", {"prompt": 123}),
        ("/v1/chat/completions", {"messages": "not-a-list"}),
        ("/v1/completions", {"prompt": "x", "max_tokens": 0}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "temperature": 0}),
    ]
    with TestClient(app) as client:
        for path, payload in cases:
            response = client.post(path, json=payload)
            assert response.status_code == 422
            assert response.status_code != 500
