"""Cloud cold starts have a separate budget; the original local deadlines stay unchanged."""
import asyncio

from tests.test_api_carry import NS, new_session


class Clock:
    def __init__(self):
        self.now = 0
        self.sleepers = []
        self.tasks = []

    def __getattr__(self, name):
        return getattr(asyncio, name)

    def create_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def sleep(self, seconds):
        future = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.now + seconds, future))
        await future

    async def advance(self, seconds):
        self.now += seconds
        for deadline, future in self.sleepers:
            if deadline <= self.now and not future.done():
                future.set_result(None)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


def session_with_speech(transport):
    session = new_session()
    session._committed_turn_id = "cloud-turn"
    session.tts.transport = transport
    session.tts._connected = True
    requests = []

    async def request(task_id, turn_id, text, language=None):
        assert task_id == session._tts_inflight["task_id"]
        assert turn_id == session._tts_inflight["tts_turn_id"]
        requests.append(task_id)
        return True

    session.tts.request = request
    sentences = ["The first sentence.", "The second sentence.", "The final sentence."]
    session._tts_sentence_queue.extend(("casual_neutral", text, "deep") for text in sentences)
    return session, requests, sentences


def audio_for(task):
    return {"task_id": task["task_id"], "turn_id": task["tts_turn_id"],
            "audio": b"audio bytes", "audio_format": "wav", "audio_duration": 2.5}


def test_cloud_cold_recovery_preserves_every_sentence(monkeypatch):
    async def scenario():
        clock = Clock()
        monkeypatch.setitem(NS, "asyncio", clock)
        session, requests, sentences = session_with_speech("modal")
        try:
            await session._pump_tts_queue()
            for index in range(3):
                pending = session._tts_inflight
                await asyncio.sleep(0)
                await clock.advance(7)
                assert session._tts_inflight is pending
                assert len(requests) == index + 1
                await session._on_tts_event("audio_ready", audio_for(pending))
            messages = [message for message in session.ws.sent if message["type"] == "tts_audio"]
            assert [message["text"] for message in messages] == sentences
            assert [message["idx"] for message in messages] == [0, 1, 2]
            assert session._tts_inflight is None and not session._tts_sentence_queue
        finally:
            await clock.close()

    asyncio.run(scenario())


def test_cloud_deadline_reports_failure_and_rejects_late_result(monkeypatch):
    async def scenario():
        clock = Clock()
        monkeypatch.setitem(NS, "asyncio", clock)
        session, requests, sentences = session_with_speech("modal")
        session.tts.request_timeout_s = 15
        try:
            await session._pump_tts_queue()
            original = session._tts_inflight
            await asyncio.sleep(0)
            await clock.advance(14)
            assert session._tts_inflight is original
            await clock.advance(1)
            failure, = [message for message in session.ws.sent if message["type"] == "tts_error"]
            assert failure["turn_id"] == original["tts_turn_id"] and failure["idx"] == 0
            following = session._tts_inflight
            assert len(requests) == 2 and following["idx"] == 1
            await session._on_tts_event("audio_ready", audio_for(original))
            assert session._tts_inflight is following
            await session._on_tts_event("audio_ready", audio_for(following))
            delivered, = [message for message in session.ws.sent if message["type"] == "tts_audio"]
            assert delivered["text"] == sentences[1]
        finally:
            await clock.close()

    asyncio.run(scenario())


def test_local_first_audio_and_sentence_deadlines_are_unchanged(monkeypatch):
    async def scenario():
        clock = Clock()
        monkeypatch.setitem(NS, "asyncio", clock)
        session, _, _ = session_with_speech("local")
        try:
            await session._pump_tts_queue()
            original = session._tts_inflight
            await asyncio.sleep(0)
            await clock.advance(1.999)
            assert session._tts_inflight is original
            await clock.advance(.001)
            assert session._tts_inflight is None and not session._tts_sentence_queue

            session, requests, _ = session_with_speech("local")
            session._tts_first_audio_sent = True
            await session._pump_tts_queue()
            original = session._tts_inflight
            await asyncio.sleep(0)
            await clock.advance(4.999)
            assert session._tts_inflight is original
            await clock.advance(.001)
            assert session._tts_inflight["idx"] == 1 and len(requests) == 2
        finally:
            await clock.close()

    asyncio.run(scenario())
