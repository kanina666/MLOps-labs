from collections.abc import Iterator
from unittest.mock import AsyncMock

from mlops_labs import source
from mlops_labs.core.config import Settings

PAYLOAD = {"recent_sales": [10.0] * 28, "day_of_week": 3, "is_snap": 0}


async def test_backpressure_retries_same_task_without_advancing_iterator() -> None:
    broker = AsyncMock()
    broker.publish.side_effect = [
        ConnectionError("full"),
        ConnectionError("full"),
        ConnectionError("full"),
        ConnectionError("full"),
        None,
        ConnectionError("full"),
        None,
    ]
    sleep = AsyncMock()

    def messages() -> Iterator[tuple[str, dict]]:
        yield "first", PAYLOAD
        assert broker.publish.await_count == 5
        yield "second", PAYLOAD

    settings = Settings(
        _env_file=None, source_retry_delay_seconds=0.1, source_retry_max_delay_seconds=0.25
    )
    stats = await source.publish_batch(messages(), broker, settings, sleep=sleep)

    assert [call.args[0]["task_id"] for call in broker.publish.await_args_list] == [
        "first",
        "first",
        "first",
        "first",
        "first",
        "second",
        "second",
    ]
    assert [call.args[0] for call in sleep.await_args_list] == [0.1, 0.2, 0.25, 0.25, 0.1]
    assert stats.published_total == 2
    assert stats.retries_total == 5
