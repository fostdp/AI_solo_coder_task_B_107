import redis.asyncio as aioredis
import json
import logging
from typing import Dict, List, Optional, Any
from shared.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20,
        )
    return _pool


async def close_redis():
    global _pool
    if _pool:
        await _pool.aclose()
        _pool = None


async def ensure_group(r: aioredis.Redis, stream: str, group: str):
    try:
        await r.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.error(f"创建消费者组失败 {stream}/{group}: {e}")


async def xadd_msg(r: aioredis.Redis, stream: str, data: Dict[str, Any],
                   max_len: Optional[int] = None):
    ml = max_len or settings.STREAM_MAX_LEN
    payload = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            payload[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, (int, float, bool)):
            payload[k] = str(v)
        elif v is None:
            payload[k] = ""
        else:
            payload[k] = str(v)
    msg_id = await r.xadd(stream, payload, maxlen=ml, approximate=True)
    return msg_id


async def xread_group(r: aioredis.Redis, stream: str, group: str,
                      consumer: str, count: int = 10,
                      block_ms: Optional[int] = None) -> List[Dict]:
    bms = block_ms if block_ms is not None else settings.STREAM_BLOCK_MS
    try:
        results = await r.xreadgroup(
            group, consumer,
            {stream: ">"},
            count=count,
            block=bms,
        )
    except Exception:
        return []

    messages = []
    if results:
        for stream_name, msgs in results:
            for msg_id, fields in msgs:
                parsed = {}
                for k, v in fields.items():
                    try:
                        parsed[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        parsed[k] = v
                parsed["_msg_id"] = msg_id
                parsed["_stream"] = stream_name
                messages.append(parsed)
    return messages


async def ack_message(r: aioredis.Redis, stream: str, group: str, msg_id: str):
    await r.xack(stream, group, msg_id)
