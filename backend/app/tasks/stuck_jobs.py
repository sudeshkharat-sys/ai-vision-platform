"""Remove stuck jobs automatically when the Celery worker starts.

Problem: if Celery stops or crashes in the middle of a job, Redis gives the
same job back on the next start, and it runs again and again.

How we fix it:
  * when a job starts, we save "task_started:<id>" in Redis (kept 7 days)
  * when the worker starts, any job in the broker whose id was already
    started once is a re-delivered (stuck) job -> we remove it

Jobs that never started (new jobs) are not touched, and neither are jobs of
other applications, because only our own jobs save "task_started:<id>".
"""

import base64
import json

import redis

from ..config import settings

QUEUE = "celery"
KEEP_SECONDS = 7 * 24 * 3600


def mark_started(task_id: str) -> None:
    """Call when a job starts."""
    try:
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.set(f"task_started:{task_id}", 1, ex=KEEP_SECONDS)
    except Exception:
        pass


def _task_id(raw) -> str:
    """Read the task id from one stored Celery message."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):          # unacked list: [message, exchange, routing_key]
            data = data[0]
        return data.get("headers", {}).get("id", "")
    except Exception:
        return ""


def _already_started(r_state, task_id: str) -> bool:
    return bool(task_id) and r_state.exists(f"task_started:{task_id}")


def remove_stuck_jobs() -> list:
    """Remove re-delivered jobs. Returns the list of removed task ids."""
    removed = []
    try:
        broker = redis.Redis.from_url(settings.celery_broker_url, socket_connect_timeout=2)
        state = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)

        # 1. Jobs a worker took but never finished (Celery crashed / was killed)
        for tag, raw in broker.hgetall("unacked").items():
            task_id = _task_id(raw)
            if _already_started(state, task_id):
                broker.hdel("unacked", tag)
                broker.zrem("unacked_index", tag)
                removed.append(task_id)

        # 2. Jobs put back in the queue when Celery was stopped mid-job
        for raw in broker.lrange(QUEUE, 0, -1):
            task_id = _task_id(raw)
            if _already_started(state, task_id):
                broker.lrem(QUEUE, 1, raw)
                removed.append(task_id)
    except Exception as e:
        print(f"[stuck_jobs] could not check for stuck jobs: {e}")
    return removed
