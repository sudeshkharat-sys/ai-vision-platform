"""Remove stuck jobs automatically when the Celery worker starts.

Problem: if Celery stops or crashes in the middle of a job, Redis gives the
same job back on the next start, and it runs again and again.

How we fix it:
  * when a job starts, we save "task_started:<id>" in Redis (kept 7 days)
  * the id is also written to logs/started_tasks.txt (survives PC off)
  * when the worker starts, any job in the broker whose id was already
    started once is a re-delivered (stuck) job -> we remove it
  * and right before a job runs we check again: if it already started once,
    it is skipped (see SkipRepeatTask in celery_app.py)

Jobs that never started (new jobs) are not touched, and neither are jobs of
other applications, because only our own jobs save "task_started:<id>".
"""

import base64
import json

import redis

from ..config import settings

QUEUE = "celery"
KEEP_SECONDS = 7 * 24 * 3600


def _started_file():
    """logs/started_tasks.txt: one task id per line. Kept on disk, so it
    survives a PC shutdown even if Redis loses its data."""
    from ..crash_logs import get_log_dir
    return get_log_dir() / "started_tasks.txt"


def mark_started(task_id: str) -> None:
    """Call when a job starts: save its id in Redis AND in a file."""
    try:
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        r.set(f"task_started:{task_id}", 1, ex=KEEP_SECONDS)
    except Exception:
        pass
    try:
        with open(_started_file(), "a", encoding="utf-8") as f:
            f.write(task_id + "\n")
    except Exception:
        pass


def was_started_before(task_id: str) -> bool:
    """True if this job id already started once (checks Redis, then the file)."""
    if not task_id:
        return False
    try:
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        if r.exists(f"task_started:{task_id}"):
            return True
    except Exception:
        pass
    try:
        path = _started_file()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip() == task_id:
                        return True
    except Exception:
        pass
    return False


def _task_id(raw) -> str:
    """Read the task id from one stored Celery message."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):          # unacked list: [message, exchange, routing_key]
            data = data[0]
        return data.get("headers", {}).get("id", "")
    except Exception:
        return ""


def remove_stuck_jobs() -> list:
    """Remove re-delivered jobs. Returns the list of removed task ids."""
    import time
    removed = []
    try:
        broker = redis.Redis.from_url(settings.celery_broker_url, socket_connect_timeout=2)
        # Redis may still be starting (e.g. right after the PC boots): wait up to ~20 s
        for _ in range(10):
            try:
                broker.ping()
                break
            except Exception:
                time.sleep(2)

        # 1. Jobs a worker took but never finished (Celery crashed / was killed)
        for tag, raw in broker.hgetall("unacked").items():
            task_id = _task_id(raw)
            if was_started_before(task_id):
                broker.hdel("unacked", tag)
                broker.zrem("unacked_index", tag)
                removed.append(task_id)

        # 2. Jobs put back in the queue when Celery was stopped mid-job
        for raw in broker.lrange(QUEUE, 0, -1):
            task_id = _task_id(raw)
            if was_started_before(task_id):
                broker.lrem(QUEUE, 1, raw)
                removed.append(task_id)
    except Exception as e:
        print(f"[stuck_jobs] could not check for stuck jobs: {e}")
    return removed
