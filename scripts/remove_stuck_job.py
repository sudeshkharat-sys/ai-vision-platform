"""List Celery jobs stuck in Redis, and remove ONE job safely.

Why: when Celery is stopped (or crashes) in the middle of a job, Redis keeps
that job in an "unacked" list and gives it back every time Celery starts.
`celery purge` does not touch that list. This script removes only the job
you choose; every other job stays.

Usage (stop Celery first):
    python scripts/remove_stuck_job.py                 # list jobs, changes nothing
    python scripts/remove_stuck_job.py <task_id>       # remove that one job

Different Redis / database number:
    python scripts/remove_stuck_job.py --url redis://127.0.0.1:6379/1
"""

import base64
import json
import sys

import redis

DEFAULT_URL = "redis://localhost:6379/1"   # same as celery_broker_url in config.py
QUEUE = "celery"


def read_job(raw):
    """Return (task_id, task_name, args) from one stored Celery message."""
    data = json.loads(raw)
    # In the unacked list the message is stored as [message, exchange, routing_key]
    if isinstance(data, list):
        data = data[0]
    headers = data.get("headers", {})
    task_id = headers.get("id", "?")
    task_name = headers.get("task", "?")
    args = ""
    try:
        body = data.get("body", "")
        if data.get("properties", {}).get("body_encoding") == "base64":
            body = base64.b64decode(body).decode("utf-8")
        args = body[:150]
    except Exception:
        pass
    return task_id, task_name, args


def main():
    url = DEFAULT_URL
    params = sys.argv[1:]
    if "--url" in params:
        i = params.index("--url")
        url = params[i + 1]
        params = params[:i] + params[i + 2:]
    remove_id = params[0] if params else None

    r = redis.Redis.from_url(url)
    print(f"Redis: {url}\n")

    found = False

    # 1. Jobs taken by a worker but never finished (these come back on restart)
    print("Stuck jobs (unacked):")
    for tag, raw in r.hgetall("unacked").items():
        task_id, task_name, args = read_job(raw)
        print(f"  {task_id}  {task_name}  {args}")
        if remove_id and task_id == remove_id:
            r.hdel("unacked", tag)
            r.zrem("unacked_index", tag)
            print("    -> REMOVED")
            found = True

    # 2. Jobs still waiting in the queue
    print(f"\nWaiting jobs (queue '{QUEUE}'):")
    for raw in r.lrange(QUEUE, 0, -1):
        task_id, task_name, args = read_job(raw)
        print(f"  {task_id}  {task_name}  {args}")
        if remove_id and task_id == remove_id:
            r.lrem(QUEUE, 1, raw)
            print("    -> REMOVED")
            found = True

    if remove_id and not found:
        print(f"\nNo job with id {remove_id} found. Nothing was changed.")
    elif not remove_id:
        print("\nNothing changed. To remove one job, run again with its task id.")


if __name__ == "__main__":
    main()
