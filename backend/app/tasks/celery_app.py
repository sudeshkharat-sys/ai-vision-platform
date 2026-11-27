from celery import Celery
from celery import signals
from ..config import settings

celery_app = Celery(
    "ai_vision",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.training",
        "app.tasks.auto_annotate",
        "app.tasks.ai_prompt",
        "app.tasks.video_processing",
        "app.tasks.active_learning",
        "app.tasks.ocr_training",
        "app.tasks.tesseract_training",
        "app.tasks.crnn_training",
        "app.tasks.value_training",
        "app.tasks.sequence_run",
        "app.tasks.crop_cls_training",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,   # ← emit STARTED state so UI shows "Running"
    worker_pool="solo",        # Windows: avoids spawn-based prefork pool issues
)


# ---------------------------------------------------------------------------
# Crash logging for the worker (see app/crash_logs.py).
# Writes logs/crash_worker.log, logs/console_worker.log, logs/resources.log
# and one line per task start / finish / error in logs/training.log.
# ---------------------------------------------------------------------------
@signals.worker_init.connect
def _start_crash_logs(**kwargs):
    from .. import crash_logs
    crash_logs.setup("worker")


@signals.task_prerun.connect
def _log_task_start(task_id=None, task=None, kwargs=None, **_):
    from .. import crash_logs
    crash_logs.log_training(f"TASK START {task.name} id={task_id} args={kwargs}")


@signals.task_postrun.connect
def _log_task_end(task_id=None, task=None, state=None, **_):
    from .. import crash_logs
    crash_logs.log_training(f"TASK END {task.name} id={task_id} state={state}")


@signals.task_failure.connect
def _log_task_error(task_id=None, exception=None, einfo=None, sender=None, **_):
    from .. import crash_logs
    crash_logs.log_training(f"TASK ERROR {sender.name} id={task_id}: {exception!r}\n{einfo}")
