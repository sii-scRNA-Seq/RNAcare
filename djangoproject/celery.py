import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "djangoproject.settings")

app = Celery(
    "djangoproject",
    broker="redis://127.0.0.1:8001",
    backend="redis://localhost:8001",
    include=["IMID.tasks"],
)

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()
app.conf.update(
    result_expires=3600,
    task_serializer="pickle",
    result_serializer="pickle",
    accept_content=["pickle", "json"],
)


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    print(f"Request: {self.request!r}")


if __name__ == "__main__":
    app.start()
