import os
from distutils.util import strtobool

wsgi_app = "app:app"
bind = os.getenv("GUNICORN_BIND", f"0.0.0.0:{os.getenv('INTERNAL_PORT', 8000)}")
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
keep_alive = int(os.getenv("GUNICORN_KEEPALIVE", timeout))
workers = int(os.getenv("GUNICORN_WORKERS", 1))
threads = int(os.getenv("GUNICORN_THREADS", 4))
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")
access_log_format = os.getenv(
    "GUNICORN_ACCESS_LOG_FORMAT",
    '{"time":"%(t)s","id":"%({X-Request-Id}i)s","ip":"%(h)s","method":"%(m)s","url":"%(U)s","status":"%(s)s","contentType":"%(Content-Type)s","userAgent":"%(a)s","referer":"%(f)s","sent":"%(B)s","duration":"%(D)s"}',
)
preload_app = bool(strtobool(os.getenv("GUNICORN_PRELOAD", "False")))
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", 1000))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", 100))
