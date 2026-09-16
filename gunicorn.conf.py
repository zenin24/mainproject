"""
Gunicorn configuration file for production deployment.

Uses Uvicorn worker class for asynchronous FastAPI request handling.
"""

import os
import multiprocessing

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
# For ML model inference (PubMedCLIP), 1-2 workers per GPU/CPU node recommended to avoid OOM
workers = int(os.getenv("WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info")

# Process naming
proc_name = "chest_xray_analyzer"
