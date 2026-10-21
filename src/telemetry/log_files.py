"""Rotating log files on a host-mounted directory.

Container stdout is lost when a container is recreated, and `docker logs`
cannot be searched across restarts. When LOG_DIR is set (the self-hosted
compose file mounts a host folder there), each process also writes its own
`<service>.log`, so operators can grep history and measure model cost.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 50 MB x 10 files bounds disk use at about 550 MB per service. One day of
# deriver INFO logs measured at about 4 MB, so this keeps months of history.
_MAX_BYTES = 50 * 1024 * 1024
_BACKUP_COUNT = 10
_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_log_file(service: str, *, log_dir: str | None) -> None:
    """Attach a rotating file handler for `service` to the root logger."""
    if not log_dir:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path / f"{service}.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(handler)
