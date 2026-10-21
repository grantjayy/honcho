"""Host log files: each Honcho process writes its logs to LOG_DIR when set."""

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from src.telemetry.log_files import configure_log_file


@pytest.fixture
def isolated_root_logger() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    try:
        yield root
    finally:
        for handler in root.handlers:
            if handler not in original_handlers:
                handler.close()
        root.handlers = original_handlers


def test_writes_service_log_file_when_log_dir_set(
    tmp_path: Path, isolated_root_logger: logging.Logger
) -> None:
    configure_log_file("deriver", log_dir=str(tmp_path))

    logging.getLogger("src.deriver.deriver").warning("queue item failed")
    for handler in isolated_root_logger.handlers:
        handler.flush()

    contents = (tmp_path / "deriver.log").read_text()
    assert "src.deriver.deriver - WARNING - queue item failed" in contents


def test_rotates_at_50_mb_keeping_10_files(
    tmp_path: Path, isolated_root_logger: logging.Logger
) -> None:
    configure_log_file("api", log_dir=str(tmp_path))

    file_handlers = [
        h for h in isolated_root_logger.handlers if isinstance(h, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    assert file_handlers[0].maxBytes == 50 * 1024 * 1024
    assert file_handlers[0].backupCount == 10


def test_no_file_handler_without_log_dir(
    isolated_root_logger: logging.Logger,
) -> None:
    before = list(isolated_root_logger.handlers)

    configure_log_file("api", log_dir=None)

    assert isolated_root_logger.handlers == before
