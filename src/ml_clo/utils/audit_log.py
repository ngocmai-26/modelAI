"""MISSING-06: Lightweight prediction audit log.

Records each prediction call to a JSONL file so backends can trace
which student/subject/lecturer was queried, the predicted score, and
a timestamp — useful for debugging and compliance. The log path defaults
to ``logs/predictions.jsonl`` next to the working directory; callers can
override via ``set_audit_log_path()``.

Design decisions:
- JSONL (one JSON object per line) rather than CSV — schema-flexible and
  append-safe under concurrent writes.
- No dependency beyond stdlib + the project logger.
- ``log_prediction`` is a no-op if audit logging is not enabled.
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_audit_path: Optional[Path] = None


def set_audit_log_path(path: str) -> None:
    """Configure the audit log file path. Creates parent dirs if needed."""
    global _audit_path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _audit_path = p
    logger.info(f"Audit log path set to {_audit_path}")


def get_audit_log_path() -> Optional[Path]:
    return _audit_path


def log_prediction(
    student_id: str,
    subject_id: str,
    lecturer_id: str,
    predicted_score: float,
    actual_score: Optional[float] = None,
    model_version: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one prediction record to the audit log.

    If ``set_audit_log_path`` has not been called, this is a no-op so
    callers do not need to guard against missing configuration.
    """
    if _audit_path is None:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "student_id": str(student_id),
        "subject_id": str(subject_id),
        "lecturer_id": str(lecturer_id),
        "predicted_score": round(float(predicted_score), 4),
    }
    if actual_score is not None:
        record["actual_score"] = round(float(actual_score), 4)
    if model_version:
        record["model_version"] = model_version
    if extra:
        record.update(extra)

    line = json.dumps(record, ensure_ascii=False) + "\n"

    with _lock:
        try:
            with open(_audit_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            logger.warning(f"Failed to write audit log: {exc}")
