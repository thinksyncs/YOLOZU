from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _JobState:
    job_id: str
    name: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    future: Future | None = None


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yolozu-mcp-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobState] = {}

    def submit(self, name: str, fn: Callable[[], dict[str, Any]]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        state = _JobState(job_id=job_id, name=name, status="queued", created_at=time.time())

        def _run() -> dict[str, Any]:
            with self._lock:
                state.status = "running"
                state.started_at = time.time()
            try:
                result = fn()
                with self._lock:
                    state.status = "completed"
                    state.result = result
                    state.finished_at = time.time()
                return result
            except Exception as exc:
                with self._lock:
                    state.status = "failed"
                    state.error = str(exc)
                    state.finished_at = time.time()
                raise

        with self._lock:
            self._jobs[job_id] = state
            state.future = self._executor.submit(_run)
        return job_id

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                }
                for job in self._jobs.values()
            ]

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
                "result": job.result,
            }

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.future and job.future.cancel():
                job.status = "cancelled"
                job.finished_at = time.time()
                return {"job_id": job_id, "cancelled": True}
            if job.status in ("completed", "failed", "cancelled"):
                return {"job_id": job_id, "cancelled": False, "reason": f"already_{job.status}"}
            return {"job_id": job_id, "cancelled": False, "reason": "running"}
