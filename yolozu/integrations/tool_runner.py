from __future__ import annotations

from typing import Any

from .layers.api import run_cli_tool
from .layers.artifacts import collect_artifact_metadata, describe_run, list_runs
from .layers.jobs import JobManager

_JOBS = JobManager()


def _with_meta(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("meta", collect_artifact_metadata())
    return payload


def doctor(*, output: str = "reports/doctor.json") -> dict[str, Any]:
    return _with_meta(run_cli_tool("doctor", ["doctor", "--output", output], artifacts={"doctor": output}))


def validate_predictions(path: str, *, strict: bool = True) -> dict[str, Any]:
    args = ["validate", "predictions", path]
    if strict:
        args.append("--strict")
    return _with_meta(run_cli_tool("validate_predictions", args))


def validate_dataset(
    dataset: str,
    *,
    split: str | None = None,
    strict: bool = True,
    mode: str = "fail",
) -> dict[str, Any]:
    args = ["validate", "dataset", dataset, "--mode", mode]
    if split:
        args.extend(["--split", split])
    if strict:
        args.append("--strict")
    return _with_meta(run_cli_tool("validate_dataset", args))


def eval_coco(
    dataset: str,
    predictions: str,
    *,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
) -> dict[str, Any]:
    args = [
        "eval-coco",
        "--dataset",
        dataset,
        "--predictions",
        predictions,
        "--output",
        output,
    ]
    if split:
        args.extend(["--split", split])
    if dry_run:
        args.append("--dry-run")
    if max_images is not None:
        args.extend(["--max-images", str(max_images)])
    return _with_meta(run_cli_tool("eval_coco", args, artifacts={"report": output}))


def run_scenarios(config: str, *, extra_args: list[str] | None = None) -> dict[str, Any]:
    args = ["test", config, *(extra_args or [])]
    return _with_meta(run_cli_tool("run_scenarios", args))


def convert_dataset(
    from_format: str,
    output: str,
    *,
    data: str | None = None,
    args_yaml: str | None = None,
    split: str | None = None,
    task: str | None = None,
    coco_root: str | None = None,
    instances_json: str | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = True,
) -> dict[str, Any]:
    args = ["migrate", "dataset", "--from", from_format, "--output", output, "--mode", mode]
    if data:
        args.extend(["--data", data])
    if args_yaml:
        args.extend(["--args", args_yaml])
    if split:
        args.extend(["--split", split])
    if task:
        args.extend(["--task", task])
    if coco_root:
        args.extend(["--coco-root", coco_root])
    if instances_json:
        args.extend(["--instances-json", instances_json])
    if include_crowd:
        args.append("--include-crowd")
    if force:
        args.append("--force")
    return _with_meta(run_cli_tool("convert_dataset", args))


def submit_job(name: str, args: list[str], *, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    job_id = _JOBS.submit(name, lambda: _with_meta(run_cli_tool(name, args, artifacts=artifacts)))
    return {
        "ok": True,
        "tool": "jobs.submit",
        "summary": f"job queued: {job_id}",
        "job_id": job_id,
        "status": "queued",
        "meta": collect_artifact_metadata(),
    }


def jobs_list() -> dict[str, Any]:
    return {
        "ok": True,
        "tool": "jobs.list",
        "summary": "listed jobs",
        "jobs": _JOBS.list(),
        "meta": collect_artifact_metadata(),
    }


def jobs_status(job_id: str) -> dict[str, Any]:
    status = _JOBS.status(job_id)
    if status is None:
        return {
            "ok": False,
            "tool": "jobs.status",
            "summary": "job not found",
            "job_id": job_id,
            "meta": collect_artifact_metadata(),
        }
    return {
        "ok": True,
        "tool": "jobs.status",
        "summary": f"job status: {status.get('status')}",
        "job": status,
        "meta": collect_artifact_metadata(),
    }


def jobs_cancel(job_id: str) -> dict[str, Any]:
    out = _JOBS.cancel(job_id)
    if out is None:
        return {
            "ok": False,
            "tool": "jobs.cancel",
            "summary": "job not found",
            "job_id": job_id,
            "meta": collect_artifact_metadata(),
        }
    return {
        "ok": bool(out.get("cancelled")),
        "tool": "jobs.cancel",
        "summary": f"cancelled={out.get('cancelled')}",
        **out,
        "meta": collect_artifact_metadata(),
    }


def runs_list(limit: int = 20) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": "runs.list",
        "summary": "listed runs",
        "runs": list_runs(limit=limit),
        "meta": collect_artifact_metadata(),
    }


def runs_describe(run_id: str) -> dict[str, Any]:
    details = describe_run(run_id)
    if details is None:
        return {
            "ok": False,
            "tool": "runs.describe",
            "summary": "run not found",
            "run_id": run_id,
            "meta": collect_artifact_metadata(),
        }
    return {
        "ok": True,
        "tool": "runs.describe",
        "summary": "run described",
        "run": details,
        "meta": collect_artifact_metadata(),
    }


def train_job(train_config: str, *, run_id: str | None = None, resume: str | None = None) -> dict[str, Any]:
    args = ["train", train_config]
    if run_id:
        args.extend(["--run-id", run_id])
    if resume:
        args.extend(["--resume", resume])
    return submit_job("train", args)


def export_onnx_job(dataset: str, output: str, *, split: str | None = None, force: bool = True) -> dict[str, Any]:
    args = ["export", "--backend", "labels", "--dataset", dataset, "--output", output]
    if split:
        args.extend(["--split", split])
    if force:
        args.append("--force")
    return submit_job("export", args, artifacts={"predictions": output})
