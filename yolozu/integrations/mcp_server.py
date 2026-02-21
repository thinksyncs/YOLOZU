from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tool_runner import (
    convert_dataset,
    doctor,
    eval_coco,
    export_onnx_job,
    jobs_cancel,
    jobs_list,
    jobs_status,
    run_scenarios,
    runs_describe,
    runs_list,
    train_job,
    validate_dataset,
    validate_predictions,
)


app = FastMCP("yolozu")


@app.tool()
def doctor_tool(output: str = "reports/doctor.json") -> dict:
    """Run yolozu doctor and return JSON payload + short summary."""
    return doctor(output=output)


@app.tool()
def validate_predictions_tool(path: str, strict: bool = True) -> dict:
    """Validate predictions JSON."""
    return validate_predictions(path=path, strict=strict)


@app.tool()
def validate_dataset_tool(dataset: str, split: str | None = None, strict: bool = True, mode: str = "fail") -> dict:
    """Validate YOLO-format dataset."""
    return validate_dataset(dataset=dataset, split=split, strict=strict, mode=mode)


@app.tool()
def eval_coco_tool(
    dataset: str,
    predictions: str,
    split: str | None = None,
    dry_run: bool = True,
    output: str = "reports/mcp_coco_eval.json",
    max_images: int | None = None,
) -> dict:
    """Evaluate predictions using eval-coco."""
    return eval_coco(
        dataset=dataset,
        predictions=predictions,
        split=split,
        dry_run=dry_run,
        output=output,
        max_images=max_images,
    )


@app.tool()
def run_scenarios_tool(config: str, extra_args: list[str] | None = None) -> dict:
    """Run scenario suite via yolozu test."""
    return run_scenarios(config=config, extra_args=extra_args)


@app.tool()
def convert_dataset_tool(
    from_format: str,
    output: str,
    data: str | None = None,
    args_yaml: str | None = None,
    split: str | None = None,
    task: str | None = None,
    coco_root: str | None = None,
    instances_json: str | None = None,
    mode: str = "manifest",
    include_crowd: bool = False,
    force: bool = True,
) -> dict:
    """Convert external dataset layout into YOLOZU descriptor via migrate dataset."""
    return convert_dataset(
        from_format=from_format,
        output=output,
        data=data,
        args_yaml=args_yaml,
        split=split,
        task=task,
        coco_root=coco_root,
        instances_json=instances_json,
        mode=mode,
        include_crowd=include_crowd,
        force=force,
    )


@app.tool()
def train_job_tool(train_config: str, run_id: str | None = None, resume: str | None = None) -> dict:
    """Queue train command as asynchronous job and return job_id."""
    return train_job(train_config=train_config, run_id=run_id, resume=resume)


@app.tool()
def export_onnx_job_tool(dataset: str, output: str, split: str | None = None, force: bool = True) -> dict:
    """Queue export command as asynchronous job and return job_id."""
    return export_onnx_job(dataset=dataset, output=output, split=split, force=force)


@app.tool()
def jobs_list_tool() -> dict:
    """List jobs."""
    return jobs_list()


@app.tool()
def jobs_status_tool(job_id: str) -> dict:
    """Get status for one job."""
    return jobs_status(job_id)


@app.tool()
def jobs_cancel_tool(job_id: str) -> dict:
    """Cancel one job if possible."""
    return jobs_cancel(job_id)


@app.tool()
def runs_list_tool(limit: int = 20) -> dict:
    """List run directories and metadata."""
    return runs_list(limit=limit)


@app.tool()
def runs_describe_tool(run_id: str) -> dict:
    """Describe run artifacts."""
    return runs_describe(run_id)


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
