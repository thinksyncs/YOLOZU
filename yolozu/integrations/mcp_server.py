from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .tool_runner import convert_dataset, doctor, eval_coco, run_scenarios, validate_dataset, validate_predictions


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


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()
