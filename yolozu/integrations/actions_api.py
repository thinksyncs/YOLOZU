from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .tool_runner import convert_dataset, doctor, eval_coco, run_scenarios, validate_dataset, validate_predictions


app = FastAPI(title="YOLOZU Actions API", version="0.1.0")


class DoctorRequest(BaseModel):
    output: str = "reports/doctor.json"


class ValidatePredictionsRequest(BaseModel):
    path: str
    strict: bool = True


class ValidateDatasetRequest(BaseModel):
    dataset: str
    split: str | None = None
    strict: bool = True
    mode: str = "fail"


class EvalCocoRequest(BaseModel):
    dataset: str
    predictions: str
    split: str | None = None
    dry_run: bool = True
    output: str = "reports/actions_coco_eval.json"
    max_images: int | None = None


class RunScenariosRequest(BaseModel):
    config: str
    extra_args: list[str] | None = None


class ConvertDatasetRequest(BaseModel):
    from_format: str
    output: str
    data: str | None = None
    args_yaml: str | None = None
    split: str | None = None
    task: str | None = None
    coco_root: str | None = None
    instances_json: str | None = None
    mode: str = "manifest"
    include_crowd: bool = False
    force: bool = True


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "summary": "yolozu-actions-api healthy"}


@app.post("/doctor")
def doctor_route(req: DoctorRequest) -> dict:
    return doctor(output=req.output)


@app.post("/validate/predictions")
def validate_predictions_route(req: ValidatePredictionsRequest) -> dict:
    return validate_predictions(path=req.path, strict=req.strict)


@app.post("/validate/dataset")
def validate_dataset_route(req: ValidateDatasetRequest) -> dict:
    return validate_dataset(dataset=req.dataset, split=req.split, strict=req.strict, mode=req.mode)


@app.post("/eval/coco")
def eval_coco_route(req: EvalCocoRequest) -> dict:
    return eval_coco(
        dataset=req.dataset,
        predictions=req.predictions,
        split=req.split,
        dry_run=req.dry_run,
        output=req.output,
        max_images=req.max_images,
    )


@app.post("/run/scenarios")
def run_scenarios_route(req: RunScenariosRequest) -> dict:
    return run_scenarios(config=req.config, extra_args=req.extra_args)


@app.post("/convert/dataset")
def convert_dataset_route(req: ConvertDatasetRequest) -> dict:
    return convert_dataset(
        from_format=req.from_format,
        output=req.output,
        data=req.data,
        args_yaml=req.args_yaml,
        split=req.split,
        task=req.task,
        coco_root=req.coco_root,
        instances_json=req.instances_json,
        mode=req.mode,
        include_crowd=req.include_crowd,
        force=req.force,
    )
