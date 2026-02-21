from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .tool_runner import (
    calibrate_predictions,
    convert_dataset,
    doctor,
    eval_coco,
    export_onnx_job,
    jobs_cancel,
    jobs_list,
    jobs_status,
    parity_check,
    predict_images,
    run_scenarios,
    runs_describe,
    runs_list,
    train_job,
    validate_dataset,
    validate_predictions,
)


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


class PredictImagesRequest(BaseModel):
    input_dir: str
    backend: str = "dummy"
    output: str = "reports/actions_predict_images.json"
    max_images: int | None = None
    dry_run: bool = True
    strict: bool = True
    force: bool = True


class ParityCheckRequest(BaseModel):
    reference: str
    candidate: str
    iou_thresh: float = 0.5
    score_atol: float = 1e-6
    bbox_atol: float = 1e-4
    max_images: int | None = None
    image_size: str | None = None


class CalibratePredictionsRequest(BaseModel):
    dataset: str
    predictions: str
    method: str = "fracal"
    split: str | None = None
    task: str = "auto"
    output: str = "reports/actions_calibrated_predictions.json"
    output_report: str = "reports/actions_calibration_report.json"
    max_images: int | None = None
    force: bool = True


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


class TrainJobRequest(BaseModel):
    train_config: str
    run_id: str | None = None
    resume: str | None = None


class ExportOnnxJobRequest(BaseModel):
    dataset: str
    output: str
    split: str | None = None
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


@app.post("/predict/images")
def predict_images_route(req: PredictImagesRequest) -> dict:
    return predict_images(
        input_dir=req.input_dir,
        backend=req.backend,
        output=req.output,
        max_images=req.max_images,
        dry_run=req.dry_run,
        strict=req.strict,
        force=req.force,
    )


@app.post("/parity/check")
def parity_check_route(req: ParityCheckRequest) -> dict:
    return parity_check(
        reference=req.reference,
        candidate=req.candidate,
        iou_thresh=req.iou_thresh,
        score_atol=req.score_atol,
        bbox_atol=req.bbox_atol,
        max_images=req.max_images,
        image_size=req.image_size,
    )


@app.post("/calibrate/predictions")
def calibrate_predictions_route(req: CalibratePredictionsRequest) -> dict:
    return calibrate_predictions(
        dataset=req.dataset,
        predictions=req.predictions,
        method=req.method,
        split=req.split,
        task=req.task,
        output=req.output,
        output_report=req.output_report,
        max_images=req.max_images,
        force=req.force,
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


@app.post("/jobs/train")
def train_job_route(req: TrainJobRequest) -> dict:
    return train_job(train_config=req.train_config, run_id=req.run_id, resume=req.resume)


@app.post("/jobs/export-onnx")
def export_onnx_job_route(req: ExportOnnxJobRequest) -> dict:
    return export_onnx_job(dataset=req.dataset, output=req.output, split=req.split, force=req.force)


@app.get("/jobs")
def jobs_list_route() -> dict:
    return jobs_list()


@app.get("/jobs/{job_id}")
def jobs_status_route(job_id: str) -> dict:
    return jobs_status(job_id)


@app.post("/jobs/{job_id}/cancel")
def jobs_cancel_route(job_id: str) -> dict:
    return jobs_cancel(job_id)


@app.get("/runs")
def runs_list_route(limit: int = 20) -> dict:
    return runs_list(limit=limit)


@app.get("/runs/{run_id}")
def runs_describe_route(run_id: str) -> dict:
    return runs_describe(run_id)
