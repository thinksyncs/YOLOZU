def run_adapter(adapter, records):
    predictions = adapter.predict(records)
    total = sum(len(entry.get("detections", [])) for entry in predictions)
    return {"images": len(records), "detections": total}
