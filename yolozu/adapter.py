class ModelAdapter:
    def predict(self, records):
        raise NotImplementedError


class DummyAdapter(ModelAdapter):
    def predict(self, records):
        return [
            {"image": record["image"], "detections": []} for record in records
        ]
