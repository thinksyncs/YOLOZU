FROM python:3.11-slim

WORKDIR /opt/yolozu

ENV PYTHONUNBUFFERED=1

RUN python -m pip install --upgrade pip

COPY pyproject.toml README.md LICENSE NOTICE COPYRIGHT ./
COPY yolozu ./yolozu

RUN python -m pip install .

ENTRYPOINT ["yolozu"]
CMD ["--help"]

