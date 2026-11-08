FROM python:3.14.7-slim-trixie@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY main.py .
COPY model/ model/
COPY utils/ utils/
COPY weights/model.keras weights/model.keras

ENTRYPOINT ["python", "main.py"]
CMD ["--input", "/data/input", "--output", "/data/output"]
