# Module 5: Deploying Machine Learning Models

- Instructions: [homework.md](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/05-deployment/homework.md)
- The model, API, lockfile and Dockerfile are the frozen 2026 course release, copied unchanged.
- My code: [`q3_predict.py`](q3_predict.py)

## Setup

```bash
uv sync --locked
uv run python smoke_test.py
```

`pipeline.bin` checksum (matches the release):

```text
1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1  pipeline.bin
```

## Q1. Environment check

```bash
uv --version
```

```text
uv 0.12.9 (9f9286029 2026-09-01 x86_64-pc-windows-msvc)
```

## Q2. Locked Scikit-Learn version

`pyproject.toml` pins `scikit-learn==1.7.2`, and `uv.lock` resolves to the same version.

## Q3. Load the model with pickle

```bash
uv run python q3_predict.py
```

```text
probability: 0.5329944186432507
rounded:     0.533
```

## Q4. Serve the model

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
python q6_test.py
```

```text
{'conversion_probability': 0.769799, 'conversion': True}
```

`GET /health`:

```json
{
  "status": "ok",
  "model_sha256": "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
  "feature_defaults_sha256": "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"
}
```

Both checksums match `model_metadata.json`.

> **Windows note:** with CRLF line endings `feature_defaults.json` gets a different checksum
> (the values are identical, so predictions don't change). `.gitattributes` keeps the
> deployment JSON files with LF endings so the checksum matches the release.

## Q5. Base image

The `Dockerfile` declares `python:3.11.15-slim-bookworm` (pinned by digest).

## Q6. Run the container

```bash
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py
```

The container runs the same `pipeline.bin` and `predict.py` as Q4, so the expected
`conversion_probability` is the same reference inference: `0.770`.
