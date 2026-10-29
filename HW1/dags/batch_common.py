import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from logging import getLogger

import numpy as np

LOGGER = getLogger(__name__)
FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()

def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_rows(rows, day):
    if not rows:
        raise ValueError("rows is empty!")

    seen_ids = set()
    for row in rows:
        customer_id = row["customer_id"]
        if not isinstance(customer_id, str) or not customer_id:
            raise ValueError(f"Invalid customer_id: {customer_id!r}")
        if customer_id in seen_ids:
            raise ValueError(f"Duplicate customer_id: {customer_id}")
        seen_ids.add(customer_id)

        if row["business_date"] != day:
            raise ValueError(f"Dates doesn't match for {customer_id}: {row['business_date']!r} != {day!r}")

        for name, (low, high) in zip(FEATURES, LIMITS):
            value = row[name]
            if not is_number(value):
                raise ValueError(f"Feature {name} of {customer_id} is not a finite number: {value!r}")
            if not low <= value <= high:
                raise ValueError(f"Feature {name} of {customer_id} out of range [{low}, {high}]: {value}")

    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError(f"Validation error for batch_size: {batch_size!r}")

    predictions = []
    for number, start in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[start:start + batch_size]
        LOGGER.info("Batch num: %s, size: %s", number, len(batch))
        predictions.extend(predict(batch, model))

    return predictions


def validate_predictions(predictions, expected_ids):
    customer_ids = [pred["customer_id"] for pred in predictions]

    if len(set(customer_ids)) != len(customer_ids):
        raise ValueError("Duplicates detected!")
    if len(customer_ids) != len(expected_ids) or set(customer_ids) != set(expected_ids):
        diff = sorted(set(customer_ids) ^ set(expected_ids))
        raise ValueError(f"Customer not equals! Diff: {diff[:10]}")

    for pred in predictions:
        score, label = pred["score"], pred["label"]
        if not is_number(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score for {pred['customer_id']}: {score!r}")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Invalid label for {pred['customer_id']}: {label!r}")
        if label != int(score >= 0.5):
            raise ValueError(f"Label doesn't match score for {pred['customer_id']}: {label} vs {score}")

def replace_day(conn, day, model_version, predictions):
    if not predictions:
        raise ValueError("Empty predictions: refusing to replace the day snapshot")

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{day}:{model_version}",))
        cur.execute(
            "DELETE FROM predictions WHERE business_date = %s AND model_version = %s",
            (day, model_version),
        )
        cur.executemany(
            "INSERT INTO predictions (business_date, model_version, customer_id, score, label)"
            " VALUES (%s, %s, %s, %s, %s)",
            [(day, model_version, p["customer_id"], p["score"], p["label"]) for p in predictions],
        )
    LOGGER.info("Replaced %s predictions for %s, model %s", len(predictions), day, model_version)
