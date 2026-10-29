import json
from datetime import datetime, timedelta
from logging import getLogger
from zoneinfo import ZoneInfo

import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, task

from batch_common import business_date, digest, infer_batches, read, replace_day, save, validate_predictions, validate_rows

TZ = ZoneInfo("Europe/Moscow")
START_DATE = datetime(2026, 9, 1, 0, 0, tzinfo=TZ)
SOURCE_CONN_ID = "source_api"
WAREHOUSE_CONN_ID = "warehouse"
MODEL_VERSION = "v1"
HTTP_TIMEOUT = 10
LOGGER = getLogger(__name__)


def _get(url: str) -> requests.Response:
	response = requests.get(url, timeout=HTTP_TIMEOUT)
	response.raise_for_status()
	return response


def _extract(context: dict) -> dict:
	day: str = business_date(context["params"]["business_date"], context.get("data_interval_start"))
	host = BaseHook.get_connection(SOURCE_CONN_ID).host
	LOGGER.info("Extract day %s from %s", day, host)

	data = _get(f"{host}/basic/{day}.json").json()
	if data.get("business_date") != day:
		raise ValueError(f"Business date doesn't match: {data.get('business_date')!r} != {day!r}")
	LOGGER.info("Got %s rows", len(data["rows"]))

	model_response = _get(f"{host}/models/{MODEL_VERSION}.json")
	model_sha256 = digest(model_response.content)
	LOGGER.info("Got model %s, sha256 %s", MODEL_VERSION, model_sha256)

	return save({
		"business_date": day,
		"rows": data["rows"],
		"model": json.loads(model_response.content),
		"model_sha256": model_sha256,
	})


def _infer(ref: dict, batch_size: int) -> dict:
	extracted = read(ref)
	rows = validate_rows(extracted["rows"], extracted["business_date"])
	predictions = infer_batches(rows, extracted["model"], batch_size)
	validate_predictions(predictions, [row["customer_id"] for row in rows])

	return save({
		"business_date": extracted["business_date"],
		"model_version": MODEL_VERSION,
		"predictions": predictions,
		"model_sha256": extracted["model_sha256"],
	})


def _publish(ref: dict) -> dict:
	result = read(ref)
	conn = PostgresHook(postgres_conn_id=WAREHOUSE_CONN_ID).get_conn()
	
	try:
		replace_day(conn, result["business_date"], result["model_version"], result["predictions"])
	finally:
		conn.close()


	return {
		"business_date": result["business_date"],
		"model_version": result["model_version"],
		"rows": len(result["predictions"]),
	}


@dag(
	dag_id="hw_basic_batch_inference",
	start_date=START_DATE,
	schedule=None,
	catchup=False,
	max_active_runs=1,
	tags=["homework", "basic"],
	params={
		"business_date": Param(START_DATE.strftime("%Y-%m-%d"), type="string", format="date"),
		"batch_size": Param(32, type="integer", minimum=1, maximum=1000),
	},
	default_args={
		"retries": 2,
		"retry_delay": timedelta(seconds=5),
		"execution_timeout": timedelta(minutes=2),
	},
)
def hw_basic_batch_inference():
	@task
	def extract(**context):
		return _extract(context)

	@task(retries=0)
	def infer(ref: dict, **context):
		return _infer(ref, context["params"]["batch_size"])

	@task
	def publish(ref: dict):
		return _publish(ref)

	publish(infer(extract()))


hw_basic_batch_inference()
