from __future__ import annotations

import json
import os
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv


DEFAULT_ENDPOINT_URL = "https://storage.yandexcloud.net"
DEFAULT_REGION = "ru-central1"


class S3Error(RuntimeError):
    """Base error for S3/Object Storage operations."""


class MissingS3CredentialsError(S3Error):
    """Raised when S3 credentials are not configured."""


class InvalidS3URIError(S3Error):
    """Raised when an S3 URI has an invalid shape."""


class S3ObjectNotFoundError(S3Error):
    """Raised when an S3 object does not exist."""


def is_s3_uri(uri: str | os.PathLike[str]) -> bool:
    return str(uri).startswith("s3://")


def parse_s3_uri(uri: str | os.PathLike[str]) -> tuple[str, str]:
    parsed = urlparse(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise InvalidS3URIError(
            f"Invalid S3 URI: {uri}. Expected format: s3://bucket/path/to/object"
        )
    return parsed.netloc, parsed.path.lstrip("/")


def get_s3_client():
    load_dotenv()
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        raise MissingS3CredentialsError(
            "Missing S3 credentials. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YANDEX_S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL),
        region_name=os.getenv("YANDEX_S3_REGION", DEFAULT_REGION),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _is_not_found_error(error: ClientError) -> bool:
    code = error.response.get("Error", {}).get("Code")
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _raise_s3_error(error: Exception, uri: str) -> None:
    if isinstance(error, NoCredentialsError):
        raise MissingS3CredentialsError(
            "Missing S3 credentials. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        ) from error
    if isinstance(error, ClientError) and _is_not_found_error(error):
        raise S3ObjectNotFoundError(f"S3 object not found: {uri}") from error
    raise S3Error(f"S3 operation failed for {uri}: {error}") from error


def _read_s3_bytes(uri: str | os.PathLike[str]) -> bytes:
    uri_str = str(uri)
    bucket, key = parse_s3_uri(uri_str)
    try:
        response = get_s3_client().get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except Exception as error:
        _raise_s3_error(error, uri_str)
        raise


def _write_s3_bytes(data: bytes, uri: str | os.PathLike[str], content_type: str | None = None) -> None:
    uri_str = str(uri)
    bucket, key = parse_s3_uri(uri_str)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    try:
        get_s3_client().put_object(**kwargs)
    except Exception as error:
        _raise_s3_error(error, uri_str)


def upload_file(local_path: str | os.PathLike[str], s3_uri: str | os.PathLike[str]) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    try:
        get_s3_client().upload_file(str(local_path), bucket, key)
    except Exception as error:
        _raise_s3_error(error, str(s3_uri))


def download_file(s3_uri: str | os.PathLike[str], local_path: str | os.PathLike[str]) -> None:
    bucket, key = parse_s3_uri(s3_uri)
    _ensure_parent_dir(local_path)
    try:
        get_s3_client().download_file(bucket, key, str(local_path))
    except Exception as error:
        _raise_s3_error(error, str(s3_uri))


def read_json(uri: str | os.PathLike[str]) -> Any:
    if is_s3_uri(uri):
        return json.loads(_read_s3_bytes(uri).decode("utf-8"))
    with Path(uri).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(obj: Any, uri: str | os.PathLike[str]) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    if is_s3_uri(uri):
        _write_s3_bytes(data, uri, content_type="application/json")
        return
    _ensure_parent_dir(uri)
    with Path(uri).open("wb") as file:
        file.write(data)


def read_parquet(uri: str | os.PathLike[str]) -> pd.DataFrame:
    if is_s3_uri(uri):
        return pd.read_parquet(BytesIO(_read_s3_bytes(uri)))
    return pd.read_parquet(uri)


def write_parquet(df: pd.DataFrame, uri: str | os.PathLike[str]) -> None:
    if is_s3_uri(uri):
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        _write_s3_bytes(buffer.getvalue(), uri, content_type="application/octet-stream")
        return
    _ensure_parent_dir(uri)
    df.to_parquet(uri, index=False)


def read_csv(uri: str | os.PathLike[str]) -> pd.DataFrame:
    if is_s3_uri(uri):
        return pd.read_csv(BytesIO(_read_s3_bytes(uri)))
    return pd.read_csv(uri)


def write_csv(df: pd.DataFrame, uri: str | os.PathLike[str]) -> None:
    if is_s3_uri(uri):
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        _write_s3_bytes(buffer.getvalue().encode("utf-8"), uri, content_type="text/csv")
        return
    _ensure_parent_dir(uri)
    df.to_csv(uri, index=False)


def read_excel(uri: str | os.PathLike[str]) -> pd.DataFrame:
    if is_s3_uri(uri):
        return pd.read_excel(BytesIO(_read_s3_bytes(uri)))
    return pd.read_excel(uri)


def joblib_dump_to_uri(obj: Any, uri: str | os.PathLike[str]) -> None:
    if is_s3_uri(uri):
        buffer = BytesIO()
        joblib.dump(obj, buffer)
        _write_s3_bytes(buffer.getvalue(), uri, content_type="application/octet-stream")
        return
    _ensure_parent_dir(uri)
    joblib.dump(obj, uri)


def joblib_load_from_uri(uri: str | os.PathLike[str]) -> Any:
    if is_s3_uri(uri):
        return joblib.load(BytesIO(_read_s3_bytes(uri)))
    return joblib.load(uri)
