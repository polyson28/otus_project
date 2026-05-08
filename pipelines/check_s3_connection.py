import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_env_or_config, load_config
from src.io.s3 import read_json, write_json


def main() -> None:
    load_dotenv()
    config = load_config()
    bucket = get_env_or_config("YANDEX_S3_BUCKET", config.get("s3", {}).get("bucket"))
    if not bucket:
        raise RuntimeError("Missing bucket. Set YANDEX_S3_BUCKET or configs.config.yaml:s3.bucket.")

    test_uri = f"s3://{bucket}/_checks/s3_connection_test.json"
    payload = {"status": "ok", "check": "s3_connection"}

    write_json(payload, test_uri)
    result = read_json(test_uri)

    if result != payload:
        raise RuntimeError(f"S3 smoke test failed: expected {payload}, got {result}")

    print(f"OK: wrote and read {test_uri}")


if __name__ == "__main__":
    main()
