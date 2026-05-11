from __future__ import annotations

import json

from app.model_loader import load_model_bundle


def main() -> None:
    bundle = load_model_bundle()
    pipeline = bundle.feature_pipeline
    payload = {
        "model_uri": bundle.model_uri,
        "model_version": bundle.model_version,
        "date_col": getattr(pipeline, "date_col", None),
        "target_col": getattr(pipeline, "target_col", None),
        "macro_columns": getattr(pipeline, "macro_columns_", []),
        "macro_numeric_columns": getattr(pipeline, "macro_numeric_columns_", []),
        "model_feature_columns": bundle.metadata.get("feature_columns"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
