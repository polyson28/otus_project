from __future__ import annotations

from app.model_loader import ModelBundle


def health_payload(bundle: ModelBundle | None) -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": bundle.model_version if bundle is not None else None,
        "model_uri": bundle.model_uri if bundle is not None else None,
    }
