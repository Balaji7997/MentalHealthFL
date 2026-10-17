from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from config import (
    CENTRALIZED_LORA_DIR,
    CENTRALIZED_MODEL_DIR,
    FEDERATED_MODEL_DIR,
    BASE_MODEL_NAME,
    ID2LABEL,
)

from models.inference import (
    get_device,
    predict_proba,
    decode_prediction,
)

from models.lora_model import load_classifier

_CACHE = {}


# ==========================================================
# MODEL RESOLUTION
# ==========================================================

def resolve_model_dir(prefer_federated=True):

    candidates = []

    if prefer_federated:
        candidates.extend(
            [
                FEDERATED_MODEL_DIR,
                CENTRALIZED_LORA_DIR,
                CENTRALIZED_MODEL_DIR,
            ]
        )
    else:
        candidates.extend(
            [
                CENTRALIZED_LORA_DIR,
                CENTRALIZED_MODEL_DIR,
                FEDERATED_MODEL_DIR,
            ]
        )

    for path in candidates:

        if (
            (path / "adapter_config.json").exists()
            or
            (path / "config.json").exists()
        ):
            return path

    raise FileNotFoundError(
        "No trained model found."
    )


# ==========================================================
# MODEL LOADING
# ==========================================================

def load_severity_bundle(
    model_dir=None,
    prefer_federated=True,
):

    model_dir = (
        Path(model_dir)
        if model_dir
        else resolve_model_dir(prefer_federated)
    )

    cache_key = str(model_dir.resolve())

    if cache_key in _CACHE:
        return _CACHE[cache_key]

    print(f"\nLoading model from: {model_dir}")

    tokenizer_source = model_dir

    if not (
        (model_dir / "tokenizer.json").exists()
        or
        (model_dir / "tokenizer_config.json").exists()
    ):
        tokenizer_source = BASE_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source)
    )

    model = load_classifier(model_dir)

    device = get_device()

    model.to(device)
    model.eval()

    bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "model_dir": model_dir,
        "source": model_dir.name,
        "id2label": ID2LABEL,
    }

    _CACHE[cache_key] = bundle

    return bundle


# ==========================================================
# PREDICTION
# ==========================================================

def predict_severity(
    text,
    model_dir=None,
    prefer_federated=True,
):

    bundle = load_severity_bundle(
        model_dir=model_dir,
        prefer_federated=prefer_federated,
    )

    probs = predict_proba(
        bundle["model"],
        bundle["tokenizer"],
        text,
        device=bundle["device"],
    )

    result = decode_prediction(probs)

    result["model_source"] = bundle["source"]
    result["model_dir"] = str(bundle["model_dir"])

    return result


# ==========================================================
# RELIABILITY HELPER
# ==========================================================

def predict_with_reliability(
    text,
    model_dir=None,
    prefer_federated=True,
):

    result = predict_severity(
        text=text,
        model_dir=model_dir,
        prefer_federated=prefer_federated,
    )

    confidence = float(result["confidence"])

    if confidence >= 0.85:
        reliability = "High"

    elif confidence >= 0.65:
        reliability = "Medium"

    else:
        reliability = "Low"

    result["reliability"] = reliability

    return result