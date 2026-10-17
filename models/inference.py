import torch

from config import ID2LABEL, MAX_LENGTH


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainable_state_dict(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model, state):
    current = model.state_dict()
    filtered = {key: value for key, value in state.items() if key in current}
    missing = model.load_state_dict(filtered, strict=False)
    return missing


def tokenize_batch(tokenizer, batch, max_length=MAX_LENGTH):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def predict_logits(model, tokenizer, texts, device=None, max_length=MAX_LENGTH):
    if isinstance(texts, str):
        texts = [texts]
    if device is None:
        device = next(model.parameters()).device
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
    return logits


def predict_proba(model, tokenizer, texts, device=None, max_length=MAX_LENGTH):
    logits = predict_logits(model, tokenizer, texts, device=device, max_length=max_length)
    return torch.softmax(logits, dim=-1)


def decode_prediction(probs):
    confidence, pred_id = torch.max(probs, dim=-1)
    pred_id = int(pred_id[0].item())
    return {
        "label_id": pred_id,
        "severity": ID2LABEL[pred_id],
        "confidence": float(confidence[0].item()),
        "probabilities": {
            ID2LABEL[i]: float(probs[0, i].item()) for i in range(probs.shape[-1])
        },
    }
