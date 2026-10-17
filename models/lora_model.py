
from pathlib import Path

from transformers import AutoModelForSequenceClassification
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

from config import (
    BASE_MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)


def build_base_classifier(model_name=BASE_MODEL_NAME):
    """Load the pretrained model with a classification head."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


def apply_lora(model, r=LORA_R):
    """Apply LoRA adapters to the classifier."""
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


def build_roberta_lora(model_name=BASE_MODEL_NAME, r=LORA_R):
    """Build a new RoBERTa + LoRA model for training."""
    base_model = build_base_classifier(model_name)
    return apply_lora(base_model, r=r)


def load_classifier(model_dir, base_model_name=BASE_MODEL_NAME):
    """
    Load a saved PEFT adapter or a complete sequence-classification model.
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}"
        )

    adapter_config = model_dir / "adapter_config.json"
    full_config = model_dir / "config.json"

    if adapter_config.exists():
        # Load the matching base architecture, then restore the saved adapter
        # and any modules saved with it, including the classifier.
        base_model = build_base_classifier(base_model_name)

        model = PeftModel.from_pretrained(
            base_model,
            str(model_dir),
            is_trainable=False,
        )

        return model

    if full_config.exists():
        # Load a complete saved model checkpoint.
        return AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )

    raise FileNotFoundError(
        f"No adapter_config.json or config.json found in {model_dir}"
    )
