from typing import Any, Dict, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from petals.constants import DEFAULT_BIO_MODEL_NAME
from petals.utils.auto_config import AutoDistributedModelForCausalLM


def load_biology_model(
    model_name: str = DEFAULT_BIO_MODEL_NAME,
    *,
    token: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    **model_kwargs,
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load tokenizer and Petals distributed model for a biology-oriented checkpoint.

    The default model is OpenBioLLM (Llama 3), which works with Petals out of the box
    because Petals already supports the ``llama`` model type.
    """

    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    if token is not None and "token" not in tokenizer_kwargs and "use_auth_token" not in tokenizer_kwargs:
        tokenizer_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    if token is not None and "token" not in model_kwargs and "use_auth_token" not in model_kwargs:
        model_kwargs["token"] = token

    model = AutoDistributedModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return tokenizer, model
