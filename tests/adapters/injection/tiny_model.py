"""A tiny, randomly initialised (seeded) DeBERTa-v2 classifier saved as a local model directory.

It lets the host run the real loading and inference path of the detector (``AutoTokenizer``,
``AutoModelForSequenceClassification``, overflowing windows with a stride, padding, softmax)
without the 300 MB Prompt Guard 2 model, with the same architecture and the same ``LABEL_0`` /
``LABEL_1`` naming as the real one. The scores mean nothing: tests that need to tell windows
apart replace the model's forward pass. Inputs are at most ``MAX_LENGTH`` tokens, so a few
sentences already need several windows.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    PreTrainedTokenizerFast,
)

MAX_LENGTH = 16
STRIDE = 4
HIDDEN = 8
LABELS = {0: "LABEL_0", 1: "LABEL_1"}
INJECTION_WORD = "ignore"
"""The word the forward-pass stubs of the tests treat as the injection signal."""
_WORDS = (
    *("ignore", "previous", "instructions", "the", "contract", "was", "signed", "by"),
    *("maria", "and", "paid", "on", "monday", "alpha", "beta", "gamma", "delta"),
)
_SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
_SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
}


def build_tiny_model(directory: Path) -> Path:
    """Save the tokenizer (``tokenizer.json``) and the weights (``model.safetensors``)."""
    vocab = {token: index for index, token in enumerate([*_SPECIAL, *_WORDS, ".", ","])}
    _tokenizer(vocab).save_pretrained(directory)
    _model(vocab).save_pretrained(directory)
    return directory


def _tokenizer(vocab: dict[str, int]) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))  # noqa: S106 - not a secret
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        model_max_length=MAX_LENGTH,
        **_SPECIAL_TOKENS,
    )


def _model(vocab: dict[str, int]) -> DebertaV2ForSequenceClassification:
    config = DebertaV2Config(
        vocab_size=len(vocab),
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=2 * HIDDEN,
        max_position_embeddings=MAX_LENGTH,
        id2label=LABELS,
        label2id={label: index for index, label in LABELS.items()},
    )
    torch.manual_seed(0)
    return DebertaV2ForSequenceClassification(config)  # type: ignore[no-untyped-call]
