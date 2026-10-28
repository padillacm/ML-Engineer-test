"""A tiny, randomly initialised (seeded) BERT encoder saved as a local model directory.

It lets the host run the real loading and inference path of the embedder (``AutoTokenizer``,
``AutoModel``, padding, truncation, pooling) without the 1 GB e5 model. The vectors mean
nothing; the tests compare them with a manual computation on the same model. Inputs are at most
``MAX_LENGTH`` tokens.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

MAX_LENGTH = 32
HIDDEN = 8
_WORDS = (
    *("passage", "query", "alpha", "beta", "gamma", "delta", "epsilon", "the", "invoice"),
    *("was", "paid", "on", "monday", "who", "signed", "contract", "a", "short", "of"),
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
    vocab = {token: index for index, token in enumerate([*_SPECIAL, *_WORDS, ":", ".", "?"])}
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


def _model(vocab: dict[str, int]) -> BertModel:
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=2 * HIDDEN,
        max_position_embeddings=MAX_LENGTH,
    )
    torch.manual_seed(0)
    return BertModel(config)  # type: ignore[no-untyped-call]
