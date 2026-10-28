"""A tiny, hand-wired token-classification model saved as a local model directory.

It lets the host run the real loading and inference path of the adapter (``AutoTokenizer``,
``AutoModelForTokenClassification``, the pipeline with ``stride``) without the 1 GB model. The
model has no transformer layers: each token's label comes from its embedding alone, so
"maria"/"peter" are ``B-PER``, "gonzalez"/"smith" are ``I-PER``, "acme"/"corporation" are an
organisation and every other token is ``O``. Inputs are at most ``MAX_LENGTH`` tokens, so a few
sentences already need several strided windows.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

MAX_LENGTH = 32
LABELS = ("O", "B-PER", "I-PER", "B-ORG", "I-ORG")
_TAGS = {"maria": 1, "peter": 1, "gonzalez": 2, "smith": 2, "acme": 3, "corporation": 4}
_WORDS = (
    *("maria", "gonzalez", "peter", "smith", "of", "acme", "corporation", "met", "in"),
    *("madrid", "on", "3", "march", "the", "agreement", "was", "signed", "by"),
)
_SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
_SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
}
_HIDDEN = 2 * len(LABELS)


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


def _model(vocab: dict[str, int]) -> BertForTokenClassification:
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=_HIDDEN,
        num_hidden_layers=0,
        num_attention_heads=1,
        intermediate_size=_HIDDEN,
        max_position_embeddings=2 * MAX_LENGTH,
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    model = BertForTokenClassification(config)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.bert.embeddings.LayerNorm.weight.fill_(1.0)
        for word, label in _TAGS.items():
            # A +1/-1 pair survives LayerNorm and points at one classifier row.
            direction = torch.zeros(_HIDDEN)
            direction[2 * label] = 1.0
            direction[2 * label + 1] = -1.0
            model.bert.embeddings.word_embeddings.weight[vocab[word]] = direction
            model.classifier.weight[label] = 10.0 * direction
        model.classifier.bias[0] = 5.0
    return model
