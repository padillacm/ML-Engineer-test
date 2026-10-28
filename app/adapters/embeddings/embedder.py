"""``Embedder`` backed by a local ``intfloat/multilingual-e5-base`` model on CPU.

The recipe is the model card's: the ``passage: `` / ``query: `` prefix, ``AutoModel``, mean
pooling of the last hidden state over the attention mask and L2 normalisation, implemented here
with ``transformers`` alone (no ``sentence-transformers``).

The model is loaded once, when the embedder is created, from ``E5EmbedderConfig.path`` with
``local_files_only=True``: nothing is fetched from the network, and a missing or unusable model
fails creation with ``ConfigurationError`` instead of failing the first request.

Thread safety: the methods may be called from several threads; inference is serialised by one
lock per embedder because the tokenizer and model are not documented as thread-safe.

Threads: a successful creation calls ``torch.set_num_threads(config.threads)``, which is
process-wide: every torch computation in the process (the recogniser's included) then uses that
many intra-op threads, and the last torch adapter created wins, so the service gives both the
same value (``TORCH_THREADS``). A failed creation leaves it unchanged.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from app.adapters.embeddings.config import E5EmbedderConfig
from app.domain.errors import ConfigurationError, ProcessingFailed, ServiceError
from app.domain.types import EmbeddingIdentity

MODEL_NAME = "intfloat/multilingual-e5-base"
PREFIX_SCHEME = "e5-query-passage"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass(frozen=True, slots=True)
class LoadedEncoder:
    """A loaded model: ``encode`` turns one batch of (already prefixed) texts into vectors.

    ``encode`` returns one L2-normalised vector of ``dimension`` floats per text, in order.
    """

    dimension: int
    encode: Callable[[Sequence[str]], list[list[float]]]


type EncoderFactory = Callable[[E5EmbedderConfig], LoadedEncoder]


class E5Embedder:
    """Embeds passages and queries with the e5 prefixes; vectors are unit length (cosine).

    ``model_factory`` builds the encoder from the configuration; the default loads the local
    model with ``transformers``. Tests pass a stub.
    """

    def __init__(
        self,
        config: E5EmbedderConfig,
        model_factory: EncoderFactory | None = None,
    ) -> None:
        factory = load_encoder if model_factory is None else model_factory
        try:
            encoder = factory(config)
        except ServiceError:
            raise
        except Exception as error:  # every other loading failure is a configuration error
            raise ConfigurationError(
                f"embedding model could not be loaded from {config.path}"
            ) from error
        if encoder.dimension <= 0:
            raise ConfigurationError(f"embedding model reports no dimension: {config.path}")
        torch.set_num_threads(config.threads)
        self._encode = encoder.encode
        self._batch_size = config.batch_size
        self._max_chars = config.max_chars
        self._identity = EmbeddingIdentity(
            model_id=f"{MODEL_NAME}@{config.revision}",
            dimension=encoder.dimension,
            prefix_scheme=PREFIX_SCHEME,
            normalized=True,
            metric="cosine",
        )
        self._lock = threading.Lock()

    @property
    def identity(self) -> EmbeddingIdentity:
        """Model and revision, dimension, e5 prefixes, unit vectors compared by cosine."""
        return self._identity

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text (``passage: `` prefix), in the same order.

        Texts are run in batches of ``batch_size`` and truncated at ``max_length`` tokens. No
        texts give ``[]``. A text longer than ``max_chars`` raises ``ProcessingFailed`` (code
        ``embedding_failed``) before the model runs, and so does any failure of the model, with
        a fixed message and the original exception kept as ``__cause__``.
        """
        self._check_sizes(texts)
        prefixed = [PASSAGE_PREFIX + text for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), self._batch_size):
            vectors.extend(self._run(prefixed[start : start + self._batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of ``text`` with the ``query: `` prefix (errors as for passages)."""
        self._check_sizes([text])
        return self._run([QUERY_PREFIX + text])[0]

    def _check_sizes(self, texts: Sequence[str]) -> None:
        if any(len(text) > self._max_chars for text in texts):
            raise ProcessingFailed(
                f"text is longer than the embedding limit of {self._max_chars} characters",
                "embedding_failed",
            )

    def _run(self, batch: Sequence[str]) -> list[list[float]]:
        with self._lock:
            try:
                vectors = self._encode(batch)
            except ServiceError:
                raise
            except Exception as error:  # every vendor failure becomes one fixed error
                raise ProcessingFailed(
                    "embedding model could not process the text", "embedding_failed"
                ) from error
        if len(vectors) != len(batch) or any(
            len(vector) != self._identity.dimension for vector in vectors
        ):
            raise ProcessingFailed(
                "embedding model returned vectors of the wrong shape", "embedding_failed"
            )
        return vectors


def load_encoder(config: E5EmbedderConfig) -> LoadedEncoder:
    """Load tokenizer and model from ``config.path`` (local files only) into an encoder.

    Raises ``ConfigurationError`` when the tokenizer is not a fast one, when ``max_length``
    exceeds the tokenizer's input length, or when the weights do not fit the declared
    architecture (missing, unexpected or mismatched tensors: ``transformers`` would otherwise
    initialise the missing ones at random and only log a warning). Other loading errors are left
    to the caller.
    """
    from transformers import (  # noqa: PLC0415 - heavy import, needed only to load a model
        AutoModel,
        AutoTokenizer,
    )

    path = str(config.path)
    # fix_mistral_regex=False: as for the NER model (an XLM-R tokenizer too), transformers may
    # flag the tokenizer as a Mistral one with a broken regex and warn; the fix would change the
    # token ids, so the tokenizer is kept as it is.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
    if not tokenizer.is_fast:
        raise ConfigurationError(f"embedding model needs a fast tokenizer: {path}")
    if config.max_length > tokenizer.model_max_length:
        raise ConfigurationError(
            f"max_length must not exceed the model input length "
            f"({tokenizer.model_max_length} tokens)"
        )
    model, loading_info = AutoModel.from_pretrained(
        path, local_files_only=True, output_loading_info=True
    )
    problems = sorted(name for name, found in loading_info.items() if found)
    if problems:
        raise ConfigurationError(
            f"embedding model weights do not match its configuration ({problems}): {path}"
        )
    model.eval()
    dimension = int(model.config.hidden_size)
    max_length = config.max_length

    def encode(texts: Sequence[str]) -> list[list[float]]:
        inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            vectors = normalize(mean_pool(hidden, inputs["attention_mask"]))
        result: list[list[float]] = vectors.tolist()
        return result

    return LoadedEncoder(dimension=dimension, encode=encode)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average of each sequence's token states over its attention mask (padding excluded).

    ``hidden`` is ``(batch, tokens, dimension)``, ``attention_mask`` ``(batch, tokens)``.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    total = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return total / count


def normalize(vectors: torch.Tensor) -> torch.Tensor:
    """Each row scaled to unit L2 length (a zero row stays zero)."""
    return torch.nn.functional.normalize(vectors, p=2.0, dim=-1)
