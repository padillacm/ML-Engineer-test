"""``Embedder`` backed by a local ``intfloat/multilingual-e5-base`` model on CPU.

The recipe is the model card's: the ``passage: `` / ``query: `` prefix, ``AutoModel``, mean
pooling of the last hidden state over the attention mask and L2 normalisation, implemented here
with ``transformers`` alone (no ``sentence-transformers``).

The model is loaded once, when the embedder is created, from ``E5EmbedderConfig.path`` with
``local_files_only=True``: nothing is fetched from the network, and a missing or unusable model
fails creation with ``ConfigurationError`` instead of failing the first request.

Thread safety: the methods may be called from several threads. Up to ``config.pool_size``
passage batches are embedded at once, all with the one loaded model and tokenizer, and a call
waits while every slot is busy. Queries (``embed_query``, one short text behind a question) do
not wait behind passages: one more slot is reserved for them, so up to ``pool_size`` + 1 calls
run at once, and a waiting query takes the next free slot before any waiting passage batch
(BENCHMARKS 18.3: without this a question waited for a passage batch to finish, 1-2 s, or was
starved). Waiting queries, and waiting passage batches, get their slots in arrival order
(``FifoSlots``), so a call with many batches does not overtake a call that waits. The model only
runs forward in ``eval`` mode under ``torch.inference_mode``, which torch allows from several
threads at once.
The fast tokenizer is not thread-safe: a call sets its truncation, padding and special-token
settings in place when they differ from the requested ones (a concurrent use then fails), so
tokenising runs under one lock, and one warm-up call at load settles those settings, which are
the same on every call afterwards; it is a small part of a call. A tokenizer per slot would
avoid the lock but costs about 300 MB each for this model's 250,002-token vocabulary (the NER
model's, measured in BENCHMARKS 14.3). ``close`` releases the model.

Threads: a successful creation calls ``torch.set_num_threads(config.threads)``, which is
process-wide: every torch computation in the process (the recogniser's included) then uses that
many intra-op threads, and the last torch adapter created wins, so the service gives both the
same value (``TORCH_THREADS``). A failed creation leaves it unchanged. ``pool_size`` concurrent
batches use up to ``pool_size`` x ``threads`` CPUs, and a query in the reserved slot ``threads``
more for about 50 ms.
"""

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.slots import FifoSlots
from app.domain.errors import ConfigurationError, ProcessingFailed, ServiceError
from app.domain.types import EmbeddingIdentity

MODEL_NAME = "intfloat/multilingual-e5-base"
PREFIX_SCHEME = "e5-query-passage"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
QUERY_LANE = 0
"""Slot lane of queries: served first, and may use the slot reserved for them."""
PASSAGE_LANE = 1
"""Slot lane of passage batches: at most ``pool_size`` at once."""


type Encode = Callable[[Sequence[str]], list[list[float]]]


@dataclass(frozen=True, slots=True)
class LoadedEncoder:
    """A loaded model: ``encode`` turns one batch of (already prefixed) texts into vectors.

    ``encode`` returns one L2-normalised vector of ``dimension`` floats per text, in order, and
    may be called from several threads at once.
    """

    dimension: int
    encode: Encode


type EncoderFactory = Callable[[E5EmbedderConfig], LoadedEncoder]


class E5Embedder:
    """Embeds passages and queries with the e5 prefixes; vectors are unit length (cosine).

    ``model_factory`` builds the encoder from the configuration; the default loads the local
    model with ``transformers``. Tests pass a stub. ``close`` releases the model.
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
        self._encode: Encode | None = encoder.encode
        # Lanes in priority order: queries may use every slot, passages all but the reserved one.
        self._slots = FifoSlots(config.pool_size + 1, (config.pool_size + 1, config.pool_size))
        self._batch_size = config.batch_size
        self._max_chars = config.max_chars
        self._identity = EmbeddingIdentity(
            model_id=f"{MODEL_NAME}@{config.revision}",
            dimension=encoder.dimension,
            prefix_scheme=PREFIX_SCHEME,
            normalized=True,
            metric="cosine",
        )

    @property
    def identity(self) -> EmbeddingIdentity:
        """Model and revision, dimension, e5 prefixes, unit vectors compared by cosine."""
        return self._identity

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per text (``passage: `` prefix), in the same order.

        Texts are run in batches of ``batch_size`` and truncated at ``max_length`` tokens. No
        texts give ``[]``. Each batch takes a slot of the pool for its run, behind the queries and
        the passage batches that were waiting before it. A text longer than
        ``max_chars`` raises ``ProcessingFailed`` (code ``embedding_failed``) before the model
        runs, and so does any failure of the model, with a fixed message and the original
        exception kept as ``__cause__``. A call on a closed embedder, or still waiting for a
        slot when it is closed, raises ``ProcessingFailed`` (code ``embedding_failed``) too.
        """
        self._check_sizes(texts)
        prefixed = [PASSAGE_PREFIX + text for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), self._batch_size):
            vectors.extend(self._run(prefixed[start : start + self._batch_size], query=False))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Return the vector of ``text`` with the ``query: `` prefix (errors as for passages).

        A query never waits for a passage slot while the slot reserved for queries is free; a
        query waiting for a slot gets the next free one before any waiting passage batch, and
        waiting queries are served in arrival order.
        """
        self._check_sizes([text])
        return self._run([QUERY_PREFIX + text], query=True)[0]

    def close(self) -> None:
        """Release the model (idempotent); calls already running finish normally."""
        self._encode = None
        self._slots.close()

    def _check_sizes(self, texts: Sequence[str]) -> None:
        if any(len(text) > self._max_chars for text in texts):
            raise ProcessingFailed(
                f"text is longer than the embedding limit of {self._max_chars} characters",
                "embedding_failed",
            )

    def _run(self, batch: Sequence[str], *, query: bool) -> list[list[float]]:
        lane = QUERY_LANE if query else PASSAGE_LANE
        if not self._slots.acquire(lane):
            raise ProcessingFailed("embedding model is closed", "embedding_failed")
        try:
            encode = self._encode
            if encode is None:  # closed after the slot was granted
                raise ProcessingFailed("embedding model is closed", "embedding_failed")
            vectors = encode(batch)
        except ServiceError:
            raise
        except Exception as error:  # every vendor failure becomes one fixed error
            raise ProcessingFailed(
                "embedding model could not process the text", "embedding_failed"
            ) from error
        finally:
            self._slots.release(lane)
        if len(vectors) != len(batch) or any(
            len(vector) != self._identity.dimension for vector in vectors
        ):
            raise ProcessingFailed(
                "embedding model returned vectors of the wrong shape", "embedding_failed"
            )
        return vectors


def load_encoder(config: E5EmbedderConfig) -> LoadedEncoder:
    """Load tokenizer and model from ``config.path`` (local files only) into an encoder.

    The encoder may be called from several threads: tokenising runs under one lock and the
    forward pass outside it (see the module docstring); one warm-up call settles the
    tokenizer's settings before any concurrent use. Raises ``ConfigurationError`` when the
    tokenizer is not a fast one, when ``max_length`` exceeds the tokenizer's input length, or
    when the weights do not fit the declared architecture (missing, unexpected or mismatched
    tensors: ``transformers`` would otherwise initialise the missing ones at random and only log
    a warning). Other loading errors are left to the caller.
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
    tokenizer_lock = threading.Lock()

    def encode(texts: Sequence[str]) -> list[list[float]]:
        with tokenizer_lock:
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

    # Every call passes the same tokenizer settings, so after this first call the tokenizer's
    # truncation, padding and special-token settings are only read, never changed again.
    encode([PASSAGE_PREFIX + _WARM_UP_TEXT])
    return LoadedEncoder(dimension=dimension, encode=encode)


_WARM_UP_TEXT = "The invoice was paid on time."


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
