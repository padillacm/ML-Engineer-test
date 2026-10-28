"""Service settings: every tunable read once from the environment (and ``.env`` when present).

Variable names are the field names in upper case, without prefix, matched case-insensitively.
Unknown variables are ignored and a variable set to the empty string counts as unset, except
the retired ``RETIRED_VARIABLES`` (``NER_THREADS``, ``EMBEDDING_THREADS``): setting one fails
with a ``ConfigurationError`` naming it, so a deployment that still tunes it is not silently
ignored. The
defaults are the benchmarked configuration and mirror the defaults of the adapter configs and
use cases (a test keeps them equal); only the values that identify a deployment (model paths
and revision, vector store target) are required.

Settings only parse types and check ranges. The detailed rules (model directory contents, URL
shapes, cross-field limits) stay in the adapter config dataclasses, which the composition root
builds from these values. ``load_settings`` turns a validation failure into a
``ConfigurationError`` naming each offending variable and the rule it breaks, never the value.
``LLM_API_KEY`` and ``QDRANT_API_KEY`` are ``SecretStr``: masked in ``repr``, ``str`` and
``model_dump_json``; ``get_secret_value()`` reveals them to the composition root only.
"""

from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_core import ErrorDetails, PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.domain.errors import ConfigurationError

PositiveInt = Annotated[int, Field(gt=0)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Threshold = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonBlank = Annotated[str, Field(min_length=1)]

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
LogFormat = Literal["json", "text"]

MAX_BUSY_WAIT_S = 300.0
"""Upper bound of ``BUSY_WAIT_S``, in seconds."""

MAX_READINESS_TIMEOUT_S = 30.0
"""Upper bound of ``READINESS_TIMEOUT_S``, in seconds: orchestrators give up on a probe after a
few seconds (Kubernetes' default probe timeout is 1 s), so a longer check only hides a hung
store."""

RETIRED_VARIABLES: dict[str, str] = {
    "ner_threads": "NER_THREADS",
    "embedding_threads": "EMBEDDING_THREADS",
}
"""Fields of variables that are no longer read -> the variable. Torch's intra-op thread count is
process-wide, so the recogniser and the embedder cannot use different values: ``TORCH_THREADS``
replaced both (the second one set used to win silently)."""

_BYTES_PER_MB = 1024 * 1024
_SETTINGS_RULE = "settings_rule"


class Settings(BaseSettings):
    """All tunables of the service, grouped as in the design (D1)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    # Models
    ner_model_path: NonBlank
    embedding_model_path: NonBlank
    embedding_model_revision: NonBlank
    # Torch intra-op threads, process-wide: shared by the recogniser and the embedder.
    torch_threads: PositiveInt = 2
    ner_pool_size: PositiveInt = 8
    embedding_batch_size: PositiveInt = 16
    # Retired (``RETIRED_VARIABLES``): read only to refuse them; never used.
    ner_threads: Annotated[Any, Field(exclude=True, repr=False)] = None
    embedding_threads: Annotated[Any, Field(exclude=True, repr=False)] = None

    # OCR; ``ocr_workers=None`` lets the adapter use min(4, cpu_count).
    ocr_dpi: PositiveInt = 200
    ocr_workers: PositiveInt | None = None
    ocr_timeout_s: PositiveSeconds = 60.0
    ocr_flatten: bool = True
    max_pages: PositiveInt = 50

    # Matching
    match_threshold: Threshold = 0.90

    # Chunking; ``chunk_overlap`` is a ratio of ``chunk_size``, not a character count.
    chunk_size: PositiveInt = 800
    chunk_overlap: Annotated[float, Field(ge=0, lt=0.5, allow_inf_nan=False)] = 0.12

    # Retrieval
    rag_threshold: Threshold = 0.808
    rag_top_k: PositiveInt = 5
    rag_max_chunks: PositiveInt = 3
    rag_max_context_chars: PositiveInt = 6000

    # Vector store: exactly one of ``qdrant_url`` and ``qdrant_location``.
    qdrant_url: NonBlank | None = None
    qdrant_location: NonBlank | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_prefix: NonBlank = "chunks"
    qdrant_timeout_s: PositiveSeconds = 10.0
    qdrant_attempts: PositiveInt = 3
    qdrant_deadline_s: PositiveSeconds = 20.0

    # Language model: unset ``llm_base_url`` means no model; when set, ``llm_model`` is required.
    llm_base_url: NonBlank | None = None
    llm_model: NonBlank | None = None
    llm_api_key: SecretStr | None = None
    llm_connect_timeout_s: PositiveSeconds = 5.0
    llm_read_timeout_s: PositiveSeconds = 60.0
    llm_total_timeout_s: PositiveSeconds = 90.0
    llm_attempts: PositiveInt = 3
    llm_max_tokens: PositiveInt = 512

    # Limits
    max_upload_mb: PositiveInt = 20
    max_names: PositiveInt = 1000
    max_concurrent_jobs: PositiveInt = 12
    # Questions wait mostly on the language model (I/O); they have their own slots and threads.
    max_concurrent_questions: PositiveInt = 16
    # At most 5 minutes: a client is better told to retry (Retry-After) than held longer, and
    # proxies and clients usually give up on a silent request well before that.
    busy_wait_s: Annotated[float, Field(gt=0, le=MAX_BUSY_WAIT_S, allow_inf_nan=False)] = 10.0
    # How long one readiness check may take; also the vector store's ping timeout.
    readiness_timeout_s: Annotated[
        float, Field(gt=0, le=MAX_READINESS_TIMEOUT_S, allow_inf_nan=False)
    ] = 2.0

    # Logging
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"
    # Local debugging only: logged tracebacks then include exception messages (personal data).
    log_exception_messages: bool = False

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("log_format", mode="before")
    @classmethod
    def _lower_format(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_combinations(self) -> "Settings":
        retired = [
            variable
            for field, variable in RETIRED_VARIABLES.items()
            if getattr(self, field) is not None
        ]
        if retired:
            raise _rule(
                ", ".join(retired),
                "no longer read; set TORCH_THREADS instead (torch threads are process-wide, "
                "shared by the recogniser and the embedder)",
            )
        if (self.qdrant_url is None) == (self.qdrant_location is None):
            raise _rule("QDRANT_URL, QDRANT_LOCATION", "set exactly one of them")
        if self.llm_base_url is not None and self.llm_model is None:
            raise _rule("LLM_MODEL", "required when LLM_BASE_URL is set")
        return self

    @property
    def max_upload_bytes(self) -> int:
        """``max_upload_mb`` in bytes (MiB, 1024 * 1024)."""
        return self.max_upload_mb * _BYTES_PER_MB


def load_settings(**overrides: Any) -> Settings:
    """Build ``Settings`` from the environment, ``.env`` and ``overrides`` (field names).

    ``_env_file`` (for example ``None`` to skip ``.env``) is passed through to pydantic-settings.
    Raises ``ConfigurationError`` naming every offending variable and its rule; input values
    are never included and the ``ValidationError`` (which shows them) is not chained.
    """
    try:
        return Settings(**overrides)
    except ValidationError as error:
        problems = "; ".join(_describe(detail) for detail in error.errors(include_url=False))
        raise ConfigurationError(f"invalid configuration: {problems}") from None


def _rule(variables: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(_SETTINGS_RULE, message, {"variables": variables})


def _describe(detail: ErrorDetails) -> str:
    if detail["type"] == _SETTINGS_RULE:
        context = detail.get("ctx") or {}
        variables = str(context.get("variables", "settings"))
    else:
        # The first location part is the field, whose upper-case name is the variable.
        variables = str(detail["loc"][0]).upper() if detail["loc"] else "settings"
    rule = "is required" if detail["type"] == "missing" else detail["msg"]
    return f"{variables}: {_without_input(rule, detail.get('input'))}"


def _without_input(message: str, value: object) -> str:
    """Remove the input from a message, in case a validator ever echoes it."""
    if isinstance(value, str) and value.strip():
        return message.replace(value, "<redacted>")
    return message
