"""Service settings: every tunable read once from the environment (and ``.env`` when present).

Variable names are the field names in upper case, without prefix, matched case-insensitively.
Unknown variables are ignored and a variable set to the empty string counts as unset, except
the retired ``RETIRED_VARIABLES`` (``NER_THREADS``, ``EMBEDDING_THREADS``): setting one fails
with a ``ConfigurationError`` naming it, so a deployment that still tunes it is not silently
ignored. The
defaults are the benchmarked configuration and mirror the defaults of the adapter configs and
use cases (a test keeps them equal); only the values that identify a deployment (model paths
and revision, vector store target) are required. ``MAX_CONCURRENT_JOBS``, ``NER_POOL_SIZE``,
``EMBEDDING_POOL_SIZE`` and ``OCR_WORKERS`` are ``None`` when unset: the composition root
derives them from the CPUs available (``app.bootstrap.resolve_concurrency``); a set value wins.

Settings only parse types and check ranges. The detailed rules (model directory contents, URL
shapes, cross-field limits) stay in the adapter config dataclasses, which the composition root
builds from these values. ``load_settings`` turns a validation failure into a
``ConfigurationError`` naming each offending variable and the rule it breaks, never the value.
``LLM_API_KEY`` and ``QDRANT_API_KEY`` are ``SecretStr``: masked in ``repr``, ``str`` and
``model_dump_json``; ``get_secret_value()`` reveals them to the composition root only.

Either key can instead come from a file, ``LLM_API_KEY_FILE`` or ``QDRANT_API_KEY_FILE`` (compose
mounts them as secrets under ``/run/secrets``, so the keys are not in the container's
environment): the file is read once while the settings load, a UTF-8 byte order mark and
surrounding whitespace are removed, and an empty file means no key. Setting both a key and its
file is an error; a file that cannot be read, is over ``MAX_KEY_FILE_BYTES`` or is not UTF-8 fails
naming the variable and the rule, never the path or the file's contents. ``key_variable`` names
the variable a key came from, so a later check of the key's characters names the right one.

``TORCH_THREADS`` is ``None`` when unset: the composition root uses ``DEFAULT_TORCH_THREADS``,
lowered to the CPUs available, so a one-CPU container does not oversubscribe its CPU.
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
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
MaxTokensField = Literal["max_tokens", "max_completion_tokens"]

MAX_BUSY_WAIT_S = 300.0
"""Upper bound of ``BUSY_WAIT_S``, in seconds."""

DEFAULT_TORCH_THREADS = 2
"""Torch intra-op threads when ``TORCH_THREADS`` is unset (at most the CPUs available): the
measured best for the recogniser and the embedder (BENCHMARKS 15)."""

KEY_FILES: dict[str, str] = {
    "llm_api_key": "llm_api_key_file",
    "qdrant_api_key": "qdrant_api_key_file",
}
"""Each key field -> the field of the file it can be read from instead (``*_FILE`` variables)."""

MAX_KEY_FILE_BYTES = 4096
"""Largest key file read (``*_API_KEY_FILE``), in bytes: far above any API key, and it keeps a
wrong path (a log file, a device) from being read whole."""

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
    # Torch intra-op threads, process-wide: shared by the recogniser and the embedder; ``None``
    # means ``DEFAULT_TORCH_THREADS``, at most the CPUs available (the composition root decides).
    torch_threads: PositiveInt | None = None
    # Concurrent recognitions and embedding batches; ``None`` derives them from the CPUs.
    ner_pool_size: PositiveInt | None = None
    embedding_pool_size: PositiveInt | None = None
    embedding_batch_size: PositiveInt = 16
    # Retired (``RETIRED_VARIABLES``): read only to refuse them; never used.
    ner_threads: Annotated[Any, Field(exclude=True, repr=False)] = None
    embedding_threads: Annotated[Any, Field(exclude=True, repr=False)] = None

    # OCR; ``ocr_workers=None`` derives the Tesseract processes per document from the CPUs.
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
    # A file holding the key instead (compose: the mounted secret); read once, while loading.
    qdrant_api_key_file: NonBlank | None = None
    qdrant_collection_prefix: NonBlank = "chunks"
    qdrant_timeout_s: PositiveSeconds = 10.0
    qdrant_attempts: PositiveInt = 3
    qdrant_deadline_s: PositiveSeconds = 20.0

    # Language model: unset ``llm_base_url`` means no model; when set, ``llm_model`` is required.
    llm_base_url: NonBlank | None = None
    llm_model: NonBlank | None = None
    llm_api_key: SecretStr | None = None
    # A file holding the key instead (compose: the mounted secret); read once, while loading.
    llm_api_key_file: NonBlank | None = None
    llm_connect_timeout_s: PositiveSeconds = 5.0
    llm_read_timeout_s: PositiveSeconds = 60.0
    llm_total_timeout_s: PositiveSeconds = 90.0
    llm_attempts: PositiveInt = 3
    llm_max_tokens: PositiveInt = 512
    # Request options for the model's parameters: ``reasoning_effort`` only when set (``none``
    # turns thinking off on Ollama), the limit's name, and whether ``temperature`` is sent
    # (hosted reasoning models reject ``max_tokens`` and any temperature but their default).
    llm_reasoning_effort: ReasoningEffort | None = None
    llm_max_tokens_field: MaxTokensField = "max_tokens"
    llm_send_temperature: bool = True

    # Limits
    max_upload_mb: PositiveInt = 20
    max_names: PositiveInt = 1000
    # ``None`` derives the job limit from the CPUs.
    max_concurrent_jobs: PositiveInt | None = None
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

    @model_validator(mode="before")
    @classmethod
    def _read_key_files(cls, data: Any) -> Any:
        """Put the contents of each ``*_API_KEY_FILE`` (``None`` when empty) in its key field."""
        if not isinstance(data, dict):
            return data
        for key_field, file_field in KEY_FILES.items():
            path = data.get(file_field)
            if not isinstance(path, str) or not path.strip():
                continue
            if data.get(key_field) not in (None, ""):
                raise _rule(f"{key_field.upper()}, {file_field.upper()}", "set at most one of them")
            data = {**data, key_field: _key_from_file(path, file_field.upper())}
        return data

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("log_format", "llm_reasoning_effort", "llm_max_tokens_field", mode="before")
    @classmethod
    def _lower_choice(cls, value: object) -> object:
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

    def key_variable(self, key_field: str) -> str:
        """The variable ``key_field``'s key came from: its ``*_FILE`` variable when that is set."""
        file_field = KEY_FILES[key_field]
        return (file_field if getattr(self, file_field) is not None else key_field).upper()

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


def _key_from_file(path: str, variable: str) -> str | None:
    """The stripped text of the key file (``variable`` names it in errors), ``None`` when it is
    empty or only whitespace. A UTF-8 byte order mark (Windows editors write one) is removed."""
    try:
        with open(path, "rb") as file:
            raw = file.read(MAX_KEY_FILE_BYTES + 1)
    except OSError as error:
        raise _rule(variable, f"cannot be read ({type(error).__name__})") from None
    if len(raw) > MAX_KEY_FILE_BYTES:
        raise _rule(variable, f"the file is over {MAX_KEY_FILE_BYTES} bytes")
    try:
        text = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        raise _rule(variable, "the file is not UTF-8 text") from None
    return text or None


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
