"""Settings read from the environment (spec: service-configuration)."""

import dataclasses
from pathlib import Path

import pytest

from app.adapters.embeddings.config import E5EmbedderConfig
from app.adapters.llm.config import LLMClientConfig
from app.adapters.ner.config import TransformersNerConfig
from app.adapters.ocr.engine import TesseractOcrConfig
from app.adapters.vector_store.config import QdrantStoreConfig
from app.application import ask_question
from app.domain.chunking import ChunkingConfig
from app.domain.errors import ConfigurationError
from app.domain.matching import DEFAULT_THRESHOLD
from app.settings import (
    MAX_BUSY_WAIT_S,
    MAX_READINESS_TIMEOUT_S,
    RETIRED_VARIABLES,
    Settings,
    load_settings,
)

API_KEY = "fake-key-0123456789abcdefghij"


def _default(config: type, field: str) -> object:
    """The default of a config dataclass field, without building (and validating) one."""
    for item in dataclasses.fields(config):
        if item.name == field:
            assert item.default is not dataclasses.MISSING, f"{config.__name__}.{field}"
            return item.default
    raise AssertionError(f"{config.__name__} has no field {field}")


# TORCH_THREADS feeds both torch adapters, so both their defaults must equal it.
CODE_DEFAULTS: list[tuple[str, object]] = [
    ("torch_threads", _default(TransformersNerConfig, "threads")),
    ("torch_threads", _default(E5EmbedderConfig, "threads")),
    ("ner_pool_size", _default(TransformersNerConfig, "pool_size")),
    ("embedding_batch_size", _default(E5EmbedderConfig, "batch_size")),
    ("ocr_dpi", _default(TesseractOcrConfig, "dpi")),
    ("ocr_workers", _default(TesseractOcrConfig, "workers")),
    ("ocr_timeout_s", _default(TesseractOcrConfig, "timeout_s")),
    ("ocr_flatten", _default(TesseractOcrConfig, "flatten")),
    ("max_pages", _default(TesseractOcrConfig, "max_pages")),
    ("match_threshold", DEFAULT_THRESHOLD),
    ("chunk_size", _default(ChunkingConfig, "max_chars")),
    ("chunk_overlap", _default(ChunkingConfig, "overlap_ratio")),
    ("rag_threshold", ask_question.DEFAULT_THRESHOLD),
    ("rag_top_k", ask_question.DEFAULT_TOP_K),
    ("rag_max_chunks", ask_question.DEFAULT_MAX_CHUNKS),
    ("rag_max_context_chars", ask_question.DEFAULT_MAX_CONTEXT_CHARS),
    ("qdrant_collection_prefix", _default(QdrantStoreConfig, "collection_prefix")),
    ("qdrant_timeout_s", _default(QdrantStoreConfig, "timeout_s")),
    ("qdrant_attempts", _default(QdrantStoreConfig, "attempts")),
    ("qdrant_deadline_s", _default(QdrantStoreConfig, "deadline_s")),
    ("readiness_timeout_s", _default(QdrantStoreConfig, "ping_timeout_s")),
    ("llm_connect_timeout_s", _default(LLMClientConfig, "connect_timeout_s")),
    ("llm_read_timeout_s", _default(LLMClientConfig, "read_timeout_s")),
    ("llm_total_timeout_s", _default(LLMClientConfig, "total_timeout_s")),
    ("llm_attempts", _default(LLMClientConfig, "attempts")),
    ("llm_max_tokens", _default(LLMClientConfig, "max_tokens")),
]

AGREED_DEFAULTS: dict[str, object] = {
    "max_upload_mb": 20,
    "max_pages": 50,
    "max_names": 1000,
    "max_concurrent_jobs": 12,
    "max_concurrent_questions": 16,
    "busy_wait_s": 10.0,
    "log_exception_messages": False,
    "log_level": "INFO",
    "log_format": "json",
    "qdrant_location": None,
    "qdrant_api_key": None,
    "llm_base_url": None,
    "llm_model": None,
    "llm_api_key": None,
}


def _error(**overrides: object) -> str:
    with pytest.raises(ConfigurationError) as caught:
        load_settings(_env_file=None, **overrides)
    assert caught.value.code == "invalid_configuration"
    return str(caught.value)


@pytest.mark.usefixtures("required_env")
@pytest.mark.parametrize(("field", "expected"), CODE_DEFAULTS)
def test_defaults_equal_the_code_defaults(field: str, expected: object) -> None:
    value = getattr(load_settings(), field)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.usefixtures("required_env")
@pytest.mark.parametrize(("field", "expected"), sorted(AGREED_DEFAULTS.items()))
def test_defaults_equal_the_agreed_values(field: str, expected: object) -> None:
    assert getattr(load_settings(), field) == expected


def test_every_setting_has_a_default_or_is_required() -> None:
    covered = {field for field, _ in CODE_DEFAULTS} | set(AGREED_DEFAULTS)
    required = {"ner_model_path", "embedding_model_path", "embedding_model_revision", "qdrant_url"}
    assert set(Settings.model_fields) == covered | required | set(RETIRED_VARIABLES)


@pytest.mark.usefixtures("required_env")
@pytest.mark.parametrize("variable", ["NER_THREADS", "EMBEDDING_THREADS", "ner_threads"])
@pytest.mark.parametrize("value", ["1", "3", "not-a-number"])
def test_a_retired_thread_variable_fails_naming_it_and_torch_threads(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    message = _error()
    assert f"{variable.upper()}: no longer read; set TORCH_THREADS instead" in message


@pytest.mark.usefixtures("required_env")
def test_both_retired_thread_variables_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_THREADS", "1")
    monkeypatch.setenv("EMBEDDING_THREADS", "3")
    assert "NER_THREADS, EMBEDDING_THREADS: no longer read" in _error()


def test_a_retired_thread_variable_in_dotenv_or_an_override_fails(
    tmp_path: Path, required_env: dict[str, str]
) -> None:
    (tmp_path / ".env").write_text("EMBEDDING_THREADS=3\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="EMBEDDING_THREADS: no longer read"):
        load_settings()
    assert "NER_THREADS: no longer read" in _error(ner_threads=2)


@pytest.mark.usefixtures("required_env")
def test_an_empty_retired_variable_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_THREADS", "")
    assert load_settings().torch_threads == 2


@pytest.mark.usefixtures("required_env")
def test_torch_threads_is_read_and_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCH_THREADS", "3")
    assert load_settings().torch_threads == 3
    monkeypatch.setenv("TORCH_THREADS", "0")
    assert "TORCH_THREADS: " in _error()


@pytest.mark.usefixtures("required_env")
def test_retired_fields_are_not_rendered() -> None:
    settings = load_settings()
    assert "ner_threads" not in repr(settings)
    assert "embedding_threads" not in settings.model_dump()


def test_required_values_are_read(required_env: dict[str, str]) -> None:
    settings = load_settings()
    assert settings.ner_model_path == required_env["NER_MODEL_PATH"]
    assert settings.embedding_model_path == required_env["EMBEDDING_MODEL_PATH"]
    assert settings.embedding_model_revision == required_env["EMBEDDING_MODEL_REVISION"]
    assert settings.qdrant_url == required_env["QDRANT_URL"]


@pytest.mark.usefixtures("required_env")
def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_THRESHOLD", "0.75")
    monkeypatch.setenv("OCR_DPI", "300")
    monkeypatch.setenv("OCR_WORKERS", "3")
    monkeypatch.setenv("OCR_FLATTEN", "false")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    monkeypatch.setenv("BUSY_WAIT_S", "0.5")
    monkeypatch.setenv("MAX_CONCURRENT_QUESTIONS", "4")
    monkeypatch.setenv("READINESS_TIMEOUT_S", "0.5")
    monkeypatch.setenv("LOG_EXCEPTION_MESSAGES", "true")

    settings = load_settings()

    assert settings.rag_threshold == 0.75
    assert settings.ocr_dpi == 300
    assert settings.ocr_workers == 3
    assert settings.ocr_flatten is False
    assert settings.chunk_overlap == 0.0
    assert settings.busy_wait_s == 0.5
    assert settings.max_concurrent_questions == 4
    assert settings.readiness_timeout_s == 0.5
    assert settings.log_exception_messages is True


@pytest.mark.usefixtures("required_env")
def test_names_are_case_insensitive_and_unknown_variables_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("match_threshold", "0.8")
    monkeypatch.setenv("SOMETHING_UNRELATED", "x")
    assert load_settings().match_threshold == 0.8


@pytest.mark.usefixtures("required_env")
def test_empty_variable_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_WORKERS", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    settings = load_settings()
    assert settings.ocr_workers is None
    assert settings.llm_api_key is None


@pytest.mark.usefixtures("required_env")
def test_log_level_and_format_are_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_FORMAT", "TEXT")
    settings = load_settings()
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "text"


@pytest.mark.usefixtures("required_env")
def test_max_upload_bytes_is_mebibytes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_settings().max_upload_bytes == 20 * 1024 * 1024
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    assert load_settings().max_upload_bytes == 1024 * 1024


@pytest.mark.usefixtures("required_env")
def test_settings_are_frozen() -> None:
    settings = load_settings()
    with pytest.raises(ValueError, match="frozen"):
        settings.ocr_dpi = 1


def test_missing_required_variables_are_named() -> None:
    message = _error()
    for name in ("NER_MODEL_PATH", "EMBEDDING_MODEL_PATH", "EMBEDDING_MODEL_REVISION"):
        assert f"{name}: is required" in message


@pytest.mark.usefixtures("required_env")
def test_bad_threshold_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MATCH_THRESHOLD", "1.5")
    message = _error()
    assert "MATCH_THRESHOLD: " in message
    assert "less than or equal to 1" in message
    assert "1.5" not in message


@pytest.mark.usefixtures("required_env")
@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RAG_THRESHOLD", "-0.1"),
        ("RAG_THRESHOLD", "nan"),
        ("CHUNK_OVERLAP", "0.5"),
        ("CHUNK_SIZE", "0"),
        ("OCR_DPI", "abc"),
        ("OCR_DPI", "2.5"),
        ("OCR_WORKERS", "0"),
        ("OCR_TIMEOUT_S", "inf"),
        ("OCR_FLATTEN", "maybe"),
        ("QDRANT_TIMEOUT_S", "0"),
        ("LLM_ATTEMPTS", "-1"),
        ("MAX_UPLOAD_MB", "0"),
        ("MAX_CONCURRENT_JOBS", "0"),
        ("BUSY_WAIT_S", "0"),
        ("BUSY_WAIT_S", "300.5"),
        ("BUSY_WAIT_S", "inf"),
        ("MAX_CONCURRENT_QUESTIONS", "0"),
        ("MAX_CONCURRENT_QUESTIONS", "2.5"),
        ("READINESS_TIMEOUT_S", "0"),
        ("READINESS_TIMEOUT_S", "30.5"),
        ("READINESS_TIMEOUT_S", "nan"),
        ("LOG_EXCEPTION_MESSAGES", "maybe"),
        ("LOG_FORMAT", "xml"),
        ("LOG_LEVEL", "verbose"),
    ],
)
def test_out_of_range_values_name_the_variable(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    message = _error()
    assert f"{variable}: " in message
    assert ";" not in message  # only that variable is reported


@pytest.mark.usefixtures("required_env")
def test_busy_wait_has_an_upper_bound_named_without_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BUSY_WAIT_S", "1e12")
    message = _error()
    assert "BUSY_WAIT_S: " in message
    assert f"less than or equal to {MAX_BUSY_WAIT_S:g}" in message
    assert "1e12" not in message
    assert "1000000000000" not in message


@pytest.mark.usefixtures("required_env")
def test_busy_wait_accepts_its_upper_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUSY_WAIT_S", "300")
    assert load_settings(_env_file=None).busy_wait_s == MAX_BUSY_WAIT_S


@pytest.mark.usefixtures("required_env")
def test_readiness_timeout_has_an_upper_bound_named_without_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("READINESS_TIMEOUT_S", "86400")
    message = _error()
    assert "READINESS_TIMEOUT_S: " in message
    assert f"less than or equal to {MAX_READINESS_TIMEOUT_S:g}" in message
    assert "86400" not in message


@pytest.mark.usefixtures("required_env")
def test_readiness_timeout_accepts_its_upper_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("READINESS_TIMEOUT_S", "30")
    assert load_settings(_env_file=None).readiness_timeout_s == MAX_READINESS_TIMEOUT_S


@pytest.mark.usefixtures("required_env")
def test_bad_log_format_lists_the_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "xml")
    message = _error()
    assert "LOG_FORMAT: " in message
    assert "'json' or 'text'" in message


@pytest.mark.usefixtures("required_env")
def test_several_problems_are_all_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MATCH_THRESHOLD", "2")
    monkeypatch.setenv("OCR_DPI", "0")
    message = _error()
    assert "MATCH_THRESHOLD: " in message
    assert "OCR_DPI: " in message


@pytest.mark.usefixtures("required_env")
def test_both_vector_store_targets_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    assert "QDRANT_URL, QDRANT_LOCATION: set exactly one of them" in _error()


@pytest.mark.usefixtures("required_env")
def test_no_vector_store_target_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL")
    assert "QDRANT_URL, QDRANT_LOCATION: set exactly one of them" in _error()


@pytest.mark.usefixtures("required_env")
def test_in_memory_location_is_accepted_by_explicit_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDRANT_URL")
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    settings = load_settings()
    assert settings.qdrant_location == ":memory:"
    assert settings.qdrant_url is None


@pytest.mark.usefixtures("required_env")
def test_llm_base_url_requires_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")
    assert "LLM_MODEL: required when LLM_BASE_URL is set" in _error()

    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")
    settings = load_settings()
    assert settings.llm_base_url == "http://ollama:11434/v1"
    assert settings.llm_model == "qwen2.5:3b"


@pytest.mark.usefixtures("required_env")
def test_no_llm_is_valid() -> None:
    settings = load_settings()
    assert settings.llm_base_url is None


def test_overrides_are_validated_like_the_environment(required_env: dict[str, str]) -> None:
    assert load_settings(ocr_dpi=150).ocr_dpi == 150
    assert "OCR_DPI: " in _error(ocr_dpi=-1)


def test_dotenv_file_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "NER_MODEL_PATH=/m/ner\n"
        "EMBEDDING_MODEL_PATH=/m/e5\n"
        "EMBEDDING_MODEL_REVISION=abc\n"
        "QDRANT_LOCATION=:memory:\n"
        "OCR_DPI=300\n"
        "LLM_API_KEY=fake-key-from-dotenv\n",
        encoding="utf-8",
    )

    settings = load_settings()

    assert settings.ner_model_path == "/m/ner"
    assert settings.ocr_dpi == 300
    assert settings.qdrant_location == ":memory:"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "fake-key-from-dotenv"


def test_environment_wins_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("OCR_DPI=300\n", encoding="utf-8")
    monkeypatch.setenv("OCR_DPI", "250")
    settings = load_settings(
        ner_model_path="a", embedding_model_path="b", embedding_model_revision="c", qdrant_url="u"
    )
    assert settings.ocr_dpi == 250


def test_dotenv_can_be_skipped(tmp_path: Path, required_env: dict[str, str]) -> None:
    (tmp_path / ".env").write_text("OCR_DPI=300\n", encoding="utf-8")
    assert load_settings(_env_file=None).ocr_dpi == 200


def test_missing_dotenv_file_is_fine(required_env: dict[str, str]) -> None:
    assert load_settings(_env_file="does-not-exist.env").ocr_dpi == 200


@pytest.mark.usefixtures("required_env")
def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", API_KEY)
    monkeypatch.setenv("QDRANT_API_KEY", API_KEY + "-qdrant")

    settings = load_settings()

    for rendering in (
        repr(settings),
        str(settings),
        settings.model_dump_json(),
        str(settings.model_dump(mode="json")),
        str(settings.model_dump()),
    ):
        assert API_KEY not in rendering
    assert "**********" in repr(settings)
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == API_KEY
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == API_KEY + "-qdrant"


@pytest.mark.usefixtures("required_env")
def test_error_message_never_contains_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", API_KEY)
    monkeypatch.setenv("OCR_DPI", API_KEY)
    monkeypatch.setenv("LOG_FORMAT", API_KEY)
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")

    with pytest.raises(ConfigurationError) as caught:
        load_settings(_env_file=None)

    error = caught.value
    for text in (str(error), repr(error), error.message):
        assert API_KEY not in text
        assert "OCR_DPI: " in text
        assert "LOG_FORMAT: " in text
    # The pydantic ValidationError shows input values, so it is not chained.
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
