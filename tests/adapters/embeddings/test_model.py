"""The embedder with the real pinned e5 model (``embedding_model``: runs in the container)."""

import logging
import math
import os
import socket
import time
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import torch

from app.adapters.embeddings import E5Embedder, E5EmbedderConfig
from app.application.ports import Embedder
from tests.application.contracts import EmbedderContract

pytestmark = pytest.mark.embedding_model

# (query, related passage, unrelated passage) per language.
TRIPLES = {
    "en": (
        "When must the invoice be paid?",
        "The invoice is payable within thirty days of delivery.",
        "The museum opens its new sculpture garden next spring.",
    ),
    "es": (
        "¿Cuándo hay que pagar la factura?",
        "La factura debe abonarse en un plazo de treinta días desde la entrega.",
        "El museo inaugura su nuevo jardín de esculturas la próxima primavera.",
    ),
    "pt": (
        "Quando a fatura deve ser paga?",
        "A fatura deve ser liquidada no prazo de trinta dias após a entrega.",
        "O museu inaugura o seu novo jardim de esculturas na próxima primavera.",
    ),
    "fr": (
        "Quand la facture doit-elle être payée ?",
        "La facture est payable dans les trente jours suivant la livraison.",
        "Le musée ouvre son nouveau jardin de sculptures au printemps prochain.",
    ),
    "de": (
        "Wann muss die Rechnung bezahlt werden?",
        "Die Rechnung ist innerhalb von dreißig Tagen nach Lieferung zu begleichen.",
        "Das Museum eröffnet im nächsten Frühjahr seinen neuen Skulpturengarten.",
    ),
}
PARAPHRASES = {
    "en": ("The contract ends on 31 December.", "The agreement expires at the end of December."),
    "es": ("El contrato termina el 31 de diciembre.", "El acuerdo vence a finales de diciembre."),
    "pt": ("O contrato termina em 31 de dezembro.", "O acordo expira no final de dezembro."),
    "fr": ("Le contrat prend fin le 31 décembre.", "L'accord expire à la fin de décembre."),
    "de": ("Der Vertrag endet am 31. Dezember.", "Die Vereinbarung läuft Ende Dezember aus."),
}
UNRELATED = {
    "en": "Penguins live mostly in the southern hemisphere.",
    "es": "Los pingüinos viven sobre todo en el hemisferio sur.",
    "pt": "Os pinguins vivem sobretudo no hemisfério sul.",
    "fr": "Les manchots vivent surtout dans l'hémisphère sud.",
    "de": "Pinguine leben vor allem auf der Südhalbkugel.",
}


def _model_path() -> Path:
    return Path(os.environ.get("EMBEDDING_MODEL_PATH", "/models/multilingual-e5-base"))


def _revision() -> str:
    return os.environ.get("EMBEDDING_MODEL_REVISION", "unknown")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def embedder() -> Iterator[E5Embedder]:
    threads = torch.get_num_threads()
    yield E5Embedder(E5EmbedderConfig(_model_path(), _revision()))
    torch.set_num_threads(threads)


class TestE5EmbedderContract(EmbedderContract):
    @pytest.fixture
    def embedder(self, embedder: E5Embedder) -> Embedder:
        return embedder


def test_identity_is_e5_base_at_the_pinned_revision(embedder: E5Embedder) -> None:
    identity = embedder.identity

    assert identity.model_id == f"intfloat/multilingual-e5-base@{_revision()}"
    assert _revision() != "unknown"
    assert (identity.dimension, identity.prefix_scheme) == (768, "e5-query-passage")
    assert (identity.normalized, identity.metric) == (True, "cosine")


def test_vectors_are_unit_length_with_768_dimensions(embedder: E5Embedder) -> None:
    vectors = [*embedder.embed_passages(list(UNRELATED.values())), embedder.embed_query("hello")]

    assert all(len(vector) == 768 for vector in vectors)
    assert all(math.hypot(*vector) == pytest.approx(1.0, abs=1e-5) for vector in vectors)


@pytest.mark.parametrize("language", sorted(PARAPHRASES))
def test_a_paraphrase_ranks_above_an_unrelated_sentence(
    embedder: E5Embedder, language: str
) -> None:
    sentence, paraphrase = PARAPHRASES[language]
    anchor, related, unrelated = embedder.embed_passages(
        [sentence, paraphrase, UNRELATED[language]]
    )

    assert _cosine(anchor, related) > _cosine(anchor, unrelated)


@pytest.mark.parametrize("language", sorted(TRIPLES))
def test_a_question_ranks_its_answer_first(embedder: E5Embedder, language: str) -> None:
    question, related, unrelated = TRIPLES[language]
    query = embedder.embed_query(question)
    passages = embedder.embed_passages([unrelated, related])

    assert _cosine(query, passages[1]) > _cosine(query, passages[0])


def test_a_spanish_question_finds_the_english_passage(embedder: E5Embedder) -> None:
    query = embedder.embed_query(TRIPLES["es"][0])
    candidates = [TRIPLES["en"][2], UNRELATED["en"], TRIPLES["en"][1], PARAPHRASES["en"][0]]
    vectors = embedder.embed_passages(candidates)

    scores = [_cosine(query, vector) for vector in vectors]

    assert max(range(len(scores)), key=scores.__getitem__) == 2


def test_an_800_character_passage_is_embedded_quickly(embedder: E5Embedder) -> None:
    passage = (TRIPLES["en"][1] + " ") * 15
    passage = passage[:800]
    embedder.embed_passages([passage])  # warm-up

    started = time.perf_counter()
    embedder.embed_passages([passage] * 16)
    per_passage = (time.perf_counter() - started) / 16

    logging.getLogger(__name__).warning("e5-base: %.3f s per 800-character passage", per_passage)
    assert per_passage < 2.0


def test_loading_and_inference_work_offline_with_configured_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[object] = []

    def refuse(self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise OSError("network disabled by the test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    threads = torch.get_num_threads()
    try:
        offline = E5Embedder(E5EmbedderConfig(_model_path(), _revision()))
        configured_threads = torch.get_num_threads()
        vector = offline.embed_query("who signed the contract?")
    finally:
        torch.set_num_threads(threads)

    assert configured_threads == 2
    assert len(vector) == 768
    assert attempts == []


def test_loading_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    threads = torch.get_num_threads()
    try:
        with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
            warnings.simplefilter("always")
            E5Embedder(E5EmbedderConfig(_model_path(), _revision()))
    finally:
        torch.set_num_threads(threads)

    assert [str(warning.message) for warning in caught] == []
    assert [record.getMessage() for record in caplog.records] == []
