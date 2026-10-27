"""Language choice with the real Tesseract: the ``-l`` model of every call is recorded."""

import threading
from collections import Counter

import pytest

from app.adapters.ocr import TesseractOcrConfig, TesseractOcrEngine
from app.adapters.ocr.tesseract import TesseractCli
from tests.adapters.ocr.pdfs import A4, PageSpec, image_only_pdf
from tests.integration.known_words import normalise

pytestmark = pytest.mark.tesseract

COMBINED = "eng+spa+por+fra+deu"

SPANISH = (
    "El contrato fue firmado por los representantes de la empresa y del",
    "proveedor en la ciudad de Madrid. Las partes han acordado que el pago",
    "se realizará cuando el servicio sea entregado sin defectos y con",
    "todas las garantías. Los documentos del proyecto fueron revisados",
    "por el comité, pero hay una cláusula que debe ser aclarada hasta",
    "el final del mes, según lo dicho en la reunión con su director.",
)
PORTUGUESE = (
    "O contrato foi assinado pelos representantes da empresa e do",
    "fornecedor na cidade de Lisboa. As partes acordaram que o pagamento",
    "será feito quando o serviço for entregue sem defeitos e com todas",
    "as garantias. Os documentos do projeto também foram revistos pela",
    "comissão, mas ainda há uma cláusula que deve ser esclarecida até o",
    "final do mês, conforme foi dito na reunião com o seu diretor.",
)
GERMAN = (
    "Der Vertrag wurde von den Vertretern der Firma und des Lieferanten",
    "in der Stadt Berlin unterzeichnet. Die Parteien haben vereinbart,",
    "dass die Zahlung erfolgt, wenn die Leistung ohne Mängel und mit",
    "allen Garantien erbracht wird. Die Unterlagen des Projekts wurden",
    "auch vom Ausschuss geprüft, aber eine Klausel muss noch bis zum",
    "Ende des Monats geklärt werden, wie in der Sitzung mit dem Direktor.",
)

FRENCH = (
    "Le contrat a été signé par les représentants de la société et du",
    "fournisseur dans la ville de Lyon. Les parties sont convenues que le",
    "paiement sera effectué dès que le service est livré sans défauts et",
    "avec toutes les garanties. Les documents du projet ont aussi été",
    "revus par le comité, mais une clause doit encore être précisée avant",
    "la fin du mois, selon ce qui a été dit lors de la réunion avec leur directeur.",
)
ENGLISH = (
    "The contract was signed by the representatives of the company and",
    "the supplier in the city of Leeds. The parties have agreed that the",
    "payment shall be made when the service is delivered without defects",
    "and with all the guarantees. The documents of the project were also",
    "reviewed by the committee, but there is a clause that should be made",
    "clear by the end of the month, as was said at the meeting with their director.",
)
NAMES_TABLE = (
    "Ana Ramírez 4512",
    "João Silva 7731",
    "Jürgen Müller 9902",
    "Chloé Martin 1187",
    "Pedro Gómez 6620",
)


class RecordingCli:
    """Wraps the real ``TesseractCli`` and records the ``-l`` models it is called with."""

    def __init__(self, config: TesseractOcrConfig) -> None:
        self._cli = TesseractCli(
            command=config.tesseract_cmd,
            dpi=config.dpi,
            psm=config.psm,
            oem=config.oem,
            timeout_s=config.timeout_s,
        )
        self._lock = threading.Lock()
        self.models: list[str] = []

    def __call__(self, png: bytes, languages: str) -> str:
        with self._lock:
            self.models.append(languages)
        return self._cli(png, languages)


def _recording_engine() -> tuple[TesseractOcrEngine, RecordingCli]:
    config = TesseractOcrConfig()
    runner = RecordingCli(config)
    return TesseractOcrEngine(config, runner=runner), runner


def _two_page_document(lines: tuple[str, ...]) -> bytes:
    return image_only_pdf(
        [PageSpec(size=A4, lines=lines), PageSpec(size=A4, lines=lines[:2])], dpi=300
    )


@pytest.mark.parametrize(
    ("lines", "model", "expected_word"),
    [
        pytest.param(SPANISH, "spa", "cláusula", id="es"),
        pytest.param(PORTUGUESE, "por", "comissão", id="pt"),
        pytest.param(GERMAN, "deu", "Mängel", id="de"),
        pytest.param(FRENCH, "fra", "société", id="fr"),
        pytest.param(ENGLISH, "eng", "guarantees", id="en"),
    ],
)
def test_without_a_hint_the_detected_language_reads_every_page(
    lines: tuple[str, ...], model: str, expected_word: str
) -> None:
    engine, runner = _recording_engine()

    pages = engine.recognize(_two_page_document(lines))

    print(f"models used: {runner.models}")
    assert runner.models[0] == COMBINED
    assert Counter(runner.models[1:]) == Counter({model: 2})
    assert [page.page for page in pages] == [0, 1]
    assert expected_word in {normalise(word.text) for word in pages[0].words}


def test_with_a_hint_every_page_is_read_with_that_model_only() -> None:
    engine, runner = _recording_engine()

    pages = engine.recognize(_two_page_document(SPANISH), language="es")

    assert runner.models == ["spa", "spa"]
    assert len(pages) == 2


def test_short_text_without_a_hint_is_read_with_all_languages_together() -> None:
    engine, runner = _recording_engine()
    document = image_only_pdf(
        [PageSpec(size=A4, lines=("Invoice 2024",)), PageSpec(size=A4, lines=("Total 450",))],
        dpi=300,
    )

    pages = engine.recognize(document)

    assert runner.models == [COMBINED, COMBINED]
    assert [page.page for page in pages] == [0, 1]
    assert "Invoice" in {normalise(word.text) for word in pages[0].words}


def test_a_names_table_on_the_first_page_does_not_decide_the_language() -> None:
    engine, runner = _recording_engine()
    document = image_only_pdf(
        [
            PageSpec(size=A4, lines=NAMES_TABLE),
            PageSpec(size=A4, lines=SPANISH),
            PageSpec(size=A4, lines=SPANISH[:3]),
        ],
        dpi=300,
    )

    pages = engine.recognize(document)

    print(f"models used: {runner.models}")
    assert runner.models[:2] == [COMBINED, COMBINED]
    assert Counter(runner.models[2:]) == Counter({"spa": 3})
    assert [page.page for page in pages] == [0, 1, 2]
