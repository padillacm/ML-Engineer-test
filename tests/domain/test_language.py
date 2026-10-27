import itertools

import pytest

from app.domain.errors import InvalidInput
from app.domain.language import (
    STOP_WORDS,
    SUPPORTED_LANGUAGES,
    detect_language,
    validate_language,
)

SAMPLES: dict[str, str] = {
    "en": (
        "This agreement is made between the parties named below and shall remain in force for "
        "a period of twelve months from the date of signature. The supplier will deliver the "
        "goods to the address given by the customer, and any delay must be reported in writing "
        "within five working days. Payment shall be made by bank transfer within thirty days "
        "of the invoice, and late payments are subject to interest at the agreed rate."
    ),
    "es": (
        "El presente contrato se celebra entre las partes indicadas al pie y tendrá una "
        "vigencia de doce meses desde la fecha de su firma. El proveedor entregará los bienes "
        "en el domicilio indicado por el cliente, y cualquier retraso deberá comunicarse por "
        "escrito en un plazo de cinco días hábiles. El pago se realizará mediante transferencia "
        "bancaria dentro de los treinta días siguientes a la emisión de la factura, y los pagos "
        "atrasados devengarán intereses según el tipo acordado."
    ),
    "pt": (
        "O presente contrato é celebrado entre as partes indicadas abaixo e terá vigência de "
        "doze meses a contar da data da sua assinatura. O fornecedor entregará os bens no "
        "endereço indicado pelo cliente, e qualquer atraso deve ser comunicado por escrito no "
        "prazo de cinco dias úteis. O pagamento será feito por transferência bancária em até "
        "trinta dias após a emissão da fatura, e os pagamentos em atraso estão sujeitos a juros "
        "pela taxa acordada."
    ),
    "fr": (
        "Le présent contrat est conclu entre les parties désignées ci-dessous et restera en "
        "vigueur pendant une durée de douze mois à compter de la date de sa signature. Le "
        "fournisseur livrera les marchandises à l'adresse indiquée par le client, et tout retard "
        "doit être signalé par écrit dans un délai de cinq jours ouvrables. Le paiement sera "
        "effectué par virement bancaire dans les trente jours suivant la facture, et les "
        "retards de paiement sont soumis à des intérêts au taux convenu."
    ),
    "de": (
        "Dieser Vertrag wird zwischen den unten genannten Parteien geschlossen und gilt für "
        "einen Zeitraum von zwölf Monaten ab dem Datum der Unterzeichnung. Der Lieferant "
        "liefert die Waren an die vom Kunden angegebene Adresse, und jede Verzögerung ist "
        "innerhalb von fünf Werktagen schriftlich mitzuteilen. Die Zahlung erfolgt per "
        "Banküberweisung innerhalb von dreißig Tagen nach Rechnungsstellung, und verspätete "
        "Zahlungen werden mit dem vereinbarten Satz verzinst."
    ),
}

MEMOS: dict[str, str] = {
    "en": (
        "Please note that the quarterly review meeting has been moved to Thursday. All team "
        "leads should bring their updated budget figures and a short summary of the risks they "
        "have identified. If you are not able to attend, send your report to the finance "
        "office before noon, and we will add it to the minutes."
    ),
    "es": (
        "Les informamos que la reunión de revisión trimestral se ha trasladado al jueves. Todos "
        "los responsables de equipo deben traer sus cifras de presupuesto actualizadas y un "
        "breve resumen de los riesgos que han identificado. Si usted no puede asistir, envíe su "
        "informe a la oficina de finanzas antes del mediodía y lo añadiremos al acta."
    ),
    "pt": (
        "Informamos que a reunião de revisão trimestral foi transferida para quinta-feira. "
        "Todos os responsáveis de equipe devem trazer os seus números de orçamento atualizados "
        "e um breve resumo dos riscos que identificaram. Se você não puder comparecer, envie o "
        "seu relatório ao departamento financeiro até o meio-dia e nós o incluiremos na ata."
    ),
    "fr": (
        "Nous vous informons que la réunion de revue trimestrielle a été déplacée à jeudi. Tous "
        "les responsables d'équipe doivent apporter leurs chiffres budgétaires mis à jour et un "
        "court résumé des risques qu'ils ont identifiés. Si vous ne pouvez pas être présent, "
        "envoyez votre rapport au service financier avant midi et nous l'ajouterons au "
        "procès-verbal."
    ),
    "de": (
        "Bitte beachten Sie, dass die vierteljährliche Besprechung auf Donnerstag verschoben "
        "wurde. Alle Teamleiter sollen ihre aktualisierten Budgetzahlen und eine kurze "
        "Zusammenfassung der erkannten Risiken mitbringen. Wenn Sie nicht teilnehmen können, "
        "senden Sie Ihren Bericht bis Mittag an die Finanzabteilung, und wir nehmen ihn in das "
        "Protokoll auf."
    ),
}


def test_supported_languages_are_the_five_engine_codes() -> None:
    assert SUPPORTED_LANGUAGES == ("en", "es", "pt", "fr", "de")
    assert set(STOP_WORDS) == set(SUPPORTED_LANGUAGES)


def test_stop_word_lists_are_disjoint_folded_and_about_forty_long() -> None:
    for words in STOP_WORDS.values():
        assert 35 <= len(words) <= 45
        assert all(word.isascii() and word.islower() for word in words)
    for first, second in itertools.combinations(STOP_WORDS.values(), 2):
        assert not first & second


@pytest.mark.parametrize("code", SUPPORTED_LANGUAGES)
@pytest.mark.parametrize("samples", [SAMPLES, MEMOS], ids=["contract", "memo"])
def test_business_text_is_detected(samples: dict[str, str], code: str) -> None:
    assert detect_language(samples[code]) == code


@pytest.mark.parametrize("code", SUPPORTED_LANGUAGES)
def test_detection_survives_lost_accents_and_upper_case(code: str) -> None:
    text = SAMPLES[code].upper().translate(str.maketrans("ÁÉÍÓÚÃÕÇÀÂÊÔÜÖÄß", "AEIOUAOCAAEOUOAS"))

    assert detect_language(text) == code


def test_short_text_is_inconclusive() -> None:
    assert detect_language("The supplier will deliver the goods to the customer.") is None


def test_min_words_is_configurable() -> None:
    sentence = "The supplier will deliver the goods to the customer."

    assert detect_language(sentence, min_words=5) == "en"


def test_evenly_mixed_text_is_inconclusive() -> None:
    mixed = (
        "The supplier will deliver the goods to the customer, and the invoice is paid within "
        "thirty days. El proveedor entregará los bienes al cliente, y el pago de la factura se "
        "hará con los intereses del contrato."
    )

    assert detect_language(mixed) is None


def test_all_languages_together_are_inconclusive() -> None:
    assert detect_language(" ".join(SAMPLES.values())) is None


def test_margin_is_configurable() -> None:
    text = SAMPLES["pt"] + " " + SAMPLES["es"]

    assert detect_language(text, min_margin=1.0) is not None
    assert detect_language(text, min_margin=100.0) is None


def test_names_only_text_is_inconclusive() -> None:
    names = (
        "María José Ramírez, João Pedro da Silva, Jean-Luc Moreau, Hans Müller, Emily Carter, "
        "Lucía Fernández, Ana Beatriz Costa, François Lefèvre, Jürgen Schmidt, Oliver Hughes, "
        "Carmen Ortega, Pedro Henrique Alves, Camille Dubois, Katrin Becker, Thomas Wright"
    )

    assert detect_language(names) is None


def test_numbers_and_punctuation_are_not_words() -> None:
    assert detect_language("1234 5678 -- ** 2024-01-31 ; 99,5 % " * 10) is None


@pytest.mark.parametrize(("raw", "expected"), [("es", "es"), (" PT ", "pt"), ("De\n", "de")])
def test_valid_language_codes_are_normalised(raw: str, expected: str) -> None:
    assert validate_language(raw) == expected


@pytest.mark.parametrize("raw", ["it", "", "eng", "spa", "en-US", "e s"])
def test_unsupported_language_codes_are_rejected(raw: str) -> None:
    with pytest.raises(InvalidInput) as caught:
        validate_language(raw)

    assert caught.value.code == "unsupported_language"
