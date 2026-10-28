"""Context selection and prompt building for grounded answering.

The context given to the language model is the full text of the parent pages of the best-ranked
chunks, deduplicated and kept within a character budget that covers the whole rendered context
(the ``[n] document <id>, page <p>`` labels and the separators included). The prompt tells the
model to answer only from that context and to reply with ``REFUSAL_SENTINEL`` when the answer
is not there; ``is_refusal`` recognises that reply, and the service then returns the fixed
``REFUSAL``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.errors import ConfigurationError
from app.domain.types import ChatMessage, SearchHit

REFUSAL = "The provided documents do not contain this information."
"""The fixed answer returned by the service when the documents do not contain the answer."""

REFUSAL_SENTINEL = "NOT_IN_DOCUMENTS"
"""The exact reply the model is asked to give when the context does not contain the answer."""

MIN_PARTIAL_CHARS = 200
"""A page after the first that does not fit is cut only if at least this much text fits."""

PAGE_SEPARATOR = "\n\n"
NO_CONTEXT = "(no context)"

_REFUSAL_PUNCTUATION = " \t\r\n\"'`*_.,;:!?()[]{}<>«»“”‘’"  # noqa: RUF001 - typographic quotes

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages given by the user. "
    "Each passage is labelled with its document and page.\n"
    "Rules:\n"
    "- Answer only from the numbered context. Do not use outside knowledge.\n"
    "- Answer in the language of the question.\n"
    "- If the context does not contain the answer, reply with exactly: "
    f"{REFUSAL_SENTINEL}\n"
    "- Do not invent document names, page numbers or quotes.\n"
    "- Be concise."
)


@dataclass(frozen=True, slots=True)
class ContextPage:
    """The (possibly cut) text of one page given to the model; ``page`` is 0-based."""

    document_id: str
    page: int
    text: str

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if self.page < 0:
            raise ValueError(f"page index must be >= 0, got {self.page}")


def select_context(
    hits: Sequence[SearchHit], *, max_chunks: int = 3, max_chars: int = 6000
) -> list[ContextPage]:
    """Return the parent pages of the best ``max_chunks`` distinct chunks, in rank order.

    ``hits`` must be best first. Chunks are distinct by ``(document_id, sequence)``; their pages
    are deduplicated by ``(document_id, page)``, so two chunks of one page give one page. Whole
    pages are added while ``render_context(pages)`` fits ``max_chars``. The first page that
    does not fit is cut at a word boundary to the room left for its text, when that room is at
    least ``MIN_PARTIAL_CHARS`` (or any room at all for the first page), and selection stops
    there. So the rendered context never exceeds ``max_chars``; when not even the first page's
    label fits, no page is returned. Pages with blank text are skipped.
    """
    if max_chunks < 1:
        raise ConfigurationError(f"max_chunks must be >= 1, got {max_chunks}")
    if max_chars < 1:
        raise ConfigurationError(f"max_chars must be >= 1, got {max_chars}")

    pages: list[ContextPage] = []
    remaining = max_chars
    for page in _parent_pages(hits, max_chunks):
        overhead = len(_render_page(len(pages) + 1, page.document_id, page.page, ""))
        if pages:
            overhead += len(PAGE_SEPARATOR)
        if overhead + len(page.text) <= remaining:
            pages.append(page)
            remaining -= overhead + len(page.text)
            continue
        room = remaining - overhead
        if room >= (MIN_PARTIAL_CHARS if pages else 1):
            cut = _cut_at_word_boundary(page.text, room)
            if cut:
                pages.append(ContextPage(page.document_id, page.page, cut))
        break
    return pages


def render_context(pages: Sequence[ContextPage]) -> str:
    """The labelled pages as given to the model: numbered from 1, pages shown 1-based."""
    return PAGE_SEPARATOR.join(
        _render_page(number, page.document_id, page.page, page.text)
        for number, page in enumerate(pages, start=1)
    )


def shown_sources(
    hits: Sequence[SearchHit], pages: Sequence[ContextPage], *, max_chunks: int = 3
) -> tuple[str, ...]:
    """The texts of the chunks the model saw, exactly as stored, in rank order.

    A chunk counts when it is one of the first ``max_chunks`` distinct chunks of ``hits`` (the
    ones ``select_context`` used) and its text lies within the text included for its page
    (which may have been cut). Each chunk is returned once.
    """
    included = {(page.document_id, page.page): page.text for page in pages}
    sources: list[str] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        if len(seen) >= max_chunks:
            break
        key = (hit.document_id, hit.sequence)
        if key in seen:
            continue
        seen.add(key)
        text = included.get((hit.document_id, hit.page))
        needle = hit.text.strip()
        if text is not None and needle and needle in text:
            sources.append(hit.text)
    return tuple(sources)


def build_messages(question: str, pages: Sequence[ContextPage]) -> list[ChatMessage]:
    """Return the system and user messages that ask ``question`` over the numbered ``pages``.

    The user message is ``"Context:\\n\\n" + render_context(pages)`` (``NO_CONTEXT`` when there
    are no pages) followed by ``"\\n\\nQuestion: "`` and the stripped question.
    """
    if not question.strip():
        raise ValueError("question must not be blank")
    context = render_context(pages) if pages else NO_CONTEXT
    user = f"Context:\n\n{context}\n\nQuestion: {question.strip()}"
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user)]


def is_refusal(reply: str) -> bool:
    """Whether a model reply says the documents do not contain the answer.

    True for a blank reply, and for one that, once stripped of surrounding whitespace, quotes
    and punctuation and casefolded, equals or starts with ``REFUSAL_SENTINEL`` or equals the
    English ``REFUSAL`` sentence.
    """
    normalised = reply.strip().casefold().strip(_REFUSAL_PUNCTUATION)
    return (
        not normalised
        or normalised.startswith(REFUSAL_SENTINEL.casefold())
        or normalised == REFUSAL.casefold().strip(_REFUSAL_PUNCTUATION)
    )


def _render_page(number: int, document_id: str, page: int, text: str) -> str:
    return f"[{number}] document {document_id}, page {page + 1}\n{text}"


def _cut_at_word_boundary(text: str, limit: int) -> str:
    """Return the longest prefix of ``text`` of at most ``limit`` characters ending at a word end.

    A single word longer than ``limit`` is cut at ``limit``.
    """
    if len(text) <= limit:
        return text
    if text[limit].isspace():
        return text[:limit].rstrip()
    boundary = next((i for i in range(limit - 1, 0, -1) if text[i].isspace()), 0)
    if boundary == 0:
        return text[:limit]
    return text[:boundary].rstrip()


def _parent_pages(hits: Sequence[SearchHit], max_chunks: int) -> list[ContextPage]:
    """The distinct, non-blank parent pages of the first ``max_chunks`` distinct chunks."""
    pages: list[ContextPage] = []
    seen_chunks: set[tuple[str, int]] = set()
    seen_pages: set[tuple[str, int]] = set()
    for hit in hits:
        if len(seen_chunks) >= max_chunks:
            break
        chunk_key = (hit.document_id, hit.sequence)
        page_key = (hit.document_id, hit.page)
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        text = hit.page_text.strip()
        if page_key in seen_pages or not text:
            continue
        seen_pages.add(page_key)
        pages.append(ContextPage(hit.document_id, hit.page, text))
    return pages
