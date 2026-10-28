"""``POST /api/ingest`` and ``POST /api/ask``.

Ingestion waits on the event loop for a job slot, then reads and validates the PDF and calls
``IngestDocument`` in a job thread (``JobLimiter.run_job``) with the request's
``Cancellation``, cancelled when the client disconnects (``request_cancelled``, logged as 499);
the response adds the document id to the fixed ``IngestResponse`` (``IngestResponseOut``) and
reports ``indexed`` or ``already_indexed``. Questions do not wait for a job slot; they wait up to
``BUSY_WAIT_S`` for one of ``MAX_CONCURRENT_QUESTIONS`` question slots (``service_busy``
otherwise) and run in the question threads (``JobLimiter.run_question``), so slow model answers
take no thread from jobs or readiness.
``AskQuestion`` rejects a blank question (``invalid_question``) and refuses with the fixed text
when nothing relevant is stored. The outcome lines (counts, ids and durations, never a question,
an answer or a source) are written by the use cases themselves and the access line by the
middleware, so the handlers log nothing.
"""

from typing import Annotated, Final, Literal

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.dependencies import JobLimiter, get_ask, get_ingest, get_limiter, get_settings
from app.api.uploads import read_pdf
from app.application.ask_question import AskQuestion
from app.application.ingest_document import IngestDocument
from app.domain.cancellation import Cancellation
from app.domain.types import IngestResult
from app.models import schemas
from app.settings import Settings

INDEXED: Final = "indexed"
ALREADY_INDEXED: Final = "already_indexed"

router = APIRouter(prefix="/api", tags=["rag"])


class IngestResponseOut(schemas.IngestResponse):
    """``IngestResponse`` plus the optional id of the stored document (allowed by the contract).

    ``status`` is ``indexed`` when the document was stored by this request and
    ``already_indexed`` when its exact bytes were stored before.
    """

    status: Literal["indexed", "already_indexed"]
    document_id: str | None = None


@router.post("/ingest", response_model=IngestResponseOut)
async def ingest(
    request: Request,
    pdf_file: Annotated[UploadFile, File(description="The scanned PDF")],
    use_case: Annotated[IngestDocument, Depends(get_ingest)],
    limiter: Annotated[JobLimiter, Depends(get_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestResponseOut:
    """Recognise, chunk, embed and store a PDF unless its exact bytes are already stored."""

    def job(cancel: Cancellation) -> IngestResult:
        return use_case(read_pdf(pdf_file, settings.max_upload_bytes), cancel=cancel)

    # The form has been read, so ``receive`` now only reports the client's disconnect.
    result = await limiter.run_job(job, request.receive)
    return IngestResponseOut(
        status=ALREADY_INDEXED if result.already_indexed else INDEXED,
        chunks_stored=result.chunks_stored,
        document_id=result.document_id,
    )


@router.post("/ask", response_model=schemas.RAGResponse)
async def ask(
    body: schemas.RAGRequest,
    use_case: Annotated[AskQuestion, Depends(get_ask)],
    limiter: Annotated[JobLimiter, Depends(get_limiter)],
) -> schemas.RAGResponse:
    """Answer a question from the stored documents only, or refuse."""
    answer = await limiter.run_question(lambda: use_case(body.question))
    return schemas.RAGResponse(answer=answer.text, sources=list(answer.sources))
