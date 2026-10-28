"""``POST /api/extract``: person names, their boxes and their matches to requested names.

The handler parses the form fields into core values, waits on the event loop for a job slot,
then reads the PDF and calls ``ExtractNames`` in a job thread (``JobLimiter.run_job``), so a
waiting request holds neither a thread nor the PDF in memory. The job gets the request's
``Cancellation``, cancelled when the client disconnects (``request_cancelled``, logged as 499),
so an abandoned extraction stops after the page in progress. It translates the result: one
``ExtractedName`` per occurrence in reading order, boxes in PDF points on 0-based pages, one
``FuzzyMatch`` per pair at or above the threshold. The outcome line (counts and durations) is
written by ``ExtractNames`` itself and the access line by the middleware, so the handler logs
nothing.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.api.dependencies import JobLimiter, get_extract, get_limiter, get_settings
from app.api.uploads import parse_language, parse_names, read_pdf
from app.application.extract_names import ExtractNames
from app.domain.cancellation import Cancellation
from app.domain.types import ExtractionResult
from app.models import schemas
from app.settings import Settings

router = APIRouter(prefix="/api", tags=["extraction"])


@router.post("/extract", response_model=schemas.ExtractionResponse)
async def extract(
    request: Request,
    pdf_file: Annotated[UploadFile, File(description="The scanned PDF")],
    use_case: Annotated[ExtractNames, Depends(get_extract)],
    limiter: Annotated[JobLimiter, Depends(get_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
    names: Annotated[
        str, Form(description='JSON array of {"first_name": ..., "last_name": ...}')
    ] = "[]",
    language: Annotated[
        str | None, Form(description="Optional language hint: en, es, pt, fr or de")
    ] = None,
) -> schemas.ExtractionResponse:
    """Extract the person names of a scanned PDF and fuzzy-match them to ``names``."""
    requested = parse_names(names, settings.max_names)
    hint = parse_language(language)

    def job(cancel: Cancellation) -> ExtractionResult:
        pdf = read_pdf(pdf_file, settings.max_upload_bytes)
        return use_case(pdf, requested, hint, cancel=cancel)

    # The form has been read, so ``receive`` now only reports the client's disconnect.
    return to_response(await limiter.run_job(job, request.receive))


def to_response(result: ExtractionResult) -> schemas.ExtractionResponse:
    """The HTTP shape of an ``ExtractionResult``."""
    return schemas.ExtractionResponse(
        extracted_names=[
            schemas.ExtractedName(
                name=mention.name,
                bounding_box=schemas.BoundingBox(
                    page_number=mention.box.page,
                    x=mention.box.x,
                    y=mention.box.y,
                    width=mention.box.width,
                    height=mention.box.height,
                ),
            )
            for mention in result.names
        ],
        fuzzy_matches=[
            schemas.FuzzyMatch(
                extracted_name=match.extracted_name,
                matched_name=match.matched_name,
                score=match.score,
            )
            for match in result.matches
        ],
    )
