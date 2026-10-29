# Spec Delta

## ADDED Requirements

### Requirement: The audit page is served besides the API
When the audit viewer is enabled, the service SHALL serve the viewer page at `GET /audit` and its own files under `/audit/`, from fixed directories, with the Content-Security-Policy of the viewer, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. Unknown file names SHALL answer 404 without touching paths outside those directories. The four API endpoints and their shapes SHALL be unchanged.

#### Scenario: Page and files
- **WHEN** `GET /audit` and the page's script and the pdf.js files are requested
- **THEN** each answers 200 with its content type and the security headers

#### Scenario: Path outside the viewer
- **WHEN** `GET /audit/static/..%2F..%2Fsettings.py` is requested
- **THEN** it answers 404

### Requirement: Viewer routes answer HEAD, conditional requests and unknown paths safely
Each viewer route SHALL answer GET and HEAD. A file response SHALL carry an ETag and answer a matching If-None-Match with 304 and the same headers. A path under `/audit/` without a route SHALL answer 404 with the viewer's headers, never a redirect.

#### Scenario: Conditional request
- **WHEN** a viewer file is requested again with the ETag of the first answer in `If-None-Match`
- **THEN** it answers 304 with the viewer's headers and no body

#### Scenario: No redirect under /audit
- **WHEN** `GET /audit/static` is requested with `Host: evil.example`
- **THEN** it answers 404 with the viewer's headers and no `Location` header
