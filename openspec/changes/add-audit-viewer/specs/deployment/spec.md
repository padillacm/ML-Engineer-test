# Spec Delta

## ADDED Requirements

### Requirement: The images carry the pinned viewer library
The test and runtime images SHALL contain the pinned pdf.js library, worker and licence, fetched at build time with a sha256 check and read-only for the service user; nothing SHALL be downloaded at run time.

#### Scenario: Image hygiene with the viewer
- **WHEN** the runtime image is inspected
- **THEN** the pdf.js files are present with the pinned hash, owned by root and not writable by uid 10001

### Requirement: The browser of the e2e image is pinned by hash
The e2e test image SHALL contain the Chrome Headless Shell that its Playwright version pins, fetched at build time with a sha256 check; the runtime and test images SHALL contain no browser.

#### Scenario: Tampered browser archive
- **WHEN** the e2e stage is built with a browser archive whose sha256 differs from the pin
- **THEN** the build fails
