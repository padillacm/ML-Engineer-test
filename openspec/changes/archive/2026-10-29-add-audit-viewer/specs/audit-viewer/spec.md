# Spec Delta

## Purpose

Defines the audit viewer: a page served by the service that sends a PDF to `POST /api/extract`, renders the PDF in the browser with an independent renderer and draws every returned box over its page, so a person can check names, boxes and matches by eye.

## ADDED Requirements

### Requirement: The viewer audits the real extraction response
The viewer SHALL send the chosen PDF, the names list and the optional language hint to `POST /api/extract` exactly as any client does, and SHALL draw the boxes of that response without changing them. It SHALL show the request id and duration of the call, and an API error with its code and message.

#### Scenario: Extraction drawn
- **WHEN** a sample PDF is chosen with the names "Maria Gonzalez" and "John Smith" and extraction is requested
- **THEN** one box is drawn per extracted name on its page, and the request id of the response is shown

#### Scenario: API error shown
- **WHEN** a file that is not a PDF is chosen and extraction is requested
- **THEN** the page shows the error code and message of the response and draws no box

### Requirement: Boxes are drawn in the page's displayed frame
The PDF SHALL be rendered in the browser by pdf.js, not by the service. A box SHALL be drawn at its point coordinates multiplied by the viewport scale on the pdf.js viewport of the page with its rotation and crop box applied, with a top-left origin, without derotation in the browser. A box that lies outside the rendered page by more than 1 point SHALL be listed as an audit warning.

#### Scenario: Position at two zoom levels
- **WHEN** a response box `x=72, y=100, width=120, height=14` on page 0 is drawn at scale 1.5 and then at scale 2
- **THEN** its element is at 108, 150 with size 180 x 21 CSS pixels relative to the page, then at 144, 200 with size 240 x 28, within 1 pixel

#### Scenario: Rotated and cropped pages
- **WHEN** a PDF whose pages have `/Rotate` 90, 180 and 270, a crop box offset from the media box, and different page sizes is audited
- **THEN** every box covers the rendered name on its page by the ink check

### Requirement: Frames the two renderers read differently are flagged
A page whose /UserUnit is not 1 SHALL be drawn in the service's frame (user units, which the service's renderer uses instead of points) and listed with a page audit warning. A page whose /Rotate pdf.js reports as something other than a number multiple of 90 SHALL be listed with a page audit warning.

#### Scenario: User unit page
- **WHEN** a page with `/UserUnit 2` is audited
- **THEN** its boxes cover its names by the ink check and the page shows the warning "page uses /UserUnit 2: the service's boxes are in user units, not points"

### Requirement: Matches are distinguishable and inspectable
Each box SHALL show whether its name matched the list, by line style and by a text label, not by colour alone. Hovering or focusing a box, or selecting its name in the side list, SHALL show the name, the 0-based page, the box in points and its best matched name and score; selecting it in the list SHALL scroll to the box and highlight it. The page SHALL offer a filter (all, matched, unmatched), zoom levels and a download of the raw response.

#### Scenario: Selecting a name
- **WHEN** a name on page 3 is selected in the side list
- **THEN** page 3 is scrolled into view and that box is highlighted with its details shown

#### Scenario: Keyboard use
- **WHEN** the user moves through the boxes with the Tab key
- **THEN** each box receives focus in page order and its details are read from its accessible label

### Requirement: Long documents stay responsive
Pages SHALL be rendered only when near the viewport and released when far from it, while the side list and the box elements exist for every page from the moment the response arrives.

#### Scenario: Fifty pages
- **WHEN** a 50-page PDF is audited
- **THEN** the side list lists every name at once, scrolling through all pages renders each page with its boxes, at most 8 page canvases hold pixels afterwards, and repeating the extraction does not grow the page's DOM nodes or event listeners

### Requirement: The page stays usable while the API is slow
While a request runs, the file, names and language inputs SHALL be disabled; a Cancel control SHALL abort the request and return the page to idle; only the start, end, error or cancel of a request SHALL be announced to assistive technology.

#### Scenario: Cancel
- **WHEN** Cancel is pressed during a request
- **THEN** the request is aborted, no box is drawn, the inputs are enabled and the status says cancelled

#### Scenario: Slow API
- **WHEN** a request takes several seconds
- **THEN** the live status changes only at its start and end

### Requirement: The page is safe with untrusted documents
The page SHALL run under a Content-Security-Policy that allows scripts, workers, styles, images and connections only from the service's own origin and forbids inline scripts, framing, plugins and form posts elsewhere. pdf.js SHALL run with eval disabled at a version without known script-execution flaws. Every value taken from the response SHALL be written as text, never interpreted as HTML. The PDF SHALL NOT be sent anywhere except `POST /api/extract`.

#### Scenario: Name containing markup
- **WHEN** the response contains the name `<img src=x onerror=alert(1)>`
- **THEN** it is shown as that literal text and no script runs

#### Scenario: No other requests
- **WHEN** a PDF is audited
- **THEN** the browser makes no request other than the page, its own files and `POST /api/extract`, and reports no CSP violation
