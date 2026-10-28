# page-geometry Specification

## Purpose
Defines how positions found on a rasterised page are expressed in PDF page space and how a piece of recognised text is mapped back to a single box on its page, so every bounding box the service reports uses one documented coordinate system.

## Requirements

### Requirement: Page space is points with a top-left origin
All boxes reported by the core SHALL be expressed in PDF points (1/72 inch), with the origin at the top-left corner of the page, x growing right, y growing down, and a 0-based page index. A box SHALL be described by its top-left corner (`x`, `y`), `width` and `height`.

#### Scenario: Box lies on an A4 page
- **WHEN** a box is created on a 595 x 842 pt page at x=72, y=100 with width 120 and height 14
- **THEN** its right edge is at 192 and its bottom edge is at 114, both measured from the top-left origin

### Requirement: Pixel to point conversion uses the real raster and page size
The conversion from raster pixels to points SHALL derive its scale from the actual pixel size of the rendered image and the actual page rectangle in points, independently for each axis. It SHALL NOT assume a DPI value.

#### Scenario: Image rendered at an unexpected resolution
- **WHEN** a 595 x 842 pt page was rendered into an image of 1653 x 2339 pixels and a pixel box of x=165.3, y=233.9, width=330.6, height=46.78 is converted
- **THEN** the point box is x=59.5, y=84.2, width=119.0, height=16.84 within 0.01

#### Scenario: Image size differs by rounding
- **WHEN** the rendered width in pixels is 1654 instead of the exact 1653.5 and the page width is 595 pt
- **THEN** the horizontal scale is computed as 595/1654 and boxes at the right edge of the image end at 595 pt within floating-point rounding (1e-9 pt)

#### Scenario: Non-finite values are rejected
- **WHEN** a box, a raster descriptor or a clamp rectangle contains NaN or infinity
- **THEN** the operation fails with a geometry error

### Requirement: Word boxes are united into one box
The core SHALL merge a non-empty set of word boxes on the same page into the smallest box that contains all of them. Merging an empty set, or boxes from different pages, SHALL be rejected with an error.

#### Scenario: Two words on one line
- **WHEN** words at (10, 20, 30, 10) and (45, 21, 40, 10) on page 0 are united
- **THEN** the result is x=10, y=20, width=75, height=11 on page 0

#### Scenario: Words on different pages
- **WHEN** a word on page 0 and a word on page 1 are united
- **THEN** the operation fails with a geometry error

### Requirement: Boxes are clamped to the page
The core SHALL clip a box to the page rectangle. A box that lies entirely outside the page SHALL be rejected instead of being turned into an empty or negative-sized box.

#### Scenario: Box slightly beyond the right edge
- **WHEN** a box ends at 597 pt on a 595 pt wide page
- **THEN** the clamped box ends at 595 pt and keeps its left edge

#### Scenario: Box fully outside the page
- **WHEN** a box starts at x=700 on a 595 pt wide page
- **THEN** the operation fails with a geometry error

### Requirement: A text span is located as one box
The core SHALL assemble the text of a page from its recognised words, in the order given and separated by single spaces, and SHALL map a character span of that text to the union box of every word the span overlaps, even partially. A span that overlaps no word SHALL yield no box.

#### Scenario: Span covers a full name
- **WHEN** the page words are "Signed", "by", "Maria", "Gonzalez", "today" and the span is the characters of "Maria Gonzalez"
- **THEN** the result is the union of the boxes of "Maria" and "Gonzalez" only

#### Scenario: Span starts in the middle of a word
- **WHEN** the span begins inside "Maria" and ends at the end of "Gonzalez"
- **THEN** the whole word "Maria" is included in the result

#### Scenario: Blank recognised words are ignored
- **WHEN** the recognised words include an entry whose text is only whitespace
- **THEN** it contributes no characters to the page text and no box to any span

#### Scenario: Span outside the text
- **WHEN** the span starts after the last character of the page text
- **THEN** no box is returned

### Requirement: Box overlap can be measured
The core SHALL provide the intersection-over-union of two boxes on the same page, and SHALL return 0 for boxes on different pages.

#### Scenario: Identical boxes
- **WHEN** a box is compared with itself
- **THEN** the overlap is 1.0

#### Scenario: Disjoint boxes
- **WHEN** two boxes on the same page do not intersect
- **THEN** the overlap is 0.0

#### Scenario: Same position on different pages
- **WHEN** two identical boxes are on pages 0 and 1
- **THEN** the overlap is 0.0
