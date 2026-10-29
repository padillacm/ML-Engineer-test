// Coordinate rule (design D3): a response box is in PDF points, top-left origin of the page
// as displayed (/Rotate applied, relative to the crop box), the frame of the pdf.js viewport.
// It is drawn at point * scale CSS pixels; no derotation, no y-flip.

export const OUT_OF_PAGE_TOLERANCE_PT = 1;
export const ZOOM_LEVELS = ["fit", "1", "1.5", "2"];
const MIN_SCALE = 0.25;
const MAX_SCALE = 4;

/** CSS pixel rectangle of `box` (points) on a page drawn at `scale`. */
export function boxRect(box, scale) {
  return {
    left: box.x * scale,
    top: box.y * scale,
    width: box.width * scale,
    height: box.height * scale,
  };
}

/** How far `box` lies outside a `widthPt` x `heightPt` page, in points (0 when inside). */
export function outsideBy(box, widthPt, heightPt) {
  return Math.max(
    0,
    -box.x,
    -box.y,
    box.x + box.width - widthPt,
    box.y + box.height - heightPt,
  );
}

export function isOutOfPage(box, widthPt, heightPt) {
  return outsideBy(box, widthPt, heightPt) > OUT_OF_PAGE_TOLERANCE_PT;
}

/** The CSS scale of a zoom value: a number, or "fit" (widest page fills `availablePx`). */
export function scaleFor(zoom, availablePx, maxWidthPt) {
  if (zoom !== "fit") return Number(zoom);
  if (!(availablePx > 0 && maxWidthPt > 0)) return 1;
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, availablePx / maxWidthPt));
}

/** Points with one decimal, as the details panel shows them. */
export function pt(value) {
  return value.toFixed(1);
}
