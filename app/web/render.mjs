// pdf.js loading and the page views: a placeholder sized from the pdf.js viewport, a canvas
// rendered on demand and an overlay for the box elements. Pages near the viewport are
// rendered, pages far from it release their canvas (design D4).

const VENDOR = new URL("../vendor/pdfjs/", import.meta.url);
const MAX_CANVAS_PIXELS = 16_777_216; // 4096 x 4096: keeps a 200 % zoom on a large page bounded
const NEAR_MARGIN = "100% 0px"; // render one viewport above and below
const FAR_MARGIN = "300% 0px"; // release beyond three viewports

let pdfjsPromise = null;

/** The pdf.js module, imported once (a rejected import is retried on the next call). */
export function loadPdfjs() {
  pdfjsPromise ??= import(new URL("pdf.min.mjs", VENDOR).href).then((lib) => {
    lib.GlobalWorkerOptions.workerSrc = new URL("pdf.worker.min.mjs", VENDOR).href;
    return lib;
  });
  pdfjsPromise.catch(() => {
    pdfjsPromise = null;
  });
  return pdfjsPromise;
}

/** Open `bytes` (a Uint8Array pdf.js may transfer) and read every page's viewport at scale 1. */
export async function openPdf(bytes) {
  const lib = await loadPdfjs();
  const doc = await lib.getDocument({
    data: bytes,
    isEvalSupported: false, // ignored by builds without the eval path, kept for older ones
    disableFontFace: false,
    useSystemFonts: false,
    useWasm: false, // the CSP has no 'wasm-unsafe-eval'
    // JBIG2, CCITT and JPEG 2000 images use pdf.js's JS fallback decoders from here.
    wasmUrl: VENDOR.href,
    enableXfa: false,
  }).promise;
  const pages = [];
  for (let number = 1; number <= doc.numPages; number += 1) {
    const page = await doc.getPage(number);
    // pdf.js applies /UserUnit to its viewport; the service's renderer (PDFium) ignores it, so
    // the response is in user units. The view is drawn in the service's frame: one user unit
    // per point, the page's /Rotate and crop box applied (design D3).
    const userUnit = Number.isFinite(page.userUnit) && page.userUnit > 0 ? page.userUnit : 1;
    const viewport = page.getViewport({ scale: 1 / userUnit });
    pages.push({
      index: number - 1,
      page,
      widthPt: viewport.width,
      heightPt: viewport.height,
      userUnit,
      rotate: page.rotate,
    });
  }
  return { doc, pages };
}

/**
 * Audit warnings of a page whose frame the service and pdf.js may not share:
 * `{ kind, text }` for a /UserUnit other than 1 and for a /Rotate that pdf.js reports as
 * something other than a number multiple of 90 (pdf.js keeps such a value as it is, the
 * service's renderer reads it as 0).
 */
export function pageWarnings({ userUnit, rotate }) {
  const warnings = [];
  if (userUnit !== 1) {
    warnings.push({
      kind: "user-unit",
      text: `page uses /UserUnit ${userUnit}: the service's boxes are in user units, not points`,
    });
  }
  if (typeof rotate !== "number" || !Number.isInteger(rotate) || rotate % 90 !== 0) {
    warnings.push({
      kind: "rotate",
      text:
        `page /Rotate is ${JSON.stringify(rotate)}, not a number multiple of 90: pdf.js and ` +
        "the service's renderer may show the page in different orientations",
    });
  }
  return warnings;
}

/** Release a document opened by `openPdf` (its worker data and pages). */
export function closePdf(doc) {
  doc.loadingTask.destroy().catch(() => {});
}

export class PageView {
  constructor({ index, page, widthPt, heightPt, userUnit = 1, rotate = 0 }) {
    this.index = index;
    this.page = page;
    this.widthPt = widthPt; // in the service's frame (user units)
    this.heightPt = heightPt;
    this.userUnit = userUnit;
    this.warnings = pageWarnings({ userUnit, rotate });
    this.scale = 1;
    this.renderedScale = null;
    this.pending = null;
    this.near = false;

    this.el = document.createElement("div");
    this.el.className = "page";
    this.el.dataset.pageIndex = String(index);
    this.el.dataset.widthPt = String(widthPt);
    this.el.dataset.heightPt = String(heightPt);
    this.el.dataset.userUnit = String(userUnit);
    this.el.dataset.rendered = "false";
    this.canvas = blankCanvas();
    this.overlay = document.createElement("div");
    this.overlay.className = "page-overlay";
    this.el.append(this.canvas, this.overlay);
  }

  /** Size the placeholder for `scale`; the old canvas stays, stretched, until re-rendered. */
  layout(scale) {
    if (this.pending && this.pending.scale !== scale) {
      this.pending.task.cancel();
      this.pending = null;
    }
    this.scale = scale;
    this.el.dataset.scale = String(scale);
    this.el.style.width = `${this.widthPt * scale}px`;
    this.el.style.height = `${this.heightPt * scale}px`;
    this.el.dataset.rendered = String(this.renderedScale === scale);
  }

  /** Render at the current scale into a new canvas, swapped in when complete. */
  render() {
    const scale = this.scale;
    if (this.pending?.scale === scale) return this.pending.promise;
    this.pending?.task.cancel(); // a render at another scale
    this.pending = null;
    if (this.renderedScale === scale) return Promise.resolve();

    const viewport = this.page.getViewport({ scale: scale / this.userUnit });
    const area = viewport.width * viewport.height;
    const output = Math.min(window.devicePixelRatio || 1, Math.sqrt(MAX_CANVAS_PIXELS / area));
    const canvas = blankCanvas();
    canvas.width = Math.max(1, Math.floor(viewport.width * output));
    canvas.height = Math.max(1, Math.floor(viewport.height * output));
    const task = this.page.render({
      canvas,
      viewport,
      transform: output === 1 ? undefined : [output, 0, 0, output, 0, 0],
    });
    const done = () => this.pending?.task === task;
    const promise = task.promise.then(
      () => {
        if (!done()) return;
        this.pending = null;
        freeCanvas(this.canvas);
        this.canvas.replaceWith(canvas);
        this.canvas = canvas;
        this.renderedScale = scale;
        this.el.dataset.rendered = "true";
      },
      (error) => {
        if (done()) this.pending = null;
        if (error?.name !== "RenderingCancelledException") throw error;
      },
    );
    this.pending = { scale, task, promise };
    return promise;
  }

  /** Drop the rendered canvas (memory) and keep the placeholder. */
  release() {
    this.pending?.task.cancel();
    this.pending = null;
    if (this.renderedScale === null) return;
    freeCanvas(this.canvas);
    this.renderedScale = null;
    this.el.dataset.rendered = "false";
    this.page.cleanup();
  }
}

/** Render the views near the viewport and release those far from it. */
export function observePages(views, onError) {
  const byElement = new Map(views.map((view) => [view.el, view]));
  const near = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const view = byElement.get(entry.target);
        view.near = entry.isIntersecting;
        if (view.near) view.render().catch(onError);
      }
    },
    { rootMargin: NEAR_MARGIN },
  );
  const far = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) byElement.get(entry.target).release();
      }
    },
    { rootMargin: FAR_MARGIN },
  );
  for (const view of views) {
    near.observe(view.el);
    far.observe(view.el);
  }
  return {
    /** After a zoom: re-render the pages that are still near. */
    refresh() {
      for (const view of views) if (view.near) view.render().catch(onError);
    },
    disconnect() {
      near.disconnect();
      far.disconnect();
    },
  };
}

function blankCanvas() {
  const canvas = document.createElement("canvas");
  canvas.className = "page-canvas";
  canvas.setAttribute("aria-hidden", "true"); // the box elements carry the labels
  canvas.width = 0;
  canvas.height = 0;
  return canvas;
}

function freeCanvas(canvas) {
  canvas.width = 0;
  canvas.height = 0;
}
