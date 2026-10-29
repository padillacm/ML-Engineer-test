// Audit viewer: sends the PDF to POST /api/extract, renders the same bytes with pdf.js and
// draws every returned box over its page (design D3, D4). Response values are written only
// with textContent and attributes: names come from OCR of untrusted documents.

import { boxRect, outsideBy, OUT_OF_PAGE_TOLERANCE_PT, pt, scaleFor } from "./geometry.mjs";
import { parseNames } from "./names.mjs";
import { closePdf, loadPdfjs, observePages, openPdf, PageView } from "./render.mjs";

const EXTRACT_URL = new URL("../../api/extract", import.meta.url).href;
const byId = (id) => document.getElementById(id);
const ui = {
  form: byId("extract-form"),
  file: byId("pdf-file"),
  names: byId("names"),
  preview: byId("names-preview"),
  problems: byId("names-problems"),
  language: byId("language"),
  extract: byId("extract"),
  cancel: byId("cancel"),
  announce: byId("names-announce"),
  status: byId("status"),
  elapsed: byId("elapsed"),
  requestId: byId("request-id"),
  error: byId("error"),
  pageWarnings: byId("page-warnings"),
  zoom: byId("zoom"),
  filter: byId("filter"),
  labels: byId("labels"),
  download: byId("download-json"),
  viewer: byId("viewer"),
  list: byId("name-list"),
  listEmpty: byId("list-empty"),
  details: byId("details"),
  pages: byId("pages"),
};

let names = parseNames("");
let pdfjsReady = false;
let run = null; // { controller, timer, started, fileName } of the extraction in progress
let raw = null; // { text, requestId } of the last response with a JSON body
let current = null; // { doc, views, entries, observer }
let selected = null; // entry
let announceTimer = 0;
let announced = "";

const ANNOUNCE_DELAY_MS = 1000; // the names' problems are read out once typing pauses
const TICK_MS = 250;

// ---------- form ----------

function updateNames() {
  names = parseNames(ui.names.value);
  ui.preview.textContent = names.names ? JSON.stringify(names.names, null, 2) : "–";
  ui.problems.replaceChildren(...names.problems.map((text) => item(text)));
  ui.names.setAttribute("aria-invalid", String(names.blocking));
  updateButton();
  clearTimeout(announceTimer);
  announceTimer = setTimeout(announceProblems, ANNOUNCE_DELAY_MS);
}

/** Put the names' problems into the polite live region, only when they changed (not per key). */
function announceProblems() {
  clearTimeout(announceTimer);
  const text = names.problems.length ? `Names: ${names.problems.join("; ")}` : "";
  if (text === announced) return;
  announced = text;
  ui.announce.textContent = text || (ui.names.value.trim() ? "Names: no problems" : "");
}

function updateButton() {
  ui.extract.disabled = Boolean(run) || !pdfjsReady || !ui.file.files.length || names.blocking;
}

async function extract() {
  const file = ui.file.files[0];
  if (names.blocking) announceProblems();
  if (!file || names.blocking || run) return;
  clearResult();
  const controller = new AbortController();
  const { signal } = controller;
  const started = performance.now();
  const tick = () => {
    ui.elapsed.textContent = seconds(performance.now() - started);
  };
  // The live status is set at the start, the end, an error or a cancel; the ticking time is
  // in a separate element that is not announced.
  run = { controller, started, fileName: file.name, timer: setInterval(tick, TICK_MS) };
  tick();
  setStatus(`Extracting ${file.name}, ${formatBytes(file.size)}…`);
  setLoading(true);
  const bytes = await file.arrayBuffer(); // read once: the API and pdf.js get the same bytes
  if (signal.aborted) return;
  const body = new FormData();
  body.append("pdf_file", new Blob([bytes], { type: file.type || "application/pdf" }), file.name);
  body.append("names", JSON.stringify(names.names));
  if (ui.language.value !== "auto") body.append("language", ui.language.value);
  const pdf = openPdf(new Uint8Array(bytes.slice(0))); // a copy: pdf.js transfers its buffer
  pdf.catch(() => {}); // awaited below, or dropped on an API error

  let response;
  let text;
  try {
    response = await fetch(EXTRACT_URL, {
      method: "POST",
      body,
      cache: "no-store",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      signal,
    });
    text = await response.text();
  } catch (error) {
    discard(pdf);
    if (signal.aborted) return; // cancel() has reset the page
    return fail(`Network error: ${error.message}`, null, performance.now() - started);
  }
  if (signal.aborted) return discard(pdf);
  const elapsed = performance.now() - started;
  const payload = parseJson(text);
  const requestId = requestIdOf(response, payload);
  ui.requestId.textContent = requestId ?? "–";
  if (payload !== undefined) {
    raw = { text, requestId };
    ui.download.disabled = false;
  }

  if (!response.ok) {
    discard(pdf);
    const code = typeof payload?.code === "string" ? payload.code : null;
    return fail(apiMessage(payload, response.status), code, elapsed, response.status);
  }
  if (!isExtraction(payload)) {
    discard(pdf);
    return fail("The response is not an extraction result", null, elapsed, response.status);
  }
  let opened;
  try {
    opened = await pdf;
  } catch (error) {
    if (signal.aborted) return;
    return fail(`pdf.js could not open the file: ${error.message}`, null, elapsed);
  }
  if (signal.aborted) return closePdf(opened.doc);
  const fileName = endRun();
  show(payload, opened);
  const matched = current.entries.filter((entry) => entry.match).length;
  const warnings = current.entries.filter((entry) => entry.warning).length;
  const pageWarnings = current.views.filter((view) => view.warnings.length).length;
  setStatus(
    `Done in ${seconds(elapsed)} (${fileName}): ${plural(current.entries.length, "name")} on ` +
      `${plural(opened.pages.length, "page")}, ${matched} matched` +
      (warnings ? `, ${plural(warnings, "audit warning")}` : "") +
      (pageWarnings ? `, ${plural(pageWarnings, "page")} with an audit warning` : ""),
  );
  setLoading(false, "done");
}

/** End the extraction in progress (its timer and controller); its file name. */
function endRun() {
  if (!run) return null;
  const { timer, fileName } = run;
  clearInterval(timer);
  run = null;
  ui.elapsed.textContent = "";
  return fileName;
}

/** Abort the extraction in progress; the page goes back to idle and says it was cancelled. */
function cancel() {
  if (!run) return;
  const { controller, started } = run;
  const fileName = endRun();
  controller.abort();
  setStatus(`Cancelled after ${seconds(performance.now() - started)} (${fileName}).`);
  setLoading(false, "idle");
}

function fail(message, code, elapsed, status) {
  const fileName = endRun();
  ui.error.replaceChildren();
  if (code) {
    ui.error.dataset.code = code;
    const strong = document.createElement("strong");
    strong.textContent = code;
    ui.error.append(strong, ": ");
  } else {
    delete ui.error.dataset.code;
  }
  ui.error.append(message);
  ui.error.hidden = false;
  setStatus(
    `Failed${status ? ` with HTTP ${status}` : ""} after ${seconds(elapsed)}` +
      (fileName ? ` (${fileName})` : ""),
  );
  setLoading(false, "error");
}

/** While loading, the inputs are locked (the result belongs to the file sent) and Cancel shows. */
function setLoading(isLoading, state = "loading") {
  document.body.dataset.state = state;
  for (const control of [ui.file, ui.names, ui.language]) control.disabled = isLoading;
  ui.cancel.hidden = !isLoading;
  ui.cancel.disabled = !isLoading;
  updateButton();
}

function clearResult() {
  if (current) {
    current.observer.disconnect();
    closePdf(current.doc);
  }
  current = null;
  selected = null;
  raw = null;
  ui.download.disabled = true;
  ui.requestId.textContent = "–";
  ui.error.hidden = true;
  ui.error.replaceChildren();
  delete ui.error.dataset.code;
  ui.pageWarnings.hidden = true;
  ui.pageWarnings.replaceChildren();
  ui.pages.replaceChildren();
  ui.list.replaceChildren();
  ui.listEmpty.hidden = false;
  ui.listEmpty.textContent = "No extraction yet.";
  showDetails(null);
}

// ---------- result ----------

function show(payload, { doc, pages }) {
  const best = bestMatches(payload.fuzzy_matches ?? []);
  const views = pages.map((info) => new PageView(info));
  const entries = payload.extracted_names.map((extracted, index) => {
    const box = extracted.bounding_box;
    const view = views[box.page_number] ?? null;
    const entry = {
      id: `box-${index}`,
      name: extracted.name,
      box,
      view,
      match: best.get(extracted.name) ?? null,
      warning: null,
    };
    if (!view) entry.warning = `page ${box.page_number} is not in the PDF`;
    else {
      const outside = outsideBy(box, view.widthPt, view.heightPt);
      if (outside > OUT_OF_PAGE_TOLERANCE_PT) {
        entry.warning = `the box lies ${pt(outside)} pt outside the page`;
      }
    }
    entry.el = view ? boxElement(entry) : null;
    return entry;
  });

  for (const view of views) {
    const frame = document.createElement("figure");
    frame.className = "page-frame";
    const caption = document.createElement("figcaption");
    caption.textContent = `Page ${view.index + 1} (page_number ${view.index})`;
    if (view.warnings.length) {
      frame.dataset.pageWarning = view.warnings.map((warning) => warning.kind).join(" ");
      for (const warning of view.warnings) {
        const note = document.createElement("span");
        note.className = "page-warning";
        note.textContent = `Audit warning: ${warning.text}`;
        caption.append(note);
        const li = item(`Page ${view.index + 1} (page_number ${view.index}): ${warning.text}`);
        li.dataset.page = String(view.index);
        li.dataset.kind = warning.kind;
        ui.pageWarnings.append(li);
      }
      ui.pageWarnings.hidden = false;
    }
    frame.append(caption, view.el);
    ui.pages.append(frame);
  }
  // Box elements in page order, then response order: the Tab order (spec: keyboard use).
  for (const entry of entries) entry.view?.overlay.append(entry.el);

  current = { doc, views, entries, byId: new Map(entries.map((entry) => [entry.id, entry])) };
  buildList(entries, views.length);
  relayout(false);
  current.observer = observePages(views, (error) => console.warn("page render failed", error));
  applyFilter();
}

function boxElement(entry) {
  const el = document.createElement("div");
  el.className = "box";
  el.id = entry.id;
  el.tabIndex = 0;
  el.setAttribute("role", "button");
  el.setAttribute("aria-label", describe(entry));
  const { box } = entry;
  el.dataset.page = String(box.page_number);
  el.dataset.x = String(box.x);
  el.dataset.y = String(box.y);
  el.dataset.w = String(box.width);
  el.dataset.h = String(box.height);
  el.dataset.matched = String(Boolean(entry.match));
  if (entry.warning) el.dataset.warning = "out-of-page";
  const tag = document.createElement("span");
  tag.className = "box-label";
  tag.setAttribute("aria-hidden", "true");
  tag.textContent = entry.match ? "matched" : "no match";
  el.append(tag);
  return el;
}

function buildList(entries, pageCount) {
  const byPage = new Map();
  for (const entry of entries) {
    const page = entry.box.page_number;
    if (!byPage.has(page)) byPage.set(page, []);
    byPage.get(page).push(entry);
  }
  for (const page of [...byPage.keys()].sort((a, b) => a - b)) {
    const group = byPage.get(page);
    const heading = document.createElement("li");
    heading.className = "list-page";
    heading.dataset.page = String(page);
    const where = page < pageCount ? "" : ", not in the PDF";
    heading.textContent = `Page ${page + 1} (page_number ${page}${where}): ${plural(group.length, "name")}`;
    ui.list.append(heading);
    for (const entry of group) ui.list.append(listItem(entry));
  }
  ui.listEmpty.hidden = entries.length > 0;
  ui.listEmpty.textContent = "No names found in this PDF.";
}

function listItem(entry) {
  const li = document.createElement("li");
  li.className = "list-name";
  li.dataset.boxId = entry.id;
  li.dataset.matched = String(Boolean(entry.match));
  const button = document.createElement("button");
  button.type = "button";
  const name = document.createElement("span");
  name.className = "item-name";
  name.textContent = entry.name;
  const match = document.createElement("span");
  match.className = "item-match";
  match.textContent = entry.match
    ? `matched: ${entry.match.matched_name} (${entry.match.score.toFixed(3)})`
    : "no match";
  button.append(name, match);
  li.append(button);
  if (entry.warning) {
    li.dataset.warning = "out-of-page";
    const warning = document.createElement("span");
    warning.className = "item-warning";
    warning.textContent = `Audit warning: ${entry.warning}`;
    li.append(warning);
  }
  entry.li = li;
  return li;
}

function bestMatches(matches) {
  const best = new Map();
  for (const match of matches) {
    const known = best.get(match.extracted_name);
    if (!known || match.score > known.score) best.set(match.extracted_name, match);
  }
  return best;
}

function describe(entry) {
  const { box, match } = entry;
  const parts = [
    entry.name,
    `page ${box.page_number}`,
    match ? `matched to ${match.matched_name}, score ${match.score.toFixed(3)}` : "no match",
    `box x ${pt(box.x)}, y ${pt(box.y)}, width ${pt(box.width)}, height ${pt(box.height)} points`,
  ];
  if (entry.warning) parts.push(`audit warning: ${entry.warning}`);
  for (const warning of entry.view?.warnings ?? []) parts.push(`page audit warning: ${warning.text}`);
  return parts.join("; ");
}

// ---------- layout, zoom, filter ----------

function relayout(keepPosition = true) {
  if (!current) return;
  const anchor = keepPosition ? scrollAnchor() : null;
  const style = getComputedStyle(ui.pages);
  const available =
    ui.pages.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
  const widest = Math.max(...current.views.map((view) => view.widthPt));
  const scale = scaleFor(ui.zoom.value, available, widest);
  for (const view of current.views) view.layout(scale);
  for (const entry of current.entries) {
    if (!entry.el) continue;
    const rect = boxRect(entry.box, entry.view.scale);
    entry.el.style.left = `${rect.left}px`;
    entry.el.style.top = `${rect.top}px`;
    entry.el.style.width = `${rect.width}px`;
    entry.el.style.height = `${rect.height}px`;
  }
  if (anchor) restoreAnchor(anchor);
  current.observer?.refresh();
}

/** The first page in view and how far down it the viewport starts. */
function scrollAnchor() {
  for (const view of current.views) {
    const rect = view.el.getBoundingClientRect();
    if (rect.bottom > 0) return { view, fraction: Math.max(0, -rect.top) / rect.height };
  }
  return null;
}

function restoreAnchor({ view, fraction }) {
  const rect = view.el.getBoundingClientRect();
  if (rect.top < 0 || fraction > 0) window.scrollBy(0, rect.top + fraction * rect.height);
}

function applyFilter() {
  const filter = ui.filter.value;
  ui.viewer.dataset.filter = filter;
  if (!current) return;
  const visible = (entry) => filter === "all" || (filter === "matched") === Boolean(entry.match);
  for (const entry of current.entries) {
    const shown = visible(entry);
    if (entry.el) entry.el.hidden = !shown;
    entry.li.hidden = !shown;
  }
  for (const heading of ui.list.querySelectorAll(".list-page")) {
    const page = Number(heading.dataset.page);
    heading.hidden = !current.entries.some((e) => e.box.page_number === page && visible(e));
  }
}

let resizeFrame = 0;
let lastWidth = 0;
new ResizeObserver(([entry]) => {
  const width = entry.contentRect.width;
  if (width === lastWidth) return;
  lastWidth = width;
  if (ui.zoom.value !== "fit") return;
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => relayout());
}).observe(ui.pages);

// ---------- selection and details ----------

function select(entry, scroll) {
  for (const el of document.querySelectorAll(".selected")) {
    el.classList.remove("selected");
    el.removeAttribute("aria-current");
  }
  selected = entry;
  for (const el of [entry.el, entry.li]) {
    if (!el) continue;
    el.classList.add("selected");
    el.setAttribute("aria-current", "true");
  }
  showDetails(entry);
  if (scroll && entry.el) entry.el.scrollIntoView({ block: "center", inline: "center" });
  revealInList(entry.li);
}

/** Scroll the list's own scroll container (not the window) so that `li` is visible. */
function revealInList(li) {
  const side = scrollParent(li);
  if (!side) return;
  const outer = side.getBoundingClientRect();
  const inner = li.getBoundingClientRect();
  if (inner.top < outer.top) side.scrollTop -= outer.top - inner.top;
  else if (inner.bottom > outer.bottom) side.scrollTop += inner.bottom - outer.bottom;
}

function scrollParent(el) {
  for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
    const overflow = getComputedStyle(node).overflowY;
    if ((overflow === "auto" || overflow === "scroll") && node.scrollHeight > node.clientHeight) {
      return node;
    }
  }
  return null;
}

function showDetails(entry) {
  if (!entry) {
    delete ui.details.dataset.boxId;
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "Select, hover or focus a name to see its box.";
    ui.details.replaceChildren(hint);
    return;
  }
  const { box, match } = entry;
  const title = document.createElement("h3");
  title.textContent = entry.name;
  const list = document.createElement("dl");
  const rows = [
    ["Status", match ? "matched" : "no match"],
    ["Page (0-based)", String(box.page_number)],
    ["Box (pt)", `x ${pt(box.x)}, y ${pt(box.y)}, w ${pt(box.width)}, h ${pt(box.height)}`],
    ["Best match", match ? match.matched_name : "–"],
    ["Score", match ? match.score.toFixed(3) : "–"],
  ];
  if (entry.warning) rows.push(["Audit warning", entry.warning]);
  for (const warning of entry.view?.warnings ?? []) rows.push(["Page audit warning", warning.text]);
  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
  ui.details.dataset.boxId = entry.id;
  ui.details.replaceChildren(title, list);
}

const entryOf = (target) => {
  const el = target.closest?.(".box, li[data-box-id]");
  return el && current?.byId.get(el.id || el.dataset.boxId);
};

ui.pages.addEventListener("click", (event) => {
  const entry = entryOf(event.target);
  if (entry) select(entry, false);
});
ui.pages.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const entry = entryOf(event.target);
  if (!entry) return;
  event.preventDefault();
  select(entry, false);
});
for (const type of ["mouseover", "focusin"]) {
  ui.pages.addEventListener(type, (event) => {
    const entry = entryOf(event.target);
    if (entry) showDetails(entry);
  });
}
for (const type of ["mouseout", "focusout"]) {
  ui.pages.addEventListener(type, (event) => {
    if (entryOf(event.target) && !event.relatedTarget?.closest?.(".box")) showDetails(selected);
  });
}
ui.list.addEventListener("click", (event) => {
  const entry = entryOf(event.target);
  if (entry) select(entry, true);
});

// ---------- download ----------

function download() {
  if (!raw) return;
  const url = URL.createObjectURL(new Blob([raw.text], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `extract-${(raw.requestId ?? "response").replace(/[^\w.-]/g, "_")}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- helpers ----------

function isExtraction(payload) {
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  return (
    Array.isArray(payload?.extracted_names) &&
    payload.extracted_names.every(
      (entry) =>
        typeof entry?.name === "string" &&
        Number.isInteger(entry.bounding_box?.page_number) &&
        entry.bounding_box.page_number >= 0 &&
        ["x", "y", "width", "height"].every((key) => finite(entry.bounding_box[key])),
    ) &&
    (payload.fuzzy_matches === undefined ||
      (Array.isArray(payload.fuzzy_matches) &&
        payload.fuzzy_matches.every(
          (match) =>
            typeof match?.extracted_name === "string" &&
            typeof match.matched_name === "string" &&
            finite(match.score),
        )))
  );
}

/** The error text of a response body: `detail` as a string, or a validation list (FastAPI's
 * `[{loc, msg}]`) read defensively, since the body is not trusted to have that shape. */
function apiMessage(payload, status) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const text = detail.map(detailEntry).filter(Boolean).join("; ");
    if (text) return text;
  }
  return `HTTP ${status}`;
}

function detailEntry(entry) {
  if (typeof entry === "string") return entry;
  const loc = Array.isArray(entry?.loc) ? entry.loc.map(asText).join(".") : "";
  const msg = asText(entry?.msg);
  return loc && msg ? `${loc}: ${msg}` : loc || msg;
}

function asText(value) {
  if (typeof value === "string") return value;
  if (value === undefined || value === null) return "";
  return JSON.stringify(value) ?? "";
}

/** The response's request id: its header, else a string or number `request_id` of the body. */
function requestIdOf(response, payload) {
  const header = response.headers.get("X-Request-ID");
  if (header) return header;
  const value = payload?.request_id;
  if (typeof value === "string" && value) return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function discard(pdf) {
  pdf.then(({ doc }) => closePdf(doc), () => {});
}

function setStatus(text) {
  ui.status.textContent = text;
}

function item(text) {
  const li = document.createElement("li");
  li.textContent = text;
  return li;
}

const seconds = (ms) => `${(ms / 1000).toFixed(2)} s`;
const plural = (count, noun) => `${count} ${noun}${count === 1 ? "" : "s"}`;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// ---------- start ----------

ui.names.addEventListener("input", updateNames);
ui.file.addEventListener("change", updateButton);
ui.cancel.addEventListener("click", cancel);
ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  extract().catch((error) => fail(`Unexpected error: ${error.message}`, null, 0));
});
ui.zoom.addEventListener("change", () => relayout());
ui.filter.addEventListener("change", applyFilter);
ui.labels.addEventListener("change", () => {
  ui.viewer.dataset.labels = ui.labels.checked ? "on" : "off";
});
ui.download.addEventListener("click", download);
updateNames();

const ready = loadPdfjs().then(
  () => {
    pdfjsReady = true;
    setStatus("Choose a PDF.");
    updateButton();
  },
  () => {
    ui.error.textContent =
      "pdf.js is not available on this server (/audit/vendor/pdfjs/); see the README to fetch it.";
    ui.error.hidden = false;
    setStatus("The viewer cannot render PDFs.");
    document.body.dataset.state = "error";
  },
);

/** Test helpers (read-only): scroll to every page in turn and wait for its rendering. */
async function scrollAllPages() {
  if (!current) return;
  for (const view of current.views) {
    view.el.scrollIntoView({ block: "start" });
    await view.render();
  }
}

Object.defineProperty(window, "__auditViewer", {
  value: Object.freeze({ scrollAllPages, ready }),
  enumerable: false,
  writable: false,
  configurable: false,
});
