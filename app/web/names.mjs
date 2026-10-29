// The names textarea -> the JSON array POST /api/extract takes in its `names` field.
//
// One name per line: "First Last" (the last word is the last name) or "Last, First".
// A text starting with "[" or "{" is read as JSON: an array is sent as it is, so the API stays
// the judge of its entries (those it would reject are only flagged here); anything else blocks.

export const MAX_NAME_CHARS = 100; // as app/api/uploads.py

/** @returns {{ names: object[] | null, problems: string[], blocking: boolean }} */
export function parseNames(text) {
  const trimmed = text.trim();
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) return parseJson(trimmed);
  const names = [];
  const problems = [];
  text.split(/\r?\n/).forEach((raw, index) => {
    const line = raw.trim();
    if (!line) return;
    const result = parseLine(line);
    if (typeof result === "string") problems.push(`Line ${index + 1}: ${result}`);
    else names.push(result);
  });
  return { names: problems.length ? null : names, problems, blocking: problems.length > 0 };
}

function parseLine(line) {
  let first;
  let last;
  if (line.includes(",")) {
    const parts = line.split(",");
    if (parts.length > 2) return 'use one comma at most ("Last, First")';
    [last, first] = parts.map((part) => part.trim().replace(/\s+/g, " "));
  } else {
    const words = line.split(/\s+/);
    last = words.pop();
    first = words.join(" ");
  }
  if (!first || !last) return 'needs a first and a last name ("First Last" or "Last, First")';
  if (first.length > MAX_NAME_CHARS || last.length > MAX_NAME_CHARS) {
    return `each part must be at most ${MAX_NAME_CHARS} characters`;
  }
  return { first_name: first, last_name: last };
}

function parseJson(text) {
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    return { names: null, problems: [`Not valid JSON: ${error.message}`], blocking: true };
  }
  if (!Array.isArray(value)) {
    return { names: null, problems: ["The JSON must be an array"], blocking: true };
  }
  const problems = [];
  value.forEach((entry, index) => {
    const ok =
      entry !== null &&
      typeof entry === "object" &&
      ["first_name", "last_name"].every(
        (key) =>
          typeof entry[key] === "string" &&
          entry[key].trim() !== "" &&
          entry[key].trim().length <= MAX_NAME_CHARS,
      );
    if (!ok) problems.push(`Entry ${index}: needs non-blank first_name and last_name strings`);
  });
  // Sent anyway: the API reports the error with its own code.
  return { names: value, problems, blocking: false };
}
