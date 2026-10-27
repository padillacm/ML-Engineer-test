# Conventional Commits 1.0.0 — reference

Source: <https://www.conventionalcommits.org/en/v1.0.0/#specification>. The numbered rules below
are a summary of the specification; where a rule says MUST, the validator enforces it.

## Specification rules

1. A commit MUST be prefixed with a type (a noun such as `feat`, `fix`), followed by an OPTIONAL
   scope, an OPTIONAL `!`, and a REQUIRED `: ` (colon and space).
2. `feat` MUST be used when a commit adds a new feature.
3. `fix` MUST be used when a commit represents a bug fix.
4. A scope MAY follow the type: a noun describing a section of the codebase, in parentheses,
   e.g. `fix(parser):`.
5. A description MUST immediately follow the `: ` and is a short summary of the change.
6. A longer body MAY follow, beginning one blank line after the description.
7. The body is free-form and MAY consist of any number of newline-separated paragraphs.
8. One or more footers MAY follow one blank line after the body. Each footer is a word token,
   then `: ` or ` #`, then a string value (inspired by git trailers).
9. A footer token MUST use `-` instead of whitespace (e.g. `Acked-by`), except
   `BREAKING CHANGE`.
10. A footer value MAY contain spaces and newlines; parsing ends at the next valid
    token/separator pair.
11. Breaking changes MUST be indicated in the type/scope prefix or as a footer.
12. As a footer: uppercase `BREAKING CHANGE`, then `: `, then a description.
13. In the prefix: `!` immediately before the `:`. If `!` is used, the `BREAKING CHANGE:` footer
    MAY be omitted and the description SHALL describe the breaking change.
14. Types other than `feat` and `fix` MAY be used.
15. Units of information are case-insensitive, except `BREAKING CHANGE`, which MUST be uppercase.
16. `BREAKING-CHANGE` is a synonym of `BREAKING CHANGE` as a footer token.

SemVer mapping: `fix` → PATCH, `feat` → MINOR, any `BREAKING CHANGE` → MAJOR (any type).

## House conventions (stricter than the spec)

These come from common practice (Angular/commitlint `config-conventional`, Chris Beams' "How to
Write a Git Commit Message") and are what this repo uses:

- types limited to: `feat fix docs style refactor perf test build ci chore revert`;
- type and scope lowercase;
- description lowercase-first, imperative, no trailing period;
- header ≤ 72 characters; body wrapped at 72;
- breaking changes carry both `!` and a `BREAKING CHANGE:` footer with migration notes;
- one logical change per commit; split when more than one type fits.

## Examples

Header only:

```
docs: correct spelling of CHANGELOG
```

Scope:

```
feat(api): add /extract endpoint for PDF uploads
```

Body and footer:

```
fix(domain): merge overlapping bounding boxes before matching

Two boxes that overlapped by less than one point were treated as
disjoint, so a value split across them was never matched to its
label. Merge them when IoU > 0 before running the matcher.

Refs: #42
```

Breaking change:

```
feat(api)!: return citations as a list of spans

The `citation` field is replaced by `citations`, a list of
{page, bbox, text} objects, so one answer can cite several regions.

BREAKING CHANGE: clients reading `citation` must read `citations[0]`
or iterate over `citations`.
```

Dependencies:

```
build(deps): add chromadb for embedded vector storage
```

Revert:

```
revert: let us never again speak of the noodle incident

Refs: 676104e, a215868
```

## Anti-patterns

| Bad | Why | Better |
|---|---|---|
| `Update files` | no type, says nothing | `refactor(domain): extract bbox helpers into geometry` |
| `feat: Added chunking.` | past tense, capitalised, period | `feat(domain): add page-aware text chunking` |
| `fix: stuff` | vague | `fix(api): return 422 when the upload is not a PDF` |
| `feat: add rag endpoint and fix typo in README and bump ruff` | three changes, three types | three commits: `feat(api)`, `docs`, `build(deps)` |
| `feat(feat): ...` | scope repeats the type | pick the code area, or drop the scope |
| `breaking change: ...` in footer | must be uppercase | `BREAKING CHANGE: ...` |
