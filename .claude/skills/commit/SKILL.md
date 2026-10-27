---
name: commit
description: Create git commits from the current working-tree changes, with messages that follow Conventional Commits 1.0.0 (type(scope)!: description, body, footers). Groups unrelated changes into separate atomic commits, picks the type and scope from the diff, validates every message, and always shows the proposal for the user to approve or correct before committing. Use when the user asks to commit, make a commit, "haz commit", "crea el commit", split changes into commits, or write a commit message.
argument-hint: "[optional hint: scope, intent or 'single' to force one commit]"
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git branch:*), Bash(git add:*), Bash(git restore --staged:*), Bash(git commit:*), Bash(git show:*), Bash(uv run --no-sync python .claude/skills/commit/scripts/validate_commit_msg.py:*)
---

Turn the pending changes into one or more commits whose messages follow
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/#specification).
The full rule set and examples are in [reference.md](reference.md); read it when a case is not
covered below (breaking changes, footers, reverts).

The user's arguments, if any: `$ARGUMENTS`. Treat them as a hint (scope, intent, "single"),
never as a reason to skip the steps below.

## 1. Inspect

Run these together and read the output before deciding anything:

```bash
git branch --show-current
git status --short
git diff --staged --stat
git diff --stat
git log --oneline -10
```

Then read the actual diffs (`git diff --staged`, `git diff -- <path>`) for the files involved.
For untracked files, read the files themselves. Never write a message from file names alone.

Stop and tell the user when:
- the branch is `main` or `master` (the `no-commit-to-main` pre-commit hook rejects it anyway);
- there is nothing to commit;
- the diff contains something that looks like a secret, a `.env` file, a large binary or
  generated output (`__pycache__/`, `.coverage`, build dirs) that is not gitignored.

## 2. Respect what is already staged

- **Something is staged** → the user curated it: commit exactly the staged set as one commit.
  Do not add or remove files. Mention unstaged leftovers at the end.
- **Nothing is staged** → you decide the grouping (step 3).

## 3. Group into atomic commits

One commit = one logical change that has one type. If the changes fit more than one type, split
them (the spec's FAQ: "go back and make multiple commits whenever possible").

Heuristics for this repo:
- Production code and its tests go in the **same** commit (`feat`/`fix` + its tests).
- Tests added alone for existing code → `test`.
- `pyproject.toml` + `uv.lock` dependency changes → their own `build(deps)` commit, unless the
  dependency exists only for a feature in the same change.
- Tooling config (`.pre-commit-config.yaml`, `.gitattributes`, `.gitignore`, `.python-version`,
  ruff/mypy settings) → `build` or `chore`; CI workflows → `ci`.
- `openspec/` changes → `docs(openspec)` when they are spec/planning only.
- `.claude/` (skills, commands) → `chore(claude)`.
- Pure formatting with no behaviour change → `style`; restructuring with no behaviour change →
  `refactor`.

Staging granularity is whole files (interactive `git add -p` is unavailable). If one file mixes
two concerns, keep it in the commit of its dominant change and say so in that commit's body.

With the hint `single`, make one commit and pick the dominant type. Whatever the plan, nothing
is committed before the user approves it in step 5.

## 4. Write the message

```
<type>[(<scope>)][!]: <description>

[body]

[footer(s)]
```

**Type** (`feat` and `fix` are defined by the spec; the rest follow the widely used Angular set):

| type | when |
|---|---|
| `feat` | new capability for the user/API (SemVer MINOR) |
| `fix` | bug fix (SemVer PATCH) |
| `docs` | documentation or specs only |
| `style` | formatting, whitespace; no code meaning change |
| `refactor` | code change that neither fixes a bug nor adds a feature |
| `perf` | performance improvement |
| `test` | adding or correcting tests only |
| `build` | build system, packaging, dependencies (`pyproject.toml`, `uv.lock`) |
| `ci` | CI configuration and scripts |
| `chore` | maintenance that touches no src or test code |
| `revert` | reverts a previous commit |

**Scope** — optional, a lowercase noun naming the area, in parentheses. Prefer the ones this repo
already has: `domain`, `application`, `adapters`, `api`, `services`, `models`, `tests`,
`openspec`, `benchmarks`, `deps`, `claude`. Omit it when the change spans the whole repo; never
invent a scope that just repeats the type.

**Description** (the spec requires it right after `: `):
- imperative mood, present tense: "add", not "added"/"adds";
- lowercase first letter, no trailing period;
- whole header ≤ 72 characters (aim for ≤ 50);
- says *what* changed in terms a reader of `git log --oneline` understands; no "update files",
  "fix stuff", "WIP".

**Body** — after one blank line. Explain *why* and any non-obvious *how*; wrap at 72 columns;
bullets are fine. Skip it when the header says everything.

**Footers** — after one blank line, `Token: value` or `Token #value`, tokens hyphenated
(`Refs: #12`, `Reviewed-by: Name`). The only token with a space is `BREAKING CHANGE`.

**Breaking changes** — anything that breaks a public API, CLI, config, data or file format: add
`!` before the colon **and** a `BREAKING CHANGE: <what breaks and how to migrate>` footer
(uppercase, required spelling).

**Language** — English, matching the repository history, regardless of the language the user
writes to you in.

**Attribution** — if the session's instructions ask for a `Co-Authored-By:` trailer, put it as
the last footer.

## 5. Validate, then let the user review and correct

This step is **mandatory, every time**, even for a single obvious commit and even when the user
said "just commit". Invoking the skill is a request to *prepare* commits, not to skip review.

1. Write each message to a temp file in the scratchpad (never into the repo) and validate it:
   ```bash
   uv run --no-sync python .claude/skills/commit/scripts/validate_commit_msg.py <msg-file>
   ```
   Fix the message until it passes, so the user only ever reviews valid messages.
2. Show the user the complete proposal, for each commit in order:
   - the files it will include (and any file left out, with the reason);
   - the **full message** in a fenced block, exactly as it will be committed (header, body,
     footers);
   - one line on why you chose that type and scope.
3. Ask for a decision with the `AskUserQuestion` tool (fall back to a plain question if it is
   unavailable), offering:
   - **Approve** — commit as shown;
   - **Edit** — the user says what to change (wording, type, scope, body, breaking change,
     which files go in which commit, merge or split commits);
   - **Cancel** — stop without committing; leave the staging area as it was before the skill
     ran.
4. On **Edit**: apply the corrections, re-validate, and show the **whole updated proposal**
   again with the same three options. Repeat until the user approves or cancels. If a requested
   change breaks the spec (e.g. a capitalised description or a type outside the list), say
   which rule it breaks and propose the closest valid form instead; commit a non-conforming
   message only if the user explicitly insists after that.

Only an explicit approval of the version currently shown counts. If the plan changes in any
way after approval (a hook rewrites files, a file moves between commits), show it and ask
again.

## 6. Stage and commit

For each approved commit, in order:

1. Stage only that commit's files by explicit path: `git add -- <path> ...`.
   Never `git add -A`, `git add .` or `git commit -a`.
2. `git commit -F <msg-file>`.
3. If a pre-commit hook fails: read its output, fix the cause (ruff-format may have rewritten
   files — re-stage them), and run `git commit -F <msg-file>` again. Never `--no-verify`, never
   `--amend` an earlier commit to absorb the fix, never disable hooks. If fixing it required
   changing code (not just re-staging formatter output), tell the user what changed before
   retrying.

Do not push, rebase, amend or reset unless the user asks for it explicitly.

## 7. Report

Finish with `git log --oneline -<n>` for the commits you made and `git status --short`, and tell
the user what is still uncommitted, if anything.
