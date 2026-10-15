# Project context

# Conventions

- Let class and function names speak for themselves.
- Use banner/section comments (e.g. `# -------- Section --------`) sparsely.
- Fail loudly, don't paper over. Throw on unexpected/missing data; don't return empty strings, fake defaults, or silent fallbacks.
- Do not duplicate code if possible. Share generic and recurring functionalities (single source of truth).
- When you notice recurring patterns, missing conventions, or other problems during a session, propose updates to this file.
- Keep docs converging toward the actual codebase practices.
- Keep comments concise and focus on the key concepts that help understand the code. It's not always necessary to comment on something just because it was the last thing you were prompted to do.
- Everything must be strongly typed whenever possible (e.g., no `Any`, `object`, or `cast`).

# Commands

see @README.md

# Workflow

- After modifying files, run `pre-commit run --files <modified files>` to auto-fix formatting and lint issues.
- Propose tests for newly added functions.
- All checks (ruff, pyright, pytest) must pass before merging to `main`.
