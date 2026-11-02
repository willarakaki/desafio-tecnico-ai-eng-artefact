---
name: tdd-commit-gate
description: Manages per-task Git branches, runs Ruff linting/formatting, enforces TDD validation via pytest, pushes the task branch to GitHub so the user can open a Pull Request manually, and generates the prompt for the next task. Trigger this skill when starting or finishing a task.
---

# TDD Quality Gate, Ruff Linter & Branch Push for Manual PR

Follow this sequential lifecycle for every task in `TASKS.md`:

## Phase A: Sync `main` & Create Task Branch (At Task Start)
1. Sync with the remote `main` branch first:
   - `git checkout main`
   - `git pull origin main` (if remote `origin/main` already exists)
2. Create and switch to the dedicated branch for the active task:
   - `Task 0` -> `git checkout -b chore/task-0-setup-ci-docker`
   - `Task 1` -> `git checkout -b feat/task-1-config-llm-factory`
   - `Task 2` -> `git checkout -b feat/task-2-duckdb-csv-tools`
   - `Task 3` -> `git checkout -b feat/task-3-hybrid-rag-pdf`
   - `Task 4` -> `git checkout -b feat/task-4-security-and-cache`
   - `Task 5` -> `git checkout -b feat/task-5-langgraph-orchestrator`
   - `Task 6` -> `git checkout -b feat/task-6-cli-and-streamlit-ui`
   - `Task 7` -> `git checkout -b docs/task-7-conversation-transcripts`
   - `Task 8` -> `git checkout -b docs/task-8-executive-readme`

## Phase B: Architecture, Security & Ruff Audit (Before Commit)
1. Verify that no real API keys (`AIza...`, `sk-...`) are hardcoded or tracked by Git (`git status`). Ensure `.env` is ignored.
2. Confirm that no LLM client is instantiated outside `src/agent/llm_factory.py`.
3. Confirm that no DuckDB SQL query uses raw string interpolation without parameterized bindings.
4. Run Ruff linter and formatter:
   - `uv run ruff check --fix .`
   - `uv run ruff format .`

## Phase C: Automated Test Suite Execution (TDD)
1. Run `uv run pytest -v` (or `pytest -v`).
2. If any test or linter check fails, fix the root cause and re-run until 100% pass.

## Phase D: Commit & Push Branch (User Opens PR Manually)
1. Mark the completed task and its sub-items with `- [x]` in `TASKS.md`.
2. Show `git status` and the semantic commit message from `TASKS.md`, then request confirmation to run ONLY:
   - `git add .`
   - `git commit -m "<semantic-commit-message-from-TASKS.md>"`
   - `git push -u origin <task-branch-name>`
3. Do NOT merge into `main` locally. Instruct the user to open the Pull Request on GitHub, wait for the GitHub Actions CI check to pass, and click "Merge pull request".

## Phase E: Mandatory Next-Task Prompt Output
After pushing the branch for `Task X`, inspect `TASKS.md` to find the next pending `[ ]` task (`Task X+1`) and print this exact block at the end of your response:

---
**NEXT TASK PROMPT (Copy and paste into a new chat AFTER you merge the PR on GitHub):**

Read @AGENTS.md, @docs/architecture.md, @TASKS.md, and @.agents/skills/tdd-commit-gate/SKILL.md.

I have merged the previous Pull Request on GitHub.
Now start ONLY **Task [X+1]: [Exact Title of Task X+1]**:
1. First, run `git checkout main && git pull origin main` to sync the merged PR, then create and switch to the dedicated branch for Task [X+1] following Phase A of `tdd-commit-gate`.
2. Implement all requirements for Task [X+1] following strict TDD and our architecture specification.
3. Run `uv run ruff check --fix .`, `uv run ruff format .`, and `uv run pytest -v` until 100% pass.
4. Execute Phases B, C, D, and E of `tdd-commit-gate` (update `TASKS.md`, commit, push the branch to `origin` so I can open the PR on GitHub, and generate the prompt for Task [X+2]).
---