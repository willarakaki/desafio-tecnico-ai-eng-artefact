# Mandatory Workspace References
@[Architecture Specification](docs/architecture.md)
@TASKS.md
@.agents/skills/tdd-commit-gate/SKILL.md

# Agent Engineering Guidelines (AGENTS.md)

## Project Context
Production-grade prototype for the Artefact Brazil AI Engineer technical challenge.
- **Client:** Empório da Música (physical store and e-commerce for musical instruments located in Campo Grande/MS, Brazil).
- **Objective:** Build an intelligent customer support agent in Python featuring deterministic routing between structured tabular data (DuckDB over CSVs) and unstructured policy text (Hybrid RAG over PDF), integrated with runtime security guardrails, Context-Aware Semantic Cache, Human-in-the-Loop (HITL) escalation, and FinOps telemetry.
- **Language Rule:** All internal code, class/function names, and technical specs must follow clean English standards, **EXCEPT** the customer-facing System Prompt, persona responses, domain filter values (e.g., `"violão"`, `"Campo Grande"`), and original dataset filenames (`data/políticas_da_loja.pdf`), which **MUST remain in Brazilian Portuguese**.

## Non-Negotiable Engineering Rules
1. **Spec-Driven & Data-First Workflow:** Before writing any SQL query or RAG chunking logic, inspect the actual files inside `data/` (`*.csv` and `políticas_da_loja.pdf`) to map exact column names, data types, formatting conventions, and document headings.
2. **Test-Driven Development (TDD):** Every tool, guardrail, cache rule, and routing mechanism must have automated `pytest` tests passing before moving to the next task in `TASKS.md`.
3. **Zero-Friction Execution:** The repository must run out-of-the-box via `uv`/`pip` or `Docker` by simply setting an API key in `.env`, without requiring manual downloads of gated Hugging Face weights or heavy system dependencies.
4. **No Raw LLM-Generated SQL:** The LLM must never generate free-form SQL strings. Expose exclusively Python tools strictly validated with Pydantic v2 that execute parameterized SQL queries against DuckDB.
5. **Privacy (LGPD) & Context-Aware Caching:** Never store interactions containing CPF, order IDs, customer emails, or live transactional order/stock lookups in the Semantic Cache. Apply PII masking to all persistent logs and telemetry outputs.

## Python Coding Standards
- **Runtime:** Python 3.11+.
- **Typing:** Mandatory type hints on all function signatures and Pydantic v2 models.
- **LLM Instantiation:** Direct instantiation of LLM SDKs across modules is strictly forbidden. Every model or embedding call must go through `src/agent/llm_factory.py` (`LLMFactory`).
- **Resilient Error Handling:** Tools must never raise unhandled exceptions to the user interface; they must return structured fallback messages so the agent can respond empathetically and suggest alternatives.

## Git & Commit Discipline
- Execute strictly one task from `TASKS.md` at a time.
- **Branch-per-Task Workflow:** Before writing code for any task, create a dedicated task branch (`git checkout -b <branch-name>`) as defined in `@.agents/skills/tdd-commit-gate/SKILL.md`.
- Upon finishing a task and passing `pytest -v`, trigger the `tdd-commit-gate` skill to mark `[x]` in `TASKS.md`, commit on the task branch, merge back into `main` with `--no-ff`, push to `origin/main`, and print the ready-to-copy prompt for the next task.
- Never squash the entire project into a single monolithic commit.